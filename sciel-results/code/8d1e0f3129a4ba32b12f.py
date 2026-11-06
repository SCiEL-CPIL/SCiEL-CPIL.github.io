import time
import random
import sys
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

def construct():
    start_time = time.time()
    deadline = start_time + 58.0

    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

    # Precompute all 4-cycles and their edge lists
    four_cycles = []
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]

    for verts in combinations(range(N), 4):
        a, b, c, d = verts
        e1 = [edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(a,d)]
        e2 = [edge_idx(a,b), edge_idx(b,d), edge_idx(c,d), edge_idx(a,c)]
        e3 = [edge_idx(a,c), edge_idx(b,c), edge_idx(b,d), edge_idx(a,d)]

        for cycle_edges in [e1, e2, e3]:
            four_cycles.append(cycle_edges)
            for e in cycle_edges:
                others = [x for x in cycle_edges if x != e]
                edge_to_cycles[e].append(others)

    def is_valid(colors):
        for cycle in four_cycles:
            if len(set(colors[e] for e in cycle)) == 4:
                return False
        return True

    def try_change(colors, edge, new_color):
        old_color = colors[edge]
        if old_color == new_color:
            return True
        for others in edge_to_cycles[edge]:
            cycle_colors = set()
            cycle_colors.add(new_color)
            for e in others:
                cycle_colors.add(colors[e])
            if len(cycle_colors) == 4:
                return False
        return True

    def get_conflict_colors(colors, edge):
        """Return set of colors that would create a 4-cycle with 4 distinct colors if assigned to edge."""
        conflicts = set()
        for others in edge_to_cycles[edge]:
            other_colors = set(colors[e] for e in others)
            if len(other_colors) >= 3:
                # If we assign a new color, we need it to be in other_colors to avoid 4 distinct
                for c in other_colors:
                    conflicts.add(c)
        return conflicts

    # Structural approach: partition vertices into groups and use structured coloring
    # Then systematically introduce new colors
    
    random.seed(123)
    
    # Start with a simple valid coloring: all edges same color
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    # Phase 1: Build up a coloring with many colors using a greedy approach
    # Process edges in an order that allows maximum color diversity
    # Use a heuristic: edges that are in fewer 4-cycles first
    
    edge_constraint_count = [len(edge_to_cycles[e]) for e in range(NUM_EDGES)]
    edges_order = sorted(range(NUM_EDGES), key=lambda e: edge_constraint_count[e])
    
    used_colors = {0}
    
    for edge in edges_order:
        if time.time() > deadline - 20:
            break
        
        # Find a color that doesn't conflict
        # Try to use a new color if possible
        best_new_color = None
        
        # First try to find a new color
        for c in range(MAX_COLOR):
            if c not in used_colors:
                if try_change(best_colors, edge, c):
                    best_new_color = c
                    break
        
        if best_new_color is not None:
            best_colors[edge] = best_new_color
            used_colors.add(best_new_color)
        else:
            # Must reuse an existing color
            for c in used_colors:
                if try_change(best_colors, edge, c):
                    best_colors[edge] = c
                    break
    
    best_count = len(set(best_colors))
    
    # Phase 2: Iterative improvement - try to replace colors with new ones
    # Use a more targeted approach: for each edge, check if it can take a new color
    # after some other edges change
    
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        
        improved = False
        used = set(best_colors)
        unused = [c for c in range(MAX_COLOR) if c not in used]
        
        if not unused:
            break
        
        # Try each edge in random order
        edges_to_try = list(range(NUM_EDGES))
        random.shuffle(edges_to_try)
        
        for edge in edges_to_try:
            if time.time() > deadline - 5:
                break
            
            # Try to assign a new color to this edge
            for new_color in unused:
                if try_change(best_colors, edge, new_color):
                    old_color = best_colors[edge]
                    best_colors[edge] = new_color
                    used.add(new_color)
                    unused.remove(new_color)
                    if old_color not in set(best_colors):
                        used.discard(old_color)
                        unused.append(old_color)
                    improved = True
                    break
        
        if not improved:
            # Try swaps to free up space
            if random.random() < 0.5:
                e1 = random.randint(0, NUM_EDGES - 1)
                e2 = random.randint(0, NUM_EDGES - 1)
                if e1 != e2:
                    c1 = best_colors[e1]
                    c2 = best_colors[e2]
                    if c1 != c2:
                        # Try swap
                        best_colors[e1] = c2
                        best_colors[e2] = c1
                        if is_valid(best_colors):
                            # Try to add new color after swap
                            used = set(best_colors)
                            unused = [c for c in range(MAX_COLOR) if c not in used]
                            if unused:
                                e3 = random.randint(0, NUM_EDGES - 1)
                                for nc in unused:
                                    if try_change(best_colors, e3, nc):
                                        best_colors[e3] = nc
                                        break
                        else:
                            # Revert
                            best_colors[e1] = c1
                            best_colors[e2] = c2
    
    # Final verification
    if not is_valid(best_colors):
        for i in range(N - 1):
            for j in range(i + 1, N):
                best_colors[edge_idx(i, j)] = i
    
    return best_colors
