import time
import random
import sys

N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    # Precompute all 4-cycles and which edges they use
    from itertools import combinations
    
    four_cycles = []
    for verts in combinations(range(N), 4):
        a, b, c, d = verts
        # Three 4-cycles on 4 vertices:
        # Cycle 1: a-b-c-d-a
        e1 = [edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(a,d)]
        # Cycle 2: a-b-d-c-a  
        e2 = [edge_idx(a,b), edge_idx(b,d), edge_idx(c,d), edge_idx(a,c)]
        # Cycle 3: a-c-b-d-a
        e3 = [edge_idx(a,c), edge_idx(b,c), edge_idx(b,d), edge_idx(a,d)]
        
        for cycle_edges in [e1, e2, e3]:
            four_cycles.append(cycle_edges)
    
    # For each edge, precompute which cycles contain it
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(four_cycles):
        for e in cycle:
            edge_to_cycles[e].append(ci)
    
    MAX_COLOR = 120
    
    # Strategy: Use a structured initial coloring based on vertex partition
    # Partition 16 vertices into 4 groups of 4
    # Within-group edges get one color, between-group edges get another pattern
    
    def make_structured_coloring():
        """Create a coloring based on 4 groups of 4 vertices."""
        colors = [0] * NUM_EDGES
        # Group assignment: vertex v -> group v // 4
        for i in range(N - 1):
            for j in range(i + 1, N):
                gi = i // 4
                gj = j // 4
                if gi == gj:
                    # Same group: use color based on position within group
                    pi = i % 4
                    pj = j % 4
                    colors[edge_idx(i, j)] = 100 + pi * 4 + pj
                else:
                    # Different groups: use color based on group pair
                    colors[edge_idx(i, j)] = gi * 4 + gj
        return colors
    
    def is_valid(colors):
        for cycle in four_cycles:
            if len(set(colors[e] for e in cycle)) == 4:
                return False
        return True
    
    def count_distinct(colors):
        return len(set(colors))
    
    # Try multiple structured initializations
    def make_modular_coloring(mod_base):
        """Create a modular arithmetic based coloring."""
        colors = [0] * NUM_EDGES
        for i in range(N - 1):
            for j in range(i + 1, N):
                colors[edge_idx(i, j)] = (i * mod_base + j) % MAX_COLOR
        return colors
    
    def make_bipartite_coloring():
        """Color based on parity of vertices."""
        colors = [0] * NUM_EDGES
        for i in range(N - 1):
            for j in range(i + 1, N):
                if i % 2 == j % 2:
                    colors[edge_idx(i, j)] = i // 2 * 8 + j // 2
                else:
                    colors[edge_idx(i, j)] = 80 + i * 2 + j
        return colors
    
    # Generate candidate initial colorings
    candidates = []
    candidates.append(make_structured_coloring())
    candidates.append(make_modular_coloring(3))
    candidates.append(make_modular_coloring(5))
    candidates.append(make_modular_coloring(7))
    candidates.append(make_bipartite_coloring())
    
    # Also try transitive
    transitive = [0] * NUM_EDGES
    for i in range(N - 1):
        for j in range(i + 1, N):
            transitive[edge_idx(i, j)] = i
    candidates.append(transitive)
    
    # Pick the best valid candidate
    best_init = None
    best_init_count = 0
    for cand in candidates:
        if is_valid(cand):
            cnt = count_distinct(cand)
            if cnt > best_init_count:
                best_init_count = cnt
                best_init = cand[:]
    
    if best_init is None:
        best_init = transitive[:]
        best_init_count = count_distinct(best_init)
    
    best_colors = best_init[:]
    best_count = best_init_count
    
    # Local search with constraint-aware moves
    def try_change(colors, edge, new_color):
        old_color = colors[edge]
        if old_color == new_color:
            return True
        for ci in edge_to_cycles[edge]:
            cycle = four_cycles[ci]
            cycle_colors = set()
            for e in cycle:
                if e == edge:
                    cycle_colors.add(new_color)
                else:
                    cycle_colors.add(colors[e])
            if len(cycle_colors) == 4:
                return False
        return True
    
    random.seed(12345)
    
    current = best_colors[:]
    current_count = best_count
    
    # Phase 1: Greedy improvement - try to add new colors
    while time.time() < deadline - 10:
        used_colors = set(current)
        unused_colors = [c for c in range(MAX_COLOR) if c not in used_colors]
        
        if not unused_colors:
            break
        
        edges = list(range(NUM_EDGES))
        random.shuffle(edges)
        
        improved = False
        for edge in edges:
            if time.time() > deadline - 10:
                break
            
            old_color = current[edge]
            
            # Try unused colors in random order
            shuffled_unused = unused_colors[:]
            random.shuffle(shuffled_unused)
            for new_color in shuffled_unused:
                if try_change(current, edge, new_color):
                    current[edge] = new_color
                    new_count = current_count + 1
                    if new_count > best_count:
                        best_count = new_count
                        best_colors = current[:]
                    current_count = new_count
                    improved = True
                    break
        
        if not improved:
            # Phase 2: Try to swap colors to free up space for new colors
            any_swap = False
            for _ in range(200):
                if time.time() > deadline - 10:
                    break
                e1 = random.randint(0, NUM_EDGES - 1)
                e2 = random.randint(0, NUM_EDGES - 1)
                if e1 == e2:
                    continue
                c1 = current[e1]
                c2 = current[e2]
                if c1 == c2:
                    continue
                
                # Try swapping
                current[e1] = c2
                current[e2] = c1
                if is_valid(current):
                    new_count = count_distinct(current)
                    if new_count > best_count:
                        best_count = new_count
                        best_colors = current[:]
                    current_count = new_count
                    any_swap = True
                else:
                    # Revert
                    current[e1] = c1
                    current[e2] = c2
            
            if not any_swap:
                # Phase 3: Random perturbation
                perturbed = False
                for _ in range(50):
                    if time.time() > deadline - 10:
                        break
                    edge = random.randint(0, NUM_EDGES - 1)
                    new_color = random.randint(0, MAX_COLOR - 1)
                    if try_change(current, edge, new_color):
                        current[edge] = new_color
                        current_count = count_distinct(current)
                        if current_count > best_count:
                            best_count = current_count
                            best_colors = current[:]
                        perturbed = True
                        break
                if not perturbed:
                    break
    
    # Final verification
    if not is_valid(best_colors):
        # Fallback
        for i in range(N - 1):
            for j in range(i + 1, N):
                best_colors[edge_idx(i, j)] = i
    
    return best_colors
