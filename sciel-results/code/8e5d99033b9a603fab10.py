import time
import random
import sys
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    def get_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    # Precompute all 4-cycles
    all_4cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        all_4cycles.append((get_idx(a,b), get_idx(b,c), get_idx(c,d), get_idx(d,a)))
        all_4cycles.append((get_idx(a,b), get_idx(b,d), get_idx(d,c), get_idx(c,a)))
        all_4cycles.append((get_idx(a,c), get_idx(c,b), get_idx(b,d), get_idx(d,a)))
    
    # For each edge, store the 4-cycles it participates in
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(all_4cycles):
        for e in cycle:
            edge_to_cycles[e].append(ci)
    
    def is_valid(colors):
        for cycle in all_4cycles:
            c1, c2, c3, c4 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
            if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
                return False
        return True
    
    def check_edge_change(colors, edge_idx, new_color):
        for ci in edge_to_cycles[edge_idx]:
            cycle = all_4cycles[ci]
            c1 = colors[cycle[0]] if cycle[0] != edge_idx else new_color
            c2 = colors[cycle[1]] if cycle[1] != edge_idx else new_color
            c3 = colors[cycle[2]] if cycle[2] != edge_idx else new_color
            c4 = colors[cycle[3]] if cycle[3] != edge_idx else new_color
            if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
                return False
        return True
    
    # Backtracking with MRV: assign colors to edges, trying new colors first
    # to maximize diversity
    
    def backtrack(colors, assigned, used_colors, depth):
        if time.time() > deadline:
            return colors, len(used_colors)
        
        if depth == NUM_EDGES:
            return colors[:], len(used_colors)
        
        # MRV: find unassigned edge with most constraints
        # For simplicity, use a fixed order but prioritize edges that appear in many cycles
        # Actually, let's use a smarter approach: pick the edge that would be most constrained
        
        # Try to assign a new color to an unassigned edge
        # Find an unassigned edge
        unassigned = [i for i in range(NUM_EDGES) if colors[i] == -1]
        if not unassigned:
            return colors[:], len(used_colors)
        
        # Pick edge with most 4-cycle constraints among unassigned
        # Precompute constraint count
        max_constraints = -1
        chosen = unassigned[0]
        for e in unassigned:
            if len(edge_to_cycles[e]) > max_constraints:
                max_constraints = len(edge_to_cycles[e])
                chosen = e
        
        # Try new colors first (to maximize diversity)
        unused = [c for c in range(120) if c not in used_colors]
        random.shuffle(unused)
        
        # Also try existing colors
        existing = list(used_colors)
        random.shuffle(existing)
        
        for color in unused + existing:
            if time.time() > deadline:
                break
            colors[chosen] = color
            if check_edge_change(colors, chosen, color):
                used_colors.add(color)
                result_colors, result_count = backtrack(colors, assigned + 1, used_colors, depth + 1)
                used_colors.discard(color)
                if result_count > 0:
                    return result_colors, result_count
            colors[chosen] = -1
        
        return colors, len(used_colors)
    
    # Better approach: greedy with local search, but smarter
    # Start with all zeros, then greedily introduce new colors
    
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    # Phase 1: Greedy introduction with constraint checking
    colors = [0] * NUM_EDGES
    used_colors = {0}
    
    random.seed(42)
    
    # Try to introduce new colors greedily
    for attempt in range(500000):
        if time.time() > deadline - 5:
            break
        
        # Pick random edge
        edge_idx = random.randint(0, NUM_EDGES - 1)
        
        # Try a new color
        unused = [c for c in range(120) if c not in used_colors]
        if not unused:
            break
        
        new_color = random.choice(unused)
        
        if check_edge_change(colors, edge_idx, new_color):
            colors[edge_idx] = new_color
            used_colors.add(new_color)
            if len(used_colors) > best_count:
                best_count = len(used_colors)
                best_colors = colors[:]
    
    # Phase 2: Local search - try to swap and introduce new colors
    for attempt in range(1000000):
        if time.time() > deadline - 3:
            break
        
        # Pick two random edges with different colors
        e1 = random.randint(0, NUM_EDGES - 1)
        e2 = random.randint(0, NUM_EDGES - 1)
        if e1 == e2 or colors[e1] == colors[e2]:
            continue
        
        c1, c2 = colors[e1], colors[e2]
        colors[e1], colors[e2] = c2, c1
        
        if is_valid(colors):
            # Try to introduce new color
            unused = [c for c in range(120) if c not in used_colors]
            if unused:
                for e in [e1, e2]:
                    nc = random.choice(unused)
                    if check_edge_change(colors, e, nc):
                        colors[e] = nc
                        used_colors.add(nc)
                        break
        else:
            colors[e1], colors[e2] = c1, c2
        
        if len(used_colors) > best_count:
            best_count = len(used_colors)
            best_colors = colors[:]
    
    # Phase 3: More aggressive search - try to find edges that can take new colors
    # by temporarily removing constraints
    for attempt in range(500000):
        if time.time() > deadline - 2:
            break
        
        # Find an edge that's currently color 0 (or most common color)
        # and try to change it to a new color
        edge_idx = random.randint(0, NUM_EDGES - 1)
        
        unused = [c for c in range(120) if c not in used_colors]
        if not unused:
            break
        
        new_color = random.choice(unused)
        
        if check_edge_change(colors, edge_idx, new_color):
            colors[edge_idx] = new_color
            used_colors.add(new_color)
            if len(used_colors) > best_count:
                best_count = len(used_colors)
                best_colors = colors[:]
    
    # Final verification
    if not is_valid(best_colors):
        best_colors = [0] * NUM_EDGES
    
    return best_colors
