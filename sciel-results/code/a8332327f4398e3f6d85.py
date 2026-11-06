import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute edge indices
edge_to_idx = {}
edges = []
for i in range(N):
    for j in range(i + 1, N):
        edge_to_idx[(i, j)] = len(edges)
        edges.append((i, j))

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return edge_to_idx[(i, j)]

# Precompute all 4-cycles: each is a tuple of 4 edge indices
all_cycles = []
for a, b, c, d in combinations(range(N), 4):
    # Cycle 1: a-b-c-d-a
    e1 = (edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a))
    all_cycles.append(e1)
    # Cycle 2: a-b-d-c-a
    e2 = (edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a))
    all_cycles.append(e2)
    # Cycle 3: a-c-b-d-a
    e3 = (edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a))
    all_cycles.append(e3)

# For each edge, which cycles contain it
edge_to_cycles = [[] for _ in range(NUM_EDGES)]
for idx, cycle in enumerate(all_cycles):
    for e in cycle:
        edge_to_cycles[e].append(idx)

# For each cycle, the 4 edges
cycle_edges = [list(c) for c in all_cycles]

def is_valid(colors):
    """Check if no 4-cycle has 4 distinct colors."""
    for cycle in all_cycles:
        c0, c1, c2, c3 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def count_violations(colors):
    """Count number of rainbow 4-cycles."""
    count = 0
    for cycle in all_cycles:
        c0, c1, c2, c3 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
        if len({c0, c1, c2, c3}) == 4:
            count += 1
    return count

def check_edge_change(colors, edge, new_color):
    """Check if setting edge to new_color would violate any constraint."""
    old_color = colors[edge]
    if old_color == new_color:
        return True
    colors[edge] = new_color
    valid = True
    for cycle_idx in edge_to_cycles[edge]:
        cycle = all_cycles[cycle_idx]
        c0, c1, c2, c3 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
        if len({c0, c1, c2, c3}) == 4:
            valid = False
            break
    colors[edge] = old_color
    return valid

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    # Strategy 1: Structured initial coloring based on modular arithmetic
    # Try several structured colorings as starting points
    structured_starts = []
    
    # Coloring based on sum mod k
    for k in range(3, 12):
        colors = [0] * NUM_EDGES
        for idx, (i, j) in enumerate(edges):
            colors[idx] = (i + j) % k
        structured_starts.append(colors[:])
    
    # Coloring based on difference mod k
    for k in range(3, 12):
        colors = [0] * NUM_EDGES
        for idx, (i, j) in enumerate(edges):
            colors[idx] = (j - i) % k
        structured_starts.append(colors[:])
    
    # Coloring based on product mod k
    for k in range(3, 12):
        colors = [0] * NUM_EDGES
        for idx, (i, j) in enumerate(edges):
            colors[idx] = (i * j) % k
        structured_starts.append(colors[:])
    
    # Random starts
    for seed in range(20):
        random.seed(seed)
        colors = [random.randint(0, 119) for _ in range(NUM_EDGES)]
        structured_starts.append(colors)
    
    # For each starting coloring, try to improve it
    for start_idx, initial in enumerate(structured_starts):
        if time.time() > deadline - 3.0:
            break
        
        colors = initial[:]
        
        # First, fix any violations using local search
        for iteration in range(500):
            if time.time() > deadline - 3.0:
                break
            
            violations = 0
            for cycle in all_cycles:
                c0, c1, c2, c3 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
                if len({c0, c1, c2, c3}) == 4:
                    violations += 1
            
            if violations == 0:
                break
            
            # Pick a random violating cycle and try to fix it
            fixed = False
            for _ in range(100):
                cycle_idx = random.randint(0, len(all_cycles) - 1)
                cycle = all_cycles[cycle_idx]
                c0, c1, c2, c3 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
                
                if len({c0, c1, c2, c3}) != 4:
                    continue
                
                # Try to change one edge's color to match another
                for pos in range(4):
                    edge = cycle[pos]
                    target_colors = [colors[cycle[q]] for q in range(4) if q != pos]
                    for tc in target_colors:
                        if check_edge_change(colors, edge, tc):
                            colors[edge] = tc
                            fixed = True
                            break
                    if fixed:
                        break
                if fixed:
                    break
            
            if not fixed:
                # Try random changes
                e = random.randint(0, NUM_EDGES - 1)
                new_c = random.randint(0, 119)
                if check_edge_change(colors, e, new_c):
                    colors[e] = new_c
        
        # Now try to add more colors
        used_colors = set(colors)
        next_color = max(used_colors) + 1 if used_colors else 0
        
        for _ in range(200):
            if time.time() > deadline - 3.0:
                break
            if next_color >= 120:
                break
            
            # Pick a random edge that has a color shared by many edges
            e = random.randint(0, NUM_EDGES - 1)
            old_c = colors[e]
            
            # Try assigning a new color
            if check_edge_change(colors, e, next_color):
                colors[e] = next_color
                used_colors.add(next_color)
                next_color += 1
        
        cur_count = len(used_colors)
        if cur_count > best_count:
            best_count = cur_count
            best_colors = colors[:]
            if best_count >= 120:
                break
    
    # Additional pass: try to squeeze out more colors from the best solution found
    if time.time() < deadline - 5.0:
        colors = best_colors[:]
        used_colors = set(colors)
        next_color = max(used_colors) + 1 if used_colors else 0
        
        # Multiple rounds of trying to add colors
        for round_num in range(10):
            if time.time() > deadline - 3.0:
                break
            if next_color >= 120:
                break
            
            progress = False
            for _ in range(500):
                if time.time() > deadline - 3.0:
                    break
                if next_color >= 120:
                    break
                
                e = random.randint(0, NUM_EDGES - 1)
                old_c = colors[e]
                
                if check_edge_change(colors, e, next_color):
                    colors[e] = next_color
                    used_colors.add(next_color)
                    next_color += 1
                    progress = True
            
            if not progress:
                break
        
        cur_count = len(used_colors)
        if cur_count > best_count:
            best_count = cur_count
            best_colors = colors[:]
    
    return best_colors
