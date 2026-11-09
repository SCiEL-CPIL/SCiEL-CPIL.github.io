import time
import itertools
import random

N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # 1. Precompute edge indices and cycles
    edge_to_idx = {}
    idx_to_edge = []
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = len(idx_to_edge)
            idx_to_edge.append((i, j))
    
    cycles = []
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    
    for v1, v2, v3, v4 in itertools.combinations(range(N), 4):
        e12 = edge_to_idx[(v1, v2)]
        e13 = edge_to_idx[(v1, v3)]
        e14 = edge_to_idx[(v1, v4)]
        e23 = edge_to_idx[(v2, v3)]
        e24 = edge_to_idx[(v2, v4)]
        e34 = edge_to_idx[(v3, v4)]
        
        c1 = [e12, e23, e34, e14]
        cycles.append(c1)
        for e in c1:
            edge_to_cycles[e].append(len(cycles) - 1)
            
        c2 = [e12, e24, e34, e13]
        cycles.append(c2)
        for e in c2:
            edge_to_cycles[e].append(len(cycles) - 1)
            
        c3 = [e13, e23, e24, e14]
        cycles.append(c3)
        for e in c3:
            edge_to_cycles[e].append(len(cycles) - 1)

    # 2. Generate 1-factorization of K16
    factors = [[] for _ in range(N - 1)]
    
    for k in range(N - 1):
        factors[k].append((15, k))
        for i in range(1, N // 2):
            u = (k + i) % (N - 1)
            v = (k - i) % (N - 1)
            factors[k].append((u, v))
            
    # Map factor index to list of edge indices
    factor_edges = [[] for _ in range(N - 1)]
    colors = [0] * NUM_EDGES
    for k in range(N - 1):
        for u, v in factors[k]:
            if u > v:
                u, v = v, u
            idx = edge_to_idx[(u, v)]
            colors[idx] = k
            factor_edges[k].append(idx)
            
    used_colors = set(colors)
    best_num_colors = len(used_colors)
    best_colors = colors[:]
    
    # Helper to check if changing edge e to color c is valid
    def is_valid_change(e, c, current_colors):
        if c == current_colors[e]:
            return True
        for c_idx in edge_to_cycles[e]:
            cycle_edges = cycles[c_idx]
            col_set = set()
            for ed in cycle_edges:
                if ed == e:
                    col_set.add(c)
                else:
                    col_set.add(current_colors[ed])
            if len(col_set) == 4:
                return False
        return True

    # 3. Structured Splitting
    # Try to split each factor into 2 parts
    random.seed(42)
    
    for k in range(N - 1):
        if time.time() > deadline:
            break
        edges_in_factor = factor_edges[k][:]
        random.shuffle(edges_in_factor)
        
        # Try to assign a new color to the first edge
        if edges_in_factor:
            e1 = edges_in_factor[0]
            # Pick a new color
            unused = [c for c in range(120) if c not in used_colors]
            if unused:
                c_new = random.choice(unused)
                if is_valid_change(e1, c_new, colors):
                    colors[e1] = c_new
                    used_colors.add(c_new)
                    if len(used_colors) > best_num_colors:
                        best_num_colors = len(used_colors)
                        best_colors = colors[:]
                        
        # Try to assign a new color to the second edge
        if len(edges_in_factor) > 1:
            e2 = edges_in_factor[1]
            unused = [c for c in range(120) if c not in used_colors]
            if unused:
                c_new = random.choice(unused)
                if is_valid_change(e2, c_new, colors):
                    colors[e2] = c_new
                    used_colors.add(c_new)
                    if len(used_colors) > best_num_colors:
                        best_num_colors = len(used_colors)
                        best_colors = colors[:]
                        
    # 4. Random Local Search
    while time.time() < deadline:
        e = random.randint(0, NUM_EDGES - 1)
        
        if len(used_colors) < 120:
            unused = [c for c in range(120) if c not in used_colors]
            if unused:
                c = random.choice(unused)
            else:
                c = random.randint(0, 119)
        else:
            c = random.randint(0, 119)
            
        if c == colors[e]:
            continue
            
        if is_valid_change(e, c, colors):
            colors[e] = c
            used_colors.add(c)
            if len(used_colors) > best_num_colors:
                best_num_colors = len(used_colors)
                best_colors = colors[:]
                
    return best_colors

