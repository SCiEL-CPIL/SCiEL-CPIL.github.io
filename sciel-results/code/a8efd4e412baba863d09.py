import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

# Precompute edge indices
EDGE_LIST = []
EDGE_TO_IDX = {}
for i in range(N):
    for j in range(i + 1, N):
        idx = len(EDGE_LIST)
        EDGE_LIST.append((i, j))
        EDGE_TO_IDX[(i, j)] = idx

# Precompute all 4-cycles
FOUR_CYCLES = []  # list of tuples of 4 edge indices
for quad in itertools.combinations(range(N), 4):
    a, b, c, d = quad
    # Three 4-cycles on {a,b,c,d}:
    # 1: a-b, b-c, c-d, d-a
    # 2: a-b, b-d, d-c, c-a
    # 3: a-c, c-b, b-d, d-a
    cycles = [
        [(a,b), (b,c), (c,d), (d,a)],
        [(a,b), (b,d), (d,c), (c,a)],
        [(a,c), (c,b), (b-d, d-a)], # Wait, typo in previous thought, correcting here
    ]
    # Correct cycles:
    # 1: (a,b), (b,c), (c,d), (d,a)
    # 2: (a,b), (b,d), (d,c), (c,a)
    # 3: (a,c), (c,b), (b,d), (d,a)
    cycles = [
        [(a,b), (b,c), (c,d), (d,a)],
        [(a,b), (b,d), (d,c), (c,a)],
        [(a,c), (c,b), (b,d), (d,a)],
    ]
    for cyc in cycles:
        edge_idxs = []
        for (u, v) in cyc:
            if u > v:
                u, v = v, u
            edge_idxs.append(EDGE_TO_IDX[(u, v)])
        FOUR_CYCLES.append(tuple(edge_idxs))

# For each edge, store the other 3 edges for each cycle containing it
edge_cycle_neighbors = [[] for _ in range(NUM_EDGES)]
for cyc in FOUR_CYCLES:
    e0, e1, e2, e3 = cyc
    edge_cycle_neighbors[e0].append([e1, e2, e3])
    edge_cycle_neighbors[e1].append([e0, e2, e3])
    edge_cycle_neighbors[e2].append([e0, e1, e3])
    edge_cycle_neighbors[e3].append([e0, e1, e2])

def is_valid(colors):
    for cyc in FOUR_CYCLES:
        if len({colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]}) == 4:
            return False
    return True

def try_recolor(colors, e, new_c):
    old_c = colors[e]
    if old_c == new_c:
        return True
    
    colors[e] = new_c
    
    for neighbors in edge_cycle_neighbors[e]:
        c1, c2, c3 = neighbors
        if len({colors[c1], colors[c2], colors[c3], new_c}) == 4:
            colors[e] = old_c
            return False
            
    return True

def greedy_construct(seed):
    random.seed(seed)
    colors = [0] * NUM_EDGES
    used_colors = {0}
    
    edges = list(range(NUM_EDGES))
    random.shuffle(edges)
    
    for e in edges:
        unused = [c for c in range(MAX_COLOR) if c not in used_colors]
        random.shuffle(unused)
        
        candidates = unused + [colors[e]]
        
        for c in candidates:
            if try_recolor(colors, e, c):
                used_colors.add(c)
                break
                
    return colors

def local_search_v2(colors, time_limit, seed):
    random.seed(seed)
    current_colors = colors[:]
    freq = [0] * MAX_COLOR
    for c in current_colors:
        freq[c] += 1
        
    distinct_count = sum(1 for f in freq if f > 0)
    best_colors = current_colors[:]
    best_distinct = distinct_count
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        e = random.randint(0, NUM_EDGES - 1)
        
        if distinct_count < MAX_COLOR:
            unused = [c for c in range(MAX_COLOR) if freq[c] == 0]
            if random.random() < 0.6:
                new_c = random.choice(unused)
            else:
                new_c = random.randint(0, MAX_COLOR - 1)
        else:
            new_c = random.randint(0, MAX_COLOR - 1)
            
        if new_c == current_colors[e]:
            continue
            
        old_c = current_colors[e]
        
        if try_recolor(current_colors, e, new_c):
            freq[old_c] -= 1
            freq[new_c] += 1
            
            if freq[old_c] == 0:
                distinct_count -= 1
            if freq[new_c] == 1:
                distinct_count += 1
                
            if distinct_count > best_distinct:
                best_distinct = distinct_count
                best_colors = current_colors[:]
            elif distinct_count < best_distinct:
                try_recolor(current_colors, e, old_c)
                freq[old_c] -= 1
                freq[new_c] += 1
                if freq[old_c] == 0:
                    distinct_count += 1
                if freq[new_c] == 1:
                    distinct_count -= 1
                    
    return best_colors

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_overall = [0] * NUM_EDGES
    best_score_overall = 1
    
    num_restarts = 100
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        seed = 42 + i
        
        g_colors = greedy_construct(seed)
        g_score = len(set(g_colors))
        
        if g_score > best_score_overall:
            best_score_overall = g_score
            best_overall = g_colors[:]
            
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        time_per_restart = remaining / (num_restarts - i)
        
        ls_time = max(0.1, time_per_restart * 0.5)
        
        ls_colors = local_search_v2(g_colors, ls_time, seed)
        ls_score = len(set(ls_colors))
        
        if ls_score > best_score_overall:
            best_score_overall = ls_score
            best_overall = ls_colors[:]
            
    if not is_valid(best_overall):
        return [0] * NUM_EDGES
        
    return best_overall
