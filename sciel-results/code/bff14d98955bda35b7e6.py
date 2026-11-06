import time
import itertools
import random

N = 16
NUM_EDGES = N * (N - 1) // 2

def get_edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def precompute_cycles():
    cycles = []
    verts = range(N)
    for combo in itertools.combinations(verts, 4):
        a, b, c, d = combo
        c1 = (get_edge_index(a, b), get_edge_index(b, c), get_edge_index(c, d), get_edge_index(d, a))
        c2 = (get_edge_index(a, b), get_edge_index(b, d), get_edge_index(d, c), get_edge_index(c, a))
        c3 = (get_edge_index(a, c), get_edge_index(c, b), get_edge_index(b, d), get_edge_index(d, a))
        cycles.append(c1)
        cycles.append(c2)
        cycles.append(c3)
    return cycles

CYCLES = precompute_cycles()

def all_distinct(a, b, c, d):
    return a != b and a != c and a != d and b != c and b != d and c != d

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Start with min(i,j) coloring: color(i,j) = i for i < j
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            idx = get_edge_index(i, j)
            colors[idx] = i
    
    # Precompute edge-to-cycle mapping
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(CYCLES):
        for e in cycle:
            edge_to_cycles[e].append(ci)
    
    best_colors = colors[:]
    best_count = len(set(colors))
    
    # Local search
    for _ in range(200000):
        if time.time() > deadline:
            break
        
        e = random.randint(0, NUM_EDGES - 1)
        orig = colors[e]
        new_c = random.randint(0, 119)
        if new_c == orig:
            continue
        
        # Check validity for all cycles containing e
        valid = True
        for ci in edge_to_cycles[e]:
            cycle = CYCLES[ci]
            c0 = new_c if cycle[0] == e else colors[cycle[0]]
            c1 = new_c if cycle[1] == e else colors[cycle[1]]
            c2 = new_c if cycle[2] == e else colors[cycle[2]]
            c3 = new_c if cycle[3] == e else colors[cycle[3]]
            if all_distinct(c0, c1, c2, c3):
                valid = False
                break
        
        if valid:
            colors[e] = new_c
            cnt = len(set(colors))
            if cnt > best_count:
                best_count = cnt
                best_colors = colors[:]
    
    return best_colors
