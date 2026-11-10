import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2

def precompute():
    c4s = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    c4s.append((edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)))
                    c4s.append((edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)))
                    c4s.append((edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)))
    
    edge_c4s = [[] for _ in range(NUM_EDGES)]
    for ci, c4 in enumerate(c4s):
        for pi, ei in enumerate(c4):
            edge_c4s[ei].append((ci, pi))
    return c4s, edge_c4s

C4S, EDGE_C4S = precompute()

def construct():
    deadline = time.time() + 55
    
    # Structural starting point: 4 groups of 4
    group = [i // 4 for i in range(N)]
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i+1, N):
            gi, gj = group[i], group[j]
            colors[edge_idx(i,j)] = gi * 4 + gj if gi < gj else gj * 4 + gi
    # This gives colors 0..10 (10 distinct colors)
    
    best_colors = colors[:]
    best_num = len(set(colors))
    used = set(colors)
    
    random.seed(777)
    
    # Try to add more colors
    for it in range(100):
        if time.time() > deadline - 5:
            break
        order = list(range(NUM_EDGES))
        random.shuffle(order)
        improved = False
        for ei in order:
            if time.time() > deadline - 5:
                break
            cur = colors[ei]
            for nc in range(120):
                if nc in used:
                    continue
                valid = True
                for ci, pi in EDGE_C4S[ei]:
                    c4 = C4S[ci]
                    c0 = nc if pi == 0 else colors[c4[0]]
                    c1 = nc if pi == 1 else colors[c4[1]]
                    c2 = nc if pi == 2 else colors[c4[2]]
                    c3 = nc if pi == 3 else colors[c4[3]]
                    if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                        valid = False
                        break
                if valid:
                    colors[ei] = nc
                    used.add(nc)
                    improved = True
                    break
        if len(used) > best_num:
            best_num = len(used)
            best_colors = colors[:]
        if not improved:
            break
    
    return best_colors
