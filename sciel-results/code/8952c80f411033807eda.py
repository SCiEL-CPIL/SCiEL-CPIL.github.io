import time
import random
import numpy as np
from itertools import combinations

# Precompute
edge_idx = {}
edges_list = []
for i in range(16):
    for j in range(i+1, 16):
        edge_idx[(i, j)] = len(edges_list)
        edges_list.append((i, j))

k4s = []
for quad in combinations(range(16), 4):
    a, b, c, d = quad
    k4s.append([
        edge_idx[(a,b)], edge_idx[(a,c)], edge_idx[(a,d)],
        edge_idx[(b,c)], edge_idx[(b,d)], edge_idx[(c,d)]
    ])

edge_to_k4 = [[] for _ in range(120)]
for k_idx, k4 in enumerate(k4s):
    for e in k4:
        edge_to_k4[e].append(k_idx)

def count_violations(edges):
    vc = 0
    for k4 in k4s:
        c0, c1, c2, c3, c4, c5 = edges[k4[0]], edges[k4[1]], edges[k4[2]], edges[k4[3]], edges[k4[4]], edges[k4[5]]
        # Count distinct
        if c0 == c1 == c2 == c3 == c4 == c5:
            vc += 1
        else:
            # Quick check
            s = {c0, c1, c2, c3, c4, c5}
            if len(s) < 5:
                vc += 1
    return vc

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    best_edges = None
    best_vc = float('inf')
    
    # Try multiple random inits with local search
    for trial in range(50):
        if time.time() > deadline - 15:
            break
            
        if trial < 20:
            K = 13
        elif trial < 40:
            K = 14
        else:
            K = 15
            
        edges = np.random.randint(0, K, 120)
        vc = count_violations(edges)
        
        if vc == 0:
            return edges.tolist()
        
        if vc < best_vc:
            best_vc = vc
            best_edges = edges.copy()
        
        # Simulated annealing
        T = 1.0
        min_T = 0.001
        for step in range(100000):
            if time.time() > deadline - 15:
                break
            if vc == 0:
                return edges.tolist()
                
            e = random.randint(0, 119)
            old_c = int(edges[e])
            new_c = random.randint(0, K-1)
            if new_c == old_c:
                continue
            
            delta = 0
            for k_idx in edge_to_k4[e]:
                k4 = k4s[k_idx]
                # Get other 5 colours
                other = []
                for ee in k4:
                    if ee != e:
                        other.append(edges[ee])
                # Old set
                old_set = set(other + [old_c])
                old_bad = 1 if len(old_set) < 5 else 0
                # New set
                new_set = set(other + [new_c])
                new_bad = 1 if len(new_set) < 5 else 0
                delta += new_bad - old_bad
            
            new_vc = vc + delta
            if new_vc < vc or (new_vc >= vc and random.random() < np.exp(-max(0, new_vc - vc) / T)):
                edges[e] = new_c
                vc = new_vc
                if vc < best_vc:
                    best_vc = vc
                    best_edges = edges.copy()
                if vc == 0:
                    return edges.tolist()
            
            T = max(min_T, T * 0.99999)
    
    if best_edges is not None:
        return best_edges.tolist()
    return edges.tolist()
