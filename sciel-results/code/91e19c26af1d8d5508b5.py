import time
import random
import numpy as np
from itertools import combinations, permutations

q3_pattern = np.array([
    [1, 1, 1, 0],
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [0, 1, 1, 1]
])

row_perms = list(permutations(range(4)))
col_perms = list(permutations(range(4)))

def has_q3_in_bipartite(M):
    for rp in row_perms:
        M_rp = M[np.array(rp), :]
        for cp in col_perms:
            M_rc = M_rp[:, np.array(cp)]
            if np.all(M_rc >= q3_pattern):
                return True
    return False

# Precompute lookup
q3_lookup = {}
for mask in range(2**16):
    M = np.zeros((4, 4), dtype=int)
    for i in range(16):
        if mask & (1 << i):
            M[i // 4, i % 4] = 1
    q3_lookup[mask] = has_q3_in_bipartite(M)

def get_mask(M):
    mask = 0
    for i in range(16):
        if M[i // 4, i % 4]:
            mask |= (1 << i)
    return mask

def find_q3s(adj):
    q3s = []
    for A in combinations(range(16), 4):
        for B in combinations(range(16), 4):
            if set(A).isdisjoint(set(B)):
                M = adj[np.ix_(A, B)]
                mask = get_mask(M)
                if q3_lookup[mask]:
                    # Find edges
                    for rp in row_perms:
                        M_rp = M[np.array(rp), :]
                        for cp in col_perms:
                            M_rc = M_rp[:, np.array(cp)]
                            if np.all(M_rc >= q3_pattern):
                                edges = []
                                for i in range(4):
                                    for j in range(4):
                                        if q3_pattern[i, j]:
                                            u = A[rp[i]]
                                            v = B[cp[j]]
                                            edges.append((u, v))
                                q3s.append(edges)
                                break
                        break
    return q3s

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_edge_count = 39
    best_edges = [[a, b] for a in range(3) for b in range(3, 16)]
    
    while time.time() < deadline:
        adj = np.ones((16, 16), dtype=int)
        np.fill_diagonal(adj, 0)
        
        for _ in range(random.randint(0, 10)):
            u = random.randint(0, 15)
            v = random.randint(0, 15)
            if u != v:
                adj[u, v] = 0
                adj[v, u] = 0
                
        while time.time() < deadline:
            q3_list = find_q3s(adj)
            
            if not q3_list:
                edge_count = int(np.sum(adj) / 2)
                if edge_count > best_edge_count:
                    best_edge_count = edge_count
                    best_edges = []
                    for i in range(16):
                        for j in range(i+1, 16):
                            if adj[i, j]:
                                best_edges.append([i, j])
                break
            
            edge_counts = {}
            for q3 in q3_list:
                for e in q3:
                    e_key = (min(e[0], e[1]), max(e[0], e[1]))
                    edge_counts[e_key] = edge_counts.get(e_key, 0) + 1
            
            max_count = max(edge_counts.values())
            candidates = [e for e, c in edge_counts.items() if c == max_count]
            e_to_remove = random.choice(candidates)
            
            adj[e_to_remove[0], e_to_remove[1]] = 0
            adj[e_to_remove[1], e_to_remove[0]] = 0

    return best_edges
