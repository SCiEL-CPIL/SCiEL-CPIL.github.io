import itertools
import random

def construct():
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))
    
    # 1. Map edges to indices
    edge_to_idx = {}
    idx_to_edge = []
    
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = idx
                idx_to_edge.append((u, v))
                idx += 1
                
    # 2. Build Conflict Graph
    # Vertices are edges of Q7 (0..n_edges-1)
    # Two vertices are connected if they share a C4 face.
    # First, find all C4 faces (sets of 4 edge indices)
    faces = set()
    for x in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v0 = x
                v1 = x ^ (1 << i)
                v2 = x ^ (1 << j)
                v3 = x ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v0, v1), max(v0, v1))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v3, v2), max(v3, v2))
                e4 = (min(v2, v0), max(v2, v0))
                
                try:
                    i1 = edge_to_idx[e1]
                    i2 = edge_to_idx[e2]
                    i3 = edge_to_idx[e3]
                    i4 = edge_to_idx[e4]
                    face = frozenset([i1, i2, i3, i4])
                    faces.add(face)
                except KeyError:
                    pass
    
    # Build adjacency list for conflict graph
    # adj[e] = set of edges that conflict with e
    adj = [set() for _ in range(n_edges)]
    for face in faces:
        f_list = list(face)
        for k in range(4):
            for l in range(k+1, 4):
                u = f_list[k]
                v = f_list[l]
                adj[u].add(v)
                adj[v].add(u)
    
    # 3. Heuristic: Greedy MIS with Local Search
    # Strategy:
    # 1. Generate multiple initial MIS using Greedy Min-Degree and Random Greedy.
    # 2. Improve each using Local Search (1-2 swap).
    # 3. Return the best one.
    
    def greedy_min_degree():
        available = set(range(n_edges))
        mis = set()
        # Precompute degrees for initial state? No, dynamic.
        # For speed, we can just pick min degree in current available set.
        while available:
            # Find vertex with min degree in available subgraph
            best_v = -1
            best_deg = n_edges + 1
            for v in available:
                deg = len(adj[v] & available)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            
            mis.add(best_v)
            # Remove best_v and its neighbors from available
            to_remove = {best_v} | (adj[best_v] & available)
            available -= to_remove
        return mis

    def greedy_random():
        available = list(range(n_edges))
        random.shuffle(available)
        mis = set()
        avail_set = set(available)
        for v in available:
            if v in avail_set:
                mis.add(v)
                # Remove v and neighbors
                to_remove = {v} | (adj[v] & avail_set)
                avail_set -= to_remove
        return mis

    def local_search(mis):
        S = set(mis)
        improved = True
        # Limit iterations to avoid infinite loops or excessive time
        max_iters = 50
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try 1-2 swap: remove u, add v1, v2
            # To speed up, we can iterate over u in S
            for u in list(S):
                S_temp = S - {u}
                # Find candidates: v not in S_temp, not conflicting with S_temp
                candidates = []
                for v in range(n_edges):
                    if v in S_temp:
                        continue
                    # Check if v conflicts with any in S_temp
                    conflict = False
                    # Optimization: if v is not neighbor of u, it might still conflict with others
                    # We must check all
                    for w in S_temp:
                        if w in adj[v]:
                            conflict = True
                            break
                    if not conflict:
                        candidates.append(v)
                
                if len(candidates) < 2:
                    continue
                
                # Try to find a pair in candidates that do not conflict with each other
                # candidates are all non-conflicting with S_temp
                # We need v1, v2 in candidates such that v1 not in adj[v2]
                found_pair = False
                for i in range(len(candidates)):
                    v1 = candidates[i]
                    for j in range(i+1, len(candidates)):
                        v2 = candidates[j]
                        if v2 not in adj[v1]:
                            # Found valid swap
                            S = S_temp | {v1, v2}
                            improved = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                if found_pair:
                    break # Restart outer loop with new S
        return S

    best_mis = set()
    
    # Run Greedy Min-Degree
    mis1 = greedy_min_degree()
    mis1 = local_search(mis1)
    if len(mis1) > len(best_mis):
        best_mis = mis1
        
    # Run several Random Greedy
    for _ in range(100):
        mis_r = greedy_random()
        mis_r = local_search(mis_r)
        if len(mis_r) > len(best_mis):
            best_mis = mis_r
            
    # Construct result
    result = []
    for e_idx in best_mis:
        u, v = idx_to_edge[e_idx]
        result.append([u, v])
        
    return result
