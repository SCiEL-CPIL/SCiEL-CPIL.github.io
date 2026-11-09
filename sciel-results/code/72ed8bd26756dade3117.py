import time
import sys
from itertools import combinations
from collections import deque
import random
import numpy as np
from scipy.optimize import linprog

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and set
    adj = {i: set() for i in range(n)}
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))

    def is_clique(S):
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                if (S_list[i], S_list[j]) not in edge_set:
                    return False
        return True

    def find_max_clique():
        best = []
        
        def expand(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = R[:]
                return
            
            if P:
                pivot = max(P | X, key=lambda v: len(adj[v] & P))
            else:
                pivot = None
            
            candidates = list(P)
            if pivot is not None:
                candidates = [v for v in P if v not in adj[pivot]]
            
            for v in candidates:
                expand(R + [v], P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
                
        expand([], set(range(n)), set())
        return best

    max_clique = find_max_clique()
    best_k = len(max_clique)
    best_S = set(max_clique)
    
    # LP Feasibility Check for Disjoint Paths
    def check_lp_feasibility(S):
        k = len(S)
        if k < 2:
            return False
        S_list = list(S)
        pairs = [(S_list[i], S_list[j]) for i in range(k) for j in range(i + 1, k)]
        num_pairs = len(pairs)
        
        # Nodes in the flow network:
        # For each v in V \ S, we have v_in and v_out.
        # For each u in S, we have u as source/sink.
        # We model vertex capacities by splitting non-branch vertices.
        
        # Map original vertices to flow node indices
        # Non-branch vertices v: v_in, v_out
        # Branch vertices u: u (single node, but acts as source/sink)
        
        # Let's define the flow network explicitly for the LP
        # Variables: flow on each directed edge for each pair.
        # To keep size small, we only consider edges in the graph.
        
        # Directed edges in graph
        directed_edges = []
        for u, v in edges:
            directed_edges.append((u, v))
            directed_edges.append((v, u))
        
        # We need to handle vertex capacities.
        # Standard trick: split each vertex v (not in S) into v_in -> v_out with capacity 1.
        # For v in S, no capacity constraint on the vertex itself, but it's a source/sink.
        
        # Let's index the variables.
        # x[p, e] where p is pair index, e is directed edge index.
        # But we must enforce that flow enters v_in and leaves v_out.
        # So, for edges (u, v) where v not in S, the flow goes into v_in.
        # For edges (v, u) where v not in S, the flow comes from v_out.
        
        # This is getting complex. Let's simplify by using a standard multi-commodity flow LP formulation.
        # Nodes: V
        # Arcs: E (directed)
        # Variables: f[p, u, v] for each pair p and directed edge (u, v).
        # Constraints:
        # 1. Flow conservation at each node w for each pair p:
        #    sum_in f[p, u, w] - sum_out f[p, w, v] = 1 if w is source of p
        #    = -1 if w is sink of p
        #    = 0 otherwise
        # 2. Vertex capacity: for each w not in S, sum_p sum_v f[p, w, v] <= 1 (outflow)
        #    Actually, capacity on vertex w means total flow passing through w is <= 1.
        #    This is sum_p (inflow to w) <= 1. Or sum_p (outflow from w) <= 1.
        # 3. f[p, u, v] >= 0.
        
        # Number of variables: num_pairs * num_directed_edges
        # num_pairs ~ 45, num_directed_edges ~ 2*70 = 140.
        # Total vars ~ 6300. This is too large for `linprog` to be fast in Python.
        
        # We need a faster check.
        # Maybe just check min-cuts for each pair individually? No, that's not sufficient for joint disjointness.
        
        # Let's stick to the backtracking but improve it.
        return True # Placeholder, we will use backtracking directly but with better pruning

    def verify_subdivision(S):
        k = len(S)
        if k < 2:
            return False, None
        
        if is_clique(S):
            S_list = list(S)
            paths = []
            for i in range(k):
                for j in range(i + 1, k):
                    paths.append([S_list[i], S_list[j]])
            return True, {'branches': S_list, 'paths': paths}
        
        S_list = list(S)
        S_set = set(S_list)
        pairs = [(S_list[i], S_list[j]) for i in range(k) for j in range(i + 1, k)]
        
        # Sort pairs by difficulty
        def pair_difficulty(p):
            u, v = p
            if (u, v) in edge_set:
                return 0
            common = (adj[u] & adj[v]) - S_set
            return -len(common)
            
        pairs.sort(key=pair_difficulty)
        
        used_internal = set()
        paths_found = {}
        
        def backtrack(idx):
            if idx == len(pairs):
                return True
            
            u, v = pairs[idx]
            forbidden = used_internal.union(S_set - {u, v})
            
            # Find all shortest paths? Or just a few?
            # Let's find up to 5 shortest paths and try them.
            found_paths = []
            visited = {u}
            queue = deque([(u, [u])])
            min_len = float('inf')
            
            while queue:
                node, path = queue.popleft()
                if len(path) > min_len:
                    break
                if node == v:
                    min_len = len(path)
                    found_paths.append(path)
                    continue
                for neighbor in adj[node]:
                    if neighbor == v:
                        found_paths.append(path + [v])
                        break
                    if neighbor not in visited and neighbor not in forbidden:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
            
            if not found_paths:
                return False
            
            # Shuffle paths to add randomness
            random.shuffle(found_paths)
            
            for path in found_paths:
                internal_nodes = set(path[1:-1])
                if internal_nodes.isdisjoint(used_internal):
                    used_internal.update(internal_nodes)
                    paths_found[(u, v)] = path
                    if backtrack(idx + 1):
                        return True
                    used_internal.difference_update(internal_nodes)
                    del paths_found[(u, v)]
            
            return False

        if backtrack(0):
            res_paths = []
            for u, v in pairs:
                res_paths.append(paths_found[(u, v)])
            return True, {'branches': S_list, 'paths': res_paths}
        else:
            return False, None

    start_time = time.time()
    deadline = start_time + 85.0
    
    degrees = [len(adj[i]) for i in range(n)]
    vertices_sorted = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    found_improvement = True
    current_k = best_k
    
    while current_k < n and time.time() < deadline:
        current_k += 1
        found = False
        
        # Strategy A: Extend best known set
        if len(best_S) == current_k - 1:
            outside = [v for v in range(n) if v not in best_S]
            outside.sort(key=lambda x: degrees[x], reverse=True)
            for v in outside:
                if time.time() > deadline:
                    break
                candidate_S = best_S.union({v})
                valid, res = verify_subdivision(candidate_S)
                if valid:
                    best_S = candidate_S
                    best_k = current_k
                    found = True
                    break
        
        # Strategy B: Random sampling
        if not found:
            pool_size = min(n, 15)
            pool = vertices_sorted[:pool_size]
            if current_k > pool_size:
                pool = vertices_sorted[:min(n, current_k + 5)]
            
            attempts = 0
            max_attempts = 200
            while attempts < max_attempts and time.time() < deadline:
                candidate_S = set(random.sample(pool, current_k))
                valid, res = verify_subdivision(candidate_S)
                if valid:
                    best_S = candidate_S
                    best_k = current_k
                    found = True
                    break
                attempts += 1
        
        if not found:
            break

    valid, res = verify_subdivision(best_S)
    if valid:
        result = res
    else:
        result = {'branches': list(max_clique), 'paths': [[max_clique[i], max_clique[j]] for i in range(len(max_clique)) for j in range(i+1, len(max_clique))]}
        
    result['branches'] = list(result['branches'])
    result['paths'] = [list(p) for p in result['paths']]
    
    return result

if __name__ == "__main__":
    res = construct()
    print(res)
