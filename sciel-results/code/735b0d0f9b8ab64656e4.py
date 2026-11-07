import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def largest_clique(vertices, adj_matrix):
        """
        Finds a largest clique in the graph induced by 'vertices'.
        adj_matrix[i][j] is True if edge (i,j) exists in the 'available' graph.
        Returns a list of vertices.
        """
        if not vertices:
            return []
        
        # Use Bron-Kerbosch with pivot for speed
        # But for n<=30, a simple recursive search with pruning might be easier to implement correctly.
        # Let's use a simple backtracking with upper bound pruning.
        
        best = []
        
        def dfs(current, candidates):
            nonlocal best
            if not candidates:
                if len(current) > len(best):
                    best = current[:]
                return
            
            # Upper bound: len(current) + len(candidates)
            if len(current) + len(candidates) <= len(best):
                return
            
            # Pick a pivot? Or just iterate.
            # To be safe and simple, iterate over candidates.
            for i, v in enumerate(candidates):
                # New candidates must be adjacent to v and in candidates[i+1:]
                new_candidates = [w for w in candidates[i+1:] if adj_matrix[v][w]]
                dfs(current + [v], new_candidates)
                
        dfs([], list(vertices))
        return best

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency list for K (0 to k-1) and I (k to n-1)
        # And adjacency matrix for K for the "available" graph
        adj_k = [[False]*k for _ in range(k)]
        neighbors_in_k = {v: [] for v in range(k, n)}
        
        edge_set = set()
        for u, v in edges:
            if u < k and v < k:
                adj_k[u][v] = True
                adj_k[v][u] = True
                edge_set.add((u, v))
            elif u >= k and v < k:
                neighbors_in_k[u].append(v)
                edge_set.add((u, v))
            elif v >= k and u < k:
                neighbors_in_k[v].append(u)
                edge_set.add((v, u))
        
        used_edges = set()
        cliques = []
        
        # Process each v in I
        for v in range(k, n):
            N_v = neighbors_in_k[v]
            if not N_v:
                continue
                
            remaining = list(N_v)
            
            while remaining:
                # Build available graph on 'remaining'
                # Edge (u, w) in available graph if (u, w) not in used_edges
                # Note: u, w are in K, so they are < k
                
                # Optimization: if time is running out, just take singletons
                if time.time() > deadline:
                    for u in remaining:
                        cliques.append([v, u])
                    break
                
                # Find largest clique in the graph where edges are "unused"
                # Vertices are 'remaining'
                # adj_avail[u][w] is True if (u, w) not in used_edges
                
                # To speed up, we can check if all pairs in a subset are unused.
                # We will find the largest subset S of 'remaining' such that all pairs in S are unused.
                
                # Simple approach: try to find largest clique in the "complement of used edges" restricted to remaining.
                # Let's build a temporary adjacency matrix for the subgraph of 'remaining'
                # Actually, we can just use the logic:
                # Two vertices u, w in 'remaining' are connected in the "available" graph if (u, w) is NOT in used_edges.
                
                # Let's map remaining vertices to indices 0..m-1 for the clique finder
                m = len(remaining)
                if m == 1:
                    u = remaining[0]
                    cliques.append([v, u])
                    remaining = []
                    continue
                
                # Build adj_matrix for the clique finder
                # We need adj_matrix[i][j] to be True if edge (remaining[i], remaining[j]) is available.
                adj_avail = [[False]*m for _ in range(m)]
                for i in range(m):
                    for j in range(i+1, m):
                        u, w = remaining[i], remaining[j]
                        # Check if edge (u, w) is in used_edges
                        if (u, w) not in used_edges and (w, u) not in used_edges:
                            # It's available
                            adj_avail[i][j] = True
                            adj_avail[j][i] = True
                
                # Find largest clique in adj_avail
                # Note: largest_clique expects a list of vertices (indices 0..m-1)
                # and the adj matrix.
                # My largest_clique implementation above used global indices.
                # Let's adapt it to use local indices 0..m-1.
                
                best_local = []
                def dfs_local(current, candidates):
                    nonlocal best_local
                    if not candidates:
                        if len(current) > len(best_local):
                            best_local = current[:]
                        return
                    if len(current) + len(candidates) <= len(best_local):
                        return
                    for i, v_idx in enumerate(candidates):
                        new_cands = [w_idx for w_idx in candidates[i+1:] if adj_avail[v_idx][w_idx]]
                        dfs_local(current + [v_idx], new_cands)
                
                dfs_local([], list(range(m)))
                
                if not best_local:
                    # Should not happen, at least singletons
                    best_local = [0]
                
                # Map back to original vertices
                S = [remaining[idx] for idx in best_local]
                
                # Add clique
                cliques.append([v] + S)
                
                # Update used_edges
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        u, w = S[i], S[j]
                        used_edges.add((u, w))
                        
                # Remove S from remaining
                S_set = set(S)
                remaining = [x for x in remaining if x not in S_set]
        
        # Cover remaining edges in K
        # E_rem = all edges in K not in used_edges
        E_rem = []
        for u in range(k):
            for w in range(u+1, k):
                if (u, w) not in used_edges:
                    E_rem.append((u, w))
        
        if E_rem:
            # Greedy clique cover for E_rem
            # Build graph of remaining edges
            adj_rem = [[False]*k for _ in range(k)]
            for u, w in E_rem:
                adj_rem[u][w] = True
                adj_rem[w][u] = True
            
            remaining_edges = set(E_rem)
            
            while remaining_edges:
                if time.time() > deadline:
                    # Fallback: add edges as pairs
                    for u, w in remaining_edges:
                        cliques.append([u, w])
                    break
                
                # Find largest clique in the graph (K, remaining_edges)
                # Vertices are 0..k-1
                # We can use the same DFS, but on the full K vertices.
                # To speed up, only consider vertices incident to remaining edges.
                active_vertices = set()
                for u, w in remaining_edges:
                    active_vertices.add(u)
                    active_vertices.add(w)
                
                if not active_vertices:
                    break
                
                # Map active vertices to 0..m-1
                v_list = list(active_vertices)
                m = len(v_list)
                idx_map = {v: i for i, v in enumerate(v_list)}
                
                adj_avail = [[False]*m for _ in range(m)]
                for i in range(m):
                    for j in range(i+1, m):
                        u, w = v_list[i], v_list[j]
                        if (u, w) in remaining_edges or (w, u) in remaining_edges:
                            adj_avail[i][j] = True
                            adj_avail[j][i] = True
                
                best_local = []
                def dfs_local2(current, candidates):
                    nonlocal best_local
                    if not candidates:
                        if len(current) > len(best_local):
                            best_local = current[:]
                        return
                    if len(current) + len(candidates) <= len(best_local):
                        return
                    for i, v_idx in enumerate(candidates):
                        new_cands = [w_idx for w_idx in candidates[i+1:] if adj_avail[v_idx][w_idx]]
                        dfs_local2(current + [v_idx], new_cands)
                
                dfs_local2([], list(range(m)))
                
                if not best_local:
                    # Take any edge
                    u, w = next(iter(remaining_edges))
                    S = [u, w]
                else:
                    S = [v_list[idx] for idx in best_local]
                
                cliques.append(S)
                
                # Remove edges in S from remaining_edges
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        u, w = S[i], S[j]
                        if (u, w) in remaining_edges:
                            remaining_edges.remove((u, w))
                        if (w, u) in remaining_edges:
                            remaining_edges.remove((w, u))
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # If we run out of time, return trivial solution for remaining
            # But we should have finished or be close.
            # Just break and return what we have? No, must be valid.
            # The trivial solution is valid.
            pass
            
        res = solve_one(*spec)
        results.append(res)
        
    return results

# EVOLVE-BLOCK-END
