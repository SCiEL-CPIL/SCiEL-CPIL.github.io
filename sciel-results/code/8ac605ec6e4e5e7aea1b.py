import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = np.zeros((n, n), dtype=bool)
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u, v] = True
                    adj[v, u] = True
        return adj

    def get_clique_edges(clique):
        edges = set()
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                if u > v:
                    u, v = v, u
                edges.add((u, v))
        return edges

    def find_greedy_clique(adj, start_vertex):
        clique = [start_vertex]
        # Candidates must be neighbors of all vertices in clique
        # Initially, neighbors of start_vertex
        candidates = set(np.where(adj[start_vertex])[0])
        candidates.discard(start_vertex)
        
        while candidates:
            # Find vertex in candidates with highest degree within candidates
            best_v = -1
            best_deg = -1
            # Optimization: if len(candidates) is small, just iterate
            for v in candidates:
                deg = sum(1 for u in candidates if adj[v, u])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                break
                
            clique.append(best_v)
            # Update candidates: intersect with neighbors of best_v
            new_candidates = set()
            for v in candidates:
                if adj[best_v, v]:
                    new_candidates.add(v)
            candidates = new_candidates
            
        return clique

    def solve_instance(n, p, seed):
        adj = generate_graph(n, p, seed)
        
        all_edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    all_edges.add((u, v))
        
        if not all_edges:
            return []

        candidates = []
        # 1. Greedy clique starting from each vertex
        for v in range(n):
            c = find_greedy_clique(adj, v)
            if len(c) >= 2:
                candidates.append(frozenset(c))
                
        # 2. Randomized restarts to find diverse/larger cliques
        # We try to find cliques by starting with a random pair or using different tie-breaking
        for _ in range(20):  # Reduce restarts to save time
            for v in range(n):
                clique = [v]
                candidates_set = set(np.where(adj[v])[0])
                candidates_set.discard(v)
                
                while candidates_set:
                    # Heuristic: pick vertex with highest degree in candidates_set
                    best_v = -1
                    best_deg = -1
                    for u in candidates_set:
                        deg = sum(1 for w in candidates_set if adj[u, w])
                        if deg > best_deg:
                            best_deg = deg
                            best_v = u
                    if best_v == -1:
                        break
                    clique.append(best_v)
                    new_candidates = set()
                    for u in candidates_set:
                        if adj[best_v, u]:
                            new_candidates.add(u)
                    candidates_set = new_candidates
                if len(clique) >= 2:
                    candidates.append(frozenset(clique))

        # Remove duplicates
        candidates = list(set(candidates))
        
        # Remove dominated candidates (subset of another)
        # This is O(M^2 * N) where M is num candidates, N is clique size.
        # M can be up to 2000. 2000^2 = 4M. N=32. 4M*32 = 128M ops. Might be slow in Python.
        # Let's skip dominance removal for now, or do it only if M is small.
        # Actually, set intersection in the main loop is O(E) where E is edges in clique.
        # Dominance removal might not be worth it.
        
        candidate_edges = [get_clique_edges(list(c)) for c in candidates]
        
        remaining_edges = all_edges.copy()
        solution = []
        
        while remaining_edges:
            if time.time() > deadline:
                for e in remaining_edges:
                    solution.append(list(e))
                break
                
            best_idx = -1
            best_overlap = 0
            
            for i, c_edges in enumerate(candidate_edges):
                if not c_edges: continue
                # Quick check: if c_edges is subset of already covered? 
                # We don't track covered, we track remaining.
                # overlap = len(c_edges & remaining_edges)
                # To speed up, we can check if any edge in c_edges is in remaining_edges
                # But len() of intersection is needed for comparison.
                
                # Optimization: if len(c_edges) < best_overlap, skip? No, overlap <= len(c_edges).
                if len(c_edges) < best_overlap:
                    continue
                    
                overlap = len(c_edges & remaining_edges)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = i
            
            if best_overlap == 0:
                # Fallback: generate new clique from a remaining edge
                e = next(iter(remaining_edges))
                u, v = e
                common_neighbors = set(np.where(adj[u] & adj[v])[0])
                common_neighbors.discard(u)
                common_neighbors.discard(v)
                
                clique = [u, v]
                # Greedily extend
                while common_neighbors:
                    best_w = -1
                    best_deg = -1
                    for w in common_neighbors:
                        deg = sum(1 for x in common_neighbors if adj[w, x])
                        if deg > best_deg:
                            best_deg = deg
                            best_w = w
                    if best_w == -1:
                        break
                    clique.append(best_w)
                    new_common = set()
                    for w in common_neighbors:
                        if adj[best_w, w]:
                            new_common.add(w)
                    common_neighbors = new_common
                
                if len(clique) >= 2:
                    solution.append(clique)
                    c_edges = get_clique_edges(clique)
                    remaining_edges -= c_edges
                else:
                    solution.append([u, v])
                    remaining_edges.discard(e)
            else:
                best_clique = list(candidates[best_idx])
                solution.append(best_clique)
                remaining_edges -= candidate_edges[best_idx]
                
        return solution

    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
