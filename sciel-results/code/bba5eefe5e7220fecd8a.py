import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def max_clique_in_subgraph(vertices, adj):
        """
        vertices: list of vertices in the subgraph
        adj: dict or list of sets representing adjacency in the FULL graph, 
             but we only care about edges within 'vertices'
        Returns a list of vertices forming a max clique (greedy heuristic)
        """
        if not vertices:
            return []
        
        # Greedy Max Clique
        # Order by degree in the subgraph
        deg = {v: 0 for v in vertices}
        v_set = set(vertices)
        for v in vertices:
            for u in adj[v]:
                if u in v_set:
                    deg[v] += 1
        
        sorted_verts = sorted(vertices, key=lambda x: deg[x], reverse=True)
        
        C = []
        candidates = set(sorted_verts)
        
        while candidates:
            # Pick vertex with max degree in candidates
            best_v = None
            max_deg = -1
            for v in candidates:
                d = sum(1 for u in candidates if u != v and u in adj[v])
                if d > max_deg:
                    max_deg = d
                    best_v = v
            
            if best_v is None:
                break
            
            # Check if best_v connects to all in C
            valid = True
            for u in C:
                if u not in adj[best_v]:
                    valid = False
                    break
            
            if valid:
                C.append(best_v)
                # Restrict candidates to neighbors of best_v
                candidates = {u for u in candidates if u != best_v and u in adj[best_v]}
            else:
                candidates.remove(best_v)
                
        return C

    def clique_partition(n, k, p, seed):
        rng = random.Random(seed)
        # Build adjacency for the full graph
        # adj[u] is a set of neighbors
        adj = [set() for _ in range(n)]
        
        # K-K edges
        for u in range(k):
            for v in range(u + 1, k):
                adj[u].add(v)
                adj[v].add(u)
        
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
                    
        # List of all edges
        all_edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    all_edges.append((u, v))
        
        uncovered = set(all_edges)
        cliques = []
        
        while uncovered:
            best_clique = None
            best_edge_count = -1
            
            # 1. Try cliques centered at I vertices
            for v in range(k, n):
                # Neighbors of v in K that are part of uncovered edges
                # Actually, we need to find a clique S in N(v) intersect K
                # such that all edges in S and (v, u) for u in S are uncovered.
                
                # Candidate vertices for S: u in N(v) such that (v, u) is uncovered
                candidates = []
                for u in adj[v]:
                    if u < k: # u is in K
                        if (min(v, u), max(v, u)) in uncovered:
                            candidates.append(u)
                
                if not candidates:
                    continue
                
                # Find max clique in the subgraph induced by 'candidates'
                # The subgraph edges are (u, w) if (u, w) is in uncovered
                # We can use the global adj, but filter by 'uncovered'
                
                # To use max_clique_in_subgraph, we need an adj structure.
                # We can just check connectivity in 'uncovered' directly in the greedy step?
                # Or build a temporary adj.
                
                # Let's modify max_clique_in_subgraph to take a predicate for edges?
                # Or just build a small adj list.
                
                temp_adj = {u: set() for u in candidates}
                c_set = set(candidates)
                for u in candidates:
                    for w in candidates:
                        if u < w and (u, w) in uncovered:
                            temp_adj[u].add(w)
                            temp_adj[w].add(u)
                            
                S = max_clique_in_subgraph(candidates, temp_adj)
                
                if S:
                    # The clique is {v} + S
                    # Number of edges covered:
                    # Cross: |S| (all (v, u) for u in S are uncovered by construction)
                    # Internal: number of edges in S
                    internal_edges = 0
                    for i in range(len(S)):
                        for j in range(i + 1, len(S)):
                            if (min(S[i], S[j]), max(S[i], S[j])) in uncovered:
                                internal_edges += 1
                                
                    total_edges = len(S) + internal_edges
                    
                    if total_edges > best_edge_count:
                        best_edge_count = total_edges
                        best_clique = [v] + S
                        
            # 2. Try cliques in K
            # Find max clique in the subgraph induced by K using uncovered edges
            k_candidates = [u for u in range(k) if any((min(u, w), max(u, w)) in uncovered for w in range(k) if w != u)]
            if k_candidates:
                temp_adj = {u: set() for u in k_candidates}
                k_set = set(k_candidates)
                for u in k_candidates:
                    for w in k_candidates:
                        if u < w and (u, w) in uncovered:
                            temp_adj[u].add(w)
                            temp_adj[w].add(u)
                            
                S = max_clique_in_subgraph(k_candidates, temp_adj)
                
                if S:
                    # Clique is S
                    internal_edges = 0
                    for i in range(len(S)):
                        for j in range(i + 1, len(S)):
                            if (min(S[i], S[j]), max(S[i], S[j])) in uncovered:
                                internal_edges += 1
                            
                    total_edges = internal_edges
                    if total_edges > best_edge_count:
                        best_edge_count = total_edges
                        best_clique = S
                        
            if best_clique is None:
                # Should not happen if uncovered is not empty
                # Fallback: pick any edge
                edge = next(iter(uncovered))
                best_clique = list(edge)
                
            cliques.append(best_clique)
            
            # Remove covered edges
            for u in best_clique:
                for v in best_clique:
                    if u < v:
                        uncovered.discard((u, v))
                        
            if time.time() > deadline:
                break

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() > deadline:
            break
        cliques = clique_partition(n, k, p, seed)
        # Ensure types are correct
        for c in cliques:
            for i in range(len(c)):
                c[i] = int(c[i])
        results.append(cliques)

    while len(results) < 3:
        results.append([])

    return results
