import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def cover_remaining_k_edges(remaining_edges, k):
        """
        Cover remaining K-edges with min cliques.
        remaining_edges: set of (u, v) tuples
        k: number of vertices in K
        """
        cliques = []
        remaining = set(remaining_edges)
        
        # Helper to find neighbors in remaining graph
        # Since k is small, we can just iterate
        
        while remaining:
            # Pick an edge
            u, v = next(iter(remaining))
            
            # Build maximal clique containing u, v
            clique = [u, v]
            used = {u, v}
            
            # Candidates: nodes x such that (u,x) and (v,x) are in remaining
            # And x not in used
            candidates = set()
            for a, b in remaining:
                if a == u and b != v:
                    candidates.add(b)
                elif b == u and a != v:
                    candidates.add(a)
                if a == v and b != u:
                    candidates.add(b)
                elif b == v and a != u:
                    candidates.add(a)
            
            # Filter candidates that are connected to ALL in clique
            # Initially clique is {u, v}. Candidates are connected to u and v.
            # So candidates are valid.
            
            while candidates:
                # Pick candidate with highest degree in remaining graph
                # Degree of c in remaining: number of edges (c, x) in remaining
                best_c = -1
                best_deg = -1
                for c in candidates:
                    deg = 0
                    for a, b in remaining:
                        if a == c or b == c:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_c = c
                
                if best_c == -1:
                    break
                
                clique.append(best_c)
                used.add(best_c)
                
                # Update candidates
                # New candidates must be connected to best_c
                new_candidates = set()
                for c in candidates:
                    if c == best_c:
                        continue
                    # Check if c is connected to best_c in remaining
                    e = (min(c, best_c), max(c, best_c))
                    if e in remaining:
                        new_candidates.add(c)
                candidates = new_candidates
            
            # Remove edges covered by this clique
            edges_to_remove = set()
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    edges_to_remove.add((min(a, b), max(a, b)))
            
            remaining -= edges_to_remove
            cliques.append(clique)
            
        return cliques

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # All K-edges
        all_k_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        
        covered_k_edges = set()
        cliques = []
        
        # Step 1: Create cliques for each v in I
        for v in range(k, n):
            N_v = neighbors[v]
            if not N_v:
                continue
            
            # Clique is {v} U N_v
            clique = [v] + N_v
            cliques.append(clique)
            
            # Mark K-edges within N_v as covered
            for i in range(len(N_v)):
                for j in range(i + 1, len(N_v)):
                    u, w = N_v[i], N_v[j]
                    covered_k_edges.add((min(u, w), max(u, w)))
        
        # Step 2: Cover remaining K-edges
        remaining_k_edges = all_k_edges - covered_k_edges
        
        if remaining_k_edges:
            remaining_cliques = cover_remaining_k_edges(remaining_k_edges, k)
            cliques.extend(remaining_cliques)
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Emergency fallback: return empty lists? No, must be valid.
            # But we expect to finish fast.
            pass
        cliques = solve_instance(*spec)
        clean_cliques = []
        for c in cliques:
            clean_cliques.append([int(x) for x in c])
        results.append(clean_cliques)
        
    return results
