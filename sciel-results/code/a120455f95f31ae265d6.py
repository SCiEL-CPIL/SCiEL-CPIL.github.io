import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        C = list(range(k))
        I = list(range(k, n))
        C_set = set(C)

        # Internal C edges (all present)
        internal_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                internal_edges.add((u, v))

        # Edges from I to C
        i_c_edges = {}
        for v in I:
            nbrs = adj[v] & C_set
            i_c_edges[v] = nbrs

        partition = []
        covered_c_edges = set()

        # Step 1: For each v in I, form clique {v} U N(v)
        for v in I:
            nbrs = i_c_edges[v]
            if nbrs:
                clique = sorted([v] + list(nbrs))
                partition.append(clique)
                # Mark C-C edges covered
                nbr_list = list(nbrs)
                for i in range(len(nbr_list)):
                    for j in range(i + 1, len(nbr_list)):
                        u, w = nbr_list[i], nbr_list[j]
                        covered_c_edges.add((min(u, w), max(u, w)))

        # Step 2: Cover remaining C-C edges
        rem_c_edges = internal_edges - covered_c_edges
        
        # Build graph H of remaining edges
        H_adj = [set() for _ in range(k)]
        for u, v in rem_c_edges:
            H_adj[u].add(v)
            H_adj[v].add(u)
            
        rem_set = set(rem_c_edges)
        
        while rem_set:
            if time.time() > deadline - 2.0:
                # Fallback: cover remaining edges individually
                for u, v in rem_set:
                    partition.append(sorted([u, v]))
                break
                
            # Find largest clique in H
            # Simple greedy: pick vertex with max degree, then intersect neighbors
            # Since k is small, we can try all vertices as start
            best_clique = []
            
            # Heuristic: sort vertices by degree in H
            degrees = [len(H_adj[u]) for u in range(k)]
            vertices_sorted = sorted(range(k), key=lambda x: -degrees[x])
            
            for start in vertices_sorted:
                if not H_adj[start]:
                    continue
                
                # Try to build a clique starting with start
                # Candidates must be neighbors of start
                candidates = set(H_adj[start])
                clique = [start]
                
                # Greedily add vertices
                # We want a large clique.
                # This is a heuristic for Max Clique.
                # For small graphs, we can be more thorough.
                # Let's use a simple recursive backtracking for Max Clique if k is small enough?
                # k <= 30. Max Clique is NP-hard, but for sparse graphs it's fast.
                # H is likely sparse.
                
                # Let's just use a greedy approach for speed and simplicity
                # Pick the neighbor with max degree in the candidate set
                while candidates:
                    # Pick candidate with max degree in candidates
                    best_cand = max(candidates, key=lambda x: len(H_adj[x] & candidates))
                    clique.append(best_cand)
                    candidates = candidates & H_adj[best_cand]
                    candidates.discard(best_cand)
                    
                if len(clique) > len(best_clique):
                    best_clique = clique
            
            if not best_clique or len(best_clique) < 2:
                # No clique of size > 1? Then just take an edge
                u, v = next(iter(rem_set))
                best_clique = [u, v]
            
            # Add best_clique to partition
            partition.append(sorted(best_clique))
            
            # Remove edges covered by best_clique
            clique_set = set(best_clique)
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    u, w = best_clique[i], best_clique[j]
                    rem_set.discard((min(u, w), max(u, w)))
            
            # Update H_adj (optional, as we use rem_set for termination)
            # But for finding next clique, we should update degrees
            for u in best_clique:
                for v in best_clique:
                    if u != v:
                        H_adj[u].discard(v)
                        H_adj[v].discard(u)

        cleaned = []
        for clique in partition:
            cleaned.append([int(x) for x in clique])
        return cleaned

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        results.append(res)

    return results
