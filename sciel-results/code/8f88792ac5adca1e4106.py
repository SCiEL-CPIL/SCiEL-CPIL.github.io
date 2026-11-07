import time
from random import Random
from itertools import combinations

def construct():
    _T0 = time.time()
    _DEADLINE = 145.0

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def greedy_clique_cover(n, k, edges):
        edge_set = set()
        for u, v in edges:
            edge_set.add((u, v) if u < v else (v, u))
        
        uncovered = set(edge_set)
        cover = []
        
        # Precompute neighbors in C for each v in I
        # C is 0..k-1, I is k..n-1
        N_C = {v: set() for v in range(k, n)}
        for u, v in edges:
            if u < k and v >= k:
                N_C[v].add(u)
            elif v < k and u >= k:
                N_C[u].add(v)
        
        # Candidate cliques
        # 1. The full clique C
        cand_C = list(range(k))
        
        # 2. For each v in I, the clique {v} U N_C(v)
        cand_I = []
        for v in range(k, n):
            if N_C[v]:
                clique = [v] + list(N_C[v])
                cand_I.append(clique)
        
        # We might also want smaller cliques? 
        # For now, let's try with just these candidates.
        # If uncovered edges remain that are not covered by these, 
        # it means some edges in C are not in any N_C(v) intersection?
        # No, C is a clique. 
        # If we use cand_C, it covers all C-C edges.
        # If we use cand_I, it covers some C-C edges.
        
        # Greedy loop
        while uncovered:
            best_clique = None
            best_count = 0
            
            # Check cand_C
            # Count how many edges in cand_C are in uncovered
            count_C = 0
            for i in range(len(cand_C)):
                for j in range(i + 1, len(cand_C)):
                    e = (cand_C[i], cand_C[j]) if cand_C[i] < cand_C[j] else (cand_C[j], cand_C[i])
                    if e in uncovered:
                        count_C += 1
            
            if count_C > best_count:
                best_count = count_C
                best_clique = cand_C
                
            # Check cand_I
            for clique in cand_I:
                # If this clique was already used, skip? 
                # We don't track used cliques, but if all its edges are covered, count will be 0.
                count = 0
                # Optimization: if the clique has no edges in uncovered, skip
                # We can check if any edge is in uncovered
                # Since we iterate, we just count.
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        e = (clique[i], clique[j]) if clique[i] < clique[j] else (clique[j], clique[i])
                        if e in uncovered:
                            count += 1
                
                if count > best_count:
                    best_count = count
                    best_clique = clique
            
            if best_clique is None:
                # Should not happen if graph is connected and we have candidates
                # Fallback: pick any edge
                e = next(iter(uncovered))
                best_clique = list(e)
                best_count = 1
            
            cover.append(best_clique)
            
            # Remove covered edges
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    e = (best_clique[i], best_clique[j]) if best_clique[i] < best_clique[j] else (best_clique[j], best_clique[i])
                    if e in uncovered:
                        uncovered.remove(e)
                        
            if _time_up():
                break

        # If we timed out, we might have uncovered edges.
        # We must cover them to be valid.
        if uncovered:
            # Group remaining edges into cliques (pairs)
            for e in uncovered:
                cover.append(list(e))
                
        return cover

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []

    for spec in specs:
        if _time_up():
            # Fallback
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    cliques.append([u, v])
            results.append(cliques)
            continue

        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        
        cliques = greedy_clique_cover(n, k, edges)
        
        # Convert to lists of ints
        final_cliques = [list(c) for c in cliques]
        results.append(final_cliques)

    return results
