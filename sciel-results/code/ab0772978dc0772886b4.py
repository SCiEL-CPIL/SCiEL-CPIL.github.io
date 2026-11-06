import time
from math import gcd

def construct():
    """
    Return a nonempty subset of [1, 255] containing 255,
    with gcd(a,b) > 1 for all distinct pairs. Maximize cardinality.
    """
    start_time = time.time()
    deadline = 80.0  # Keep 10s margin for safety

    # Candidate pool: numbers in [1, 255] that share a factor with 255
    # 255 = 3 * 5 * 17
    # x must be divisible by 3, 5, or 17
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    # Ensure 255 is in candidates
    if 255 not in candidates:
        candidates.append(255)
        
    n = len(candidates)
    idx_map = {val: i for i, val in enumerate(candidates)}
    
    # Build adjacency list
    # adj[i] is a set of indices j such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        vi = candidates[i]
        for j in range(i + 1, n):
            vj = candidates[j]
            if gcd(vi, vj) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Bron-Kerbosch algorithm with pivot to find maximum clique
    best_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        
        if time.time() > deadline:
            return
            
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        
        # Pivot selection: choose u in P union X with maximum degree in P
        # To optimize, we can just pick an arbitrary one or the one with max len(adj[u] & P)
        # Since n is small, we can afford to be a bit less optimal in pivot selection
        # Standard heuristic: pick u from P ∪ X that maximizes |N(u) ∩ P|
        pivot = None
        max_degree = -1
        for u in list(P) + list(X):
            deg = len(adj[u] & P)
            if deg > max_degree:
                max_degree = deg
                pivot = u
        
        if pivot is not None:
            # Recurse on P \ N(pivot)
            # We iterate over P \ N(pivot)
            # Optimization: if P \ N(pivot) is empty, return
            candidates_for_recursion = P - adj[pivot]
            for v in list(candidates_for_recursion):
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        else:
            # If no pivot (P and X empty handled above, so this shouldn't happen if P not empty)
            for v in list(P):
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)

    # Start with empty R, all candidates in P, empty X
    # Note: We want the max clique.
    # Since 255 is a universal vertex in this graph, it will be part of the max clique.
    # We can just run the algorithm on the whole graph.
    P = set(range(n))
    X = set()
    R = set()
    
    bron_kerbosch(R, P, X)
    
    # If best_clique is empty (should not happen), fallback
    if not best_clique:
        # Fallback to multiples of 3
        result_vals = [i for i in candidates if i % 3 == 0]
        if 255 not in result_vals:
            result_vals.append(255)
        return result_vals
    
    result_vals = [candidates[i] for i in best_clique]
    
    # Ensure 255 is in the result (it should be, but just in case)
    if 255 not in result_vals:
        result_vals.append(255)
        
    # Final verification
    for i in range(len(result_vals)):
        for j in range(i+1, len(result_vals)):
            if gcd(result_vals[i], result_vals[j]) == 1:
                # Fallback to known good set
                result_vals = [i for i in candidates if i % 3 == 0]
                if 255 not in result_vals:
                    result_vals.append(255)
                break

    return result_vals
