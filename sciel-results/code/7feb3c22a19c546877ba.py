import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds safety margin
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_edges = set()
    best_count = 0
    
    # Run multiple restarts of the greedy algorithm
    while time.time() < deadline:
        random.shuffle(all_edges)
        current_edges = set()
        
        # Helper to check if adding (u, v) creates a K3,3
        def can_add(u, v):
            # N(u) and N(v) are sets of neighbors
            # We need to check if there exist v1, v2 in N(u) and u1, u2 in N(v)
            # such that u1, u2 are connected to v1, v2, and the sets are disjoint.
            
            # Optimization:
            # For each pair (v1, v2) in N(u), check common neighbors in N(v)
            # that are also connected to both v1 and v2.
            
            nu = [w for w in range(n) if w != u and (min(u, w), max(u, w)) in current_edges]
            nv = [w for w in range(n) if w != v and (min(v, w), max(v, w)) in current_edges]
            
            # If we add (u, v), u will be connected to v, and v to u.
            # The new neighbors of u will be nu + [v]
            # The new neighbors of v will be nv + [u]
            
            # We need to find v1, v2 in (nu + [v]) and u1, u2 in (nv + [u])
            # such that:
            # 1. {u, u1, u2} and {v, v1, v2} are disjoint.
            #    This means u1, u2 != v, v1, v2.
            #    Since u1, u2 are in nv + [u], and v1, v2 are in nu + [v].
            #    u1, u2 cannot be v (since v is not in nv unless self-loop, which is not).
            #    u1, u2 cannot be v1 or v2.
            #    Similarly v1, v2 cannot be u1 or u2.
            
            # Let's simplify:
            # We iterate over pairs (v1, v2) from the new N(u).
            # New N(u) = nu + [v]
            # We iterate over pairs (u1, u2) from the new N(v).
            # New N(v) = nv + [u]
            
            # To be efficient, we can precompute common neighbors for pairs.
            # But n=20 is small.
            
            # Let's just check all pairs.
            new_nu = set(nu)
            new_nu.add(v)
            new_nv = set(nv)
            new_nv.add(u)
            
            # List of vertices in new_nu
            list_nu = list(new_nu)
            list_nv = list(new_nv)
            
            for i in range(len(list_nu)):
                for j in range(i + 1, len(list_nu)):
                    v1 = list_nu[i]
                    v2 = list_nu[j]
                    
                    # Check if u1, u2 exist in new_nv
                    # such that u1, u2 are connected to v1 and v2
                    # and u1, u2 are not v1, v2 (disjointness)
                    
                    candidates = []
                    for w in list_nv:
                        if w == v1 or w == v2:
                            continue
                        # Check if w is connected to v1 and v2
                        if (min(w, v1), max(w, v1)) in current_edges and (min(w, v2), max(w, v2)) in current_edges:
                            candidates.append(w)
                    
                    if len(candidates) >= 2:
                        return False
            return True

        for u, v in all_edges:
            if (u, v) in current_edges:
                continue
            if can_add(u, v):
                current_edges.add((u, v))
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges.copy()
            
    # Convert set of tuples to list of lists
    return [[u, v] for u, v in best_edges]
