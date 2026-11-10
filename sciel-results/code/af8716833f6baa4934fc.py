import time
import math

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    N = 512
    # nodes are 1..N, indices 0..N-1
    
    # Precompute adjacency as bitsets
    # adj[i] is an integer with bit j set if node i+1 and node j+1 are compatible
    adj = [0] * N
    for i in range(N):
        a = i + 1
        mask = 0
        for j in range(N):
            b = j + 1
            # Optimization: if a*b > 512 * gcd, it's false. 
            # lcm(a,b) = a*b // gcd(a,b) <= 512
            g = math.gcd(a, b)
            if a * b // g <= 512:
                mask |= (1 << j)
        adj[i] = mask
        
    best_clique = []
    
    def bron_kerbosch(P, X, clique):
        nonlocal best_clique
        if time.time() - start_time > timeout:
            return False # Signal timeout
        
        # If P and X are empty, we have a maximal clique
        if P == 0 and X == 0:
            if len(clique) > len(best_clique):
                best_clique = clique[:]
            return True
            
        if P == 0:
            return True
            
        # Choose pivot u from P | X
        # We want u that maximizes |P & adj[u]|
        # To do this efficiently, we can iterate through candidates in P|X
        # Since N is small (512), we can just find the max.
        
        union = P | X
        if union == 0:
            return True
            
        # Find pivot
        # We can iterate through bits in union
        # To speed up, we can just pick a random one or the first one, but max is better for BK
        # Let's find the one with max intersection with P
        max_inter = -1
        pivot = -1
        u = union
        while u:
            # Get lowest set bit
            lsb = u & -u
            idx = lsb.bit_length() - 1
            inter_count = (P & adj[idx]).bit_count()
            if inter_count > max_inter:
                max_inter = inter_count
                pivot = idx
            u ^= lsb
            
        # Candidates are P \ N(pivot)
        candidates = P & ~adj[pivot]
        
        # Iterate through candidates
        while candidates:
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            candidates ^= lsb
            
            # Include v
            new_clique = clique + [v+1]
            
            # Recurse
            P_new = P & adj[v]
            X_new = X & adj[v]
            
            # Remove v from P and add to X for subsequent iterations in this loop
            # This is handled by the standard BK loop structure where we update P and X
            # But here we are iterating over a fixed set of candidates derived from the original P.
            # Standard BK:
            # for v in P \ N(u):
            #    BK(P \ {v} \ N(u) ... no, P' = P \ {v} ... wait.
            # Standard BK:
            # for v in P:
            #   BK(P \ {v} ∩ N(v), X ∩ N(v))
            #   P = P \ {v}
            #   X = X ∪ {v}
            
            # My loop iterates over `candidates` which is `P \ N(pivot)`.
            # I need to be careful with P and X updates.
            
            # Let's stick to the standard iterative structure to avoid bugs.
            # I will implement a standard recursive BK that modifies local P and X copies (bitwise ops are cheap).
            
            # Actually, the standard efficient implementation iterates over a list of vertices to add.
            
            pass # I will rewrite the core loop properly below.

    # Let's use a cleaner implementation structure for BK with bitsets.
    
    def bk(P, X, clique):
        nonlocal best_clique
        if time.time() - start_time > timeout:
            return False
        
        if P == 0 and X == 0:
            if len(clique) > len(best_clique):
                best_clique = clique[:]
            return True
        if P == 0:
            return True
            
        # Pivot selection
        # We want u in P ∪ X maximizing |P ∩ N(u)|
        # Optimization: If P is small, we can just pick any.
        # For N=512, finding max is O(N^2) worst case per node, which might be slow if deep.
        # But N is small.
        
        union = P | X
        if union == 0:
             return True

        # Simple pivot: pick vertex with max degree in P
        # Iterate through bits in union
        max_deg = -1
        pivot = -1
        u = union
        while u:
            lsb = u & -u
            idx = lsb.bit_length() - 1
            deg = (P & adj[idx]).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot = idx
            u ^= lsb
            
        # Candidates: P \ N(pivot)
        # If pivot is -1 (should not happen if union!=0), candidates = P
        if pivot == -1:
            candidates = P
        else:
            candidates = P & ~adj[pivot]
            
        # If candidates is empty, all nodes in P are in N(pivot), so we must pick one?
        # No, if P \ N(pivot) is empty, it means P is a subset of N(pivot).
        # In that case, we must pick a node from P.
        # Wait, the theorem says we can restrict to P \ N(u). If that set is empty,
        # it implies P is empty? No.
        # If P \ N(u) is empty, then P ⊆ N(u).
        # Then any maximal clique containing current clique must contain a node from P.
        # But BK says we can iterate over P \ N(u). If it's empty, we do nothing?
        # No, if P \ N(u) is empty, we can just pick any node from P?
        # Actually, if P \ N(u) is empty, it means u is connected to all nodes in P.
        # Then we can just add u? No, u might be in X.
        # If u is in X, we can't add it.
        # If u is in P, we can add it.
        
        # Standard BK implementation handles this by just iterating P \ N(u).
        # If P \ N(u) is empty, the loop doesn't execute, and the branch terminates?
        # That would be wrong if P is not empty.
        
        # Correction: The set of candidates is P \ N(u).
        # If this set is empty, it means P ⊆ N(u).
        # In this case, we can simply pick one vertex from P, say v, and recurse?
        # Or rather, if P \ N(u) is empty, we can just return? No.
        
        # Let's check: If P \ N(u) is empty, then for any v in P, v is in N(u).
        # This doesn't restrict P to be empty.
        # However, the pivot selection is an optimization.
        # If we pick u such that P \ N(u) is empty, we might miss cliques?
        # No, the property is that any maximal clique in P must intersect P \ N(u) OR ...?
        # Actually, if P \ N(u) is empty, we can just pick any v in P.
        
        # To be safe and simple, if candidates is empty but P is not, we set candidates = P.
        # This reduces the efficiency gain of pivoting in that specific case but ensures correctness.
        
        if candidates == 0 and P != 0:
            candidates = P
            
        # Iterate over candidates
        while candidates:
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            candidates ^= lsb
            
            # New P: P \ {v} ∩ N(v)
            # P & ~lsb & adj[v]
            P_new = (P & ~lsb) & adj[v]
            # New X: X ∩ N(v)
            X_new = X & adj[v]
            
            if not bk(P_new, X_new, clique + [v+1]):
                return False # Timeout propagated
            
            # Update P and X for next iteration of this loop
            # P = P \ {v}
            P &= ~lsb
            # X = X ∪ {v}
            X |= lsb
            
        return True

    # Initial call
    # P = all nodes
    initial_P = (1 << N) - 1
    initial_X = 0
    initial_clique = []
    
    bk(initial_P, initial_X, initial_clique)
    
    if not best_clique:
        # Fallback if something went wrong or timeout immediately
        return list(range(1, 23))
        
    return best_clique
