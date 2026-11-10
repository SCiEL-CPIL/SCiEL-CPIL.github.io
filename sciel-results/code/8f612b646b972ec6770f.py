import time
import random

def construct():
    """
    Find the largest subset of {2, ..., 40} such that all subset products are distinct.
    Uses conflict-graph branch-and-bound with random restarts as fallback.
    """
    deadline = time.time() + 45
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Precompute prime exponent vectors for 2..40
    vectors = {}
    for n in range(2, 41):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        vectors[n] = tuple(v)
    
    zero_vec = tuple([0] * num_primes)
    candidates = list(range(2, 41))
    n_cands = len(candidates)
    
    # Build conflict graph: two numbers conflict if their vectors are linearly dependent
    # (i.e., one is a rational power of the other), meaning including both could cause collisions
    # Actually, the real conflict is: for any two distinct subsets A, B of our set, prod(A) != prod(B)
    # This is equivalent to: no non-trivial subset of our set has product 1 in the multiplicative group
    # which means: no non-empty subset has vector sum = zero_vec (since all vectors are non-negative,
    # this only happens if some vector is zero, which it isn't)
    # Wait - the vectors are all non-negative, so a subset product collision means:
    # sum of vectors in subset A = sum of vectors in subset B
    # which means sum of vectors in (A \ B) = sum of vectors in (B \ A)
    # So we need: no two disjoint non-empty subsets have the same vector sum.
    
    # For the conflict graph approach: two numbers a, b conflict if there exist
    # subsets S1 containing a but not b, and S2 containing b but not a, such that
    # sum(S1) = sum(S2). This is complex.
    
    # Simpler approach: use the incremental check with better heuristics.
    # Let's try a smarter greedy + local search.
    
    best_set = []
    best_len = 0
    
    def _check_distinct_products(s):
        """Verify all subset products are distinct."""
        if not s:
            return True
        products = {1}
        for x in s:
            new_products = set()
            for p in products:
                np = p * x
                if np in products or np in new_products:
                    return False
                new_products.add(np)
            products.update(new_products)
        return True
    
    def _can_add_incremental(current_set, cand, vectors, primes):
        """Check if adding cand to current_set maintains distinct subset products."""
        v = vectors[cand]
        # Get all current subset sums
        sums = {zero_vec}
        for x in current_set:
            vx = vectors[x]
            new_sums = set()
            for s in sums:
                ns = tuple(s[i] + vx[i] for i in range(num_primes))
                if ns in sums or ns in new_sums:
                    # This shouldn't happen if current_set is valid, but check
                    pass
                new_sums.add(ns)
            sums.update(new_sums)
        
        # Check if adding cand causes collision
        for s in sums:
            ns = tuple(s[i] + v[i] for i in range(num_primes))
            if ns in sums:
                return False
        return True
    
    # Strategy 1: Branch and bound on conflict graph
    # Two numbers i, j are in conflict if there exist disjoint non-empty subsets
    # S1 ⊆ {i} ∪ (some others) and S2 ⊆ {j} ∪ (some others) with equal product
    # Simplified: i and j conflict if vec(i) = vec(j) (same number, not possible)
    # or if vec(i) is a positive multiple of vec(j) or vice versa (power relationship)
    # More precisely: conflict if ∃ k≥1 such that vec(i)*k = vec(j) or vec(j)*k = vec(i)
    # This captures: if a = b^k, then including both a and b means {a} and {b,b,...,b} (k times) collide
    
    # Build conflict matrix
    conflict = [[False] * n_cands for _ in range(n_cands)]
    for i in range(n_cands):
        for j in range(i + 1, n_cands):
            vi = vectors[candidates[i]]
            vj = vectors[candidates[j]]
            # Check if vi is a positive integer multiple of vj or vice versa
            # vi = k * vj for some integer k >= 1
            # This means all components of vi are multiples of vj and ratio is constant
            # Handle zero components: if vj has a zero, vi must also have zero there
            # and for non-zero vj components, vi/vj must be constant integer
            is_multiple = False
            ratio = None
            for a, b in zip(vi, vj):
                if b == 0:
                    if a != 0:
                        break
                else:
                    if a % b != 0:
                        break
                    r = a // b
                    if ratio is None:
                        ratio = r
                    elif r != ratio:
                        break
            else:
                if ratio is not None and ratio >= 1:
                    is_multiple = True
            
            # Check reverse
            is_multiple_rev = False
            ratio_rev = None
            for a, b in zip(vj, vi):
                if b == 0:
                    if a != 0:
                        break
                else:
                    if a % b != 0:
                        break
                    r = a // b
                    if ratio_rev is None:
                        ratio_rev = r
                    elif r != ratio_rev:
                        break
            else:
                if ratio_rev is not None and ratio_rev >= 1:
                    is_multiple_rev = True
            
            if is_multiple or is_multiple_rev:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # Branch and bound for maximum independent set in conflict graph
    # Independent set = no two nodes are connected by an edge (no conflicts)
    
    def _branch_and_bound(adj, n):
        """Find maximum independent set using branch and bound."""
        best = set()
        
        def _bbs(current, available, depth):
            nonlocal best
            if len(current) + len(available) <= len(best):
                return
            if not available:
                if len(current) > len(best):
                    best = set(current)
                return
            
            # Choose a vertex to branch on
            v = available[0]
            remaining = available[1:]
            
            # Branch 1: include v
            new_current = current + [v]
            # Available vertices that don't conflict with v
            new_available = [u for u in remaining if not adj[v][u]]
            _bbs(new_current, new_available, depth + 1)
            
            # Branch 2: exclude v
            _bbs(current, remaining, depth + 1)
        
        _bbs([], list(range(n)), 0)
        return best
    
    # Try branch and bound
    if time.time() < deadline - 5:
        bis = _branch_and_bound(conflict, n_cands)
        best_set = [candidates[i] for i in sorted(bis)]
        best_len = len(best_set)
        
        # Verify
        if _check_distinct_products(best_set):
            if best_len >= 15:
                return list(best_set)
    
    # Strategy 2: Random restarts with greedy + local search
    num_restarts = 500
    
    for r in range(num_restarts):
        if time.time() > deadline - 2:
            break
        
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = []
        sums = {zero_vec}
        
        for cand in shuffled:
            v = vectors[cand]
            
            # Check for collision
            collision = False
            for s in sums:
                ns = tuple(s[i] + v[i] for i in range(num_primes))
                if ns in sums:
                    collision = True
                    break
            
            if not collision:
                current_set.append(cand)
                new_sums = set()
                for s in sums:
                    ns = tuple(s[i] + v[i] for i in range(num_primes))
                    new_sums.add(ns)
                sums.update(new_sums)
                
                if len(current_set) > best_len:
                    best_len = len(current_set)
                    best_set = current_set[:]
                    
                    if best_len == 18:
                        return list(best_set)
        
        # Local search: try swapping out elements
        if len(current_set) < best_len:
            continue
        
        # Try to improve by removing one and adding another
        for i in range(len(current_set)):
            if time.time() > deadline - 2:
                break
            removed = current_set[i]
            trial_set = current_set[:i] + current_set[i+1:]
            trial_sums = {zero_vec}
            for x in trial_set:
                vx = vectors[x]
                new_sums = set()
                for s in trial_sums:
                    ns = tuple(s[j] + vx[j] for j in range(num_primes))
                    new_sums.add(ns)
                trial_sums.update(new_sums)
            
            # Try adding each candidate not in trial_set
            for cand in candidates:
                if cand in trial_set:
                    continue
                v = vectors[cand]
                can_add = True
                for s in trial_sums:
                    ns = tuple(s[j] + v[j] for j in range(num_primes))
                    if ns in trial_sums:
                        can_add = False
                        break
                if can_add:
                    new_set = trial_set + [cand]
                    if len(new_set) > best_len:
                        best_len = len(new_set)
                        best_set = new_set[:]
                        if best_len == 18:
                            return list(best_set)
                    break  # Only try first valid replacement per removal
    
    # Fallback
    if not best_set:
        return primes
        
    return list(best_set)
