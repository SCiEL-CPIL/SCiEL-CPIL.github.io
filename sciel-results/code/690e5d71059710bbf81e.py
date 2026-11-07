import time
import random

def construct():
    deadline = time.time() + 85
    candidates = list(range(2, 41))
    
    # Precompute prime factorizations for all candidates
    def factorize(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    # Get all primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Convert each candidate to a prime exponent vector (tuple)
    def to_vector(n):
        factors = factorize(n)
        vec = [0] * len(primes)
        for p, exp in factors.items():
            vec[prime_to_idx[p]] = exp
        return tuple(vec)
    
    vectors = {n: to_vector(n) for n in candidates}
    
    # Check if a set of vectors is linearly independent over rationals
    # Using Gaussian elimination
    def is_independent(vecs):
        if not vecs:
            return True
        # Build matrix and do row reduction
        mat = [list(v) for v in vecs]
        rows = len(mat)
        cols = len(mat[0]) if rows > 0 else 0
        rank = 0
        for col in range(cols):
            # Find pivot
            pivot = -1
            for row in range(rank, rows):
                if mat[row][col] != 0:
                    pivot = row
                    break
            if pivot == -1:
                continue
            # Swap rows
            mat[rank], mat[pivot] = mat[pivot], mat[rank]
            # Normalize pivot row
            pivot_val = mat[rank][col]
            mat[rank] = [x / pivot_val for x in mat[rank]]
            # Eliminate other rows
            for row in range(rows):
                if row != rank and mat[row][col] != 0:
                    factor = mat[row][col]
                    mat[row] = [a - factor * b for a, b in zip(mat[row], mat[rank])]
            rank += 1
        return rank == rows
    
    # Better: check independence incrementally using the product set approach
    # but with vectors for faster collision detection
    def build_product_vectors(current_set):
        """Build set of all subset product exponent vectors."""
        vectors_set = {tuple([0]*len(primes))}
        for x in current_set:
            v = vectors[x]
            new_vecs = set()
            for pv in vectors_set:
                new_v = tuple(pv[i] + v[i] for i in range(len(primes)))
                new_vecs.add(new_v)
            vectors_set.update(new_vecs)
        return vectors_set
    
    def can_add(current_vec_set, candidate):
        """Check if adding candidate keeps all subset product vectors distinct."""
        v = vectors[candidate]
        for pv in current_vec_set:
            new_v = tuple(pv[i] + v[i] for i in range(len(primes)))
            if new_v in current_vec_set:
                return False
        return True
    
    # Branch and bound to find maximum independent set
    def branch_and_bound(remaining, current_set, current_vec_set, depth, best_size):
        if time.time() > deadline - 1:
            return current_set
        
        # Upper bound: current size + remaining count
        if len(current_set) + len(remaining) <= best_size:
            return current_set
        
        if not remaining:
            return current_set
        
        # Try adding each remaining element
        best_result = current_set[:]
        
        for i in range(len(remaining)):
            if time.time() > deadline - 1:
                break
            
            candidate = remaining[i]
            new_remaining = remaining[:i] + remaining[i+1:]
            
            if can_add(current_vec_set, candidate):
                # Add it
                new_vec_set = set()
                v = vectors[candidate]
                for pv in current_vec_set:
                    new_v = tuple(pv[i] + v[i] for i in range(len(primes)))
                    new_vec_set.add(new_v)
                new_vec_set.update(current_vec_set)
                
                new_set = current_set + [candidate]
                result = branch_and_bound(new_remaining, new_set, new_vec_set, depth + 1, len(best_result))
                
                if len(result) > len(best_result):
                    best_result = result
        
        return best_result
    
    # Direct construction using primes as initial best
    direct_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    best_set = direct_primes[:]
    best_size = len(best_set)
    
    # Use branch and bound to try to improve
    remaining = [x for x in candidates if x not in best_set]
    current_vec_set = build_product_vectors(best_set)
    
    # Try to add elements using branch and bound
    result = branch_and_bound(remaining, best_set, current_vec_set, 0, best_size)
    if len(result) > best_size:
        best_size = len(result)
        best_set = result[:]
    
    # Also try randomized greedy approaches for diversity
    def greedy_from_order(ordered_candidates):
        current_set = []
        current_vec_set = {tuple([0]*len(primes))}
        for x in ordered_candidates:
            if can_add(current_vec_set, x):
                current_set.append(x)
                v = vectors[x]
                new_vecs = set()
                for pv in current_vec_set:
                    new_v = tuple(pv[i] + v[i] for i in range(len(primes)))
                    new_vecs.add(new_v)
                current_vec_set.update(new_vecs)
        return current_set
    
    random.seed(42)
    
    # Phase 1: Randomized greedy with multiple restarts
    for restart in range(200):
        if time.time() > deadline - 2:
            break
        
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = greedy_from_order(shuffled)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            if best_size >= 17:
                break
    
    # Phase 2: Branch and bound from best set
    if time.time() < deadline - 5:
        remaining = [x for x in candidates if x not in best_set]
        current_vec_set = build_product_vectors(best_set)
        result = branch_and_bound(remaining, best_set, current_vec_set, 0, best_size)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Phase 3: More randomized restarts with time remaining
    while time.time() < deadline - 1:
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = greedy_from_order(shuffled)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    return [int(x) for x in best_set]
