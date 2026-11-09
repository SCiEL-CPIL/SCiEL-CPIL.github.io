"""Rewritten program: search for a large set of distinct integers in 2..40 with distinct subset products."""
# EVOLVE-BLOCK-START
import time

DEADLINE = 90.0
MAX_TIME = 60.0  # safety margin before 90s kill

def _factorize(n, primes):
    """Return dict of prime->exponent for n."""
    f = {}
    for p in primes:
        if p * p > n:
            break
        while n % p == 0:
            f[p] = f.get(p, 0) + 1
            n //= p
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f

def _subset_products_distinct(nums, primes):
    """Check if all 2^k subset products of nums are distinct."""
    k = len(nums)
    if k == 0:
        return True
    
    all_primes = set()
    vecs = []
    for num in nums:
        f = _factorize(num, primes)
        all_primes.update(f.keys())
        vecs.append(f)
    
    prime_list = sorted(all_primes)
    prime_index = {p: i for i, p in enumerate(prime_list)}
    dim = len(prime_list)
    
    exp_vecs = []
    for f in vecs:
        v = [0] * dim
        for p, e in f.items():
            v[prime_index[p]] = e
        exp_vecs.append(tuple(v))
    
    sums = {tuple([0]*dim)}
    for i, v in enumerate(exp_vecs):
        new_sums = set()
        for s in sums:
            new_sums.add(tuple(s[j] + v[j] for j in range(dim)))
        sums.update(new_sums)
        if len(sums) != (1 << (i + 1)):
            return False
            
    return True

def construct():
    start_time = time.time()
    
    # Generate primes up to 40
    primes = []
    for n in range(2, 41):
        is_prime = True
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
    
    # Candidate numbers: all integers from 2 to 40
    candidates = list(range(2, 41))
    
    # Precompute factorization vectors for all candidates
    # This will be used for a faster independence check
    num_primes = len(primes)
    prime_index = {p: i for i, p in enumerate(primes)}
    
    vectors = {}
    for n in range(2, 41):
        f = _factorize(n, primes)
        vec = [0] * num_primes
        for p, e in f.items():
            if p in prime_index:
                vec[prime_index[p]] = e
        vectors[n] = tuple(vec)
    
    def is_valid_set(nums):
        """Fast check using the precomputed vectors."""
        # For a set of numbers to have distinct subset products,
        # their exponent vectors must be linearly independent over the rationals.
        # We can check this by computing the rank of the matrix formed by these vectors.
        # Since the dimension is small (12), we can do Gaussian elimination.
        
        if not nums:
            return True
        
        # Extract vectors
        vecs = [vectors[n] for n in nums]
        
        # Perform Gaussian elimination to find rank
        # We'll work with a copy of the vectors
        matrix = [list(v) for v in vecs]
        rows = len(matrix)
        cols = len(matrix[0])
        
        rank = 0
        for col in range(cols):
            # Find pivot
            pivot = -1
            for row in range(rank, rows):
                if matrix[row][col] != 0:
                    pivot = row
                    break
            if pivot == -1:
                continue
            
            # Swap rows
            matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
            
            # Eliminate
            for row in range(rank + 1, rows):
                factor = matrix[row][col]
                if factor != 0:
                    for c in range(col, cols):
                        matrix[row][c] -= factor * matrix[rank][c]
            
            rank += 1
            
        return rank == len(nums)
    
    # Strategy: Greedy search with multiple orderings, then local search
    # Key insight from inspirations: Sorting in reverse order (larger first) 
    # allows the DFS to find valid larger sets more quickly.
    
    best_set = []
    
    # Ordering 1: reverse natural (larger first) - observed to be effective
    ordering1 = list(reversed(candidates))
    
    # Ordering 2: primes first
    ordering2 = primes + [n for n in candidates if n not in primes]
    
    # Ordering 3: numbers with fewer prime factors first
    def num_prime_factors(n):
        f = _factorize(n, primes)
        return sum(f.values())
    
    ordering3 = sorted(candidates, key=lambda x: (num_prime_factors(x), x))
    
    # Ordering 4: natural order
    ordering4 = candidates
    
    # Try each ordering with greedy selection
    for ordering in [ordering1, ordering2, ordering3, ordering4]:
        if time.time() - start_time > MAX_TIME:
            break
        current = []
        for num in ordering:
            if time.time() - start_time > MAX_TIME:
                break
            test = current + [num]
            if is_valid_set(test):
                current = test
        if len(current) > len(best_set):
            best_set = current
    
    # Local search: try to swap elements to improve the set
    improved = True
    while improved and (time.time() - start_time < MAX_TIME):
        improved = False
        best_size = len(best_set)
        for i in range(len(best_set)):
            if time.time() - start_time > MAX_TIME:
                break
            # Remove best_set[i]
            reduced = best_set[:i] + best_set[i+1:]
            # Try to add each candidate not in reduced
            for num in candidates:
                if num in reduced:
                    continue
                if time.time() - start_time > MAX_TIME:
                    break
                test = reduced + [num]
                if len(test) > best_size:
                    if is_valid_set(test):
                        best_set = test
                        best_size = len(test)
                        improved = True
                        break
                elif len(test) == best_size:
                    # Same size, but maybe leads to better future? Try anyway
                    if is_valid_set(test):
                        # Swap: replace best_set[i] with num
                        new_set = reduced + [num]
                        # Check if this swap allows adding more later
                        # For now, just accept if valid
                        pass
    
    # Final polish: try to add any remaining candidates
    if time.time() - start_time < MAX_TIME:
        for num in candidates:
            if num not in best_set:
                test = best_set + [num]
                if is_valid_set(test):
                    best_set = test
    
    # Ensure we return a list of Python ints
    result = [int(x) for x in best_set]
    return result
# EVOLVE-BLOCK-END
