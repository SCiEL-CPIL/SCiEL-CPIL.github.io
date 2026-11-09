import time
import random
import numpy as np

def construct():
    deadline = time.time() + 55.0
    n = 20
    max_val = 10000
    
    # Generate 5-smooth numbers (2^a * 3^b * 5^c) <= 10000
    # These are good for minimizing product set size.
    # We also consider 7-smooth for slightly more flexibility.
    
    def get_smooth_numbers(primes, limit, count):
        """
        Generate the smallest 'count' numbers <= limit whose prime factors are in 'primes'.
        """
        smooth = set([1])
        # Simple generation: iterate 1..limit and check factors
        # This is fast enough for limit=10000
        smooth_list = []
        for x in range(1, limit + 1):
            t = x
            valid = True
            for p in primes:
                while t % p == 0:
                    t //= p
            if t != 1:
                valid = False
                break
            if valid:
                smooth_list.append(x)
                if len(smooth_list) == count:
                    break
        return smooth_list

    # Candidate pools
    # 5-smooth numbers
    pool_5 = get_smooth_numbers([2, 3, 5], max_val, 200)
    # 7-smooth numbers
    pool_7 = get_smooth_numbers([2, 3, 5, 7], max_val, 200)
    # Small integers (1..100)
    pool_small = list(range(1, 101))
    
    # Combine pools for local search candidates
    # We want distinct integers.
    all_candidates = list(set(pool_5 + pool_7 + pool_small))
    all_candidates.sort()
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A."""
        # A is a list of ints
        # Using sets is fast for N=20
        sum_set = set()
        prod_set = set()
        for i in range(len(A)):
            ai = A[i]
            for j in range(i, len(A)): # Symmetric, but sum/product commutative
                aj = A[j]
                sum_set.add(ai + aj)
                prod_set.add(ai * aj)
        return max(len(sum_set), len(prod_set))

    def local_search(start_set, max_iter, pool):
        """
        Local search by swapping elements.
        """
        current = start_set[:]
        current_score = evaluate(current)
        best_set = current[:]
        best_score = current_score
        
        # Track visited states to avoid cycles? Maybe not necessary with randomization.
        # Use a simple restart strategy if stuck.
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
                
            # Pick an index to replace
            idx = random.randint(0, n - 1)
            old_val = current[idx]
            
            # Pick a new value from the pool
            # Bias towards values in the pool that are "close" or just random
            new_val = random.choice(pool)
            
            if new_val == old_val:
                continue
            
            # Check if new_val is already in current (excluding old_val)
            # We need distinct integers
            if new_val in current and new_val != old_val:
                continue
            
            # Perform swap
            current[idx] = new_val
            new_score = evaluate(current)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_set = current[:]
            else:
                # Simulated Annealing: accept worse moves occasionally
                # Simple heuristic: accept with probability 0.05 if not too bad
                if random.random() < 0.05 and new_score < current_score * 1.2:
                    current_score = new_score
                else:
                    # Revert
                    current[idx] = old_val
        
        return best_set, best_score

    # Initial Candidates
    candidates = []
    
    # 1. Smallest 20 5-smooth numbers
    if len(pool_5) >= n:
        candidates.append(pool_5[:n])
        
    # 2. Random subsets of 5-smooth numbers
    for _ in range(10):
        if len(pool_5) >= n:
            candidates.append(sorted(random.sample(pool_5, n)))
            
    # 3. Smallest 20 7-smooth numbers
    if len(pool_7) >= n:
        candidates.append(pool_7[:n])
        
    # 4. Random subsets of 7-smooth numbers
    for _ in range(10):
        if len(pool_7) >= n:
            candidates.append(sorted(random.sample(pool_7, n)))
            
    # 5. Small integers
    candidates.append(list(range(1, n + 1)))
    
    # 6. Mixed: Some smooth, some small
    for _ in range(5):
        cand = random.sample(pool_5, 10) + random.sample(pool_small, 10)
        # Ensure distinctness
        cand = list(set(cand))
        if len(cand) < n:
            # Fill with other smooth numbers
            remaining = [x for x in pool_5 if x not in cand]
            cand += random.sample(remaining, n - len(cand))
        candidates.append(sorted(cand[:n]))

    best_set = None
    best_score = float('inf')
    
    # Run local search on each candidate
    # Allocate time budget among candidates
    # Simple strategy: run a fixed number of iterations for each, or time-sliced
    # Let's do time-sliced: give each candidate ~1/10 of the time, or run until deadline
    
    for cand in candidates:
        if time.time() > deadline:
            break
            
        # Validate candidate
        if len(set(cand)) != n or max(cand) > max_val or min(cand) < 1:
            continue
            
        # Determine pool for local search
        # Use a larger pool for better search
        search_pool = all_candidates
        
        # Run local search
        # We don't know how many iterations fit in remaining time, so we just loop until deadline
        # But we want to try multiple candidates.
        # Let's give each candidate a chunk of time, e.g., 3 seconds per candidate if many, or more if few.
        # There are ~26 candidates. 55s / 26 ~ 2s each.
        
        res_set, res_score = local_search(cand, 50000, search_pool)
        
        if res_score < best_score:
            best_score = res_score
            best_set = res_set[:]
            
    # Final polish: Run a longer local search on the best found set
    if best_set is not None and time.time() < deadline - 5.0:
        # Use a larger pool and more iterations
        res_set, res_score = local_search(best_set, 200000, all_candidates)
        if res_score < best_score:
            best_score = res_score
            best_set = res_set[:]

    if best_set is None:
        best_set = list(range(1, n + 1))
        
    return best_set
