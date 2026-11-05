import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    # 1. Precompute primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)

    # Helper: Verify a set of classes covers 1..L and respects budget
    def verify_classes(classes, L):
        if not classes:
            return L <= 0
        
        # Check distinct primes and valid residues
        primes_used = set()
        for p, r in classes:
            if p in primes_used:
                return False
            primes_used.add(p)
            if r < 0 or r >= p:
                return False
        
        # Check budget using exact rationals
        s = Fraction(0)
        for p, r in classes:
            s += Fraction(1, p)
        if s > 1:
            return False
        
        # Check coverage
        covered = np.zeros(L + 1, dtype=bool)
        for p, r in classes:
            start = p if r == 0 else r
            if start <= L:
                idx = np.arange(start, L + 1, p)
                covered[idx] = True
        
        return np.all(covered[1:])

    # Helper: Calculate the coverage length L for a given set of classes
    # Returns the largest L such that 1..L are all covered.
    def calc_coverage_length(classes, max_check=10000):
        if not classes:
            return 0
        
        current_limit = 10000
        while True:
            covered = np.zeros(current_limit + 1, dtype=bool)
            for p, r in classes:
                start = p if r == 0 else r
                if start <= current_limit:
                    idx = np.arange(start, current_limit + 1, p)
                    covered[idx] = True
            
            uncovered = np.where(~covered[1:])[0]
            if len(uncovered) == 0:
                if current_limit >= 100000:
                    return 100000
                current_limit *= 2
                continue
            else:
                return uncovered[0]

    # Helper: Greedy assignment of residues for a fixed set of primes
    # Returns list of [p, r]
    def greedy_assign(primes_set, max_check=10000):
        primes_sorted = sorted(primes_set)
        residues = {}
        covered = np.zeros(max_check + 1, dtype=bool)
        
        for p in primes_sorted:
            best_r = 0
            best_gain = -1
            
            for r in range(p):
                start = p if r == 0 else r
                if start > max_check:
                    continue
                idx = np.arange(start, max_check + 1, p)
                gain = np.sum(~covered[idx])
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            residues[p] = best_r
            start = p if best_r == 0 else best_r
            if start <= max_check:
                idx = np.arange(start, max_check + 1, p)
                covered[idx] = True
        
        return [[p, residues[p]] for p in primes_sorted]

    # Helper: Local search to improve residue assignment
    # Try changing one residue at a time to increase coverage length
    def local_search(classes, max_iter=200, max_check=10000):
        current_classes = [list(c) for c in classes]
        current_L = calc_coverage_length(current_classes, max_check)
        
        if current_L == 0:
            return current_classes, 0
            
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            
            for i, (p, r) in enumerate(current_classes):
                best_r = r
                best_L = current_L
                
                for r_new in range(p):
                    if r_new == r:
                        continue
                    
                    old_r = current_classes[i][1]
                    current_classes[i][1] = r_new
                    
                    new_L = calc_coverage_length(current_classes, max_check)
                    
                    if new_L > best_L:
                        best_L = new_L
                        best_r = r_new
                    else:
                        current_classes[i][1] = old_r
                
                if best_r != r:
                    current_classes[i][1] = best_r
                    current_L = best_L
                    improved = True
                    
        return current_classes, current_L

    # Main Strategy:
    # 1. Generate candidate sets of primes that satisfy sum(1/p) <= 1.
    # 2. For each set, run greedy assignment + local search.
    # 3. Keep the best result.
    
    best_classes = []
    best_L = 0
    
    # Candidate generation
    def smart_trim(prime_list):
        sorted_asc = sorted(prime_list)
        result = []
        current_sum = Fraction(0)
        for p in sorted_asc:
            if current_sum + Fraction(1, p) <= 1:
                result.append(p)
                current_sum += Fraction(1, p)
        return result
    
    # Generate candidates
    candidates = []
    
    # Base sets
    base_sets = [
        [p for p in primes if p >= 13],
        [p for p in primes if p >= 11],
        [p for p in primes if p >= 7],
        [p for p in primes if p >= 5],
        [p for p in primes if p >= 3],
        primes,
        # Mixed sets
        [2] + [p for p in primes if p >= 13],
        [3] + [p for p in primes if p >= 13],
        [5] + [p for p in primes if p >= 13],
        [2, 3] + [p for p in primes if p >= 17],
        [2, 5] + [p for p in primes if p >= 17],
        [3, 5] + [p for p in primes if p >= 17],
    ]
    
    for bs in base_sets:
        trimmed = smart_trim(bs)
        if trimmed:
            candidates.append(trimmed)
            
    # Remove duplicates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = frozenset(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
            
    # Evaluate candidates
    max_check = 20000 # Increased check range for local search
    
    for i, cand_primes in enumerate(unique_candidates):
        if time.time() > deadline - 15:
            break
            
        # Greedy Assignment
        classes = greedy_assign(cand_primes, max_check)
        
        # Local Search
        classes, L = local_search(classes, max_iter=50, max_check=max_check)
        
        if L > best_L:
            best_L = L
            best_classes = classes
            
    # If no good candidate found, fallback to a known good small set
    if best_L == 0:
        test_classes = [[2, 1], [3, 0]] # Covers odds, multiples of 3
        L = calc_coverage_length(test_classes, 1000)
        if L > best_L:
            best_L = L
            best_classes = test_classes

    # Final Verification
    if best_classes and verify_classes(best_classes, best_L):
        return best_classes
    
    # Fallback: Return a valid single class
    return [[2, 1]]
