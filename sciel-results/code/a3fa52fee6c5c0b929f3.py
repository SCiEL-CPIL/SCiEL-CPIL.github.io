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
    def calc_coverage_length(classes, max_check=100000):
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
                return int(uncovered[0])

    # Helper: Fast local search maintaining coverage counts
    def fast_local_search(classes, max_iter=100, initial_limit=50000):
        current_classes = [list(c) for c in classes]
        
        # Build initial coverage counts
        counts = np.zeros(initial_limit + 1, dtype=np.int32)
        for p, r in current_classes:
            start = p if r == 0 else r
            if start <= initial_limit:
                idx = np.arange(start, initial_limit + 1, p)
                counts[idx] += 1
        
        def get_L_from_counts():
            zeros = np.where(counts[1:] == 0)[0]
            if len(zeros) == 0:
                return initial_limit
            return int(zeros[0])
        
        current_L = get_L_from_counts()
        
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            
            for i, (p, r) in enumerate(current_classes):
                best_r = r
                best_L = current_L
                
                # Try all other residues
                for r_new in range(p):
                    if r_new == r:
                        continue
                    
                    start_old = p if r == 0 else r
                    start_new = p if r_new == 0 else r_new
                    
                    # Apply change
                    if start_old <= initial_limit:
                        idx_old = np.arange(start_old, initial_limit + 1, p)
                        counts[idx_old] -= 1
                    if start_new <= initial_limit:
                        idx_new = np.arange(start_new, initial_limit + 1, p)
                        counts[idx_new] += 1
                        
                    new_L = get_L_from_counts()
                    
                    if new_L > best_L:
                        best_L = new_L
                        best_r = r_new
                    else:
                        # Revert
                        if start_new <= initial_limit:
                            counts[idx_new] -= 1
                        if start_old <= initial_limit:
                            counts[idx_old] += 1
                
                if best_r != r:
                    current_classes[i][1] = best_r
                    current_L = best_L
                    improved = True
                    
        return current_classes, current_L

    # Helper: Greedy assignment of residues
    def greedy_assign(primes_set, max_check=20000):
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

    def smart_trim(prime_list):
        sorted_asc = sorted(prime_list)
        result = []
        current_sum = Fraction(0)
        for p in sorted_asc:
            if current_sum + Fraction(1, p) <= 1:
                result.append(p)
                current_sum += Fraction(1, p)
        return result

    best_classes = []
    best_L = 0
    
    candidates = []
    base_sets = [
        [p for p in primes if p >= 13],
        [p for p in primes if p >= 11],
        [p for p in primes if p >= 7],
        [p for p in primes if p >= 5],
        [p for p in primes if p >= 3],
        primes,
        [2] + [p for p in primes if p >= 13],
        [3] + [p for p in primes if p >= 13],
        [5] + [p for p in primes if p >= 13],
        [2, 3] + [p for p in primes if p >= 17],
        [2, 5] + [p for p in primes if p >= 17],
        [3, 5] + [p for p in primes if p >= 17],
        [2, 3, 7] + [p for p in primes if p >= 19],
        [2, 5, 11] + [p for p in primes if p >= 19],
    ]
    
    for bs in base_sets:
        trimmed = smart_trim(bs)
        if trimmed:
            candidates.append(trimmed)
            
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = frozenset(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
            
    max_check = 50000
    
    for i, cand_primes in enumerate(unique_candidates):
        if time.time() > deadline - 15:
            break
            
        classes = greedy_assign(cand_primes, max_check)
        classes, L = fast_local_search(classes, max_iter=50, initial_limit=max_check)
        
        if L > best_L:
            best_L = L
            best_classes = classes
            
    if best_L == 0:
        test_classes = [[2, 1], [3, 0]]
        L = calc_coverage_length(test_classes, 1000)
        if L > best_L:
            best_L = L
            best_classes = test_classes

    if best_classes and verify_classes(best_classes, best_L):
        return best_classes
    
    return [[2, 1]]
