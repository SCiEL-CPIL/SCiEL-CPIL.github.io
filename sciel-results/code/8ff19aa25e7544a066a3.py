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
            # Indices i in 1..L such that i % p == r
            # If r == 0, first is p. If r > 0, first is r (if r >= 1).
            # Note: i % p == r implies i = k*p + r.
            # Smallest positive i: if r==0, i=p. If r>0, i=r.
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
        
        # Estimate a reasonable upper bound for L to check
        # Sum of 1/p is roughly 1. The product of primes is huge, but coverage is local.
        # Let's check up to 10000 initially, or until we find a gap.
        # We can extend if 1..10000 are covered.
        
        # For efficiency, use a boolean array
        # If L is large, this array might be big, but 10000 is fine.
        # If we cover 10000, we might check more.
        
        current_limit = 10000
        while True:
            covered = np.zeros(current_limit + 1, dtype=bool)
            for p, r in classes:
                start = p if r == 0 else r
                if start <= current_limit:
                    idx = np.arange(start, current_limit + 1, p)
                    covered[idx] = True
            
            # Find first gap
            uncovered = np.where(~covered[1:])[0]
            if len(uncovered) == 0:
                # All 1..current_limit covered
                if current_limit >= 100000:
                    return 100000
                current_limit *= 2
                continue
            else:
                return uncovered[0]

    # Helper: Greedy assignment of residues for a fixed set of primes
    # Returns list of [p, r]
    def greedy_assign(primes_set, max_check=10000):
        # Sort primes ascending? Or descending?
        # Usually, larger primes have more "flexibility" in residue choice?
        # Or smaller primes cover more numbers, so fixing them first is better?
        # Let's try ascending (small primes first) as they have higher density.
        
        primes_sorted = sorted(primes_set)
        residues = {}
        
        # We track coverage counts to maximize "new" coverage
        # Or simply track boolean coverage.
        # Maximizing new coverage is a standard greedy heuristic.
        
        # To be fast, we can just use boolean coverage.
        # However, checking "new coverage" requires knowing what's already covered.
        
        covered = np.zeros(max_check + 1, dtype=bool)
        
        for p in primes_sorted:
            best_r = 0
            best_gain = -1
            
            # Optimization: We only care about coverage up to some reasonable L.
            # If we are targeting L~1000, checking up to 10000 is safe.
            
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
            
        # Map prime to index in list for quick update
        prime_to_idx = {p: i for i, (p, r) in enumerate(current_classes)}
        
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
                    
                    # Temporarily update
                    old_r = current_classes[i][1]
                    current_classes[i][1] = r_new
                    
                    # Recalculate L
                    # This is expensive if done naively.
                    # Optimization: We can just check if the new residue covers the current gap
                    # or extends the coverage.
                    # For simplicity and correctness, recalc L.
                    # To speed up, we can limit the check range to current_L + p + 10
                    # because changing residue of p can only affect coverage in blocks of p.
                    
                    new_L = calc_coverage_length(current_classes, max_check)
                    
                    if new_L > best_L:
                        best_L = new_L
                        best_r = r_new
                    else:
                        # Revert if no improvement
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
    # We know that including small primes is generally good for coverage density.
    # But budget is tight.
    
    # Strategy A: Start with all primes >= 13, add small ones if budget allows.
    # Strategy B: Start with small primes, add large ones if budget allows.
    
    # Let's generate a diverse set of candidates.
    
    # Candidate 1: Primes >= 13
    cand1 = [p for p in primes if p >= 13]
    s1 = sum(Fraction(1, p) for p in cand1)
    if s1 <= 1:
        yield_cand = cand1
    else:
        yield_cand = []
        # This set is too dense? No, sum 1/p for p>=13 is small.
        # 1/13+1/17+... is definitely < 1.
    
    # Candidate 2: Primes >= 11
    cand2 = [p for p in primes if p >= 11]
    
    # Candidate 3: Primes >= 7
    cand3 = [p for p in primes if p >= 7]
    
    # Candidate 4: Primes >= 5
    cand4 = [p for p in primes if p >= 5]
    
    # Candidate 5: Primes >= 3
    cand5 = [p for p in primes if p >= 3]
    
    # Candidate 6: All primes
    cand6 = primes
    
    # We need to trim these to fit budget if they exceed 1.
    # Trimming: Remove largest primes (least impact on density? No, largest primes have smallest 1/p, so removing them saves the least budget. Wait.
    # To fit budget, we should remove primes with LARGEST 1/p? No, that saves the most budget.
    # But small primes are more useful for covering.
    # So if we exceed budget, we should remove LARGE primes first (smallest 1/p) to save the least budget?
    # No. If sum > 1, we need to reduce sum.
    # Removing a prime p reduces sum by 1/p.
    # To reduce sum by the required amount with minimal loss of coverage, we should remove primes that are least useful.
    # Large primes are less useful for covering 1..L (for small L).
    # So, sort by prime size descending, and remove until sum <= 1.
    
    def trim_to_budget(prime_list):
        s = sum(Fraction(1, p) for p in prime_list)
        if s <= 1:
            return prime_list
        
        # Sort descending by prime value
        sorted_primes = sorted(prime_list, reverse=True)
        trimmed = []
        current_sum = s
        # We need to remove primes until sum <= 1
        # Iterate from largest to smallest
        for p in sorted_primes:
            if current_sum <= 1:
                # Keep the rest
                # We need to add back the ones we haven't processed yet?
                # This logic is messy.
                pass
        
        # Better: Start with empty set, add primes in order of "value".
        # Value: Small primes are high value.
        # So sort ascending. Add until sum would exceed 1.
        sorted_asc = sorted(prime_list)
        result = []
        current_sum = Fraction(0)
        for p in sorted_asc:
            if current_sum + Fraction(1, p) <= 1:
                result.append(p)
                current_sum += Fraction(1, p)
            else:
                # Skip this prime and try larger ones?
                # If we skip a small prime, we might fit a large one.
                # But large one is less useful.
                # So just skip.
                pass
        return result
    
    # Let's refine trim_to_budget to be smarter.
    # If we can't fit a small prime, we definitely can't fit a larger one?
    # 1/small > 1/large.
    # So if current_sum + 1/small > 1, then current_sum + 1/large < current_sum + 1/small.
    # It *might* be <= 1.
    # Example: Sum = 0.9. 1/2 = 0.5. 0.9+0.5 > 1. Skip 2.
    # 1/3 = 0.33. 0.9+0.33 > 1. Skip 3.
    # 1/7 = 0.14. 0.9+0.14 <= 1. Add 7.
    # So we should continue checking.
    
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
            
        # Optional: Try a few random restarts of greedy if time permits?
        # For now, stick to one greedy pass + local search per candidate.
        
    # If no good candidate found, fallback to a known good small set
    if best_L == 0:
        # Try a very small set
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
