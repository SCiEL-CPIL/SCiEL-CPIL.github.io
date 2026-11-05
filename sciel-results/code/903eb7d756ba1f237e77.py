import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

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

    # Helper: Calculate the coverage length L for a given mask
    # mask is a Python int where bit i is 1 if number i is covered.
    # We care about numbers 1..MAX_CHECK.
    def get_coverage_length(mask, max_check):
        # target_mask has bits 1..max_check set
        # Bit 0 is ignored.
        target_mask = (1 << (max_check + 1)) - 2
        missing = target_mask & ~mask
        if missing == 0:
            return max_check
        # Find the lowest set bit in missing
        # missing & -missing isolates the lowest set bit
        lowest_bit = missing & -missing
        # bit_length gives the position + 1
        pos = lowest_bit.bit_length() - 1
        return pos - 1

    # Precompute class masks for all primes and residues up to MAX_CHECK
    # This is expensive to store for all primes/residues if MAX_CHECK is large,
    # but for MAX_CHECK=20000, it's 100 primes * 1000 residues * 20000 bits ~ 25MB?
    # Actually, we can compute the mask for a specific (p, r) on the fly or cache it.
    # Computing on the fly:
    # mask = 0
    # for x in range(start, MAX_CHECK+1, p): mask |= (1 << x)
    # This loop is O(MAX_CHECK/p). Sum over primes is MAX_CHECK * sum(1/p) ~ MAX_CHECK * log(log(1000)) ~ 20000 * 2 ~ 40000 ops.
    # Doing this inside the search loop (100 primes * 1000 residues * 100 iters) = 10^7 * 40000 is too slow.
    # We MUST precompute the masks for each prime/residue pair.
    
    MAX_CHECK = 20000
    class_masks = {}
    for p in primes:
        class_masks[p] = []
        for r in range(p):
            m = 0
            start = p if r == 0 else r
            if start <= MAX_CHECK:
                # Use a loop to set bits
                # Optimization: Use a precomputed pattern?
                # For now, simple loop is fine for precomputation.
                for x in range(start, MAX_CHECK + 1, p):
                    m |= (1 << x)
            class_masks[p].append(m)

    # Helper: Check if a set of primes fits the budget
    def fits_budget(prime_list):
        s = Fraction(0)
        for p in prime_list:
            s += Fraction(1, p)
        return s <= 1

    # Main Strategy
    best_classes = []
    best_L = 0

    # Candidate Generation
    candidates = []
    
    # 1. All primes >= 11
    c1 = [p for p in primes if p >= 11]
    if fits_budget(c1):
        candidates.append(c1)
        
    # 2. All primes >= 13
    c2 = [p for p in primes if p >= 13]
    if fits_budget(c2):
        candidates.append(c2)

    # 3. Small primes + large primes
    small_primes = [2, 3, 5, 7]
    for sp in small_primes:
        c3 = [sp] + [p for p in primes if p >= 13]
        if fits_budget(c3):
            candidates.append(c3)
            
    # 4. Pairs of small primes + large primes
    for i in range(len(small_primes)):
        for j in range(i+1, len(small_primes)):
            c4 = [small_primes[i], small_primes[j]] + [p for p in primes if p >= 17]
            if fits_budget(c4):
                candidates.append(c4)
                
    # 5. Specific known good sets
    c5 = [2, 3, 7] + [p for p in primes if p >= 17]
    if fits_budget(c5):
        candidates.append(c5)
        
    c6 = [2, 5, 7] + [p for p in primes if p >= 17]
    if fits_budget(c6):
        candidates.append(c6)
        
    c7 = [3, 5, 7] + [p for p in primes if p >= 17]
    if fits_budget(c7):
        candidates.append(c7)
    
    # 6. More systematic
    for cutoff in [11, 13, 17, 19, 23]:
        base = [p for p in primes if p >= cutoff]
        if fits_budget(base):
            candidates.append(base)
        for sp in small_primes:
            c_new = [sp] + [p for p in primes if p >= cutoff and p != sp]
            if fits_budget(c_new):
                candidates.append(c_new)
            
    # Remove duplicates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = frozenset(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
            
    # Sort candidates by number of primes
    unique_candidates.sort(key=lambda x: -len(x))
    
    # Evaluate candidates
    for i, cand_primes in enumerate(unique_candidates):
        if time.time() > deadline:
            break
        
        # Greedy Assignment with Bitset
        residues = {}
        current_mask = 0
        current_L = 0
        
        # Process primes in order of increasing p? Or decreasing?
        # Larger primes have fewer residues to try? No, p residues.
        # Smaller primes have more "impact" on low numbers.
        # Let's try increasing p first to secure the small numbers.
        primes_sorted = sorted(cand_primes)
        
        for p in primes_sorted:
            best_r = 0
            best_L_local = -1
            best_mask_local = 0
            
            masks_for_p = class_masks[p]
            
            for r in range(p):
                m = masks_for_p[r]
                new_mask = current_mask | m
                L = get_coverage_length(new_mask, MAX_CHECK)
                if L > best_L_local:
                    best_L_local = L
                    best_r = r
                    best_mask_local = new_mask
            
            residues[p] = best_r
            current_mask = best_mask_local
            current_L = best_L_local
            
        classes = [[p, residues[p]] for p in sorted(cand_primes)]
        
        # Local Search with Bitset
        improved = True
        iters = 0
        max_iter = 50
        
        while improved and iters < max_iter:
            improved = False
            iters += 1
            
            for i_idx, p in enumerate(sorted(cand_primes)):
                r_current = residues[p]
                masks_for_p = class_masks[p]
                current_mask_without_p = current_mask & ~masks_for_p[r_current]
                
                best_r = r_current
                best_L = current_L
                best_mask = current_mask
                
                for r_new in range(p):
                    if r_new == r_current:
                        continue
                    m_new = masks_for_p[r_new]
                    new_mask = current_mask_without_p | m_new
                    L = get_coverage_length(new_mask, MAX_CHECK)
                    
                    if L > best_L:
                        best_L = L
                        best_r = r_new
                        best_mask = new_mask
                
                if best_r != r_current:
                    residues[p] = best_r
                    current_mask = best_mask
                    current_L = best_L
                    improved = True
        
        if current_L > best_L:
            best_L = current_L
            best_classes = [[p, residues[p]] for p in sorted(cand_primes)]
            
    # Final Verification
    if best_classes:
        if fits_budget([p for p, r in best_classes]):
            return best_classes
            
    return [[2, 1]]
