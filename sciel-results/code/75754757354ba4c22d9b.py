# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

P = {'n': 210, 'maximum_window': 10000, 'maximum_size': 100}

def _factorize(n):
    """Return dict of prime -> exponent for n."""
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

def _squarefree_signature(n):
    """Return frozenset of primes with odd exponent in n."""
    factors = _factorize(n)
    sig = frozenset(p for p, e in factors.items() if e % 2 == 1)
    return sig

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s
    
    target_n = P['n']  # 210
    target_sig = _squarefree_signature(target_n)  # frozenset({2,3,5,7})
    
    min_val = target_n + 1  # 211
    max_val = target_n + P['maximum_window']  # 10210
    
    # Precompute signatures
    sigs = {}
    for i in range(min_val, max_val + 1):
        sigs[i] = _squarefree_signature(i)
        
    # Search for minimal M < 240
    for M in range(min_val, 240):
        if time.time() > deadline:
            break
        needed_sig = target_sig ^ sigs[M]
        candidates = list(range(min_val, M))
        
        # Build sig_to_nums for candidates
        sig_to_nums = defaultdict(list)
        for c in candidates:
            sig_to_nums[sigs[c]].append(c)
            
        found_subset = None
        
        # Size 1
        if needed_sig in sig_to_nums and sig_to_nums[needed_sig]:
            found_subset = [sig_to_nums[needed_sig][0]]
            
        # Size 2
        if not found_subset:
            for c1 in candidates:
                needed_c2_sig = needed_sig ^ sigs[c1]
                if needed_c2_sig in sig_to_nums:
                    nums = sig_to_nums[needed_c2_sig]
                    # Find a num != c1
                    for c2 in nums:
                        if c2 != c1:
                            found_subset = [c1, c2]
                            break
                if found_subset:
                    break
                    
        # Size 3
        if not found_subset:
            # Iterate pairs of candidates
            # To avoid duplicates and ensure distinctness, we can just iterate all pairs
            # and check if a third exists
            for i in range(len(candidates)):
                c1 = candidates[i]
                for j in range(len(candidates)):
                    if i == j: continue
                    c2 = candidates[j]
                    needed_c3_sig = needed_sig ^ sigs[c1] ^ sigs[c2]
                    if needed_c3_sig in sig_to_nums:
                        nums = sig_to_nums[needed_c3_sig]
                        for c3 in nums:
                            if c3 != c1 and c3 != c2:
                                found_subset = [c1, c2, c3]
                                break
                    if found_subset:
                        break
                if found_subset:
                    break

        # Size 4
        if not found_subset:
            # Iterate triples? Might be slow if candidates are many.
            # But candidates < 29.
            # Number of triples ~ 29*28*27 / 6 ~ 3600.
            # For each, check dict. Fast.
            for i in range(len(candidates)):
                c1 = candidates[i]
                for j in range(len(candidates)):
                    if i == j: continue
                    c2 = candidates[j]
                    for k in range(len(candidates)):
                        if k == i or k == j: continue
                        c3 = candidates[k]
                        needed_c4_sig = needed_sig ^ sigs[c1] ^ sigs[c2] ^ sigs[c3]
                        if needed_c4_sig in sig_to_nums:
                            nums = sig_to_nums[needed_c4_sig]
                            for c4 in nums:
                                if c4 != c1 and c4 != c2 and c4 != c3:
                                    found_subset = [c1, c2, c3, c4]
                                    break
                        if found_subset:
                            break
                    if found_subset:
                        break
                if found_subset:
                    break

        if found_subset:
            result = found_subset + [M]
            # Verify
            prod = target_n
            for v in result:
                prod *= v
            if _is_perfect_square(prod):
                return [int(x) for x in result]
                
    # Fallback to known solution
    result_set = [224, 240]
    prod = target_n
    for v in result_set:
        prod *= v
    if not _is_perfect_square(prod):
        return [224, 240]
        
    return [int(x) for x in result_set]
# EVOLVE-BLOCK-END
