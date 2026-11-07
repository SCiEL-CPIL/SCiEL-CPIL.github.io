# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

def _is_perfect_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    MAX_VAL = 10210
    N = 210
    
    # Compute smallest prime factor sieve
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL ** 0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute squarefree signature as frozenset for each number
    sigs = [frozenset() for _ in range(MAX_VAL + 1)]
    for n in range(2, MAX_VAL + 1):
        m = n
        sig = set()
        while m > 1:
            p = spf[m]
            cnt = 0
            while m % p == 0:
                m //= p
                cnt ^= 1
            if cnt:
                sig.add(p)
        sigs[n] = frozenset(sig)
    
    sig_210 = frozenset({2, 3, 5, 7})
    
    # Build lookup: signature -> list of numbers with that signature (sorted)
    sig_to_nums = defaultdict(list)
    for i in range(211, MAX_VAL + 1):
        sig_to_nums[sigs[i]].append(i)
    
    best_result = None
    best_max = MAX_VAL + 1
    
    # Search for minimum M such that there exists a valid subset of {211..M}
    for M in range(211, MAX_VAL + 1):
        if time.time() > deadline:
            break
        if M >= best_max:
            break
        
        sig_M = sigs[M]
        
        # Check 1-element: sig(M) == sig_210
        if sig_M == sig_210:
            best_max = M
            best_result = [M]
            break
        
        # Check 2-element: need a in (210, M) with sigs[a] == sig_M ^ sig_210
        target2 = sig_M ^ sig_210
        if target2 in sig_to_nums:
            for a in sig_to_nums[target2]:
                if 210 < a < M:
                    best_max = M
                    best_result = [a, M]
                    break
        if best_result is not None and best_max == M:
            break
        
        # Check 3-element: need a, b in (210, M) with sigs[a] ^ sigs[b] == sig_M ^ sig_210
        target3 = sig_M ^ sig_210
        seen = {}
        found = False
        for a in range(211, M):
            if time.time() > deadline:
                break
            need = sigs[a] ^ target3
            if need in seen:
                b = seen[need]
                if b != a:
                    best_max = M
                    best_result = [a, b, M]
                    found = True
                    break
            else:
                seen[sigs[a]] = a
        if found:
            break
        
        # Check 4-element: need a,b,c in (210,M) with sigs[a]^sigs[b]^sigs[c] == target3
        # Only do this for small M to keep it fast
        if M <= 500:
            target4 = sig_M ^ sig_210
            # Build a list of (sig, num) for numbers in (210, M)
            nums_range = list(range(211, M))
            n_elems = len(nums_range)
            if n_elems >= 3:
                # For each pair (a,b), compute sig[a]^sig[b], store in dict
                # Then for each c, check if sig[c] ^ pair_sig == target4
                # i.e., pair_sig == sig[c] ^ target4
                pair_sigs = {}
                found4 = False
                for i_idx in range(n_elems):
                    if time.time() > deadline:
                        break
                    a = nums_range[i_idx]
                    sa = sigs[a]
                    for j_idx in range(i_idx + 1, n_elems):
                        b = nums_range[j_idx]
                        sb = sigs[b]
                        pair_sig = sa ^ sb
                        # For each c > b, check if sigs[c] ^ pair_sig == target4
                        # i.e., sigs[c] == pair_sig ^ target4
                        needed_c_sig = pair_sig ^ target4
                        # We need a number c in (b, M) with sigs[c] == needed_c_sig
                        # This is expensive to check directly. Instead, use a different approach:
                        # Store pair_sigs and then iterate c
                        pass
                # Alternative: iterate c from 211 to M-1, and for each c,
                # need a,b in (210,c) with sigs[a]^sigs[b] == sigs[c]^target4
                # This is the same as 3-element search but nested
                # Too slow. Skip 4-element for now.
                pass
    
    # If nothing found (shouldn't happen), fallback
    if best_result is None:
        best_result = [840]
    
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
