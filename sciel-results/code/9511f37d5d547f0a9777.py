"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _spf_sieve(limit):
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _pf_set(n, spf):
    s = set()
    while n > 1:
        p = spf[n]
        s.add(p)
        while n % p == 0:
            n //= p
    return s

def _score_reduced(vals, spf):
    """Count distinct prime factors of all pairwise sums of vals."""
    ps = set()
    n = len(vals)
    for i in range(n):
        vi = vals[i]
        for j in range(i + 1, n):
            s = vi + vals[j]
            if s > 1:
                ps |= _pf_set(s, spf)
    return len(ps)

def _score_full(arr, spf):
    """Count distinct prime factors of all pairwise sums of arr."""
    ps = set()
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            s = ai + arr[j]
            if s > 1:
                ps |= _pf_set(s, spf)
    return len(ps)

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # 5s safety margin before 90s kill
    
    MAX_SUM = 20000
    spf = _spf_sieve(MAX_SUM)
    
    best_set = list(range(1, 13))
    best_score = _score_full(best_set, spf)
    
    def try_update(arr):
        nonlocal best_set, best_score
        s = _score_full(arr, spf)
        if s < best_score:
            best_score = s
            best_set = arr[:]
        return s
    
    # Phase 1: Systematic multi-modulus search
    # For each modulus M, find 12 distinct multiples of M (value = M * k)
    # such that the k-values have pairwise sums with few distinct prime factors.
    # The actual prime factors of M*k1 + M*k2 = M*(k1+k2) are primes(M) union primes(k1+k2).
    # So we want: primes(M) small, AND primes(k1+k2) for all pairs small.
    
    # Try moduli from a curated list of highly composite / smooth numbers
    candidate_moduli = []
    for M in range(2, 2001):
        pfM = _pf_set(M, spf)
        if len(pfM) <= 4:  # M has at most 4 distinct prime factors
            candidate_moduli.append((M, len(pfM), pfM))
    
    # Sort by number of prime factors, then by M
    candidate_moduli.sort(key=lambda x: (x[1], x[0]))
    
    # For each promising modulus, search for good k-sets
    # k values must be distinct, and M*k <= 10000, so k <= 10000//M
    # We need 12 distinct k values from 1..max_k
    
    # Precompute: for small max_k, enumerate all 12-subsets is infeasible.
    # Instead, use local search on k-values.
    
    random.seed(12345)
    
    # Strategy: for each modulus M, do a local search on k-values
    # Objective: minimize distinct primes in {primes(M)} union {primes(ki+kj) for all i<j}
    
    def score_with_modulus(k_vals, M, pfM, spf):
        """Score = |primes(M) union primes(ki+kj) for all pairs|"""
        ps = set(pfM)
        n = len(k_vals)
        for i in range(n):
            ki = k_vals[i]
            for j in range(i + 1, n):
                s = ki + k_vals[j]
                if s > 1:
                    ps |= _pf_set(s, spf)
        return len(ps)
    
    # For each modulus, try to find a good set of 12 k-values
    # Use greedy + local search
    
    # Limit work per modulus
    max_k_searches = 50  # number of local search restarts per modulus
    
    processed_moduli = 0
    for M, npfM, pfM in candidate_moduli:
        if time.time() > deadline:
            break
        if processed_moduli >= 300:  # cap total moduli to try
            break
        
        max_k = 10000 // M
        if max_k < 12:
            continue
        
        processed_moduli += 1
        
        # Quick upper bound: even if all ki+kj are smooth, we have at least npfM primes
        # Lower bound is npfM. If npfM >= best_score, skip
        if npfM >= best_score:
            continue
        
        # Local search on k-values
        # Start with k = 1, 2, ..., 12 (if max_k >= 12)
        best_k = None
        best_k_score = float('inf')
        
        # Try several initial configurations
        for restart in range(min(max_k_searches, 10)):
            if time.time() > deadline:
                break
            
            if restart == 0:
                k_vals = list(range(1, 13))
            elif restart == 1:
                # Spread out
                step = max(1, max_k // 12)
                k_vals = [1 + i * step for i in range(12)]
                if k_vals[-1] > max_k:
                    k_vals = list(range(1, 13))
            else:
                # Random
                k_vals = random.sample(range(1, max_k + 1), 12)
                k_vals.sort()
            
            # Ensure all k*M <= 10000
            if k_vals[-1] * M > 10000:
                continue
            
            ks = score_with_modulus(k_vals, M, pfM, spf)
            if ks < best_k_score:
                best_k_score = ks
                best_k = k_vals[:]
            
            # Local search: swap one k-value
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for idx in range(12):
                    if time.time() > deadline:
                        break
                    old_k = k_vals[idx]
                    # Try replacing with random k
                    for _ in range(15):
                        if time.time() > deadline:
                            break
                        new_k = random.randint(1, max_k)
                        if new_k == old_k:
                            continue
                        if new_k in k_vals:
                            continue
                        new_k_vals = k_vals[:]
                        new_k_vals[idx] = new_k
                        new_k_vals.sort()
                        if new_k_vals[-1] * M > 10000:
                            continue
                        nks = score_with_modulus(new_k_vals, M, pfM, spf)
                        if nks < best_k_score:
                            best_k_score = nks
                            best_k = new_k_vals[:]
                            k_vals = new_k_vals
                            improved = True
                            break
                        if nks < ks:
                            k_vals = new_k_vals
                            ks = nks
                            improved = True
                            break
        
        # Convert best k-set to actual values
        if best_k is not None:
            arr = [k * M for k in best_k]
            # Verify distinct and <= 10000
            if len(set(arr)) == 12 and max(arr) <= 10000:
                try_update(arr)
    
    # Phase 2: General local search on the best set found
    if time.time() < deadline - 5:
        # Try to improve by replacing individual elements
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                if time.time() > deadline - 5:
                    break
                old = best_set[i]
                # Try a range of replacements
                for _ in range(100):
                    if time.time() > deadline - 5:
                        break
                    new_val = random.randint(1, 10000)
                    if new_val == old:
                        continue
                    if new_val in best_set:
                        continue
                    test = best_set[:]
                    test[i] = new_val
                    test.sort()
                    ns = _score_full(test, spf)
                    if ns < best_score:
                        best_score = ns
                        best_set = test
                        improved = True
                        break
    
    # Phase 3: Final polish - try nearby values for each element
    if time.time() < deadline - 3:
        for i in range(12):
            if time.time() > deadline - 3:
                break
            old = best_set[i]
            for delta in range(-20, 21):
                if delta == 0:
                    continue
                new_val = old + delta
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in best_set:
                    continue
                test = best_set[:]
                test[i] = new_val
                test.sort()
                ns = _score_full(test, spf)
                if ns < best_score:
                    best_score = ns
                    best_set = test
    
    return best_set
# EVOLVE-BLOCK-END
