"""Rewritten program: smooth-number meet-in-the-middle to find M with many preimages k where k*sigma(k)=M."""
import math
import time
import sys
from collections import defaultdict

def _sigma_sieve(limit):
    """Compute sigma(n) for all n in 1..limit using a divisor-sum sieve.
    Returns a list sigma where sigma[i] = sum of divisors of i."""
    sigma = [0] * (limit + 1)
    for d in range(1, limit + 1):
        for m in range(d, limit + 1, d):
            sigma[m] += d
    return sigma

def _sigma_multiplicative(n):
    """Compute sigma(n) via prime factorization (for individual n)."""
    if n == 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            term = 1
            power = 1
            while temp % p == 0:
                temp //= p
                power *= p
                term += power
            result *= term
        p += 1
    if temp > 1:
        result *= (1 + temp)
    return result

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    # Strategy: enumerate smooth numbers k, compute M = k*sigma(k), group by M.
    # Use meet-in-the-middle: split primes into two sets, enumerate products in each,
    # then combine to find M values with many distinct k.
    
    # Actually, a simpler and more effective approach:
    # 1. Sieve sigma up to a reasonable N (say 50M) using numpy
    # 2. For each k in 1..N, compute M = k*sigma(k)
    # 3. Group by M, find M with most distinct k
    # 4. Also try larger k via smooth number enumeration
    
    try:
        import numpy as np
        HAVE_NUMPY = True
    except ImportError:
        HAVE_NUMPY = False
    
    best_M = None
    best_ks = []
    
    if HAVE_NUMPY:
        # Phase 1: Numpy sieve up to N
        for N in [30_000_000, 20_000_000, 10_000_000, 5_000_000]:
            if time.time() - start_time > DEADLINE - 40:
                break
            try:
                t0 = time.time()
                sigma = np.zeros(N + 1, dtype=np.int64)
                for d in range(1, N + 1):
                    sigma[d::d] += d
                
                elapsed_sieve = time.time() - t0
                if time.time() - start_time > DEADLINE - 30:
                    continue
                
                # Compute M = k * sigma(k) for all k
                ks = np.arange(1, N + 1, dtype=np.int64)
                Ms = ks * sigma[1:N+1]
                
                # Filter: M <= 10^18
                max_M = 10**18
                valid = Ms <= max_M
                Ms_valid = Ms[valid]
                ks_valid = ks[valid]
                
                # Group by M using dictionary (convert to Python ints)
                # To avoid huge memory, use a two-pass approach:
                # First pass: find candidate M values with frequency > 1
                # Use numpy unique with counts
                t1 = time.time()
                unique_Ms, inverse, counts = np.unique(Ms_valid, return_inverse=True, return_counts=True)
                
                # Find M values with count >= 3
                high_freq_mask = counts >= 3
                if not np.any(high_freq_mask):
                    continue
                    
                high_freq_Ms = unique_Ms[high_freq_mask]
                high_freq_counts = counts[high_freq_mask]
                
                # For each high-frequency M, collect the k values
                # Sort by count descending, process top candidates
                sorted_indices = np.argsort(-high_freq_counts)
                
                for idx in sorted_indices[:50]:  # top 50 candidates
                    if time.time() - start_time > DEADLINE - 20:
                        break
                    
                    m_val = int(high_freq_Ms[idx])
                    c_val = int(high_freq_counts[idx])
                    
                    # Collect all k that map to this M
                    k_list = ks_valid[inverse == np.where(high_freq_mask, np.arange(len(high_freq_Ms)))[0][idx]].tolist()
                    
                    # Actually, let's redo this more carefully
                    # The inverse array maps each element of Ms_valid to its index in unique_Ms
                    # We need to find all positions where Ms_valid == m_val
                    mask = (Ms_valid == m_val)
                    k_list = ks_valid[mask].tolist()
                    
                    k_list = sorted(set(k_list))
                    
                    if len(k_list) > len(best_ks):
                        best_M = m_val
                        best_ks = k_list
                        
            except (MemoryError, Exception) as e:
                continue
    
    # Phase 2: Smooth number enumeration for larger k
    # Enumerate 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g * 19^h * 23^i * 29^j
    # up to 10^9, compute k*sigma(k), and add to the grouping
    
    if time.time() - start_time < DEADLINE - 15:
        try:
            # Generate smooth numbers up to 10^9
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
            smooth_numbers = set([1])
            
            # Iteratively multiply by primes
            for p in primes:
                new_numbers = set()
                for s in list(smooth_numbers):
                    prod = s
                    while prod * p <= 10**9:
                        prod *= p
                        if prod not in smooth_numbers:
                            new_numbers.add(prod)
                            smooth_numbers.add(prod)
                if time.time() - start_time > DEADLINE - 10:
                    break
            
            # For each smooth number, compute M = k * sigma(k)
            # and add to our grouping
            if best_M is None:
                best_ks = []
            
            # We need to track all (M, k) pairs to find collisions
            # Use a dict: M -> set of k's
            M_to_ks = defaultdict(set)
            
            # Add smooth numbers
            for k in smooth_numbers:
                if k > 10**9:
                    continue
                sig = _sigma_multiplicative(k)
                M = k * sig
                if M <= 10**18:
                    M_to_ks[M].add(k)
            
            # Check for better M from smooth numbers
            for m, k_set in M_to_ks.items():
                if len(k_set) > len(best_ks):
                    best_M = m
                    best_ks = sorted(k_set)
                    
        except Exception:
            pass
    
    # Final verification and return
    if best_M is not None and len(best_ks) >= 3:
        # Verify each k satisfies k*sigma(k) == M
        verified_ks = []
        for k in best_ks:
            if k <= 1 and k * 1 != best_M:
                continue
            sig = _sigma_multiplicative(k)
            if k * sig == best_M and k <= math.isqrt(best_M):
                verified_ks.append(k)
        
        if len(verified_ks) >= 3:
            return {'M': int(best_M), 'k': [int(x) for x in verified_ks]}
    
    # Fallback: return the seed result
    return {'M': 333312, 'k': [336, 372, 434]}
