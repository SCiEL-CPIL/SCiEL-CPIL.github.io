import math
import time
import numpy as np
from collections import defaultdict

def _sigma_one(n):
    """Exact sum of divisors of n by trial division up to sqrt(n)."""
    if n < 1:
        return 0
    res = 1
    tmp = n
    p = 2
    while p * p <= tmp:
        if tmp % p == 0:
            s = 1
            pw = 1
            while tmp % p == 0:
                tmp //= p
                pw *= p
                s += pw
            res *= s
        p += 1 if p == 2 else 2
    if tmp > 1:
        res *= (1 + tmp)
    return res

def _is_prime(n):
    """Deterministic Miller-Rabin for n < 3.3e24, or simple trial division for smaller."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _find_prime_in_range(low, high):
    """Find a prime in [low, high]. Returns None if not found."""
    if low < 2:
        low = 2
    # Ensure odd
    if low % 2 == 0:
        low += 1
    for p in range(low, high + 1, 2):
        if _is_prime(p):
            return p
    return None

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # 5s safety margin before 150s kill

    best_M = 0
    best_ks = []
    best_count = 0
    
    # Fallback
    fallback_M = 1512000
    fallback_ks = [756, 1260, 1680, 2520]
    best_M = fallback_M
    best_ks = list(fallback_ks)
    best_count = len(fallback_ks)

    # Try progressively larger sieves
    # NumPy sieve is O(N log N) but with low constant factor
    # 10^7 ~ 1-2s
    # 5*10^7 ~ 5-10s
    # 10^8 ~ 15-20s
    sizes = [10**6, 5 * 10**6, 10**7, 2 * 10**7, 5 * 10**7, 10**8]

    for N in sizes:
        elapsed = time.time() - t0
        if elapsed > DEADLINE - 10:
            break
            
        try:
            # Sieve sigma using NumPy
            sigma = np.zeros(N + 1, dtype=np.int64)
            for i in range(1, N + 1):
                sigma[i::i] += i
            
            # Compute H' = a * sigma(a) for a in 1..N
            a_vals = np.arange(1, N + 1, dtype=np.int64)
            H_vals = a_vals * sigma[1:]
            
            # Find the most frequent H'
            # unique returns sorted unique values and their counts
            unique_H, counts = np.unique(H_vals, return_counts=True)
            
            # Find the index of the max count
            max_count_idx = np.argmax(counts)
            max_count = counts[max_count_idx]
            
            if max_count > best_count:
                best_H = unique_H[max_count_idx]
                
                # Get the a's that produce this best_H
                # We need to filter a_vals where H_vals == best_H
                # This creates a boolean mask
                mask = (H_vals == best_H)
                a_list = a_vals[mask].tolist()
                
                # Verify that a <= sqrt(H') is not strictly required for the intermediate step,
                # but the final k <= sqrt(M) is.
                # The condition k <= sqrt(M) was proven to hold automatically.
                
                # Now find a prime P such that:
                # 1. P > max(a_list)
                # 2. P * (P + 1) * best_H <= 10**18
                # 3. P is prime
                
                max_a = max(a_list)
                min_P = max_a + 1
                
                # Max P allowed
                # P^2 * best_H <= 10**18  => P <= sqrt(10**18 / best_H)
                if best_H > 0:
                    max_P = int(math.isqrt(10**18 // best_H))
                else:
                    max_P = 10**9 # Fallback, shouldn't happen for H>0
                
                if min_P > max_P:
                    # Cannot find such P, skip this H'
                    continue
                    
                # Find a prime in [min_P, max_P]
                # We can just check the largest candidate downwards or smallest upwards.
                # Largest P gives largest M, but any works.
                # Let's try to find one quickly.
                P = _find_prime_in_range(min_P, max_P)
                
                if P is not None:
                    M = P * (P + 1) * best_H
                    if M <= 10**18 and M >= 1:
                        # Construct k's
                        ks = [a * P for a in a_list]
                        
                        # Verify
                        valid = True
                        seen = set()
                        for k in ks:
                            if not isinstance(k, int):
                                valid = False
                                break
                            if k < 1 or k > math.isqrt(M):
                                valid = False
                                break
                            if k in seen:
                                valid = False
                                break
                            seen.add(k)
                            if k * _sigma_one(k) != M:
                                valid = False
                                break
                        
                        if valid:
                            best_M = M
                            best_ks = ks
                            best_count = len(ks)
                            # Update best_count to reflect verified solutions
                            # We can break early if we find a very good one?
                            # No, keep searching for better H' in larger N if time permits.
                            # But if we found 100 solutions, it's unlikely to find more.
                            # However, larger N might find H' with more collisions.
                            
        except MemoryError:
            break
        except Exception:
            break

    # Final safety check
    if not best_ks:
        best_M = fallback_M
        best_ks = list(fallback_ks)

    # Ensure types
    M = int(best_M)
    ks = [int(k) for k in best_ks]

    # Final verification
    valid = True
    seen = set()
    for k in ks:
        if not isinstance(k, int) or isinstance(k, bool):
            valid = False
            break
        if k < 1 or k > math.isqrt(M):
            valid = False
            break
        if k in seen:
            valid = False
            break
        seen.add(k)
        if k * _sigma_one(k) != M:
            valid = False
            break
            
    if not valid or M < 1 or M > 10**18:
        M = int(fallback_M)
        ks = [int(k) for k in fallback_ks]

    return {'M': M, 'k': ks}
