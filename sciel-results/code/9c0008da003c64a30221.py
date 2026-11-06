"""Improved program: NumPy vectorized segmented sieve for long runs of distinct totients."""
import time
import math
import random
import sys
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    # Precompute small primes for trial division and segmented sieve
    def sieve_primes(limit):
        is_prime = np.ones(limit + 1, dtype=np.uint8)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = 0
        return np.array([i for i in range(2, limit + 1) if is_prime[i]], dtype=np.int64)

    small_primes = sieve_primes(100000)  # primes up to 100000

    def phi_trial(x, primes=None):
        """Compute phi(x) using trial division by primes."""
        if primes is None:
            primes = small_primes
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result

    def find_longest_distinct_run(phi_vals, offset=0):
        """Find the longest run of consecutive distinct values in phi_vals.
        Returns (best_k, best_start) where best_start is the actual number (offset + index)."""
        best_k = 0
        best_start = offset
        seen = set()
        run_start = 0
        run_len = 0

        for i, v in enumerate(phi_vals):
            if v not in seen:
                seen.add(v)
                run_len += 1
            else:
                if run_len > best_k:
                    best_k = run_len
                    best_start = offset + run_start
                seen = {v}
                run_start = i
                run_len = 1

        if run_len > best_k:
            best_k = run_len
            best_start = offset + run_start

        return best_k, best_start

    def check_window(start, length, primes=None):
        """Check if a window has all distinct phi values, return max distinct prefix length."""
        if primes is None:
            primes = small_primes
        seen = set()
        k = 0
        for i in range(length):
            v = phi_trial(start + i, primes)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    def segmented_phi_numpy(low, high, primes=None):
        """Compute phi(n) for all n in [low, high] using NumPy segmented sieve.
        Returns a list of phi values indexed by (n - low)."""
        if primes is None:
            primes = small_primes
        
        n = high - low + 1
        # Initialize phi array
        phi = np.arange(low, high + 1, dtype=np.int64)
        
        # For each prime p <= sqrt(high), apply the (1 - 1/p) factor to multiples of p
        max_p = int(math.isqrt(high)) + 1
        for p in primes:
            if p > max_p:
                break
            p = int(p)
            # Find first multiple of p in [low, high]
            first = ((low + p - 1) // p) * p
            if first < p:
                first = p
            if first > high:
                continue
            # Apply phi[j] -= phi[j] // p for all multiples of p in range
            indices = np.arange(first - low, n, p)
            phi[indices] = phi[indices] - phi[indices] // p
        
        # Handle remaining prime factors > sqrt(high)
        # For each position, check if there's a remaining prime factor
        remaining = phi.copy()
        # We need to identify which numbers still have a prime factor > sqrt(high)
        # A number n has a remaining prime factor if after removing all small prime factors,
        # the remaining part > 1 and is prime (or n itself was prime)
        
        # For numbers where phi changed (had small factors), check if remaining > 1
        # For numbers where phi didn't change, the number itself might be prime or have a large factor
        
        # Strategy: for each i, compute what's left after removing all small prime factors
        # This is expensive to do naively, so we use a different approach:
        # If phi[i] == original value, then n had no small prime factors -> n is prime or product of large primes
        # Actually, let's just do trial division for the remaining part
        
        # More efficient: track which numbers still need processing
        # For each i, if the number had no prime factor <= sqrt(high), it's either 1 or prime
        # But we need to handle composites with two large prime factors too
        
        # Let's use a simpler approach: for each position, divide out all small prime factors
        # and check if remainder > 1
        # This is O(n * num_primes) which could be slow, but let's try with numpy tricks
        
        # Actually, the most reliable approach for the remaining factor:
        # For each i, we need to know if (low+i) has a prime factor > sqrt(high)
        # that wasn't already accounted for.
        
        # Let's do this: for each i, compute y = low+i, divide by all small primes,
        # if y > 1, then apply phi[i] -= phi[i] // y
        # But this is O(n * pi(sqrt(high))) which for n=10^6 and pi(10^6)~78000 is too slow
        
        # Alternative: use the fact that if n has no prime factor <= sqrt(n), it's prime.
        # But we already processed all p <= sqrt(high) which >= sqrt(n) for n <= high.
        # So after processing all p <= sqrt(high), any remaining factor > 1 must be prime.
        # Wait, that's not quite right. If n = p*q where p <= sqrt(high) < q, 
        # then after removing p, q remains. And q > sqrt(high) >= sqrt(n/p) but q could be < sqrt(n).
        # Actually no: if p <= sqrt(high) and n <= high, then q = n/p >= ... 
        # Let me think again. After removing all prime factors <= sqrt(high), 
        # the remaining factor y satisfies: if y > 1, then y is prime OR y is a product of primes all > sqrt(high).
        # But if y is a product of two primes both > sqrt(high), then y > high, which contradicts y <= n <= high.
        # So y must be prime (or 1).
        
        # So: for each i, compute the remaining factor by dividing out all small primes.
        # We can do this more efficiently by tracking the "reduced" value.
        
        # Let me use a different strategy: maintain a "reduced" array
        reduced = np.arange(low, high + 1, dtype=np.int64)
        for p in primes:
            if p > max_p:
                break
            p = int(p)
            first = ((low + p - 1) // p) * p
            if first < p:
                first = p
            if first > high:
                continue
            indices = np.arange(first - low, n, p)
            while np.any(reduced[indices] % p == 0):
                mask = reduced[indices] % p == 0
                reduced[indices[mask]] = reduced[indices[mask]] // p
        
        # Now for each i, if reduced[i] > 1, apply phi[i] -= phi[i] // reduced[i]
        mask = reduced > 1
        phi[mask] = phi[mask] - phi[mask] // reduced[mask]
        
        return phi

    best_k = 1
    best_start = 1

    # Phase 1: Segmented sieve search using NumPy over large ranges
    CHUNK_SIZE = 2 * 10**6  # 2 million numbers per chunk
    
    # Search from 1 up to as far as time permits
    search_start = 1
    while time_remaining() > 15.0:
        chunk_end = min(search_start + CHUNK_SIZE - 1, 10**12)
        try:
            phi_vals = segmented_phi_numpy(search_start, chunk_end)
            # Convert to list for the distinct run finder (or use numpy-based approach)
            k, s = find_longest_distinct_run(phi_vals.tolist(), search_start)
            if k > best_k:
                best_k = k
                best_start = s
                if best_k >= 200:
                    break
        except MemoryError:
            # Try smaller chunk
            CHUNK_SIZE = 500000
            continue
        
        search_start = chunk_end + 1
        if search_start > 10**12:
            break

    # Phase 2: Targeted search in specific regions
    random.seed(42)
    
    # Try windows around interesting locations
    primorials = [2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870]
    
    for prim in primorials:
        if time_remaining() < 5.0:
            break
        max_mult = min(10**12 // prim, 10000)
        for _ in range(min(30, max_mult)):
            if time_remaining() < 3.0:
                break
            mult = random.randint(1, max_mult)
            start = mult * prim
            if start < 1 or start + 200 > 10**12:
                continue
            k = check_window(start, 200)
            if k > best_k:
                best_k = k
                best_start = start
            for offset in [-5, -1, 1, 5]:
                s2 = start + offset
                if s2 < 1 or s2 + 200 > 10**12:
                    continue
                k2 = check_window(s2, 200)
                if k2 > best_k:
                    best_k = k2
                    best_start = s2

    # Phase 3: Random search in mid-range
    if time_remaining() > 5.0 and best_k < 200:
        for _ in range(100):
            if time_remaining() < 2.0:
                break
            s = random.randint(10**6, 10**11)
            if s + 200 > 10**12:
                s = 10**12 - 200
            k = check_window(s, 200)
            if k > best_k:
                best_k = k
                best_start = s

    # Phase 4: Local refinement around best found
    if best_k > 1 and time_remaining() > 3.0:
        for delta in range(-30, 31):
            if time_remaining() < 1.0:
                break
            s = best_start + delta
            if s < 1 or s + 200 > 10**12:
                continue
            k = check_window(s, 200)
            if k > best_k:
                best_k = k
                best_start = s

    # Ensure constraints
    if best_k > 200:
        best_k = 200
    if best_start + best_k - 1 > 10**12:
        best_start = 10**12 - best_k + 1
    if best_start < 1:
        best_start = 1
    if best_k < 1:
        best_k = 1

    return {'start': best_start, 'length': best_k}
# EVOLVE-BLOCK-END
