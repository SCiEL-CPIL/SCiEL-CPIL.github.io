"""Rewritten program: find long runs of consecutive integers with distinct totients."""
import time
import math
import random
import sys

def construct():
    start_time = time.time()
    TIME_LIMIT = 115.0  # safety margin before 120s kill

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    def sieve_primes(limit):
        """Return list of primes up to limit."""
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]

    # Precompute primes up to 10^6 for fast factorization
    primes = sieve_primes(1000000)

    def phi_factorize(n, primes):
        """Compute phi(n) using trial division by precomputed primes."""
        if n == 1:
            return 1
        result = n
        y = n
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

    def phi_batch(start, length, primes):
        """Compute phi for range [start, start+length)."""
        vals = [0] * length
        for i in range(length):
            vals[i] = phi_factorize(start + i, primes)
        return vals

    def find_best_run_in_range(start, end, primes):
        """Find the best (longest) run of distinct phi values in [start, end)."""
        best_k = 1
        best_s = start
        # We need to check runs; use a sliding window approach
        # But for large ranges, we sample strategically
        # First, compute phi values for a chunk
        chunk_size = 5000
        seen = set()
        k = 0
        s = start
        for pos in range(start, end):
            if time_left() < 5.0:
                break
            v = phi_factorize(pos, primes)
            if v not in seen:
                seen.add(v)
                k += 1
                if k > best_k:
                    best_k = k
                    best_s = pos - k + 1
            else:
                # Reset: start new window
                # But we need to be smarter - use a different approach
                # For now, just reset
                # Actually, let's use a proper sliding window
                pass
        return best_k, best_s

    def find_best_run_sliding(start, end, primes):
        """Find best run using sliding window with phi computation."""
        best_k = 1
        best_s = start
        # Compute phi values in chunks
        chunk_size = 2000
        all_vals = []
        all_starts = []
        for cs in range(start, end, chunk_size):
            if time_left() < 5.0:
                break
            ce = min(cs + chunk_size, end)
            vals = phi_batch(cs, ce - cs, primes)
            all_vals.extend(vals)
            all_starts.append(cs)
        
        if not all_vals:
            return 1, start
        
        # Now find longest run of distinct values
        # Use a dictionary to track last seen position
        last_seen = {}
        k = 0
        s = all_starts[0]
        # We need to map indices back to positions
        # all_vals[i] corresponds to position all_starts[0] + i if we had one chunk
        # But with multiple chunks, let's just use index
        for i, v in enumerate(all_vals):
            pos = start + i
            if v in last_seen:
                # Collision: reset window
                # We need to find a new start
                # Simple approach: restart from i
                last_seen = {}
                k = 0
                s = pos
            last_seen[v] = pos
            k += 1
            if k > best_k:
                best_k = k
                best_s = pos - k + 1
        return best_k, best_s

    # Strategy:
    # 1. Search in several strategic ranges
    # 2. Use larger ranges for better chances of finding long runs
    # 3. Prioritize ranges where numbers have varied factorizations

    best_k = 1
    best_s = 1

    # Phase 1: Search small ranges quickly
    # Phase 2: Search larger ranges

    # Try several starting points
    # Known: runs of distinct phi values tend to occur where numbers have diverse prime factors
    # Let's search in ranges around highly composite numbers or random ranges

    # First, do a quick scan of [1, 100000]
    if time_left() > 30:
        k, s = find_best_run_sliding(1, 100000, primes)
        if k > best_k:
            best_k = k
            best_s = s

    # Search in larger ranges
    # Try ranges around multiples of large primes
    search_ranges = []
    
    # Add some strategic ranges
    # Range near 10^6
    search_ranges.append((10**6, 10**6 + 50000))
    # Range near 10^7
    search_ranges.append((10**7, 10**7 + 50000))
    # Range near 10^8
    search_ranges.append((10**8, 10**8 + 50000))
    # Random ranges
    random.seed(42)
    for _ in range(20):
        lo = random.randint(10**6, 10**11)
        search_ranges.append((lo, lo + 50000))

    for lo, hi in search_ranges:
        if time_left() < 10:
            break
        if hi > 10**12:
            hi = 10**12
        k, s = find_best_run_sliding(lo, hi, primes)
        if k > best_k:
            best_k = k
            best_s = s
            # If we found a good run, try to extend it
            # Search around best_s with larger window
            if best_k >= 5 and time_left() > 20:
                ext_lo = max(1, best_s - 10000)
                ext_hi = min(10**12, best_s + best_k + 10000)
                k2, s2 = find_best_run_sliding(ext_lo, ext_hi, primes)
                if k2 > best_k:
                    best_k = k2
                    best_s = s2

    # Phase 3: If we have time, do more extensive search
    if time_left() > 30:
        # Search in larger chunks
        for base in [10**5, 10**6, 10**7, 10**8, 10**9, 10**10]:
            if time_left() < 10:
                break
            for offset in range(0, 1000000, 100000):
                if time_left() < 5:
                    break
                lo = base + offset
                hi = lo + 100000
                if hi > 10**12:
                    break
                k, s = find_best_run_sliding(lo, hi, primes)
                if k > best_k:
                    best_k = k
                    best_s = s

    # Ensure constraints
    if best_s < 1:
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1
    if best_k > 200:
        best_k = 200

    return {'start': best_s, 'length': best_k}
