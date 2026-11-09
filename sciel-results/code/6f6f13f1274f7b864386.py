import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin

    # Sieve primes up to 10^6
    limit = 1000000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    prime_squares = [p * p for p in primes]

    def compute_phi_segment(start, end):
        """Compute phi(n) for n in [start, end] using segmented sieve."""
        size = end - start + 1
        phi = np.arange(start, end + 1, dtype=np.int64)
        rem = phi.copy()
        
        for i, p in enumerate(primes):
            if prime_squares[i] > end:
                break
            first = ((start + p - 1) // p) * p
            if first > end:
                continue
            idxs = np.arange(first - start, size, p)
            if len(idxs) == 0:
                continue
            mask = rem[idxs] % p == 0
            if not mask.any():
                continue
            sel = idxs[mask]
            phi[sel] = phi[sel] // p * (p - 1)
            # Remove all factors of p
            while True:
                still = rem[sel] % p == 0
                if not still.any():
                    break
                sel = sel[still]
                rem[sel] //= p
        
        # Handle remaining prime factors > 1
        mask = rem > 1
        if mask.any():
            phi[mask] = phi[mask] // rem[mask] * (rem[mask] - 1)
        
        return phi

    def find_longest_run_sliding(phi_vals):
        """Find longest consecutive run of distinct phi values using sliding window."""
        n = len(phi_vals)
        if n == 0:
            return 1, 0
        best_k = 1
        best_s = 0
        seen = {}
        left = 0
        for right in range(n):
            val = int(phi_vals[right])
            if val in seen and seen[val] >= left:
                left = seen[val] + 1
            seen[val] = right
            cur_k = right - left + 1
            if cur_k > best_k:
                best_k = cur_k
                best_s = left
            if best_k >= 200:
                break
        return best_k, best_s

    best_k = 1
    best_s = 1

    # Phase 1: Quick baseline with smaller segment
    if time.time() - start_time < deadline - 30:
        phi_vals = compute_phi_segment(1, 100000)
        k, s = find_longest_run_sliding(phi_vals)
        if k > best_k:
            best_k = k
            best_s = 1 + s

    # Phase 2: Search at various high ranges where collisions are rare
    # Use larger segments for efficiency
    seg_size = 500000
    candidates = []
    
    # High ranges: 10^11 to 10^12 with coarser sampling
    for base in range(10**11, 10**12 + 1, 10**11):
        candidates.append(base)
    
    # Add more granular sampling in the highest range
    for base in range(9*10**11, 10**12, 2*10**10):
        candidates.append(base)
    
    # Add some mid-range candidates
    for base in [10**10, 5*10**10, 10**11]:
        candidates.append(base)
    
    # Deduplicate and sort
    candidates = sorted(set(candidates))

    for base in candidates:
        if time.time() - start_time > deadline - 15:
            break
        if base < 1 or base > 10**12:
            continue
        seg_start = max(1, int(base))
        seg_end = min(seg_start + seg_size - 1, 10**12)
        if seg_end < seg_start:
            continue
        try:
            phi_vals = compute_phi_segment(seg_start, seg_end)
            k, s = find_longest_run_sliding(phi_vals)
            actual_s = seg_start + s
            if k > best_k:
                best_k = k
                best_s = actual_s
                print(f"Found k={best_k} at s={best_s}")
        except Exception:
            pass

    # Phase 3: Finer search around best found location
    if time.time() - start_time < deadline - 10 and best_k > 1:
        small_seg = 300000
        offsets = [-300000, -150000, -75000, 0, 75000, 150000, 300000]
        for offset in offsets:
            if time.time() - start_time > deadline - 5:
                break
            test_s = best_s + offset
            if test_s < 1:
                continue
            seg_end = min(test_s + small_seg - 1, 10**12)
            if seg_end < test_s:
                continue
            try:
                phi_vals = compute_phi_segment(test_s, seg_end)
                k, s = find_longest_run_sliding(phi_vals)
                actual_s = test_s + s
                if k > best_k:
                    best_k = k
                    best_s = actual_s
            except Exception:
                pass

    # Final validation
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k > 200:
        best_k = 200
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1

    return {'start': int(best_s), 'length': int(best_k)}
