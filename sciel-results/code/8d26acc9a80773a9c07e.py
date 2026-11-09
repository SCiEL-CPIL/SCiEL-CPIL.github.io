import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin (150s kill, 10s buffer)

    # Sieve primes up to 10^6 for factorization
    limit = 1000000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def compute_phi_segment(start, end):
        """Compute phi(n) for n in [start, end] using segmented approach."""
        size = end - start + 1
        phi = np.arange(start, end + 1, dtype=np.int64)
        rem = phi.copy()
        
        for p in primes:
            if p * p > end:
                break
            # Find the first multiple of p in [start, end]
            first = ((start + p - 1) // p) * p
            if first > end:
                continue
            # All indices in the segment that are multiples of p
            idxs = np.arange(first - start, size, p)
            if len(idxs) == 0:
                continue
            # Check which of these still have p as a factor
            mask = rem[idxs] % p == 0
            if not mask.any():
                continue
            sel = idxs[mask]
            phi[sel] = phi[sel] // p * (p - 1)
            # Remove ALL factors of p from rem in a loop
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
            val = phi_vals[right]
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

    # Phase 1: Quick baseline in small range
    if time.time() - start_time < deadline - 40:
        phi_vals = compute_phi_segment(1, 50000)
        k, s = find_longest_run_sliding(phi_vals)
        if k > best_k:
            best_k = k
            best_s = 1 + s

    # Phase 2: Search at various scales, prioritizing higher ranges
    seg_size = 300000
    candidates = [
        # High ranges where collisions are rarer
        10**12 - seg_size, 9*10**11, 8*10**11, 7*10**11, 6*10**11,
        5*10**11, 4*10**11, 3*10**11, 2*10**11, 10**11,
        # Medium ranges
        5*10**10, 2*10**10, 10**10, 5*10**9, 2*10**9, 10**9,
        5*10**8, 10**8, 5*10**7, 10**7,
        # Lower ranges
        5*10**6, 10**6,
    ]
    # Add diverse offsets within high ranges
    for base in [10**11, 5*10**11, 10**12 - seg_size]:
        for off in [1000, 50000, 100000, 500000, 1000000]:
            c = base + off
            if 1 <= c <= 10**12 - seg_size:
                candidates.append(c)

    for base in candidates:
        if time.time() - start_time > deadline - 15:
            break
        if base < 1 or base > 10**12:
            continue
        seg_start = max(1, base)
        seg_end = min(base + seg_size - 1, 10**12)
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
        except Exception as e:
            pass

    # Phase 3: Finer search around the best found location
    if time.time() - start_time < deadline - 10 and best_k > 1:
        small_seg = 100000
        for offset in [-200000, -100000, -50000, 0, 50000, 100000, 200000]:
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

    # Final validation and clamping
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k > 200:
        best_k = 200
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1

    return {'start': int(best_s), 'length': int(best_k)}
