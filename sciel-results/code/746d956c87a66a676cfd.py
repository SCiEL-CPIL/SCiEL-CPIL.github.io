import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 142.0  # Safety margin (150s kill, 8s buffer)

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
            # Find indices where rem is divisible by p
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
            phi[sel] -= phi[sel] // p
            rem[sel] //= p
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
            phi[mask] -= phi[mask] // rem[mask]
        
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

    # Phase 1: Small range to get a baseline
    if time.time() - start_time < deadline - 30:
        phi_vals = compute_phi_segment(1, 100000)
        k, s = find_longest_run_sliding(phi_vals)
        if k > best_k:
            best_k = k
            best_s = 1 + s

    # Phase 2: Dense contiguous high-range scan with overlap to bridge segment boundaries
    # Use 250K segments with 50K overlap for cross-boundary run detection
    seg_size = 250000
    overlap = 50000
    scan_start = 9 * 10**11
    scan_end = 10**12
    # Pre-compute how many segments we can afford
    # Each segment ~ 250K: phi computation ~ 0.5-1s, sliding window ~ 0.1s
    # With 142s deadline and ~30s already used, we have ~100s
    # At ~1.2s per segment, that's ~80 segments, covering 80*200K = 16M
    # But we want to be safe: scan as far as time allows
    
    import random
    random.seed(42)
    
    # First: dense contiguous scan from 9e11 upward
    pos = scan_start
    segments_done = 0
    max_segments = 100  # cap to avoid infinite loop
    while pos + seg_size <= scan_end and segments_done < max_segments:
        if time.time() - start_time > deadline - 20:
            break
        seg_start = pos
        seg_end = min(pos + seg_size - 1, 10**12)
        try:
            phi_vals = compute_phi_segment(seg_start, seg_end)
            k, s = find_longest_run_sliding(phi_vals)
            actual_s = seg_start + s
            if k > best_k:
                best_k = k
                best_s = actual_s
                print(f"Found k={best_k} at s={best_s}")
            # Also check the overlap region with previous segment if applicable
            # (handled by the overlap in pos advancement)
            pos += seg_size - overlap
            segments_done += 1
        except Exception:
            pos += seg_size  # skip ahead on error
    
    # Phase 2b: Random high-range probes (different regions)
    random_bases = set()
    for _ in range(30):
        b = random.randint(10**11, 10**12 - seg_size)
        # Round to segment boundary for consistency
        b = (b // seg_size) * seg_size
        if b >= scan_start and b < pos:
            continue  # already scanned
        random_bases.add(b)
    
    for base in sorted(random_bases):
        if time.time() - start_time > deadline - 15:
            break
        if base < 1 or base > 10**12 - seg_size:
            continue
        seg_start = base
        seg_end = min(base + seg_size - 1, 10**12)
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

    # Phase 3: Collision-density guided search
    # Sample small windows at various points to estimate local collision density,
    # then focus on the lowest-density regions for larger searches
    if time.time() - start_time < deadline - 20 and best_k < 200:
        sample_size = 10000
        # Sample at diverse locations in high range
        sample_positions = []
        for i in range(20):
            pos = random.randint(10**11, 10**12 - sample_size)
            pos = (pos // sample_size) * sample_size
            sample_positions.append(pos)
        # Also include some structured positions
        for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            pos = int(10**11 + frac * (10**12 - 10**11 - sample_size))
            pos = (pos // sample_size) * sample_size
            sample_positions.append(pos)
        
        # Score each sample by collision count
        sample_scores = []
        for sp in sample_positions:
            if time.time() - start_time > deadline - 15:
                break
            try:
                phi_sample = compute_phi_segment(sp, sp + sample_size - 1)
                # Count collisions in this window
                seen = set()
                collisions = 0
                for v in phi_sample:
                    vi = int(v)
                    if vi in seen:
                        collisions += 1
                    seen.add(vi)
                # Also compute the longest run in this sample
                k_s, s_s = find_longest_run_sliding(phi_sample)
                sample_scores.append((collisions, k_s, sp, s_s))
            except Exception:
                pass
        
        # Sort by fewest collisions and search the best regions
        sample_scores.sort(key=lambda x: x[0])
        for collisions, k_s, sp, s_s in sample_scores[:5]:
            if time.time() - start_time > deadline - 10:
                break
            # Do a larger search around this low-collision region
            large_seg = 500000
            search_start = max(1, sp - large_seg // 2)
            search_end = min(10**12, search_start + large_seg - 1)
            try:
                phi_large = compute_phi_segment(search_start, search_end)
                k_l, s_l = find_longest_run_sliding(phi_large)
                actual_s = search_start + s_l
                if k_l > best_k:
                    best_k = k_l
                    best_s = actual_s
                    print(f"Found k={best_k} at s={best_s} (collision-guided)")
            except Exception:
                pass

    # Phase 4: Try to extend the best run with wider windows
    if time.time() - start_time < deadline - 10 and best_k > 1:
        # Try to find a longer run starting near best_s with larger windows
        for offset in [-5000, -2000, -1000, -500, -200, 0, 200, 500, 1000, 2000, 5000]:
            if time.time() - start_time > deadline - 5:
                break
            test_s = best_s + offset
            if test_s < 1:
                continue
            window = 1000  # larger window to find longer runs
            seg_end = min(test_s + window - 1, 10**12)
            try:
                phi_vals = compute_phi_segment(test_s, seg_end)
                k, s = find_longest_run_sliding(phi_vals)
                actual_s = test_s + s
                if k > best_k:
                    best_k = k
                    best_s = actual_s
                    print(f"Extended to k={best_k} at s={best_s}")
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
