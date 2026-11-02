import time
import math
import random
import numpy as np
from itertools import combinations

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    arr = np.array(U, dtype=np.int64)
    if arr[0] != 0:
        arr = np.concatenate(([0], arr))
    arr = np.unique(arr)
    n = len(arr)
    mx = int(arr.max())
    if mx == 0:
        return 0.0
    
    # Size of sumset U+U: values in [0, 2*mx]
    # Size of diffset U-U: values in [-mx, mx], shift by mx -> [0, 2*mx]
    N = 2 * mx + 1
    
    # Efficient counting via FFT
    # Create signal of ones at positions in U
    sig = np.zeros(N, dtype=np.float64)
    sig[arr] = 1.0
    
    # Sumset: convolution of sig with sig
    # Result index k corresponds to sum k
    # Diffset: convolution of sig with reversed sig
    
    # To avoid circular convolution artifacts, we need size >= 2*N - 1? 
    # No, max sum is 2*mx = N-1. Max diff is mx.
    # Linear convolution of two length N signals has length 2N-1.
    # We only care about indices 0..2*mx for sum (which is 0..N-1).
    # For diff, we care about indices 0..2*mx in the reversed convolution?
    # Let's just do full linear convolution via FFT size 2N or next power of 2.
    
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    F = np.fft.rfft(sig, n=fft_size)
    
    # Sumset
    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6) # Sums range 0..2*mx which is 0..N-1
    
    # Diffset
    # rev_sig[i] = sig[N-1-i]
    # We want counts of a-b.
    # Standard trick: conv(sig, flip(sig))
    # Let's construct flip_sig
    flip_sig = np.zeros(N, dtype=np.float64)
    flip_sig[:] = sig[::-1]
    
    F_flip = np.fft.rfft(flip_sig, n=fft_size)
    S_diff = np.fft.irfft(F * F_flip, n=fft_size)
    
    # The index k in S_diff corresponds to a + (N-1-b) = k => a-b = k - (N-1).
    # We want to know if a-b exists.
    # a-b ranges from -mx to mx.
    # k ranges from 0 to 2*mx.
    # Let's check non-zero entries in S_diff.
    # Specifically, we want to map back to valid differences.
    # Since U contains 0, min diff is -mx, max diff is mx.
    # The convolution covers all combinations.
    # diffset_size = np.sum(S_diff > 1e-6) is safe because only valid diffs will have mass.
    # Wait, is it possible for invalid indices to have mass?
    # a in [0, mx], b in [0, mx].
    # a + (N-1-b) = a - b + N - 1.
    # Min: 0 - mx + N - 1 = N - 1 - mx.
    # Max: mx - 0 + N - 1 = N - 1 + mx.
    # So the non-zero part is in [N-1-mx, N-1+mx].
    # Total range is 2*mx+1 = N.
    # So diffset_size is simply the count of non-zero entries in S_diff.
    
    diffset_size = np.sum(S_diff > 1e-6)
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
        
    denom = np.log(2 * mx + 1)
    if denom == 0:
        return 0.0
        
    return 1.0 + np.log(ratio) / denom

def _score_set_small(U):
    """Fast O(n^2) score for small sets (n < 15)."""
    if not U or 0 not in U:
        return 0.0
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
            
    num_sums = len(sum_set)
    num_diffs = len(diff_set)
    if num_sums == 0:
        return 0.0
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return 1.0
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return 1.0
    return 1.0 + math.log(ratio) / log_denom

def _local_search(initial, deadline, max_steps=2000):
    """
    Local search to improve a set.
    Uses fast O(n^2) scoring for small sets, FFT for larger.
    """
    current_U = list(initial)
    current_U.sort()
    current_score = _score_set_small(current_U) if len(current_U) < 15 else _score_set(current_U)
    best_U = list(current_U)
    best_score = current_score
    
    # Pre-compute some candidate pools
    # Powers of 2, 3, etc.
    powers = set()
    for base in [2, 3, 4, 5, 7]:
        p = 1
        while p < 100000:
            powers.add(p)
            p *= base
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        n = len(current_U)
        max_val = max(current_U) if current_U else 0
        
        # Generate candidates
        candidates = set()
        
        # 1. Small random integers
        for _ in range(10):
            candidates.add(random.randint(1, max(10, max_val // 2)))
            
        # 2. Neighbors of max
        for d in range(1, 5):
            candidates.add(max_val + d)
            if max_val - d > 0:
                candidates.add(max_val - d)
                
        # 3. From powers set, pick some random ones
        if powers:
            p_list = list(powers)
            random.shuffle(p_list)
            for p in p_list[:5]:
                if p not in current_U:
                    candidates.add(p)
                    
        # 4. Differences/sums of existing elements (structure-aware)
        if n < 20:
            for a in current_U:
                for b in current_U:
                    if a != b:
                        candidates.add(a + b)
                        if a > b:
                            candidates.add(a - b)
        
        # Filter candidates
        valid_candidates = [c for c in candidates if c > 0 and c not in current_U]
        
        if not valid_candidates:
            # If no candidates, try removing
            pass
        else:
            # Score candidates
            scored_candidates = []
            for c in valid_candidates:
                new_U = current_U + [c]
                new_U.sort()
                if len(new_U) < 15:
                    s = _score_set_small(new_U)
                else:
                    s = _score_set(new_U)
                scored_candidates.append((s, c))
            
            scored_candidates.sort(reverse=True, key=lambda x: x[0])
            
            # Try the best candidate
            if scored_candidates:
                best_s, best_c = scored_candidates[0]
                if best_s > current_score + 1e-9:
                    current_U.append(best_c)
                    current_U.sort()
                    current_score = best_s
                    if current_score > best_score:
                        best_score = current_score
                        best_U = list(current_U)
                    continue
            
        # If no improvement from adding, try removing
        if n > 2:
            # Try removing a random element (not 0)
            # Or the one that, when removed, yields the best score
            # For small n, check all
            if n < 15:
                best_remove_idx = -1
                best_remove_score = current_score
                for i in range(1, n):
                    new_U = current_U[:i] + current_U[i+1:]
                    s = _score_set_small(new_U)
                    if s > best_remove_score:
                        best_remove_score = s
                        best_remove_idx = i
                
                if best_remove_idx != -1:
                    current_U = current_U[:best_remove_idx] + current_U[best_remove_idx+1:]
                    current_score = best_remove_score
                    if current_score > best_score:
                        best_score = current_score
                        best_U = list(current_U)
                    continue
            else:
                # For larger n, just try a few random removals
                for _ in range(3):
                    idx = random.randint(1, n-1)
                    new_U = current_U[:idx] + current_U[idx+1:]
                    s = _score_set(new_U)
                    if s > current_score + 1e-9:
                        current_U = new_U
                        current_score = s
                        if current_score > best_score:
                            best_score = current_score
                            best_U = list(current_U)
                        break
                        
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_U = [0, 1, 3]
    best_score = _score_set_small(best_U)
    
    # Phase 1: Exhaustive search for small sets
    # n=3
    # Already have [0,1,3]. Let's check a few others.
    # Actually, let's run a quick local search from a few seeds.
    
    seeds = [
        [0, 1, 3],
        [0, 1, 2],
        [0, 1, 4],
        [0, 2, 3],
        [0, 1, 5],
        [0, 2, 5],
        [0, 3, 4],
        [0, 1, 6],
        [0, 2, 7],
        [0, 3, 7],
        [0, 4, 7],
    ]
    
    for seed in seeds:
        if time.time() > deadline - 60:
            break
        U, s = _local_search(seed, deadline - 30, max_steps=500)
        if s > best_score:
            best_score = s
            best_U = U
    
    # Phase 2: Try geometric progressions as seeds for local search
    for base in [2, 3, 4, 5]:
        if time.time() > deadline - 30:
            break
        for n in range(3, 10):
            if time.time() > deadline - 30:
                break
            U = [0]
            val = 1
            for i in range(1, n):
                U.append(val)
                val *= base
            U, s = _local_search(U, deadline - 10, max_steps=200)
            if s > best_score:
                best_score = s
                best_U = U
    
    # Phase 3: Random restarts with local search
    # Use more time here
    while time.time() < deadline - 5:
        # Generate a random seed
        n = random.randint(3, 8)
        seed = [0]
        vals = set()
        while len(seed) < n:
            # Mix of small and large
            if random.random() < 0.5:
                v = random.randint(1, 50)
            else:
                v = random.randint(1, 200)
            if v not in vals and v != 0:
                vals.add(v)
                seed.append(v)
        seed.sort()
        
        U, s = _local_search(seed, deadline - 5, max_steps=300)
        if s > best_score:
            best_score = s
            best_U = U
            
    # Final polish: Run local search on the best found set with more steps
    if time.time() < deadline - 2:
        U, s = _local_search(best_U, deadline - 1, max_steps=1000)
        if s > best_score:
            best_score = s
            best_U = U

    return sorted(best_U)

# Test locally
if __name__ == "__main__":
    # Quick test
    s = construct()
    print(f"Result: {s[:10]}... Size: {len(s)}, Score: {_score_set(s)}")
