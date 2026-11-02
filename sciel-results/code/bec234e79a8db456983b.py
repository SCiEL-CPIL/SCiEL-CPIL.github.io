import time
import math
import numpy as np
from itertools import combinations
import random

def _score_set_direct(U):
    """Direct scoring for small sets - O(n^2) but faster for small n."""
    if not U or len(U) < 2:
        return 0.0
    arr = sorted(set(U))
    if arr[0] != 0:
        arr = [0] + arr
    n = len(arr)
    if n < 2:
        return 0.0
    mx = arr[-1]
    if mx == 0:
        return 0.0
    
    # Compute sumset directly
    sumset = set()
    diffset = set()
    for i in range(n):
        for j in range(i, n):  # i <= j for sumset (commutative)
            sumset.add(arr[i] + arr[j])
        for j in range(n):
            diffset.add(arr[i] - arr[j])
    
    sumset_size = len(sumset)
    diffset_size = len(diffset)
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
        
    denom = math.log(2 * mx + 1)
    if denom == 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / denom

def _score_set_fft(U):
    """FFT-based scoring for larger sets."""
    if not U or len(U) < 2:
        return 0.0
    arr = np.array(U, dtype=np.int64)
    if arr[0] != 0:
        arr = np.concatenate(([0], arr))
    arr = np.unique(arr)
    n = len(arr)
    if n < 2:
        return 0.0
    mx = int(arr.max())
    if mx == 0:
        return 0.0
    
    # Size of sumset U+U: values in [0, 2*mx]
    # Size of diffset U-U: values in [-mx, mx], shift by mx -> [0, 2*mx]
    N = 2 * mx + 1
    
    # Efficient counting via FFT
    sig = np.zeros(N, dtype=np.float64)
    sig[arr] = 1.0
    
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    F = np.fft.rfft(sig, n=fft_size)
    
    # Sumset
    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6) # Sums range 0..2*mx which is 0..N-1
    
    # Diffset
    flip_sig = np.zeros(N, dtype=np.float64)
    flip_sig[:] = sig[::-1]
    
    F_flip = np.fft.rfft(flip_sig, n=fft_size)
    S_diff = np.fft.irfft(F * F_flip, n=fft_size)
    
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

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    arr = sorted(set(U))
    if arr[0] != 0:
        arr = [0] + arr
    n = len(arr)
    if n < 2:
        return 0.0
    mx = arr[-1]
    if mx == 0:
        return 0.0
    
    # Use direct method for small sets, FFT for larger
    if n <= 30 and mx <= 500:
        return _score_set_direct(arr)
    else:
        return _score_set_fft(arr)

def _grow_set(U, target_size, time_limit=2.0):
    """Grow set U to target_size by adding elements that maximize score."""
    start_time = time.time()
    current = sorted(U)
    current_score = _score_set(current)
    
    while len(current) < target_size:
        if time.time() - start_time > time_limit:
            break
        
        best_new_val = None
        best_new_score = current_score
        
        # Try adding elements in range
        max_val = current[-1] if current else 0
        search_range = min(max_val + 30, 200)
        
        for new_val in range(max_val + 1, search_range + 1):
            candidate = current + [new_val]
            candidate = sorted(set(candidate))
            if len(candidate) != len(current) + 1:
                continue
            score = _score_set(candidate)
            if score > best_new_score + 1e-10:
                best_new_score = score
                best_new_val = new_val
        
        if best_new_val is None:
            # No improvement found, break
            break
        
        current = sorted(current + [best_new_val])
        current_score = best_new_score
    
    return current, current_score

def _local_search_fixed_size(U, max_iter=500, time_limit=5.0):
    """Local search with fixed set size."""
    current = sorted(U)
    current_score = _score_set(current)
    start_time = time.time()
    
    small_deltas = list(range(-5, 6))
    large_deltas = list(range(-20, 21, 4)) + list(range(20, 51, 8))
    all_deltas = small_deltas + large_deltas
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(len(current)):
            for delta in all_deltas:
                new_val = current[i] + delta
                if new_val < 0:
                    continue
                
                new_set = current[:]
                new_set[i] = new_val
                new_set = sorted(set(new_set))
                
                if len(new_set) != len(current):
                    continue
                
                new_score = _score_set(new_set)
                
                if new_score > current_score + 1e-10:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break
            
            if improved:
                break
                
    return current, current_score

def _local_search_variable_size(U, max_iter=300, time_limit=10.0):
    """Local search that allows set size to change."""
    current = sorted(U)
    current_score = _score_set(current)
    start_time = time.time()
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Try moving elements
        for i in range(len(current)):
            for delta in range(-10, 11):
                new_val = current[i] + delta
                if new_val < 0:
                    continue
                
                new_set = current[:]
                new_set[i] = new_val
                new_set = sorted(set(new_set))
                
                if len(new_set) != len(current):
                    continue
                
                new_score = _score_set(new_set)
                
                if new_score > current_score + 1e-10:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break
            
            if improved:
                break
        
        if improved:
            continue
        
        # Try adding an element
        if len(current) < 30:
            max_val = current[-1]
            for new_val in range(max_val + 1, max_val + 15):
                candidate = sorted(current + [new_val])
                score = _score_set(candidate)
                if score > current_score + 1e-10:
                    current = candidate
                    current_score = score
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try removing an element (if size > 2)
        if len(current) > 3:
            for i in range(1, len(current)):  # Keep 0
                candidate = current[:i] + current[i+1:]
                score = _score_set(candidate)
                if score > current_score + 1e-10:
                    current = candidate
                    current_score = score
                    improved = True
                    break
        
        if not improved:
            break
    
    return current, current_score

def _exhaustive_small_search(max_size=12, max_val=15, time_limit=5.0):
    """Exhaustive search over small subsets of 0..max_val."""
    start_time = time.time()
    best_set = None
    best_score = 0.0
    
    all_vals = list(range(max_val + 1))
    
    for size in range(2, max_size + 1):
        if time.time() - start_time > time_limit:
            break
        
        for subset in combinations(all_vals, size):
            if 0 not in subset:
                continue
            
            U = sorted(list(subset))
            score = _score_set(U)
            
            if score > best_score:
                best_score = score
                best_set = U[:]
    
    return best_set, best_score

# Known optimal Golomb rulers (minimal max element) for n=1 to 20
_OPTIMAL_GOLOMB = [
    [0],
    [0, 1],
    [0, 1, 3],
    [0, 1, 4, 6],
    [0, 1, 4, 9, 11],
    [0, 1, 4, 10, 12, 17],
    [0, 1, 4, 10, 12, 17, 25],
    [0, 1, 4, 10, 18, 23, 29, 34],
    [0, 1, 4, 10, 18, 23, 29, 37, 44],
    [0, 1, 4, 10, 18, 23, 29, 37, 41, 55],
    [0, 1, 4, 10, 18, 23, 29, 37, 41, 55, 72],
    [0, 1, 4, 10, 18, 23, 29, 37, 41, 55, 72, 85],
    [0, 1, 4, 10, 18, 23, 29, 37, 41, 55, 72, 85, 106],
    [0, 1, 4, 10, 18, 23, 29, 37, 41, 55, 72, 85, 106, 129],
    [0, 1, 4, 10, 18, 23, 29, 37, 41, 55, 72, 85, 106, 129, 153],
    [0, 1, 4, 10, 18, 23, 29, 37, 41, 55, 72, 85, 106, 129, 153, 179],
    [0, 1, 4, 10, 18, 23, 29, 37, 41, 55, 72, 85, 106, 129, 153, 179, 208],
    [0, 1, 4, 10, 18, 23, 29, 37, 41, 55, 72, 85, 106, 129, 153, 179, 208, 238],
    [0, 1, 4, 10, 18, 23, 29, 37, 41, 55, 72, 85, 106, 129, 153, 179, 208, 238, 275],
    [0, 1, 4, 10, 18, 23, 29, 37, 41, 55, 72, 85, 106, 129, 153, 179, 208, 238, 275, 319]
]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    best_set = None
    best_score = 0.0
    
    # 1. Exhaustive search over small sets (subsets of 0..15)
    if time.time() < deadline - 20:
        ex_set, ex_score = _exhaustive_small_search(max_size=10, max_val=15, time_limit=8.0)
        if ex_set is not None and ex_score > best_score:
            best_score = ex_score
            best_set = ex_set[:]
    
    # 2. Evaluate known optimal Golomb rulers
    for n, U in enumerate(_OPTIMAL_GOLOMB):
        if time.time() > deadline - 15:
            break
        if len(U) < 2:
            continue
        score = _score_set(U)
        if score > best_score:
            best_score = score
            best_set = U[:]
    
    # 3. Local search with variable size on best candidate
    if best_set is not None and time.time() < deadline - 10:
        improved_set, improved_score = _local_search_variable_size(best_set, max_iter=300, time_limit=20.0)
        if improved_score > best_score:
            best_score = improved_score
            best_set = improved_set
    
    # 4. Try growing small sets to larger sizes
    if time.time() < deadline - 15:
        # Start from small good sets and grow them
        for base_size in [4, 5, 6, 7, 8]:
            if time.time() > deadline - 10:
                break
            
            # Find best small set of this size from Golomb rulers
            base_set = None
            base_score = 0.0
            for U in _OPTIMAL_GOLOMB:
                if len(U) == base_size:
                    s = _score_set(U)
                    if s > base_score:
                        base_score = s
                        base_set = U[:]
            
            if base_set is None:
                continue
            
            # Grow it
            grown_set, grown_score = _grow_set(base_set, base_size + 3, time_limit=5.0)
            if grown_score > best_score:
                best_score = grown_score
                best_set = grown_set[:]
            
            # Local search on grown set
            if time.time() < deadline - 8:
                improved_set, improved_score = _local_search_fixed_size(grown_set, max_iter=200, time_limit=5.0)
                if improved_score > best_score:
                    best_score = improved_score
                    best_set = improved_set
    
    # 5. If time remains, try more local search with variable size
    if best_set is not None and time.time() < deadline - 5:
        improved_set, improved_score = _local_search_variable_size(best_set, max_iter=200, time_limit=10.0)
        if improved_score > best_score:
            best_score = improved_score
            best_set = improved_set
    
    # 6. Final polish with fixed-size local search
    if best_set is not None and time.time() < deadline - 2:
        improved_set, improved_score = _local_search_fixed_size(best_set, max_iter=100, time_limit=5.0)
        if improved_score > best_score:
            best_score = improved_score
            best_set = improved_set
    
    if best_set is None:
        return [0, 1]
        
    return best_set

# Test locally
if __name__ == "__main__":
    s = construct()
    print(f"Result: {s[:10]}... Size: {len(s)}, Max: {max(s)}, Score: {_score_set(s)}")
