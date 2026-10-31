import time
import math
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0

def _score_from_set(u):
    """
    Compute the score for a set of integers u using FFT for exact cardinality.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    
    u_arr = np.array(u, dtype=np.int64)
    n = len(u_arr)
    
    # Use FFT to compute sum and difference set sizes
    # For sums: indicator vector f, then f*f (convolution) gives sum counts
    # For diffs: indicator vector f, then f * f_rev gives diff counts
    
    max_val = m
    # Size of FFT: need at least 2*max_val + 1 for sums, 2*max_val + 1 for diffs
    fft_size = 1
    while fft_size < 2 * max_val + 1:
        fft_size *= 2
    # Add some padding for safety
    fft_size = next_power_of_two(2 * max_val + 1)
    
    # Create indicator vector
    f = np.zeros(fft_size, dtype=np.float64)
    f[u_arr] = 1.0
    
    # Compute autocorrelation for differences
    # diff counts: for each d, count pairs (a,b) in U such that a - b = d
    # This is equivalent to cross-correlation of f with itself
    # Using FFT: F = FFT(f), then autocorr = IFFT(|F|^2)
    F = np.fft.fft(f)
    autocorr = np.fft.ifft(np.abs(F) ** 2).real
    # autocorr[k] gives count of pairs with difference k (cyclic)
    # We need to handle the wrap-around: for d in [-max_val, max_val]
    # autocorr gives cyclic correlation, so we need to extract the right part
    
    # For sums: convolution of f with f
    conv = np.fft.ifft(F * F).real
    # conv[k] gives count of pairs (a,b) with a+b = k
    
    # Count unique sums: positions where conv[k] > 0 for k in [0, 2*max_val]
    sum_counts = conv[:2 * max_val + 1]
    unique_sums = np.sum(sum_counts > 0)
    
    # Count unique differences: positions where autocorr[d] > 0 for d in [-max_val, max_val]
    # autocorr is cyclic with period fft_size
    # For difference d, the count is autocorr[d] if d >= 0, or autocorr[fft_size + d] if d < 0
    diff_counts = np.zeros(2 * max_val + 1, dtype=np.float64)
    for d in range(-max_val, max_val + 1):
        idx = d % fft_size
        diff_counts[d + max_val] = autocorr[idx]
    unique_diffs = np.sum(diff_counts > 0)
    
    if unique_sums == 0:
        return 0.0
        
    ratio = unique_diffs / float(unique_sums)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def next_power_of_two(n):
    """Return the next power of two >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

def _construct_greedy_golomb(n):
    """
    Construct a greedy Golomb ruler of length n.
    A Golomb ruler has all pairwise differences distinct.
    This maximizes |U-U| for a given size n (|U-U| = n*(n-1) + 1).
    """
    if n <= 1:
        return [0]
    
    # Greedy construction
    ruler = [0]
    for i in range(1, n):
        best_pos = None
        for pos in range(1, i * i * 4):  # Upper bound search
            # Check if pos is a valid Golomb ruler position
            valid = True
            for j in range(len(ruler)):
                diff = pos - ruler[j]
                if diff <= 0:
                    continue
                # Check if this difference already exists
                for k in range(len(ruler)):
                    if k == j:
                        continue
                    d2 = abs(ruler[j] - ruler[k])
                    if d2 == diff:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_pos = pos
                break
        if best_pos is None:
            # Fallback: just extend
            best_pos = ruler[-1] + i
        ruler.append(best_pos)
    
    return ruler

def _construct_powers_of_base(B, n):
    """
    Construct a set as powers of B: {B^0, B^1, ..., B^(n-1)}.
    """
    if n < 1:
        return [0]
    result = [0]
    val = 1
    for i in range(n - 1):
        result.append(val)
        val *= B
    return result

def _construct_sparse_random(n, max_val, seed=42):
    """
    Generate a sparse random set with large gaps to minimize collisions.
    """
    rng = random.Random(seed)
    if n < 1:
        return [0]
    
    vals = [0]
    current = 0
    min_gap = max(1, int(max_val ** 0.6 / n))
    
    for i in range(1, n):
        gap = rng.randint(min_gap, min_gap * 3)
        current += gap
        if current > max_val:
            current = max_val
        vals.append(current)
    
    return sorted(set(vals))

def _local_search_golomb(u, max_steps=2000, timeout=10.0):
    """
    Local search specifically for Golomb-like structures.
    Move: replace one element with a nearby valid Golomb position.
    """
    u = sorted(set(u))
    if len(u) < 2:
        u = [0, 1]
        
    current_score = _score_from_set(u)
    best_u = u[:]
    best_score = current_score
    
    start_time = time.time()
    
    for step in range(max_steps):
        if time.time() > _DEADLINE - 5.0 or time.time() > start_time + timeout:
            break
            
        idx = random.randint(1, len(u) - 1)  # Don't replace 0
        old_val = u[idx]
        
        # Try replacing with a nearby value that maintains Golomb property
        found = False
        for delta in range(-50, 51):
            if delta == 0:
                continue
            new_val = old_val + delta
            if new_val < 0:
                continue
            if new_val in u:
                continue
            
            # Check Golomb property
            valid = True
            for j in range(len(u)):
                if j == idx:
                    continue
                diff1 = new_val - u[j]
                if diff1 <= 0:
                    continue
                for k in range(len(u)):
                    if k == idx or k == j:
                        continue
                    diff2 = abs(u[j] - u[k])
                    if diff2 == diff1:
                        valid = False
                        break
                    # Also check diff with new_val and other pairs
                if not valid:
                    break
                # Check diff between new_val and u[j] against all existing diffs
                for k in range(len(u)):
                    if k == j or k == idx:
                        continue
                    d = abs(u[j] - u[k])
                    if d == diff1:
                        valid = False
                        break
            if valid:
                # Also need to check diffs involving new_val against all existing diffs
                # This is expensive, so we rely on the local search to fix things
                test_u = u[:]
                test_u[idx] = new_val
                test_u = sorted(set(test_u))
                if len(test_u) != len(u):
                    continue
                s = _score_from_set(test_u)
                if s > current_score:
                    u = test_u
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_u = u[:]
                    found = True
                    break
        
        # If no valid replacement, try adding a new element
        if not found and random.random() < 0.3:
            max_val = max(u)
            new_val = max_val + random.randint(1, max(1, max_val // 10))
            test_u = sorted(u + [new_val])
            s = _score_from_set(test_u)
            if s > current_score and len(test_u) <= 30:
                u = test_u
                current_score = s
                if current_score > best_score:
                    best_score = current_score
                    best_u = u[:]
            
            # Try removing an element
            if len(u) > 3:
                idx_rm = random.randint(1, len(u) - 1)
                test_u = u[:]
                test_u.pop(idx_rm)
                s = _score_from_set(test_u)
                if s > current_score:
                    u = test_u
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_u = u[:]

    return best_u, best_score

def construct():
    global _DEADLINE
    
    random.seed(42)
    np.random.seed(42)
    
    best_u = [0, 1]
    best_s = 0.0
    
    candidates = []
    
    # Phase 1: Generate diverse initial candidates
    
    # Golomb rulers
    for n in [5, 8, 10, 12, 15, 20]:
        if time.time() > _DEADLINE - 30.0:
            break
        u = _construct_greedy_golomb(n)
        candidates.append(u)
        
    # Powers of base
    for B in [3, 4, 5, 10]:
        for n in [5, 8, 10, 12]:
            if time.time() > _DEADLINE - 20.0:
                break
            u = _construct_powers_of_base(B, n)
            candidates.append(u)
            
    # Sparse random
    for n in [10, 15, 20, 25]:
        for max_val in [1000, 10000, 100000]:
            if time.time() > _DEADLINE - 15.0:
                break
            for seed in [1, 2, 3]:
                u = _construct_sparse_random(n, max_val, seed)
                candidates.append(u)
                
    # Evaluate candidates
    scored_candidates = []
    for u in candidates:
        if time.time() > _DEADLINE - 10.0:
            break
        s = _score_from_set(u)
        scored_candidates.append((s, u))
        if s > best_s:
            best_s = s
            best_u = u[:]
            
    # Sort by score
    scored_candidates.sort(reverse=True)
    
    # Phase 2: Local search on top candidates
    for i, (init_s, init_u) in enumerate(scored_candidates[:5]):
        if time.time() > _DEADLINE - 5.0:
            break
        if init_s < best_s * 0.95:
            continue
            
        u_new, s_new = _local_search_golomb(init_u, max_steps=1000, timeout=10.0)
        if s_new > best_s:
            best_s = s_new
            best_u = u_new[:]
            
    # Phase 3: Final polish
    if time.time() < _DEADLINE - 2.0:
        u_new, s_new = _local_search_golomb(best_u, max_steps=500, timeout=5.0)
        if s_new > best_s:
            best_s = s_new
            best_u = u_new[:]
            
    # Final sanity check
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
    if len(best_u) < 2:
        best_u = [0, 1]
        
    return best_u
