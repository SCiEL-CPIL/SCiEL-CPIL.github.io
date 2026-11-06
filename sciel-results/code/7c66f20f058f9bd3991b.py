"""Rewritten program for low discrepancy on all arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 256

def _precompute():
    """Precompute AP indices and position-to-AP mapping using a compact representation."""
    # For efficiency, store APs as (start, step, length) tuples and compute indices on the fly
    # But for incremental updates, we need to know which APs each position belongs to
    # and be able to update sums quickly.
    
    # Approach: Precompute all APs as lists of indices, but group by (step, start)
    # For each position p, store the list of (ap_index, offset_in_ap) so we can update sums.
    
    # Actually, let's use a different approach:
    # For each (step, start), the AP is positions [start-1, start-1+step, start-1+2*step, ...]
    # We can maintain a 2D array of sums: sums[step][start] = current sum of that AP
    # But the number of (step, start) pairs is manageable.
    
    # Number of distinct (step, start) pairs:
    # step from 1 to N//2
    # start from 1 to N - step + 1
    # Total ~ N^2/4 = 16384 for N=256, which is fine.
    
    # For each position p, we need to know all (step, start) pairs such that p is in the AP
    # p is in AP(start, step) if p >= start and (p - start) % step == 0
    # And p < N + 1 (1-indexed)
    
    # Let's use 0-indexed: positions 0 to N-1
    # AP with start s (0-indexed) and step d: positions s, s+d, s+2d, ...
    
    # Precompute pos_to_aps: for each position, list of (step_idx, start_idx) 
    # where step_idx is the step value and start_idx is the start position
    
    # Actually, let's just precompute all APs as index lists and also build
    # pos_to_aps: for each position, which APs (by index) contain it
    
    ap_indices = []  # List of lists of positions
    pos_to_aps = [[] for _ in range(N)]  # For each position, list of ap_indices
    
    for step in range(1, N):
        for start in range(N - step + 1):
            # AP: start, start+step, start+2*step, ...
            indices = list(range(start, N, step))
            if len(indices) < 2:
                continue
            ap_idx = len(ap_indices)
            ap_indices.append(indices)
            for pos in indices:
                pos_to_aps[pos].append(ap_idx)
    
    return ap_indices, pos_to_aps

def _compute_initial_sums(signs, ap_indices):
    """Compute the sum for each AP."""
    sums = np.zeros(len(ap_indices), dtype=np.int32)
    for ap_idx, indices in enumerate(ap_indices):
        s = 0
        for pos in indices:
            s += signs[pos]
        sums[ap_idx] = s
    return sums

def _get_max_abs(sums):
    return int(np.max(np.abs(sums)))

def _construct_thue_morse():
    """Thue-Morse sequence."""
    signs = []
    for i in range(N):
        c = bin(i).count('1')
        signs.append(1 if c % 2 == 0 else -1)
    return signs

def _construct_random(seed):
    rng = random.Random(seed)
    return [rng.choice((-1, 1)) for _ in range(N)]

def _construct_pattern(seed):
    """Various structured patterns."""
    rng = random.Random(seed)
    method = rng.randint(0, 3)
    if method == 0:
        return _construct_thue_morse()
    elif method == 1:
        # Alternating
        return [1 if i % 2 == 0 else -1 for i in range(N)]
    elif method == 2:
        # Block of 2
        signs = []
        for _ in range(N // 2):
            b = rng.choice((-1, 1))
            signs.extend([b, b])
        return signs
    else:
        return _construct_random(seed)

def _hill_climb(signs, ap_indices, pos_to_aps, time_limit, rng, allow_neutral=False):
    """Hill climbing with incremental sum updates."""
    signs = np.array(signs, dtype=np.int32)
    sums = _compute_initial_sums(signs, ap_indices)
    current_max = _get_max_abs(sums)
    
    deadline = time.time() + time_limit
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline - 1.0:
        pos = rng.randint(0, N - 1)
        old_sign = signs[pos]
        new_sign = -old_sign
        signs[pos] = new_sign
        
        # Update all APs containing this position
        affected = pos_to_aps[pos]
        for ap_idx in affected:
            sums[ap_idx] += 2 * new_sign
        
        new_max = _get_max_abs(sums)
        
        if new_max < current_max:
            current_max = new_max
            no_improve = 0
        elif new_max == current_max and allow_neutral and rng.random() < 0.3:
            no_improve += 1
        else:
            # Revert
            signs[pos] = old_sign
            for ap_idx in affected:
                sums[ap_idx] -= 2 * new_sign
            no_improve += 1
        
        if no_improve > max_no_improve:
            # Perturbation: flip a few random positions
            for _ in range(3):
                p = rng.randint(0, N - 1)
                old = signs[p]
                signs[p] = -old
                for ap_idx in pos_to_aps[p]:
                    sums[ap_idx] += 2 * signs[p]
            current_max = _get_max_abs(sums)
            no_improve = 0
            max_no_improve = 100
    
    return [int(s) for s in signs], current_max

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 90s limit with 5s margin
    
    # Precompute
    ap_indices, pos_to_aps = _precompute()
    num_aps = len(ap_indices)
    print(f"Precomputed {num_aps} APs in {time.time() - start_time:.2f}s", flush=True)
    
    rng = random.Random(12345)
    
    # Generate diverse initial candidates
    candidates = []
    for i in range(30):
        if i < 5:
            candidates.append(_construct_pattern(i))
        else:
            candidates.append(_construct_random(10000 + i))
    
    # Evaluate all candidates
    best_signs = None
    best_val = float('inf')
    
    for cand in candidates:
        cand_arr = np.array(cand, dtype=np.int32)
        sums = _compute_initial_sums(cand_arr, ap_indices)
        val = _get_max_abs(sums)
        if val < best_val:
            best_val = val
            best_signs = cand[:]
    
    print(f"Best initial: {best_val}", flush=True)
    
    # Hill climb from the best candidate
    if time.time() < deadline - 5.0:
        hc_time = min(25.0, (deadline - time.time()) * 0.7)
        best_signs, best_val = _hill_climb(best_signs, ap_indices, pos_to_aps, hc_time, rng)
        print(f"After hill climb: {best_val}", flush=True)
    
    # Try other good candidates
    candidate_scores = []
    for cand in candidates:
        cand_arr = np.array(cand, dtype=np.int32)
        sums = _compute_initial_sums(cand_arr, ap_indices)
        val = _get_max_abs(sums)
        candidate_scores.append((val, cand))
    candidate_scores.sort()
    
    for val, cand in candidate_scores[:5]:
        if time.time() > deadline - 5.0:
            break
        hc_time = min(8.0, (deadline - time.time()) * 0.3)
        if hc_time < 1.0:
            break
        hs, hv = _hill_climb(cand, ap_indices, pos_to_aps, hc_time, rng)
        if hv < best_val:
            best_val = hv
            best_signs = hs
            print(f"Found better: {best_val}", flush=True)
    
    # Final verification
    final_arr = np.array(best_signs, dtype=np.int32)
    final_sums = _compute_initial_sums(final_arr, ap_indices)
    final_val = _get_max_abs(final_sums)
    print(f"Final value: {final_val}", flush=True)
    
    return best_signs

# EVOLVE-BLOCK-END
