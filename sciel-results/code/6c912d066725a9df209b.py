"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

N = 256
DEADLINE_FRAC = 0.9  # safety margin: stop at 90% of 90s = 81s

def _precompute_aps():
    """Precompute all arithmetic progressions within [1, N].
    Returns a list of (start, step, length) tuples, but we'll store
    them as index arrays for fast evaluation."""
    aps = []
    for step in range(1, N + 1):
        for start in range(1, N + 1):
            # Length is the maximum k such that start + (k-1)*step <= N
            length = (N - start) // step + 1
            if length >= 1:
                # We'll represent this AP by its indices (0-based)
                idx = np.arange(start - 1, start - 1 + length * step, step)
                aps.append(idx)
    return aps

def _eval_max_abs_sum(signs, ap_indices):
    """Compute max absolute signed sum over all APs.
    signs: array of +1/-1 of length N
    ap_indices: list of numpy arrays of indices
    Returns the maximum absolute sum."""
    max_abs = 0
    for idx in ap_indices:
        s = np.sum(signs[idx])
        abs_s = abs(s)
        if abs_s > max_abs:
            max_abs = int(abs_s)
    return max_abs

def _eval_max_abs_sum_fast(signs, ap_indices):
    """Faster evaluation using precomputed structure.
    We group APs by (start, step) and compute prefix sums for each step."""
    # For each step, we can compute all AP sums efficiently using
    # the fact that APs with the same step and different starts/lengths
    # can be computed from a cumulative sum along that step's residue classes.
    
    # Actually, let's just do it directly but with numpy vectorization per step
    max_abs = 0
    for idx in ap_indices:
        s = np.dot(signs, idx.astype(np.float64) * 0)  # placeholder
        # This won't work; let's just use np.sum
        s = np.sum(signs[idx])
        if abs(s) > max_abs:
            max_abs = abs(s)
    return int(max_abs)

def _eval_max_abs_sum_v2(signs, ap_indices):
    """Efficient evaluation: for each step, compute all AP sums in O(N) total."""
    max_abs = 0
    # Group APs by step
    # For a fixed step d, APs are: for each start s in 1..N, the sequence
    # s, s+d, s+2d, ... up to N.
    # The sum of signs on this AP is sum of signs at positions s-1, s-1+d, ...
    
    # Let's precompute for each step d, for each residue r mod d,
    # the cumulative sum along that residue class.
    
    # But iterating over all steps and all starts is O(N^2) which for N=256
    # is about 65536 operations per evaluation, which should be fast enough.
    
    for d in range(1, N + 1):
        for start in range(1, N + 1):
            # AP: start, start+d, start+2d, ...
            # indices: start-1, start-1+d, ...
            idx = np.arange(start - 1, N, d)
            s = np.sum(signs[idx])
            abs_s = abs(s)
            if abs_s > max_abs:
                max_abs = int(abs_s)
    return max_abs

def _eval_max_abs_sum_v3(signs):
    """Most efficient: precompute cumulative sums per step/residue."""
    max_abs = 0
    for d in range(1, N + 1):
        # For each residue class r in 0..d-1, compute the sequence of signs
        # and their prefix sums, then find max subarray-like sums.
        # Actually, we need sums over contiguous segments in the residue class,
        # starting from any position.
        # For residue r, the positions are r, r+d, r+2d, ...
        # We need max over all contiguous sub-sequences of these positions.
        
        for r in range(d):
            if r >= N:
                continue
            # Get all positions in this residue class
            pos = np.arange(r, N, d)
            if len(pos) == 0:
                continue
            # Get signs at these positions
            vals = signs[pos]
            # Compute prefix sums
            prefix = np.cumsum(vals)
            # The sum from index i to j in pos is prefix[j] - prefix[i-1] (with prefix[-1]=0)
            # We want max absolute value of any such sum.
            # This is equivalent to: max over j of prefix[j] - min over i<j of prefix[i-1]
            # and min over j of prefix[j] - max over i<j of prefix[i-1]
            
            # Let's compute min and max prefix values
            # prefix has length L, representing sums from pos[0] to pos[i] for i=0..L-1
            # A sub-sum from pos[a] to pos[b] (a<=b) = prefix[b] - (prefix[a-1] if a>0 else 0)
            
            # Let's build an array where element i is the sum from pos[0] to pos[i]
            # Then sub-sum from a to b = prefix[b] - prefix[a-1] where prefix[-1]=0
            
            # For max positive sub-sum:
            # max over b of prefix[b] - min over a-1 < b of (prefix[a-1] if a>0 else 0)
            # Let min_prefix_before[b] = min(0, prefix[0], prefix[1], ..., prefix[b-1])
            # Then max sub-sum = max over b of prefix[b] - min_prefix_before[b]
            
            # Similarly for max negative sub-sum (i.e., min sub-sum):
            # min over b of prefix[b] - max over a-1 < b of (prefix[a-1] if a>0 else 0)
            # Let max_prefix_before[b] = max(0, prefix[0], ..., prefix[b-1])
            # Then min sub-sum = min over b of prefix[b] - max_prefix_before[b]
            
            L = len(prefix)
            if L == 0:
                continue
            
            # min_prefix_before[b] for b=0..L-1
            # For b=0: min_prefix_before[0] = 0
            # For b>0: min_prefix_before[b] = min(0, prefix[0], ..., prefix[b-1])
            
            min_pref_before = np.zeros(L)
            max_pref_before = np.zeros(L)
            
            current_min = 0
            current_max = 0
            for b in range(L):
                min_pref_before[b] = current_min
                max_pref_before[b] = current_max
                if b < L - 1:
                    current_min = min(current_min, prefix[b])
                    current_max = max(current_max, prefix[b])
            
            # Max positive sub-sum
            max_pos = np.max(prefix - min_pref_before)
            # Min sub-sum (most negative)
            min_sub = np.min(prefix - max_pref_before)
            
            # The maximum absolute sum for this residue class is max(|max_pos|, |min_sub|)
            # But actually, we want the max absolute value of any sub-sum.
            # max_pos >= 0, min_sub <= 0
            # So max absolute is max(max_pos, -min_sub)
            
            abs_max_here = max(max_pos, -min_sub)
            if abs_max_here > max_abs:
                max_abs = int(abs_max_here)
    
    return max_abs

def construct():
    start_time = time.time()
    deadline = start_time + 81  # 90s * 0.9
    
    # Start with a random assignment
    rng = random.Random(42)
    signs = np.array([rng.choice((-1, 1)) for _ in range(N)], dtype=np.float64)
    
    # Evaluate initial
    best_score = _eval_max_abs_sum_v3(signs)
    best_signs = signs.copy()
    
    # Local search: flip one sign at a time, accept if it improves (or doesn't worsen)
    # Use simulated annealing or just hill climbing
    
    # Precompute for each position i, which APs contain it? That's complex.
    # Instead, just evaluate the full objective after each flip.
    
    # For speed, let's use a simpler evaluation for local search
    # and only do full evaluation periodically.
    
    # Actually, _eval_max_abs_sum_v3 should be fast enough for N=256.
    # Let's time it.
    
    t0 = time.time()
    _eval_max_abs_sum_v3(signs)
    t1 = time.time()
    eval_time = t1 - t0
    
    # If eval is fast enough, do hill climbing
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try flipping each position
        for i in range(N):
            if time.time() >= deadline:
                break
            old_val = signs[i]
            signs[i] = -old_val
            new_score = _eval_max_abs_sum_v3(signs)
            if new_score < best_score:
                best_score = new_score
                best_signs = signs.copy()
                improved = True
            else:
                signs[i] = old_val
    
    # If we have time, do random restarts
    while time.time() < deadline - 5:
        # Random restart
        signs = np.array([rng.choice((-1, 1)) for _ in range(N)], dtype=np.float64)
        score = _eval_max_abs_sum_v3(signs)
        if score < best_score:
            best_score = score
            best_signs = signs.copy()
        
        # Hill climb from this starting point
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(N):
                if time.time() >= deadline - 5:
                    break
                old_val = signs[i]
                signs[i] = -old_val
                new_score = _eval_max_abs_sum_v3(signs)
                if new_score < best_score:
                    best_score = new_score
                    best_signs = signs.copy()
                    improved = True
                else:
                    signs[i] = old_val
    
    # Convert back to list of integers
    result = [int(s) for s in best_signs]
    return result

# EVOLVE-BLOCK-END
