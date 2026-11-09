import numpy as np
import time
from scipy.optimize import minimize

N = 100

def calculate_max_overlap(h):
    """Calculate maximum linear overlap of h with shifted 1-h."""
    max_ov = 0.0
    for k in range(N):
        if k == 0:
            ov = np.sum(h * (1 - h))
        else:
            ov = np.sum(h[k:] * (1 - h[:-k]))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def soft_max_overlaps(h, beta=20.0):
    """Smooth surrogate for max overlap using log-sum-exp."""
    overlaps = np.zeros(N)
    for k in range(N):
        if k == 0:
            overlaps[k] = np.sum(h * (1 - h))
        else:
            overlaps[k] = np.sum(h[k:] * (1 - h[:-k]))
    # Log-sum-exp for numerical stability
    m = np.max(overlaps)
    if m < 1e-10:
        return 0.0
    return m + np.log(np.sum(np.exp((overlaps - m) * beta))) / beta

def _construct_2level_block(a, block_len, N):
    """
    Construct a 2-level sequence with level a and 1-a, arranged in blocks of length block_len.
    Pattern: a repeated block_len times, then (1-a) repeated block_len times, etc.
    Returns the sequence and whether it's valid (sum close to N/2).
    """
    h = np.zeros(N)
    # Number of full blocks
    num_full_blocks = N // (2 * block_len)
    # Fill full blocks
    for b in range(num_full_blocks):
        start = b * 2 * block_len
        h[start:start + block_len] = a
        h[start + block_len:start + 2 * block_len] = 1 - a
    
    # Handle remainder
    rem_start = num_full_blocks * 2 * block_len
    rem = N - rem_start
    if rem > 0:
        # First fill with a
        fill_a = min(rem, block_len)
        h[rem_start:rem_start + fill_a] = a
        rem -= fill_a
        if rem > 0:
            h[rem_start + fill_a:rem_start + fill_a + rem] = 1 - a
    
    # Check sum
    total = np.sum(h)
    # Count of a's and (1-a)'s
    count_a = np.sum(h == a)
    count_1ma = np.sum(h == 1 - a)
    # For exact sum = N/2, we need: count_a * a + count_1ma * (1-a) = N/2
    # This is automatically satisfied if count_a = count_1ma = N/2
    # But with blocks, this may not hold exactly.
    return h, total

def _optimize_2level_block(a, block_len, N):
    """
    For given a and block_len, try to adjust the last few elements to make sum exactly N/2.
    Returns the optimized h or None if not possible.
    """
    h, total = _construct_2level_block(a, block_len, N)
    
    # If sum is already correct, return
    if abs(total - N / 2.0) < 1e-10:
        return h
    
    # Try to fix by adjusting individual elements
    # The sum deviation
    diff = N / 2.0 - total
    
    # We can adjust elements that are not at the boundary (0 or 1)
    # For 2-level, all elements are either a or 1-a
    # If a is in (0,1), we can adjust any element
    
    # Strategy: find an element and adjust it
    # If diff > 0, we need to increase some elements
    # If diff < 0, we need to decrease some elements
    
    # Let's try adjusting the last element first
    idx = N - 1
    new_val = h[idx] + diff
    if 0 <= new_val <= 1:
        h[idx] = new_val
        return h
    
    # Try adjusting the second to last
    idx = N - 2
    new_val = h[idx] + diff
    if 0 <= new_val <= 1:
        h[idx] = new_val
        return h
    
    # If neither works, try distributing
    # Split diff among last few elements
    for split_count in [2, 3, 4]:
        if split_count > N:
            continue
        per_elem = diff / split_count
        valid = True
        for j in range(split_count):
            idx = N - 1 - j
            new_val = h[idx] + per_elem
            if not (0 <= new_val <= 1):
                valid = False
                break
        if valid:
            for j in range(split_count):
                idx = N - 1 - j
                h[idx] += per_elem
            return h
    
    return None

def _search_2level_blocks(N, deadline):
    """
    Search over 2-level block patterns to find the one with minimum max overlap.
    """
    best_h = None
    best_score = float('inf')
    
    # Search over block lengths
    # block_len must be such that we can have roughly equal counts of a and 1-a
    # For simplicity, search block_len from 1 to N//2
    
    for block_len in range(1, N // 2 + 1):
        if time.time() > deadline - 10.0:
            break
        
        # For each block_len, search over a
        # The sum constraint: if we have equal numbers of a and (1-a), sum = N/2 automatically
        # With blocks of length block_len, we have pairs of blocks
        # Let's search a in (0, 1)
        
        # Coarse search
        a_coarse = np.linspace(0.01, 0.99, 50)
        best_a_for_bl = None
        best_score_for_bl = float('inf')
        
        for a in a_coarse:
            h_candidate = _optimize_2level_block(a, block_len, N)
            if h_candidate is not None:
                score = calculate_max_overlap(h_candidate)
                if score < best_score_for_bl:
                    best_score_for_bl = score
                    best_a_for_bl = a
        
        # Fine search around best_a_for_bl
        if best_a_for_bl is not None:
            a_fine = np.linspace(max(0.001, best_a_for_bl - 0.02), min(0.999, best_a_for_bl + 0.02), 50)
            for a in a_fine:
                h_candidate = _optimize_2level_block(a, block_len, N)
                if h_candidate is not None:
                    score = calculate_max_overlap(h_candidate)
                    if score < best_score_for_bl:
                        best_score_for_bl = score
                        best_a_for_bl = a
        
        if best_score_for_bl < best_score:
            best_score = best_score_for_bl
            h_candidate = _optimize_2level_block(best_a_for_bl, block_len, N)
            if h_candidate is not None:
                best_h = h_candidate
    
    return best_h, best_score

def _search_interleaved_2level(N, deadline):
    """
    Search over interleaved 2-level patterns.
    Pattern: a, 1-a, a, 1-a, ... or a, a, 1-a, 1-a, ... etc.
    """
    best_h = None
    best_score = float('inf')
    
    # Try different interleaving patterns
    # Pattern 1: alternating a, 1-a
    # Pattern 2: a, a, 1-a, 1-a (block size 2)
    # etc.
    
    for block_size in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 10.0:
            break
        
        # For this block size, search over a
        a_coarse = np.linspace(0.01, 0.99, 30)
        best_a = None
        best_score_bl = float('inf')
        
        for a in a_coarse:
            # Construct pattern
            h = np.zeros(N)
            pattern = [a] * block_size + [1 - a] * block_size
            pattern_len = len(pattern)
            
            for i in range(N):
                h[i] = pattern[i % pattern_len]
            
            # Check sum
            total = np.sum(h)
            diff = N / 2.0 - total
            
            # Try to fix sum
            if abs(diff) < 1e-10:
                score = calculate_max_overlap(h)
                if score < best_score_bl:
                    best_score_bl = score
                    best_a = a
            else:
                # Try adjusting last element
                new_val = h[-1] + diff
                if 0 <= new_val <= 1:
                    h[-1] = new_val
                    score = calculate_max_overlap(h)
                    if score < best_score_bl:
                        best_score_bl = score
                        best_a = a
        
        if best_score_bl < best_score:
            best_score = best_score_bl
            # Reconstruct
            a = best_a
            h = np.zeros(N)
            pattern = [a] * block_size + [1 - a] * block_size
            pattern_len = len(pattern)
            for i in range(N):
                h[i] = pattern[i % pattern_len]
            total = np.sum(h)
            diff = N / 2.0 - total
            if abs(diff) >= 1e-10:
                new_val = h[-1] + diff
                if 0 <= new_val <= 1:
                    h[-1] = new_val
            best_h = h
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    # Base solution
    h_base = np.full(N, 0.5)
    base_score = calculate_max_overlap(h_base)
    
    best_h = h_base.copy()
    best_score = base_score
    
    # --- Phase 1: 2-Level Block Search ---
    # This is the primary strategy per the directive
    if time.time() < deadline - 30.0:
        h_block, score_block = _search_2level_blocks(N, deadline)
        if h_block is not None and score_block < best_score:
            best_h = h_block
            best_score = score_block
    
    # --- Phase 2: Interleaved 2-Level Search ---
    if time.time() < deadline - 15.0:
        h_inter, score_inter = _search_interleaved_2level(N, deadline)
        if h_inter is not None and score_inter < best_score:
            best_h = h_inter
            best_score = score_inter
    
    # --- Phase 3: Local Optimization ---
    # Use the best 2-level solution as starting point for local refinement
    if time.time() < deadline - 10.0:
        def objective_vector(h_flat):
            h = h_flat.reshape(N)
            sum_dev = np.sum(h) - N / 2.0
            penalty = 1000.0 * sum_dev**2
            bound_penalty = 0.0
            if np.min(h) < 0: bound_penalty += 1000.0 * (np.min(h)**2)
            if np.max(h) > 1: bound_penalty += 1000.0 * (np.max(h)-1)**2
            
            val = soft_max_overlaps(h, beta=20.0) + penalty + bound_penalty
            return val
        
        bounds = [(0.0, 1.0)] * N
        x0 = best_h.copy()
        
        try:
            res = minimize(objective_vector, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success:
                h_opt = res.x
                # Final sum correction
                diff = N / 2.0 - np.sum(h_opt)
                if abs(diff) > 1e-12:
                    weights = np.abs(h_opt - 0.5) + 1e-6
                    weights /= np.sum(weights)
                    h_opt = h_opt + diff * weights
                    h_opt = np.clip(h_opt, 0.0, 1.0)
                    diff2 = N / 2.0 - np.sum(h_opt)
                    if abs(diff2) > 1e-12:
                        idx = np.argmin(np.abs(h_opt - 0.5))
                        h_opt[idx] += diff2
                        h_opt = np.clip(h_opt, 0.0, 1.0)
                        
                if np.min(h_opt) >= 0 and np.max(h_opt) <= 1:
                    opt_score = calculate_max_overlap(h_opt)
                    if opt_score < best_score:
                        best_h = h_opt
                        best_score = opt_score
        except Exception:
            pass
    
    # --- Phase 4: Final Polish ---
    if time.time() < deadline - 5.0:
        for trial in range(30):
            if time.time() > deadline - 2.0:
                break
            noise = np.random.normal(0, 0.005, N)
            h_trial = best_h + noise
            h_trial = np.clip(h_trial, 0.0, 1.0)
            
            # Correct sum
            diff = N / 2.0 - np.sum(h_trial)
            if abs(diff) > 1e-12:
                weights = np.abs(h_trial - 0.5) + 1e-6
                weights /= np.sum(weights)
                h_trial = h_trial + diff * weights
                h_trial = np.clip(h_trial, 0.0, 1.0)
                diff2 = N / 2.0 - np.sum(h_trial)
                if abs(diff2) > 1e-12:
                    idx = np.argmin(np.abs(h_trial - 0.5))
                    h_trial[idx] += diff2
                    h_trial = np.clip(h_trial, 0.0, 1.0)
                    
            if np.min(h_trial) >= 0 and np.max(h_trial) <= 1:
                trial_score = calculate_max_overlap(h_trial)
                if trial_score < best_score:
                    best_h = h_trial
                    best_score = trial_score

    return best_h
