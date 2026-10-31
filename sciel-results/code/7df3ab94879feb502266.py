"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START

def _autoconvolve(x):
    """Full autoconvolution of x (length 2N-1)."""
    return np.convolve(x, x, mode='full')

def _score(x):
    """Score: max |autoconv| / (sum x)^2. Lower is better."""
    ac = _autoconvolve(x)
    denom = (np.sum(x)) ** 2
    if denom == 0:
        return np.inf
    return np.max(np.abs(ac)) / denom

def _flip_gain(x, i, current_ac, current_sum):
    """
    Compute the change in score if we flip x[i] (i.e., x[i] -> -x[i]).
    We track the change in max|ac| and denom.
    
    Flipping x[i] changes ac by: -2 * x[i] * (x shifted by i)
    Specifically, ac[k] changes by -2*x[i]*x[k-i] for k in [i, i+2N-2]... 
    Actually ac = convolve(x,x). ac[j] = sum_l x[l]x[j-l].
    If we change x[i] to -x[i], the change in ac[j] is:
      delta_ac[j] = -2*x[i]*x[j-i]  (for the term where l=i) 
                   + (-2*x[i])*x[j-i] (for the term where j-l=i => l=j-i)
    Wait, if i == j-i, it's counted twice? No.
    ac[j] = sum_{l} x[l] x[j-l].
    Term with l=i: x[i] x[j-i].
    Term with j-l=i => l=j-i: x[j-i] x[i].
    If we replace x[i] with -x[i]:
    New term l=i: -x[i] x[j-i]. Change: -2 x[i] x[j-i].
    New term l=j-i: x[j-i] (-x[i]). Change: -2 x[j-i] x[i].
    If i != j-i, total change is -4 x[i] x[j-i].
    If i == j-i (j=2i), total change is -2 x[i]^2 - 2 x[i]^2 = -4 x[i]^2? 
    Wait. If j=2i, the term is x[i]*x[i]. 
    Original: x[i]^2.
    New: (-x[i])^2 = x[i]^2. Change is 0?
    Let's re-evaluate.
    ac[j] = sum_l x[l] x[j-l].
    If we flip x[i] to x'[i] = -x[i]:
    ac'[j] = sum_{l != i} x[l] x[j-l] + x'[i] x[j-i] + (if j-i != i) x[j-i] x'[i]
    
    Case 1: i != j-i.
    ac'[j] = ac[j] - x[i]x[j-i] - x[j-i]x[i] + (-x[i])x[j-i] + x[j-i](-x[i])
           = ac[j] - 2 x[i]x[j-i] - 2 x[i]x[j-i]
           = ac[j] - 4 x[i] x[j-i]
    
    Case 2: i == j-i => j = 2i.
    ac'[j] = ac[j] - x[i]x[i] + (-x[i])x[i]
           = ac[j] - x[i]^2 - x[i]^2
           = ac[j] - 2 x[i]^2
    
    So:
    if j == 2*i: delta_ac[j] = -2 * x[i]**2
    else: delta_ac[j] = -4 * x[i] * x[j-i]  (valid only if 0 <= j-i < N)
    
    Also, sum changes by -2*x[i].
    denom changes from S^2 to (S-2x[i])^2.
    """
    xi = x[i]
    # Indices in ac affected: j such that 0<=j-i<N and 0<=j<N+i
    # j ranges from i to i+2N-2.
    # But we only care about j where x[j-i] is defined (0 <= j-i < N) and x[j-(N-1-i)]? No.
    # ac index j corresponds to sum_{l=0}^{N-1} x[l] x[j-l].
    # The term involving x[i] is x[i]x[j-i]. This is non-zero only if 0 <= j-i <= N-1.
    # So j-i in [0, N-1] => j in [i, i+N-1].
    # Also, is there another term? x[j-i]x[i]. Same condition.
    # So affected j are in [i, i+N-1].
    
    j_start = i
    j_end = i + N  # exclusive
    
    # Get slice of x that interacts with x[i]
    # x[j-i] for j in [i, i+N-1] is x[0:N]
    # So delta_ac[j] depends on x[j-i].
    
    # We need to update ac[j] for j in [j_start, j_end-1].
    # Let's compute the new ac slice.
    # old_ac_slice = current_ac[j_start:j_end]
    # For each j in slice:
    #   k = j - i (index in x)
    #   if k == i: delta = -2*xi*xi
    #   else: delta = -4*xi*x[k]
    
    # Vectorized:
    k_indices = np.arange(N) # 0 to N-1
    deltas = np.full(N, -4.0 * xi)
    deltas *= x # -4 * xi * x[k]
    # Correct the center where k == i
    deltas[i] = -2.0 * xi * xi
    
    new_ac_slice = current_ac[j_start:j_end] + deltas
    
    # Check for any out-of-bounds in current_ac?
    # j_start >= 0, j_end <= i+N <= 2N-1. So it's safe.
    
    # New sum
    new_sum = current_sum - 2.0 * xi
    new_denom = new_sum * new_sum
    
    if new_denom == 0:
        return np.inf # Avoid division by zero, or handle separately
    
    # We need max |new_ac|.
    # new_ac = current_ac with slice replaced.
    # To avoid copying full array, we can compute max of:
    # 1. max |current_ac[:j_start]|
    # 2. max |new_ac_slice|
    # 3. max |current_ac[j_end:]|
    
    m1 = np.max(np.abs(current_ac[:j_start])) if j_start > 0 else 0.0
    m2 = np.max(np.abs(new_ac_slice))
    m3 = np.max(np.abs(current_ac[j_end:])) if j_end < len(current_ac) else 0.0
    
    new_max_ac = max(m1, m2, m3)
    
    new_score = new_max_ac / new_denom
    return new_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initialize with random binary sequence
    # Or start with all 1s and perturb?
    # Random start is better for diversity.
    x = np.random.choice([-1.0, 1.0], size=N)
    
    # Normalize? The score is scale invariant if we scale x by c:
    # ac -> c^2 ac, sum -> c sum, score -> c^2 max|ac| / (c^2 sum^2) = same.
    # So binary is fine.
    
    current_ac = _autoconvolve(x)
    current_sum = np.sum(x)
    current_score = np.max(np.abs(current_ac)) / (current_sum**2)
    
    best_x = x.copy()
    best_score = current_score
    
    # Local Search Loop
    # Use a list of indices to shuffle
    indices = np.arange(N)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle indices for random order
        np.random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            # Try flip
            new_score = _flip_gain(x, i, current_ac, current_sum)
            
            if new_score < current_score:
                # Accept flip
                x[i] = -x[i]
                
                # Update current_ac incrementally
                # This is the expensive part if done naively, but we already computed deltas in _flip_gain?
                # No, _flip_gain only returned the score. We need to update current_ac.
                # Let's re-derive the update for current_ac to keep it consistent.
                
                xi = -x[i] # This is the NEW value? No, x[i] is already flipped.
                # Let's use the OLD value for calculation.
                # Actually, let's just recompute the slice update.
                
                # Old x[i] was -x[i] (since we just flipped it).
                old_xi = -x[i]
                
                j_start = i
                j_end = i + N
                
                k_indices = np.arange(N)
                deltas = np.full(N, -4.0 * old_xi)
                deltas *= x # Wait, x has changed. We need x[k] where k != i?
                # The formula delta_ac[j] = -4 * x_old[i] * x[j-i] assumes x[j-i] is the CURRENT value.
                # If j-i == i, we handled it separately.
                # If j-i != i, x[j-i] is unchanged by the flip.
                # So using current x is correct for k != i.
                # But we multiplied by x which includes x[i].
                # For k=i, we overwrite deltas[i] anyway.
                # So this is safe.
                
                deltas[i] = -2.0 * old_xi * old_xi
                
                current_ac[j_start:j_end] += deltas
                
                current_sum -= 2.0 * old_xi
                current_score = new_score
                
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
            
            # If no improvement in this pass, we could break early?
            # No, standard local search tries all.
    
    return best_x

# EVOLVE-BLOCK-END
