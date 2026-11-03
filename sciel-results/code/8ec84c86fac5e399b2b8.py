"""
Strategy: Structured Parametric Optimization.
Instead of optimizing 100 independent variables, we assume the optimal solution
has a periodic block structure: h[i] is 1 if (i*phase + offset) mod period < width, else 0.
We optimize the continuous parameters (period, width, offset) using Nelder-Mead.
This reduces the search space from 100D to 3D and allows for rapid, precise optimization.
"""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START

def _max_overlap_fast(h):
    """
    Compute the maximum overlap of h with shifted (1-h).
    Overlap at shift k is sum_i h[i] * (1 - h[(i+k) % N]).
    We want to minimize the maximum of this over all k.
    
    Note: The task description says "minimize the maximum overlap of h with shifted 1-h".
    Let's verify the formula.
    Overlap(k) = sum_i h[i] * (1 - h[i+k]) = sum_i h[i] - sum_i h[i]h[i+k].
    Since sum_i h[i] = N/2 is constant, minimizing max Overlap(k) is equivalent to
    maximizing min_k (sum_i h[i]h[i+k]).
    sum_i h[i]h[i+k] is the circular autocorrelation at lag k.
    So we want to maximize the minimum circular autocorrelation.
    
    The current code uses:
    autocorr = real(ifft(fft(h) * conj(fft(h))))
    min_autocorr = min(autocorr)
    max_overlap = N/2 - min_autocorr
    
    This matches the logic above.
    """
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    min_autocorr = np.min(autocorr)
    max_overlap = N / 2.0 - min_autocorr
    return max_overlap

def _construct_block_signal(period, width, offset):
    """
    Construct a periodic block signal.
    period: length of the repeating pattern (float, will be rounded)
    width: length of the '1' block within the period (float)
    offset: phase shift (float)
    
    Returns an array of length N with values in [0, 1].
    To ensure the sum is exactly N/2, we might need to adjust.
    However, a pure block signal of 0s and 1s with exactly 50 ones is ideal for minimizing overlap?
    Actually, the problem allows h in [0,1]. 
    Let's stick to 0/1 blocks first, as they are extreme points and often optimal for such problems.
    If the ratio of width to period doesn't yield exactly 50 ones, we can adjust one element or use a fractional last block.
    
    A more robust approach:
    Generate the pattern for N elements.
    For each i in 0..N-1:
        pos = (i + offset) % period
        if pos < width: h[i] = 1 else 0
    
    Then, adjust to ensure sum(h) == N/2.
    """
    h = np.zeros(N)
    # Ensure period is at least 1
    p = max(1.0, period)
    w = max(0.0, min(width, p))
    
    for i in range(N):
        pos = (i + offset) % p
        if pos < w:
            h[i] = 1.0
    
    # Adjust sum to be exactly N/2
    current_sum = np.sum(h)
    target_sum = N / 2.0
    diff = target_sum - current_sum
    
    if abs(diff) > 1e-9:
        # We need to add or remove 'mass'.
        # Strategy: find elements closest to the boundary or just modify the last element?
        # Better: distribute the difference? No, h must be in [0,1].
        # If diff > 0, we need to increase some h[i].
        # If diff < 0, we need to decrease some h[i].
        
        # Simple approach: modify the element at index 0? Or spread it?
        # Spreading might break the block structure.
        # Let's try to adjust the 'width' effectively by modifying the fractional part.
        # Or, simpler: just clip and project?
        
        # Let's try a projection that preserves the "blockiness" as much as possible.
        # Actually, for the optimization, it's better to allow fractional values in the construction
        # if the integer block count doesn't match.
        
        # Alternative construction:
        # Use a continuous "ramp" at the edges? No, keep it simple.
        # Let's just project to the feasible set [0,1] with sum N/2.
        # But projection might ruin the structure.
        
        # Let's try this: If we need to add mass, add it to the 1s (making them >1? No, max is 1).
        # So we can only add mass to 0s (making them >0) or remove from 1s (making them <1).
        
        # Let's just do a standard projection for now, but try to keep the structure.
        # Actually, the Nelder-Mead will optimize the parameters. If the discrete count is off by 1,
        # the error is small.
        
        # Let's modify the construction to allow fractional width effectively.
        # If current_sum < target_sum, we need more ones.
        # If current_sum > target_sum, we need fewer ones.
        
        # Let's just use a simple correction on the last element to satisfy the constraint strictly.
        # This might create a "spike" but it's valid.
        h = np.clip(h, 0, 1)
        current_sum = np.sum(h)
        diff = target_sum - current_sum
        if abs(diff) > 1e-9:
            # Add/subtract diff to h[0]?
            # Better to distribute among the most "central" elements of the blocks?
            # For now, just adjust h[0] to ensure validity.
            h[0] += diff
            # If h[0] goes out of bounds, it's a problem.
            # Let's check bounds.
            if h[0] < 0 or h[0] > 1:
                # Fallback: uniform adjustment? No, that ruins structure.
                # Let's try to adjust the element closest to the boundary of the block.
                # For simplicity and robustness in the optimizer, let's just clip and hope the optimizer
                # finds parameters where the integer count is close.
                # Actually, let's just project properly.
                h = np.clip(h, 0, 1)
                # Re-calculate sum
                current_sum = np.sum(h)
                diff = target_sum - current_sum
                if diff > 0:
                    # Add to zeros
                    zeros = np.where(h == 0)[0]
                    if len(zeros) > 0:
                        # Add diff to one zero? Or spread?
                        # Spreading is safer for staying in [0,1]
                        # But we only have diff amount.
                        # Let's just add to the first zero.
                        h[zeros[0]] += diff
                else:
                    # Subtract from ones
                    ones = np.where(h == 1)[0]
                    if len(ones) > 0:
                        h[ones[0]] += diff # diff is negative
                
                h = np.clip(h, 0, 1)
                
    return h

def _objective(params):
    """
    Objective function for Nelder-Mead.
    params: [period, width, offset]
    Returns the max overlap score.
    """
    period, width, offset = params
    h = _construct_block_signal(period, width, offset)
    return _max_overlap_fast(h)

def _nelder_mead_3d(f, x0, max_iter=5000, tol=1e-8):
    """
    Simple Nelder-Mead implementation for 3D.
    """
    n = len(x0)
    # Create initial simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        x = np.array(x0, dtype=float)
        x[i] += 0.1 if x[i] == 0 else x[i] * 0.1
        simplex.append(x)
    
    fvals = [f(x) for x in simplex]
    
    for _ in range(max_iter):
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        
        # Reflection
        x_r = centroid + (centroid - worst)
        f_r = f(x_r)
        
        if f_r < fvals[0]:
            # Expansion
            x_e = centroid + 2 * (centroid - worst)
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            # Contraction
            x_c = centroid + 0.5 * (worst - centroid)
            f_c = f(x_c)
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
    
    return simplex[0], fvals[0]

def _generate_initial_guesses():
    """
    Generate diverse initial guesses for (period, width, offset).
    """
    guesses = []
    # Try different periods
    for p in [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 25, 50, 100]:
        # Width should be roughly p/2 to get sum ~ N/2
        # If p divides N, width = p/2 gives exactly 50 ones.
        # If not, we adjust.
        w = p / 2.0
        # Try a few offsets
        for o in [0.0, p/4.0, p/2.0, p/3.0]:
            guesses.append([float(p), w, o])
    return guesses

def construct():
    start_time = time.time()
    deadline = 140.0  # 10 second safety margin
    
    best_h = None
    best_score = float('inf')
    
    guesses = _generate_initial_guesses()
    
    # Shuffle guesses to ensure diversity in case of timeout
    np.random.seed(42)
    np.random.shuffle(guesses)
    
    # Limit number of guesses to fit in time
    # Each NM run takes ~0.1-0.5s depending on convergence.
    # We have ~140s. Let's allow up to 100 runs? Or just all 40?
    # Let's prioritize and run as many as possible.
    
    for guess in guesses:
        if time.time() > deadline - 5.0:
            break
            
        try:
            x_opt, score_opt = _nelder_mead_3d(_objective, guess, max_iter=2000)
            
            if score_opt < best_score:
                best_score = score_opt
                # Reconstruct the best h
                best_h = _construct_block_signal(x_opt[0], x_opt[1], x_opt[2])
                
        except Exception:
            continue
    
    # If no valid h was found (unlikely), fallback to uniform
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    # Final check and projection
    # Ensure sum is exactly N/2 and bounds are respected
    # The _construct_block_signal tries to do this, but let's double check.
    current_sum = np.sum(best_h)
    target_sum = N / 2.0
    diff = target_sum - current_sum
    if abs(diff) > 1e-9:
        # Simple correction
        best_h[0] += diff
        best_h = np.clip(best_h, 0, 1)
        # If clipping changed the sum, do a second pass
        current_sum = np.sum(best_h)
        diff = target_sum - current_sum
        if abs(diff) > 1e-9:
             # Distribute evenly?
             # This is a rare edge case.
             best_h += diff / N
             best_h = np.clip(best_h, 0, 1)
    
    return best_h

# EVOLVE-BLOCK-END
