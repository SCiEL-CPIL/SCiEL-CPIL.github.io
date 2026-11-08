import numpy as np
import time

N = 400

def _calc_metrics(h):
    """
    Calculate the autoconvolution, integral, and the objective score.
    Returns: max_abs_autoconv, integral, score
    """
    # Autoconvolution
    # np.convolve(h, h, mode='full') gives length 2N-1
    # Lag 0 is at index N-1
    C = np.convolve(h, h, mode='full')
    # We consider lags k != 0.
    # C indices: 0 to 2N-2.
    # Lag 0 is C[N-1].
    # Sidelobes are C[0:N-1] and C[N:2N-1].
    max_abs_sid = np.max(np.abs(np.concatenate([C[:N-1], C[N:]])))
    
    integral = np.sum(h)
    int_sq = integral * integral
    
    if int_sq == 0:
        return max_abs_sid, integral, float('inf')
        
    score = max_abs_sid / int_sq
    return max_abs_sid, integral, score

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Initial sequence: All 1s
    h = np.ones(N, dtype=np.int8)
    
    # Get initial metrics
    max_abs_sid, integral, best_score = _calc_metrics(h)
    best_h = h.copy()
    
    # Hill Climbing
    # We try to flip bits to reduce the score
    # To speed up, we can estimate the change, but for N=400, full recalc is ~0.1ms?
    # np.convolve is fast.
    
    stuck_count = 0
    
    while time.time() < deadline:
        improved = False
        # Try all possible single flips
        best_flip_idx = -1
        best_flip_score = best_score
        
        # Optimization: Only check a subset if N is large, but N=400 is small.
        # We can vectorize the flip evaluation?
        # Flipping h[i] changes C[k] by -2*h[i]*(h[i-k] + h[i+k]).
        # This is complex to vectorize for max.
        
        for i in range(N):
            # Check time
            if time.time() > deadline:
                break
                
            # Flip
            h[i] *= -1
            
            # Calculate new score
            # We can update C incrementally?
            # C_new[k] = C_old[k] - 2*h_old[i]*(h_old[i-k] + h_old[i+k])
            # But we don't have C_old stored globally in an easy way for all k without array ops.
            # Let's just recalc.
            m, _, s = _calc_metrics(h)
            
            if s < best_flip_score:
                best_flip_score = s
                best_flip_idx = i
                improved = True
            else:
                # Revert
                h[i] *= -1
        
        if improved:
            # Apply the best flip
            h[best_flip_idx] *= -1
            best_score = best_flip_score
            best_h = h.copy()
            stuck_count = 0
        else:
            # No improvement found in this round
            stuck_count += 1
            if stuck_count > 5:
                # Try a random restart with a small number of -1s
                # Or just break if we've been stuck too long
                # Let's try a different initial pattern: 
                # e.g., flip first K bits
                # But we might lose the good state.
                # Let's just break to be safe, or try a random perturbation.
                
                # Random perturbation: flip 10 random bits
                indices = np.random.choice(N, 10, replace=False)
                h[indices] *= -1
                m, _, s = _calc_metrics(h)
                if s < best_score:
                    best_score = s
                    best_h = h.copy()
                    stuck_count = 0
                else:
                    # Revert to best_h
                    h = best_h.copy()
                    # If we are stuck at All-Ones or near it, break
                    if stuck_count > 20:
                        break

    return best_h

# Test locally if allowed? No, just output.
# The single flip at index 1 (0-indexed) is a known improvement.
# The hill climber should find it.
