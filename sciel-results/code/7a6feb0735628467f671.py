"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 120.0

# EVOLVE-BLOCK-START
def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    The metric is defined as: max_k |conv(h, h)[k]| / (sum_i |h_i|)^2
    where conv is the discrete autoconvolution.
    """
    # signed_step is an array of N values
    conv = np.convolve(signed_step, signed_step, mode='full')
    sum_abs_h = np.sum(np.abs(signed_step))
    
    if sum_abs_h < 1e-12:
        return float('inf')
        
    return np.max(np.abs(conv)) / (sum_abs_h ** 2)

def _c3_upper_bound(h_init):
    """
    Optimize the signed step function to minimize the C3 upper bound.
    Uses a combination of continuous tanh-relaxation optimization (L-BFGS-B)
    and discrete hill-climbing refinement.
    """
    start_time = time.time()
    
    def objective(x):
        # Deadline check inside objective to prevent hanging
        if time.time() - start_time > DEADLINE - 5.0:
            raise ValueError("Deadline exceeded")
            
        h = np.tanh(x)  # Bounded in (-1, 1)
        
        # Compute autoconvolution
        conv = np.convolve(h, h, mode='full')
        
        # Smooth absolute value
        eps = 1e-10
        abs_conv = np.sqrt(conv**2 + eps)
        
        # Soft max with temperature T
        # We want to approximate max(abs_conv).
        # Using log-sum-exp: log(sum(exp(x/T))) * T
        # For stability, subtract max
        T = 5.0
        max_abs = np.max(abs_conv)
        # If max_abs is very large, exp might overflow if we don't shift properly.
        # (abs_conv - max_abs) / T is <= 0.
        shifted = (abs_conv - max_abs) / T
        log_sum_exp = np.log(np.sum(np.exp(shifted))) + max_abs / T
        soft_max = T * log_sum_exp
        
        # Sum of absolute values of h
        sum_abs_h = np.sum(np.sqrt(h**2 + eps))
        
        # Objective: soft_max / (sum_abs_h)^2
        value = soft_max / (sum_abs_h ** 2)
        return value

    def refine_discrete(h, max_iters=2000):
        """
        Discrete hill-climbing refinement.
        Tries to flip individual elements to improve the score.
        """
        h = h.copy()
        current_val = _compute_c3(h)
        best_val = current_val
        best_h = h.copy()
        
        # Precompute convolution for delta calculation? 
        # Flipping h[i] from v to -v changes conv by:
        # conv_new = conv_old - 2*v * (h shifted by i) ... roughly.
        # Actually, conv[k] = sum_j h[j] h[k-j].
        # If we change h[i] to h[i] + delta, conv changes by delta * (h + reverse(h) shifted).
        # Let's just recompute for simplicity in the first pass, or use a fast delta.
        # For N=400, recompute is fast enough if we limit iters.
        
        # To make it faster, we can try random flips first, then local.
        # But let's stick to a simple stochastic local search with restarts or just greedy.
        
        # Greedy local search: find the single flip that gives the biggest improvement.
        # This is O(N^2) per step if we recompute, which is 160k ops.
        # 2000 iters * 160k = 320M ops. Might be tight in Python/NumPy loop overhead.
        # Let's limit iters based on time.
        
        iters_done = 0
        while iters_done < max_iters:
            if time.time() - start_time > DEADLINE - 5.0:
                break
                
            improved = False
            # Try flipping each element
            # Optimization: Instead of trying all N, try a random subset or use a smarter heuristic?
            # For now, let's try all N but break early if we find a good one? No, greedy needs best.
            # Let's just do a few random flips to escape local minima, then greedy.
            
            # Random restart within the discrete space? No, just random flips.
            idx = np.random.randint(N)
            old_val = h[idx]
            h[idx] = -h[idx]
            new_val = _compute_c3(h)
            
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
                improved = True
            else:
                h[idx] = old_val # Revert
            
            iters_done += 1
            
            # If no improvement for a while, might be stuck. 
            # But with random flips, it can escape.
            
        return best_h, best_val

    # 1. Continuous Optimization Phase
    # Initialize with a few diverse starting points
    best_h = None
    best_score = float('inf')
    
    # Seed: all ones
    x0 = np.full(N, 5.0)
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
        if res.success or res.fun < best_score:
            h_opt = np.tanh(res.x)
            score = _compute_c3(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt
    except Exception:
        pass
    
    # Alternating signs
    x0_alt = np.zeros(N)
    x0_alt[::2] = 2.0
    x0_alt[1::2] = -2.0
    try:
        res = minimize(objective, x0_alt, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
        if res.success or res.fun < best_score:
            h_opt = np.tanh(res.x)
            score = _compute_c3(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt
    except Exception:
        pass

    # Random initialization
    np.random.seed(42)
    x0_rand = np.random.randn(N) * 2.0
    try:
        res = minimize(objective, x0_rand, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
        if res.success or res.fun < best_score:
            h_opt = np.tanh(res.x)
            score = _compute_c3(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt
    except Exception:
        pass

    if best_h is None:
        # Fallback to seed
        best_h = np.ones(N)
        best_score = _compute_c3(best_h)
        
    # 2. Discrete Refinement Phase
    # Use the best continuous result as a starting point for discrete search
    # Quantize to +/- 1? Or keep continuous but flip signs?
    # The task says "signed step function", usually implies discrete signs if we want to minimize max conv.
    # Let's try refining the signs of the continuous solution.
    # Map continuous h to signs: h_sign = sign(h)
    # But maybe the magnitudes matter? The metric is scale invariant?
    # max |conv| / (sum |h|)^2.
    # If we scale h by c, conv scales by c^2, sum|h| scales by c.
    # Ratio: c^2 / c^2 = 1. So magnitude doesn't matter, only the shape and signs.
    # Thus, we can restrict to h_i in {-1, 1} without loss of generality for the optimal value.
    # However, the continuous optimization might find a "better" shape that isn't just +/-1?
    # No, because of scale invariance, we can always normalize.
    # But wait, if we have mixed magnitudes, e.g., 0.5 and 1.0, does that help?
    # Let h' = h / max|h|. Then h'_i in [-1, 1].
    # The ratio is invariant.
    # So we can just work with h_i in {-1, 1} or [-1, 1].
    # The discrete search on signs is a subset of the continuous space.
    # Let's refine the signs of the best continuous solution.
    
    h_signs = np.sign(best_h)
    # Handle zeros
    h_signs[h_signs == 0] = 1.0
    
    h_refined, score_refined = refine_discrete(h_signs, max_iters=5000)
    
    if score_refined < best_score:
        best_h = h_refined
        best_score = score_refined
        
    # Try refining a random discrete start as well
    h_rand_signs = np.random.choice([-1.0, 1.0], size=N)
    h_refined2, score_refined2 = refine_discrete(h_rand_signs, max_iters=5000)
    
    if score_refined2 < best_score:
        best_h = h_refined2
        best_score = score_refined2

    return best_h

def construct():
    return _c3_upper_bound(np.full(N, 5.0))
# EVOLVE-BLOCK-END
