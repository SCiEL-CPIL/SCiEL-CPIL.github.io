import numpy as np
import time

N = 50
DEADLINE = 120.0  # seconds

# EVOLVE-BLOCK-START
def _compute_ratio(h):
    """Compute the ratio L2^2 / (L1 * ConvInf) for a discrete sequence."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-15):
        return -1.0  # invalid
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return -1.0
    l2_sq = np.sum(h ** 2)
    # Compute autoconvolution (linear convolution with itself)
    conv = np.convolve(h, h, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return -1.0
    return l2_sq / (l1 * conv_inf)

def _optimize_2point(w, x, max_iters=200, tol=1e-10):
    """
    Optimize a 2-point cluster with weights w=(w1,w2) and positions x=(x1,x2).
    The continuous model: h(t) = w1 * top_hat(t-x1) + w2 * top_hat(t-x2)
    where top_hat is 1 on [0,1) and 0 otherwise.
    We sample this at N points on [-1/4, 1/4].
    """
    w = np.array(w, dtype=np.float64)
    x = np.array(x, dtype=np.float64)
    
    # Ensure w >= 0 and x in valid range
    w = np.maximum(w, 0.0)
    x = np.clip(x, -0.25, 0.25)
    
    def evaluate(w, x):
        # Build the discrete sequence from top-hat basis functions
        # Sample points: t_i = -1/4 + i * (1/2)/(N-1) for i=0..N-1
        t = np.linspace(-0.25, 0.25, N)
        h = np.zeros(N, dtype=np.float64)
        for wi, xi in zip(w, x):
            # top_hat(t - xi) = 1 if 0 <= t - xi < 1, else 0
            mask = (t - xi >= 0) & (t - xi < 1)
            h[mask] += wi
        return h
    
    h = evaluate(w, x)
    best_ratio = _compute_ratio(h)
    best_w, best_x = w.copy(), x.copy()
    best_h = h.copy()
    
    # Gradient-free local search (Nelder-Mead style)
    # Parameters: w1, w2, x1, x2
    params = np.concatenate([w, x])
    best_params = params.copy()
    best_val = best_ratio
    
    # Simple coordinate descent with line search
    for it in range(max_iters):
        improved = False
        for dim in range(len(params)):
            step = 0.01
            for _ in range(20):
                trial = best_params.copy()
                trial[dim] += step
                # Clamp
                trial[0] = max(trial[0], 0)
                trial[1] = max(trial[1], 0)
                trial[2] = np.clip(trial[2], -0.25, 0.25)
                trial[3] = np.clip(trial[3], -0.25, 0.25)
                
                h_trial = evaluate(trial[:2], trial[2:])
                val_trial = _compute_ratio(h_trial)
                if val_trial > best_val + 1e-12:
                    best_params = trial
                    best_val = val_trial
                    best_w = trial[:2].copy()
                    best_x = trial[2:].copy()
                    best_h = h_trial.copy()
                    improved = True
                    step *= 1.5
                else:
                    step *= 0.5
                    # Try negative direction
                    trial_neg = best_params.copy()
                    trial_neg[dim] -= step
                    trial_neg[0] = max(trial_neg[0], 0)
                    trial_neg[1] = max(trial_neg[1], 0)
                    trial_neg[2] = np.clip(trial_neg[2], -0.25, 0.25)
                    trial_neg[3] = np.clip(trial_neg[3], -0.25, 0.25)
                    h_trial_neg = evaluate(trial_neg[:2], trial_neg[2:])
                    val_trial_neg = _compute_ratio(h_trial_neg)
                    if val_trial_neg > best_val + 1e-12:
                        best_params = trial_neg
                        best_val = val_trial_neg
                        best_w = trial_neg[:2].copy()
                        best_x = trial_neg[2:].copy()
                        best_h = h_trial_neg.copy()
                        improved = True
                        step *= 1.5
                    else:
                        break
        
        if not improved:
            break
        if time.time() > DEADLINE - 30:
            break
    
    return best_h, best_val

def construct():
    """
    Constructs a sequence of 50 nonnegative heights.
    Strategy: Optimize a 2-point top-hat cluster using local search.
    """
    start_time = time.time()
    
    # Initial candidates for 2-point cluster
    # w = (w1, w2), x = (x1, x2)
    # Try various initial conditions
    candidates = [
        ([2.0/3, 1.0/3], [-0.25, -0.25]),  # original best
        ([1.0, 1.0], [-0.25, -0.15]),
        ([1.5, 0.5], [-0.25, -0.2]),
        ([1.0, 0.5], [-0.25, -0.1]),
        ([2.0, 1.0], [-0.2, -0.1]),
        ([1.0, 1.0], [-0.25, 0.0]),
        ([1.5, 0.5], [-0.1, -0.05]),
    ]
    
    best_h = None
    best_val = -1.0
    
    for w_init, x_init in candidates:
        if time.time() > DEADLINE - 30:
            break
        h, val = _optimize_2point(w_init, x_init, max_iters=100)
        if val > best_val:
            best_val = val
            best_h = h
    
    # If no valid result, fall back to known good
    if best_h is None or best_val < 0:
        h = [0.0] * N
        h[0] = 2.0 / 3.0
        h[1] = 1.0 / 3.0
        return h
    
    return list(best_h)
# EVOLVE-BLOCK-END
