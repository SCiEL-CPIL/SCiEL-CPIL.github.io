"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_MARGIN = 10.0  # seconds to spare before 150s kill
TARGET_SUM = N / 2.0

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    if n == 0:
        return 0.0
    # shifted[x] = h[x+1] for x in [0, N-2], shifted[N-1] = 0
    # This matches the non-circular shift interpretation used in the seed and inspiration
    shifted = np.zeros(n)
    if n > 1:
        shifted[:-1] = h[1:]
    # shifted[-1] remains 0
    
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _fix_sum(h, target=None):
    """Adjust h so that sum(h) == target (default N/2), preserving [0,1] bounds."""
    if target is None:
        target = TARGET_SUM
    h = np.asarray(h, dtype=float).copy()
    diff = target - h.sum()
    if abs(diff) > 1e-12:
        # Distribute correction uniformly, but be careful with clipping
        h = h + diff / len(h)
        h = np.clip(h, 0.0, 1.0)
        # If clipping broke the sum, do a second pass on non-saturated elements
        diff2 = target - h.sum()
        if abs(diff2) > 1e-12:
            mask = (h > 1e-9) & (h < 1.0 - 1e-9)
            if np.any(mask):
                h[mask] += diff2 / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
                # Third pass if necessary
                diff3 = target - h.sum()
                if abs(diff3) > 1e-9:
                    mask2 = (h > 0) & (h < 1.0)
                    if np.any(mask2):
                        h[mask2] += diff3 / np.sum(mask2)
                        h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_max_and_grad(x, beta):
    """
    Compute soft max of x and its gradient using log-sum-exp.
    Returns (value, gradient_vector).
    """
    x = np.asarray(x, dtype=float)
    max_x = np.max(x)
    if max_x < 1e-12:
        return 0.0, np.zeros_like(x)
    
    # Shift for numerical stability
    shifted = x - max_x
    # exp(beta * shifted) might overflow if beta is large and shifted is not small enough relative to max
    # But shifted <= 0, so exp(beta * shifted) <= 1. Safe.
    weights = np.exp(beta * shifted)
    wsum = np.sum(weights)
    
    if wsum == 0:
        return max_x, np.zeros_like(x)
        
    value = max_x + math.log(wsum) / beta
    grad = weights / wsum
    return value, grad

def _objective_and_grad(h, beta):
    """
    Compute the smooth objective value and its gradient.
    Objective: smooth_max(overlap) + penalty * (sum(h) - N/2)^2
    """
    n = len(h)
    h = np.clip(h, 0.0, 1.0)
    
    # Calculate overlap vector: o_i = h_i * (1 - h_{i+1})
    # Note: h_{N} is considered 0 for the shift at the end? 
    # The seed code uses: shifted[:-1] = h[1:], shifted[-1]=0.
    # So overlap[i] = h[i] * (1 - (h[i+1] if i < N-1 else 0))
    
    overlap = np.zeros(n)
    if n > 1:
        # h[i] * (1 - h[i+1]) for i in 0..N-2
        overlap[:-1] = h[:-1] * (1.0 - h[1:])
        # h[N-1] * (1 - 0) = h[N-1]
        overlap[-1] = h[-1]
    else:
        overlap[0] = h[0]
        
    val_smooth, grad_overlap = _smooth_max_and_grad(overlap, beta)
    
    # Gradient of overlap w.r.t h
    # o_i depends on h_i and h_{i+1}
    # d(o_i)/d(h_i) = 1 - h_{i+1}  (for i < N-1)
    # d(o_i)/d(h_{i+1}) = -h_i      (for i < N-1)
    # d(o_{N-1})/d(h_{N-1}) = 1
    
    grad_h = np.zeros(n)
    if n > 1:
        # Contribution from o_i to h_i
        grad_h[:-1] += grad_overlap[:-1] * (1.0 - h[1:])
        # Contribution from o_i to h_{i+1}
        grad_h[1:] += grad_overlap[:-1] * (-h[:-1])
        # Contribution from o_{N-1} to h_{N-1}
        grad_h[-1] += grad_overlap[-1] * 1.0
    else:
        grad_h[0] += grad_overlap[0] * 1.0
        
    # Penalty term for sum constraint
    penalty_weight = 100.0
    sum_error = h.sum() - TARGET_SUM
    penalty_val = penalty_weight * sum_error * sum_error
    penalty_grad = 2.0 * penalty_weight * sum_error
    
    total_val = val_smooth + penalty_val
    total_grad = grad_h + penalty_grad
    
    return total_val, total_grad

def _optimize_direct(h0, deadline, beta_start=50.0, beta_end=500.0, max_iter=500):
    """
    Optimize h using L-BFGS-B with analytic gradients and adaptive beta.
    """
    h = h0.copy()
    h = _fix_sum(h)
    current_score = _score(h)
    
    # We will run a few stages with increasing beta
    stages = [
        (beta_start, max_iter // 2),
        (beta_end // 2, max_iter // 4),
        (beta_end, max_iter // 4)
    ]
    
    for beta, iters in stages:
        if time.time() > deadline - 2.0:
            break
            
        def obj_func(x):
            val, _ = _objective_and_grad(x, beta)
            return val
        
        def grad_func(x):
            _, grad = _objective_and_grad(x, beta)
            return grad
        
        try:
            res = minimize(
                obj_func, 
                h, 
                jac=grad_func,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': iters, 'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0}
            )
            h_new = res.x
            h_new = _fix_sum(h_new)
            new_score = _score(h_new)
            if new_score < current_score:
                h = h_new
                current_score = new_score
        except Exception:
            pass
            
    return current_score, h

def _local_search_polish(h, max_iters=200, deadline=None):
    """
    Deterministic local search to polish the solution.
    Tries small adjustments and swaps.
    """
    h = h.copy()
    current_score = _score(h)
    
    if deadline is None:
        deadline = time.time() + 5.0
        
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        improved = False
        # Try small adjustments
        for i in range(N):
            for delta in [0.005, -0.005, 0.01, -0.01]:
                h_trial = h.copy()
                h_trial[i] += delta
                if h_trial[i] < 0 or h_trial[i] > 1:
                    continue
                h_trial = _fix_sum(h_trial)
                s = _score(h_trial)
                if s < current_score - 1e-12:
                    h = h_trial
                    current_score = s
                    improved = True
                    break
            if improved:
                break
        
        # Try swaps
        if not improved:
            for i in range(N - 1):
                h_trial = h.copy()
                h_trial[i], h_trial[i+1] = h_trial[i+1], h_trial[i]
                h_trial = _fix_sum(h_trial)
                s = _score(h_trial)
                if s < current_score - 1e-12:
                    h = h_trial
                    current_score = s
                    improved = True
                    break
        if not improved:
            break
            
    return current_score, h

def _generate_initial_guesses():
    """Generate diverse initial guesses."""
    guesses = []
    
    # 1. Sinusoidal
    for A in [0.2, 0.3, 0.4]:
        for phi in [0, np.pi/2, np.pi, 3*np.pi/2]:
            i = np.arange(N)
            h = 0.5 + A * np.sin(2.0 * np.pi * i / N + phi)
            h = np.clip(h, 0.0, 1.0)
            h = _fix_sum(h)
            guesses.append(h)
            
    # 2. Block patterns
    for k in [10, 20, 25, 30, 50]:
        for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
            h = np.zeros(N)
            n_high = int(k * frac)
            if n_high > 0:
                h[:n_high] = 1.0
            # Adjust sum
            h = _fix_sum(h)
            guesses.append(h)
            
    # 3. Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        h = rng.random(N)
        h = _fix_sum(h)
        guesses.append(h)
        
    return guesses

def construct():
    deadline = time.time() + (150.0 - DEADLINE_MARGIN)
    
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    guesses = _generate_initial_guesses()
    
    for guess in guesses:
        if time.time() > deadline - 5.0:
            break
        try:
            score, h = _optimize_direct(guess, deadline)
            if score < best_score:
                best_score = score
                best_h = h
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 1.0:
        score, h = _local_search_polish(best_h, deadline=deadline)
        if score < best_score:
            best_score = score
            best_h = h
            
    return best_h
