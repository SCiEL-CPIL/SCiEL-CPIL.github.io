import numpy as np
import time
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 140.0  # Safety margin for 150s limit

def _max_overlap(h):
    """Calculate the exact maximum overlap for a given sequence h."""
    # Overlap[k] = sum_i h[i] * (1 - h[(i-k)%N])
    # = sum_i h[i] - sum_i h[i]h[i-k]
    # sum_i h[i] is fixed at N/2 if feasible.
    # We compute all shifts using FFT for speed, though N=100 is small.
    # h * (1 - shift(h))
    # Let's just use the direct vectorized approach for clarity and correctness first.
    # If N is larger, FFT is better. For N=100, direct is fine.
    max_ov = 0.0
    for k in range(N):
        shifted = np.roll(h, k)
        ov = np.sum(h * (1.0 - shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _logsumexp_overlap(h, beta=50.0):
    """
    Smooth approximation of max overlap using LogSumExp.
    This makes the objective differentiable everywhere, suitable for L-BFGS-B.
    """
    overlaps = np.zeros(N)
    for k in range(N):
        shifted = np.roll(h, k)
        overlaps[k] = np.sum(h * (1.0 - shifted))
    
    # LogSumExp: (1/beta) * log(sum(exp(beta * x)))
    # To avoid overflow, subtract max
    m = np.max(overlaps)
    if m == -np.inf:
        return 0.0
    return (1.0 / beta) * np.log(np.sum(np.exp(beta * (overlaps - m)))) + m

def _project_to_feasible(h):
    """
    Project h to the feasible set: sum(h) = N/2, 0 <= h[i] <= 1.
    Simple projection: adjust mean, then clip. 
    This is not a true Euclidean projection but a practical heuristic.
    """
    h = np.clip(h, 0.0, 1.0)
    current_sum = np.sum(h)
    target_sum = N / 2.0
    diff = target_sum - current_sum
    h = h + diff / N
    # Final clip to ensure bounds, then re-adjust sum slightly if needed
    h = np.clip(h, 0.0, 1.0)
    # If clipping broke the sum significantly, distribute the error to elements not at bounds
    current_sum = np.sum(h)
    diff = target_sum - current_sum
    if abs(diff) > 1e-9:
        # Identify indices not at 0 or 1
        mask = (h > 1e-5) & (h < 1.0 - 1e-5)
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    start_time = time.time()
    
    # 1. Baseline: Uniform
    h_uniform = np.full(N, 0.5)
    best_h = h_uniform.copy()
    best_score = _max_overlap(best_h)
    
    # 2. Multi-restart L-BFGS-B with LogSumExp smoothing
    # We use a high beta for the smoothing to approximate max closely,
    # but not so high that it causes numerical instability.
    beta = 50.0
    
    # Generate diverse initial guesses
    inits = []
    
    # a) Uniform (already handled)
    # b) Sine wave
    x = np.arange(N)
    h_sine = 0.5 + 0.4 * np.sin(2 * np.pi * x / N)
    inits.append(_project_to_feasible(h_sine))
    
    # c) Cosine wave
    h_cos = 0.5 + 0.4 * np.cos(2 * np.pi * x / N)
    inits.append(_project_to_feasible(h_cos))
    
    # d) Random
    np.random.seed(42)
    h_rand = np.random.rand(N)
    inits.append(_project_to_feasible(h_rand))
    
    # e) Block structure (e.g., 0.8 and 0.2 alternating)
    h_block = np.where(x % 2 == 0, 0.8, 0.2)
    inits.append(_project_to_feasible(h_block))
    
    # f) Another random
    np.random.seed(123)
    h_rand2 = np.random.rand(N)
    inits.append(_project_to_feasible(h_rand2))

    bounds = [(0, 1)] * N
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2}]

    def objective(x):
        return _logsumexp_overlap(x, beta)

    def gradient(x):
        # L-BFGS-B can approximate gradient, but providing it is faster.
        # However, implementing the exact gradient of LogSumExp of cyclic autocorrelation
        # is complex. We will let L-BFGS-B use finite differences or numerical approx.
        # Actually, scipy's L-BFGS-B allows approx_grad=True.
        return None # Let scipy handle it or use finite diff

    # To speed up, we can provide the gradient of the sum of overlaps, but LogSumExp needs specific handling.
    # For N=100, finite difference gradient is acceptable if we limit iterations.
    
    for h0 in inits:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        try:
            # Use L-BFGS-B with maxiter limit
            res = minimize(
                objective, 
                h0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if res.success or res.fun < best_score:
                # Check if the result is close to feasible
                h_candidate = _project_to_feasible(res.x)
                score_candidate = _max_overlap(h_candidate)
                
                if score_candidate < best_score:
                    best_score = score_candidate
                    best_h = h_candidate
        except Exception:
            continue

    # 3. Final refinement with SLSQP on the best found so far, if time permits
    if time.time() - start_time < TIME_LIMIT - 5:
        try:
            # Use the exact max overlap for the final polish, but SLSQP struggles with non-smoothness.
            # We stick to the L-BFGS-B result as it's robust.
            # If we want to try SLSQP, we do it on the best_h.
            # But SLSQP is slow. We will just return the best L-BFGS-B result.
            pass
        except Exception:
            pass

    # Ensure final output is strictly feasible
    best_h = _project_to_feasible(best_h)
    
    return best_h
