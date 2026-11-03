"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DELTA = 1.0 / N  # width of each interval

def compute_ratio(h):
    """
    h: array of N nonnegative heights
    Returns the ratio ||f||_2^2 / (||f||_1 * ||f*f||_infty)
    where f is the step function with height h[i] on interval i.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return 0.0
    
    l2_sq = DELTA * np.sum(h**2)
    l1 = DELTA * np.sum(h)
    
    if l1 <= 0:
        return 0.0
    
    # Compute autoconvolution f*f
    # f is defined on [-1/4, 1/4] with N intervals
    # The autoconvolution f*f is defined on [-1/2, 1/2]
    # We need to compute (f*f)(x) = integral f(y) f(x-y) dy
    # For step functions, this can be computed exactly
    
    # The intervals are from -1/4 to 1/4, each of width DELTA
    # Interval i covers [x_i, x_i + DELTA] where x_i = -1/4 + i*DELTA
    
    # For the autoconvolution, we can use the fact that for step functions,
    # the autoconvolution is piecewise linear (trapezoidal-like)
    
    # A simpler approach: discretize the autoconvolution on a fine grid
    # and take the max
    
    # The support of f is [-1/4, 1/4], so f*f has support [-1/2, 1/2]
    # Let's compute on a fine grid
    
    # Actually, let's compute the autoconvolution more carefully.
    # f*f at point t = sum_i h_i * (measure of {y: y in interval i AND t-y in interval j for some j})
    # This is complex. Let's use a numerical approach.
    
    # Create a fine grid for evaluation
    n_fine = 1000
    t_grid = np.linspace(-0.5, 0.5, n_fine)
    
    # For each t, compute (f*f)(t) = integral f(y) f(t-y) dy
    # f(y) = h_i if y in interval i
    # We can compute this by summing over pairs of intervals
    
    # Interval i: [a_i, b_i] where a_i = -0.25 + i*DELTA, b_i = a_i + DELTA
    a = -0.25 + np.arange(N) * DELTA
    b = a + DELTA
    
    # For each t, (f*f)(t) = sum_i sum_j h_i h_j * length of intersection of [a_i, b_i] and [t-b_j, t-a_j]
    # This is O(N^2 * n_fine) which is 50*50*1000 = 2.5M, acceptable
    
    ff = np.zeros(n_fine)
    for i in range(N):
        for j in range(N):
            if h[i] == 0 or h[j] == 0:
                continue
            # Intersection of [a_i, b_i] and [t - b_j, t - a_j]
            # For each t, the length of intersection
            # Let's vectorize over t
            left = np.maximum(a[i], t_grid - b[j])
            right = np.minimum(b[i], t_grid - a[j])
            length = np.maximum(0, right - left)
            ff += h[i] * h[j] * length
    
    ff_max = np.max(ff)
    
    if ff_max <= 0:
        return 0.0
    
    return l2_sq / (l1 * ff_max)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Start with uniform
    h0 = np.ones(N)
    best_h = h0.copy()
    best_score = compute_ratio(h0)
    
    # Try some heuristic initializations
    # 1. Uniform (already done)
    # 2. Gaussian-like
    x = np.linspace(-0.25, 0.25, N)
    h_gauss = np.exp(-10 * x**2)
    h_gauss /= np.sum(h_gauss) * DELTA  # normalize so ||f||_1 = 1
    score_gauss = compute_ratio(h_gauss)
    if score_gauss > best_score:
        best_score = score_gauss
        best_h = h_gauss.copy()
    
    # 3. Try a few random initializations
    np.random.seed(42)
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        h_rand = np.random.exponential(1, N)
        h_rand /= np.sum(h_rand) * DELTA
        score_rand = compute_ratio(h_rand)
        if score_rand > best_score:
            best_score = score_rand
            best_h = h_rand.copy()
    
    # Use scipy optimization to refine
    # We want to maximize the ratio, which is equivalent to minimizing the negative
    # The ratio is scale-invariant in a certain way, so let's fix ||f||_1 = 1
    # i.e., sum(h) * DELTA = 1, so sum(h) = N = 50
    
    def objective(h):
        if time.time() > deadline - 2:
            return 0  # penalty
        h = np.maximum(0, h)
        s = np.sum(h)
        if s <= 0:
            return 0
        # Normalize to ||f||_1 = 1
        h_norm = h / (s * DELTA)
        return -compute_ratio(h_norm)
    
    # Try optimization from best_h
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_h, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            h_opt = np.maximum(0, res.x)
            s = np.sum(h_opt)
            if s > 0:
                h_opt_norm = h_opt / (s * DELTA)
                score_opt = compute_ratio(h_opt_norm)
                if score_opt > best_score:
                    best_score = score_opt
                    best_h = h_opt_norm.copy()
        except Exception:
            pass
    
    # Final polish: try a few more local optimizations
    if time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            try:
                h_init = best_h.copy()
                # Add small perturbation
                h_init = h_init + np.random.normal(0, 0.01, N)
                h_init = np.maximum(0, h_init)
                res = minimize(objective, h_init, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
                h_opt = np.maximum(0, res.x)
                s = np.sum(h_opt)
                if s > 0:
                    h_opt_norm = h_opt / (s * DELTA)
                    score_opt = compute_ratio(h_opt_norm)
                    if score_opt > best_score:
                        best_score = score_opt
                        best_h = h_opt_norm.copy()
            except Exception:
                pass
    
    # Return the best heights found
    # The task says "sequence of nonnegative step heights"
    # We should return heights that give the best ratio
    # Note: the ratio depends on the actual heights, not just the shape
    # But we normalized to ||f||_1 = 1. Let's return the normalized version.
    return list(best_h)
# EVOLVE-BLOCK-END
