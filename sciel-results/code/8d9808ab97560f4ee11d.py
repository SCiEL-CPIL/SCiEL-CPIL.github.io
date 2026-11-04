"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

# Precompute the correlation structure
# h is a vector of N values in [0,1] summing to N/2.
# Overlap at shift k is sum_{i} h[i] * (1 - h[(i+k) % N])
# We want to minimize the maximum overlap over all shifts k.

# Let's define the overlap function more carefully.
# The problem states: "minimize the maximum overlap of h with shifted 1-h"
# Overlap at shift k: O_k = sum_{i=0}^{N-1} h[i] * (1 - h[(i+k) % N])
# Note that h[(i+k)%N] is the shifted version.
# Since h[i] is in [0,1], 1-h is also in [0,1].

# We can rewrite O_k = sum h[i] - sum h[i] h[(i+k)%N] = N/2 - C_k
# where C_k = sum h[i] h[(i+k)%N] is the autocorrelation of h at lag k.
# Minimizing max O_k is equivalent to minimizing max (N/2 - C_k).
# Since N/2 is constant, this is equivalent to maximizing min C_k.
# Or, to keep the minimization form, we can minimize max (N/2 - C_k).

# Let's use the epigraph formulation:
# Minimize t
# Subject to: N/2 - C_k <= t for all k
#             0 <= h_i <= 1
#             sum h_i = N/2

# We can use scipy.optimize.minimize with SLSQP or trust-constr.
# To handle the max, we use the epigraph variable t.

def construct():
    start_time = time.time()
    deadline = 140  # Leave 10s margin
    
    # Variables: h (N) and t (1)
    # Objective: t
    # Constraints:
    #   g_k(h, t) = N/2 - C_k(h) - t <= 0 for k=0..N-1
    #   h_i >= 0, h_i <= 1
    #   sum(h) = N/2
    
    def objective(x):
        t = x[N]
        return t
    
    def constraints(x):
        h = x[:N]
        t = x[N]
        # Autocorrelation
        # C_k = sum h[i] h[(i+k)%N]
        # We can compute all C_k via FFT for speed, but N=100 is small.
        # Let's just compute directly or via np.roll.
        cons = []
        for k in range(N):
            shifted_h = np.roll(h, -k)
            C_k = np.dot(h, shifted_h)
            # Constraint: N/2 - C_k - t <= 0
            cons.append({'type': 'ineq', 'fun': lambda x, k=k: x[N] + np.dot(x[:N], np.roll(x[:N], -k)) - N/2})
        return cons
    
    # This lambda inside constraint is slow because it's re-evaluated.
    # Let's try a different approach: smooth max.
    # f(x) = logsumexp(beta * (N/2 - C_k)) / beta
    # This is a smooth approximation of max(N/2 - C_k).
    
    def smooth_max_overlap(x):
        h = x[:N]
        # Compute all C_k
        # Using FFT: autocorrelation
        # But for N=100, direct is fine.
        c_vals = []
        for k in range(N):
            shifted_h = np.roll(h, -k)
            c_vals.append(np.dot(h, shifted_h))
        c_vals = np.array(c_vals)
        overlaps = N/2 - c_vals
        beta = 10.0
        # logsumexp
        m = np.max(overlaps)
        if m == -np.inf:
            return 0
        lse = m + np.log(np.sum(np.exp(overlaps - m)))
        return lse / beta
    
    def constraint_sum(x):
        h = x[:N]
        return np.sum(h) - N/2
    
    # Initial guess: 0.5 everywhere
    h0 = np.full(N, 0.5)
    t0 = N/2 - np.dot(h0, h0)  # max overlap at start
    x0 = np.concatenate([h0, [t0]])
    
    bounds = [(0, 1)] * N + [(0, N/2)]
    
    constraints_list = [
        {'type': 'eq', 'fun': constraint_sum}
    ]
    
    # Try to optimize
    best_x = x0
    best_val = smooth_max_overlap(x0)
    
    try:
        res = minimize(
            smooth_max_overlap,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success and res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        # Return best so far
        h_best = best_x[:N]
        # Enforce constraints strictly
        h_best = np.clip(h_best, 0, 1)
        # Adjust to sum exactly N/2
        diff = N/2 - np.sum(h_best)
        if abs(diff) > 1e-10:
            # Distribute difference
            h_best += diff / N
            h_best = np.clip(h_best, 0, 1)
            # Re-adjust
            diff2 = N/2 - np.sum(h_best)
            h_best += diff2 / N
            h_best = np.clip(h_best, 0, 1)
        
        return h_best
    
    # If time permits, do a more precise epigraph optimization
    # to get the exact max overlap, not the smooth approx.
    # And try to push it lower.
    
    # Let's try SLSQP with exact max (using epigraph) on the result
    # But SLSQP with N constraints might be slow.
    
    # Alternative: Projected Gradient Descent on the smooth objective
    # with a high beta for the last part.
    
    # Let's try one more optimization with higher beta
    beta_high = 50.0
    def smooth_max_overlap_high(x):
        h = x[:N]
        c_vals = np.array([np.dot(h, np.roll(h, -k)) for k in range(N)])
        overlaps = N/2 - c_vals
        m = np.max(overlaps)
        lse = m + np.log(np.sum(np.exp(overlaps - m)))
        return lse / beta_high
    
    try:
        res2 = minimize(
            smooth_max_overlap_high,
            best_x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        if res2.success and res2.fun < best_val:
            best_x = res2.x
            best_val = res2.fun
    except Exception:
        pass
    
    if time.time() > deadline:
        h_best = best_x[:N]
        h_best = np.clip(h_best, 0, 1)
        diff = N/2 - np.sum(h_best)
        if abs(diff) > 1e-10:
            h_best += diff / N
            h_best = np.clip(h_best, 0, 1)
            diff2 = N/2 - np.sum(h_best)
            h_best += diff2 / N
            h_best = np.clip(h_best, 0, 1)
        return h_best
    
    # Final polish: ensure exact sum
    h_best = best_x[:N]
    h_best = np.clip(h_best, 0, 1)
    diff = N/2 - np.sum(h_best)
    if abs(diff) > 1e-10:
        h_best += diff / N
        h_best = np.clip(h_best, 0, 1)
        diff2 = N/2 - np.sum(h_best)
        h_best += diff2 / N
        h_best = np.clip(h_best, 0, 1)
    
    return h_best
# EVOLVE-BLOCK-END
