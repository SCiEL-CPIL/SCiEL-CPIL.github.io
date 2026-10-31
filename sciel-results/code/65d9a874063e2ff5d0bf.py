"""Erdos minimum overlap - optimize step heights to minimize max overlap.

Strategy:
- Exploit translational symmetry: restrict to periodic functions with period 1 on [0,2].
  This means h[i] = h[i + N/2] for i < N/2, so we only optimize N/2 free variables.
- The overlap o(s) = integral of h(x)(1-h(x+s)) dx for s in [0,1] (by periodicity, 
  we only need s in [0,1] since the function is 1-periodic and the domain is [0,2]).
- Use a smooth max (log-sum-exp) surrogate to make the objective differentiable.
- Use L-BFGS-B with the sum constraint handled by reparameterization:
  h[i] = sigmoid(z[i]) for i < N/2, then h[i + N/2] = h[i].
  The sum constraint becomes: sum(sigmoid(z)) = N/4 (since total sum = N/2 and half are copies).
  Actually, let me re-think: if h is 1-periodic on [0,2], then h[x] = h[x+1] for x in [0,1].
  So h[i] for i in [0, N/2) determines h[i + N/2] = h[i] for i in [0, N/2).
  The integral constraint: sum_{i=0}^{N-1} h[i] * (2/N) = 1 => sum_{i=0}^{N-1} h[i] = N/2.
  With periodicity: 2 * sum_{i=0}^{N/2-1} h[i] = N/2 => sum_{i=0}^{N/2-1} h[i] = N/4.
  
- We optimize z in R^{N/2} where h[i] = sigmoid(z[i]) for i < N/2.
- The overlap for shift s (in [0,1]) is:
  o(s) = integral_0^2 h(x)(1-h(x+s)) dx
  By periodicity, this equals 2 * integral_0^1 h(x)(1-h(x+s)) dx where h is extended periodically.
  For step functions, we compute this discretely.

- Use scipy.optimize.minimize with L-BFGS-B on the smooth max of o(s) over s in [0,1].
- Multiple random restarts.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import time

N = 100  # Total number of steps on [0, 2]
M = N // 2  # Number of free variables (periodic with period 1)

def _smooth_max(vals, temperature=0.05):
    """Compute a smooth approximation to max(vals) using log-sum-exp."""
    if len(vals) == 0:
        return 0.0
    m = np.max(vals)
    if m < -1e10:
        return 0.0
    return m + temperature * np.log(np.sum(np.exp((vals - m) / temperature)))

def compute_overlap_max_periodic(h_free, n_eval=300):
    """Compute the maximum overlap for a 1-periodic step function.
    
    h_free has M entries, defining h[i] for i in [0, M).
    The full h on [0,2] is h_full[i] = h_free[i % M].
    """
    h_full = np.tile(h_free, 2)  # Length N
    step_w = 2.0 / N
    
    # By periodicity, we only need to evaluate shifts s in [0, 1]
    # (shifts outside this range give the same overlap due to periodicity and the [0,2] domain)
    shifts = np.linspace(0.0, 1.0, n_eval, endpoint=False)
    
    max_ov = 0.0
    for s in shifts:
        # o(s) = integral_0^2 h(x)(1-h(x+s)) dx
        # For step functions, approximate by summing over x-steps
        total = 0.0
        for i in range(N):
            x0 = 2.0 * i / N
            x1 = 2.0 * (i + 1) / N
            x_mid = (x0 + x1) / 2.0
            xs = x_mid + s
            # Wrap xs to [0, 2)
            xs_wrapped = xs % 2.0
            # Find which step xs_wrapped falls in
            j = int(xs_wrapped / step_w)
            if j >= N:
                j = N - 1
            total += h_full[i] * (1.0 - h_full[j]) * step_w
        ov = total
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _overlap_values_periodic(h_free, n_eval=300):
    """Return the overlap values for all shifts (for smooth max)."""
    h_full = np.tile(h_free, 2)
    step_w = 2.0 / N
    shifts = np.linspace(0.0, 1.0, n_eval, endpoint=False)
    ovals = np.zeros(n_eval)
    for idx, s in enumerate(shifts):
        total = 0.0
        for i in range(N):
            x_mid = (2.0 * i / N + 2.0 * (i + 1) / N) / 2.0
            xs = (x_mid + s) % 2.0
            j = int(xs / step_w)
            if j >= N:
                j = N - 1
            total += h_full[i] * (1.0 - h_full[j]) * step_w
        ovals[idx] = total
    return ovals

def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin
    
    best_h = np.full(N, 0.5)
    best_score = compute_overlap_max_periodic(np.full(M, 0.5))
    
    np.random.seed(12345)
    
    # Optimization: minimize smooth max of overlap, with constraint sum(h_free) = N/4
    # Reparameterize: h_free[i] = sigmoid(z[i])
    # Constraint: sum(sigmoid(z)) = N/4 = 25
    
    n_eval = 200  # Use fewer eval points during optimization for speed
    
    def objective(z):
        h_free = expit(z)
        # Enforce sum constraint via penalty
        s = np.sum(h_free)
        target = N / 4.0
        penalty = 1000.0 * (s - target) ** 2
        ovals = _overlap_values_periodic(h_free, n_eval)
        return _smooth_max(ovals, 0.03) + penalty
    
    def objective_grad(z):
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(z)
        f0 = objective(z)
        for i in range(len(z)):
            z_plus = z.copy()
            z_plus[i] += eps
            grad[i] = (objective(z_plus) - f0) / eps
        return grad
    
    # Try multiple random initializations
    max_attempts = 30
    for attempt in range(max_attempts):
        if time.time() > deadline - 15:
            break
        
        if attempt == 0:
            # Start from uniform: sigmoid(z) = 0.5 => z = 0
            z_init = np.zeros(M)
        else:
            # Random initialization
            z_init = np.random.randn(M) * 2.0
            # Scale to roughly satisfy constraint
            h_test = expit(z_init)
            scale = (N / 4.0) / np.sum(h_test)
            # Adjust z so that sum(sigmoid(z)) ≈ N/4
            # This is approximate; the optimizer will fix it
            z_init = z_init * 1.0  # Keep as is, let optimizer handle
        
        try:
            result = minimize(
                objective, z_init, method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-5, 'ftol': 1e-8}
            )
        except Exception:
            continue
        
        h_free_opt = expit(result.x)
        # Fix sum constraint
        diff = N / 4.0 - np.sum(h_free_opt)
        if abs(diff) > 1e-10:
            # Distribute adjustment
            h_free_opt = h_free_opt + diff / M
            h_free_opt = np.clip(h_free_opt, 1e-6, 1 - 1e-6)
            # Re-normalize
            h_free_opt = h_free_opt / np.sum(h_free_opt) * (N / 4.0)
            h_free_opt = np.clip(h_free_opt, 1e-6, 1 - 1e-6)
        
        # Build full h
        h_full = np.tile(h_free_opt, 2)
        
        # Final score with higher resolution
        score = compute_overlap_max_periodic(h_free_opt, n_eval=400)
        
        if score < best_score:
            best_score = score
            best_h = h_full.copy()
    
    # Also try a non-periodic refinement on the best periodic solution
    # Use the best periodic h as initial guess for a non-periodic optimization
    if time.time() < deadline - 20:
        # Refine without periodicity constraint
        h_init = best_h.copy()
        
        def obj_nonperiodic(h_flat):
            h = np.clip(h_flat, 1e-8, 1 - 1e-8)
            # Sum constraint penalty
            s = np.sum(h)
            penalty = 1000.0 * (s - N / 2.0) ** 2
            # Compute overlap
            step_w = 2.0 / N
            n_eval_local = 150
            shifts = np.linspace(0.0, 1.0, n_eval_local, endpoint=False)
            ovals = np.zeros(n_eval_local)
            for idx, s in enumerate(shifts):
                total = 0.0
                for i in range(N):
                    x_mid = (2.0 * i / N + 2.0 * (i + 1) / N) / 2.0
                    xs = (x_mid + s) % 2.0
                    j = int(xs / step_w)
                    if j >= N:
                        j = N - 1
                    total += h[i] * (1.0 - h[j]) * step_w
                ovals[idx] = total
            return _smooth_max(ovals, 0.03) + penalty
        
        try:
            result = minimize(
                obj_nonperiodic, h_init, method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-5}
            )
            h_opt = np.clip(result.x, 1e-8, 1 - 1e-8)
            # Fix sum
            diff = N / 2.0 - np.sum(h_opt)
            h_opt += diff / N
            h_opt = np.clip(h_opt, 1e-8, 1 - 1e-8)
            
            # Compute final score
            step_w = 2.0 / N
            n_eval_final = 400
            shifts = np.linspace(0.0, 1.0, n_eval_final, endpoint=False)
            max_ov = 0.0
            for s in shifts:
                total = 0.0
                for i in range(N):
                    x_mid = (2.0 * i / N + 2.0 * (i + 1) / N) / 2.0
                    xs = (x_mid + s) % 2.0
                    j = int(xs / step_w)
                    if j >= N:
                        j = N - 1
                    total += h_opt[i] * (1.0 - h_opt[j]) * step_w
                if total > max_ov:
                    max_ov = total
            
            if max_ov < best_score:
                best_score = max_ov
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Ensure exact sum constraint
    diff = N / 2.0 - np.sum(best_h)
    best_h += diff / N
    best_h = np.clip(best_h, 0.0, 1.0)
    # Final tiny adjustment for exactness
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        idx_max = np.argmax(best_h) if diff < 0 else np.argmin(best_h)
        best_h[idx_max] += diff
    
    return best_h
