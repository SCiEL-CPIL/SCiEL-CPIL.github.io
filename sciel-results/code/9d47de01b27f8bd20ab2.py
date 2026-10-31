"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START

def _compute_overlap(h, N, dx):
    """Compute the maximum linear overlap of h with shifted (1-h)."""
    # h is defined on [0, 2] with N steps, each of width dx = 2/N
    # For linear (non-circular) shifts, we consider shifts where the overlap region is non-empty
    # Overlap at shift s: integral over x of h(x) * (1 - h(x+s)) where both are defined
    # We'll evaluate at discrete shifts
    
    max_overlap = 0.0
    # Shifts from 0 to 2 (linear, non-wrapping)
    # For shift s, x ranges from 0 to 2-s (so x+s is in [s, 2])
    # Discretize shifts
    n_shifts = N  # evaluate at N discrete shifts
    for k in range(n_shifts + 1):
        s = k * dx
        # x from 0 to 2-s
        # Discretize x
        n_x = max(1, int((2 - s) / dx))
        x_vals = np.linspace(0, 2 - s, n_x)
        # h(x) and h(x+s)
        # Use interpolation or direct index
        # h is defined at centers: (i + 0.5) * dx for i in range(N)
        # For x_vals, find corresponding h values
        idx_x = np.clip(np.floor(x_vals / dx).astype(int), 0, N - 1)
        idx_x_plus_s = np.clip(np.floor((x_vals + s) / dx).astype(int), 0, N - 1)
        h_x = h[idx_x]
        h_x_plus_s = h[idx_x_plus_s]
        # Overlap: h(x) * (1 - h(x+s))
        overlap_vals = h_x * (1.0 - h_x_plus_s)
        overlap = np.trapz(overlap_vals, x_vals)
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap


def _objective(params):
    """Objective function for optimization."""
    # params encodes a smooth function: use a combination of basis functions
    # h(x) = 0.5 + sum_i a_i * cos(i * pi * x / 2) for i in 1..K
    # This ensures h is smooth and we can clip to [0,1]
    K = len(params)
    x = np.linspace(0, 2, N)
    h = np.zeros(N)
    for i in range(K):
        h += params[i] * np.cos((i + 1) * np.pi * x / 2)
    h = 0.5 + h
    # Clip to [0, 1]
    h = np.clip(h, 0, 1)
    # Normalize to have integral 1 (sum = N/2 since dx = 2/N, integral = sum * dx = N/2 * 2/N = 1)
    # Actually: integral = sum(h_i) * dx = sum(h_i) * 2/N should equal 1
    # So sum(h_i) should be N/2
    current_sum = np.sum(h)
    if current_sum > 0:
        h = h * (N / 2.0) / current_sum
        h = np.clip(h, 0, 1)
    return _compute_overlap(h, N, 2.0 / N)


def _objective_and_grad(params):
    """Objective with numerical gradient for better optimization."""
    f0 = _objective(params)
    # Numerical gradient
    grad = np.zeros_like(params)
    eps = 1e-6
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _objective(params_plus)
        grad[i] = (f_plus - f0) / eps
    return f0, grad


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Use a parameterized smooth function with cosine basis
    K = 8  # number of cosine terms
    
    # Initial guess: start with a cosine wave that has been shown to work well
    # h(x) = 0.5 + a * cos(pi * x / 2) gives h(0)=0.5+a, h(2)=0.5-a
    # For a=0.3: h ranges from 0.2 to 0.8
    best_params = np.zeros(K)
    best_params[0] = 0.3  # First cosine term
    best_params[1] = -0.1  # Second term for asymmetry
    
    best_h = None
    best_score = float('inf')
    
    # Evaluate initial
    x = np.linspace(0, 2, N)
    h_init = 0.5 + best_params[0] * np.cos(np.pi * x / 2) + best_params[1] * np.cos(2 * np.pi * x / 2)
    h_init = np.clip(h_init, 0, 1)
    s = np.sum(h_init)
    if s > 0:
        h_init = h_init * (N / 2.0) / s
        h_init = np.clip(h_init, 0, 1)
    best_h = h_init.copy()
    best_score = _compute_overlap(h_init, N, 2.0 / N)
    
    # Try multiple optimization approaches
    attempts = 0
    while time.time() < deadline and attempts < 5:
        attempts += 1
        
        if time.time() > deadline - 20:
            break
        
        # Try different initial conditions
        if attempts == 1:
            x0 = best_params.copy()
        elif attempts == 2:
            x0 = np.zeros(K)
            x0[0] = 0.4
            x0[2] = 0.1
        elif attempts == 3:
            x0 = np.zeros(K)
            x0[0] = 0.2
            x0[1] = 0.15
            x0[3] = -0.05
        elif attempts == 4:
            x0 = np.zeros(K)
            x0[0] = 0.35
            x0[1] = -0.1
            x0[2] = 0.05
        else:
            x0 = np.random.uniform(-0.3, 0.3, K)
        
        # Use Nelder-Mead for robustness (no gradient needed)
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                params_opt = result.x
                x = np.linspace(0, 2, N)
                h_opt = 0.5
                for i in range(len(params_opt)):
                    h_opt += params_opt[i] * np.cos((i + 1) * np.pi * x / 2)
                h_opt = np.clip(h_opt, 0, 1)
                s = np.sum(h_opt)
                if s > 0:
                    h_opt = h_opt * (N / 2.0) / s
                    h_opt = np.clip(h_opt, 0, 1)
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final local refinement with BFGS if time permits
    if time.time() < deadline - 10 and best_h is not None:
        try:
            # Reconstruct params from best_h via least squares
            x = np.linspace(0, 2, N)
            A = np.zeros((N, K))
            for i in range(K):
                A[:, i] = np.cos((i + 1) * np.pi * x / 2)
            target = best_h - 0.5
            params_refined, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
            
            result2 = minimize(
                _objective,
                params_refined,
                method='BFGS',
                options={'maxiter': 200, 'gtol': 1e-6}
            )
            
            if result2.fun < best_score:
                best_score = result2.fun
                params_opt = result2.x
                h_opt = 0.5
                for i in range(len(params_opt)):
                    h_opt += params_opt[i] * np.cos((i + 1) * np.pi * x / 2)
                h_opt = np.clip(h_opt, 0, 1)
                s = np.sum(h_opt)
                if s > 0:
                    h_opt = h_opt * (N / 2.0) / s
                    h_opt = np.clip(h_opt, 0, 1)
                best_h = h_opt.copy()
        except Exception:
            pass
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    # Ensure exact sum = N/2
    # Adjust to make sum exactly N/2 while staying in [0,1]
    target_sum = N / 2.0
    current_sum = np.sum(best_h)
    diff = target_sum - current_sum
    
    if abs(diff) > 1e-10:
        # Distribute adjustment evenly, respecting bounds
        for _ in range(100):
            new_h = best_h.copy()
            # Add diff/N to each element
            new_h += diff / N
            # Clip
            new_h = np.clip(new_h, 0, 1)
            new_sum = np.sum(new_h)
            if abs(new_sum - target_sum) < 1e-10:
                best_h = new_h
                break
            # If we hit bounds, try to fix by adjusting free elements
            free_mask = (new_h > 1e-12) & (new_h < 1 - 1e-12)
            if np.any(free_mask):
                remaining = target_sum - new_sum
                n_free = np.sum(free_mask)
                new_h[free_mask] += remaining / n_free
                new_h = np.clip(new_h, 0, 1)
                best_h = new_h
                if abs(np.sum(best_h) - target_sum) < 1e-10:
                    break
            else:
                break
        else:
            # Final fix: adjust one element
            if np.sum(best_h) < target_sum:
                # Increase an element that has room
                for i in range(N):
                    room = 1.0 - best_h[i]
                    if room > 1e-12:
                        add = min(room, target_sum - np.sum(best_h))
                        best_h[i] += add
                        if abs(np.sum(best_h) - target_sum) < 1e-10:
                            break
            else:
                for i in range(N):
                    if best_h[i] > 1e-12:
                        sub = min(best_h[i], np.sum(best_h) - target_sum)
                        best_h[i] -= sub
                        if abs(np.sum(best_h) - target_sum) < 1e-10:
                            break
    
    return best_h

# EVOLVE-BLOCK-END
