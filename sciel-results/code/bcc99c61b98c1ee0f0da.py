import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 600
X_GRID = np.linspace(-0.25, 0.25, N, endpoint=False)

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    total = np.sum(h)
    if total < 1e-15:
        return 1e10
    # Use FFT for speed
    n_fft = 2 * N
    f_hat = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(f_hat * f_hat, n=n_fft)
    return np.max(conv) / (total * total)

def soft_max_objective(h, T=0.01):
    """Smooth surrogate for max autoconvolution using log-sum-exp."""
    h = np.asarray(h, dtype=float)
    total = np.sum(h)
    if total < 1e-15:
        return 1e10
    n_fft = 2 * N
    f_hat = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(f_hat * f_hat, n=n_fft)
    conv_max = np.max(conv)
    shifted = (conv - conv_max) / T
    shifted = np.clip(shifted, -100, 100)
    soft_max = conv_max + T * np.log(np.sum(np.exp(shifted)))
    return soft_max / (total * total)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Initial guess: uniform
    best_h = np.ones(N)
    best_score = compute_autoconv_max(best_h)

    # --- Phase 1: Gaussian Mixture Representation + Differential Evolution ---
    # Represent h(x) as sum of K Gaussian bumps. This reduces dimensionality
    # and exploits the smooth structure of the optimal solution.
    
    def gauss_params_to_h(params, K):
        """Convert [amp_1, center_1, width_1, ..., amp_K, center_K, width_K] to heights."""
        h = np.zeros(N)
        for i in range(K):
            amp = params[3*i]
            center = params[3*i+1]
            width = np.exp(params[3*i+2])  # Ensure positive width
            h += amp * np.exp(-0.5 * ((X_GRID - center) / width) ** 2)
        return np.maximum(h, 0)

    def gauss_objective(params, K):
        h = gauss_params_to_h(params, K)
        return compute_autoconv_max(h)

    # Try different numbers of Gaussian components
    for K in [3, 5, 8, 12]:
        if time.time() > deadline - 30:
            break
        
        n_params = 3 * K
        # Bounds: amplitudes in [0, 5], centers in [-0.25, 0.25], log-widths in [-5, 2]
        bounds = []
        for _ in range(K):
            bounds.append((0.0, 5.0))  # amplitude
            bounds.append((-0.25, 0.25))  # center
            bounds.append((-5.0, 2.0))  # log-width

        try:
            result = differential_evolution(
                lambda p: gauss_objective(p, K),
                bounds=bounds,
                maxiter=30,
                popsize=15,
                tol=1e-5,
                seed=42 + K,
                polish=False,
                init='latinhypercube'
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_h = gauss_params_to_h(result.x, K)
        except Exception:
            pass

    # --- Phase 2: Full-dimension L-BFGS-B with Soft-Max Surrogate ---
    # Now that we have a good initial guess from Phase 1, refine in the full
    # 600-dim space using a smooth surrogate to navigate the non-smooth max.
    
    def soft_obj(x, T):
        h = np.maximum(x, 0)
        return soft_max_objective(h, T)

    # Stage 2a: Coarse refinement with higher temperature
    T1 = 0.05
    try:
        res1 = minimize(
            lambda x: soft_obj(x, T1),
            best_h,
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        h1 = np.maximum(res1.x, 0)
        s1 = compute_autoconv_max(h1)
        if s1 < best_score:
            best_score = s1
            best_h = h1
    except Exception:
        pass

    # Stage 2b: Fine refinement with lower temperature
    if time.time() < deadline - 20:
        T2 = 0.005
        try:
            res2 = minimize(
                lambda x: soft_obj(x, T2),
                best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h2 = np.maximum(res2.x, 0)
            s2 = compute_autoconv_max(h2)
            if s2 < best_score:
                best_score = s2
                best_h = h2
        except Exception:
            pass

    # --- Phase 3: Water-Filling / Remez-like Iterative Refinement ---
    # The optimal condition for minimizing max autoconvolution is that the
    # convolution is flat where h > 0. We can use this to guide updates.
    # This is a heuristic local search that adjusts h to flatten the conv.
    
    if time.time() < deadline - 10:
        h = best_h.copy()
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            
            # Compute convolution
            n_fft = 2 * N
            f_hat = np.fft.rfft(h, n=n_fft)
            conv = np.fft.irfft(f_hat * f_hat, n=n_fft)
            total = np.sum(h)
            conv_norm = conv / (total * total)
            
            # Find max and min of conv where h > 0
            mask = h > 1e-8
            if not np.any(mask):
                break
            
            conv_max = np.max(conv_norm)
            conv_min = np.min(conv_norm[mask])
            
            if conv_max - conv_min < 1e-8:
                break
            
            # Gradient-like step: increase h where conv is low, decrease where high
            # This is a simplified water-filling step
            step = 0.01
            h_new = h.copy()
            h_new[mask] += step * (conv_min - conv_norm[mask])
            h_new = np.maximum(h_new, 0)
            
            # Normalize to keep sum constant (optional, but helps stability)
            if np.sum(h_new) > 0:
                h_new = h_new * (total / np.sum(h_new))
            
            s_new = compute_autoconv_max(h_new)
            if s_new < best_score:
                best_score = s_new
                best_h = h_new
                h = h_new
            else:
                # If it got worse, reduce step size or break
                step *= 0.5
                if step < 1e-6:
                    break

    # --- Phase 4: Final Polish with Exact Objective ---
    if time.time() < deadline - 5:
        try:
            res_final = minimize(
                lambda x: compute_autoconv_max(np.maximum(x, 0)),
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 50, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            h_final = np.maximum(res_final.x, 0)
            s_final = compute_autoconv_max(h_final)
            if s_final < best_score:
                best_score = s_final
                best_h = h_final
        except Exception:
            pass

    return best_h.tolist()
