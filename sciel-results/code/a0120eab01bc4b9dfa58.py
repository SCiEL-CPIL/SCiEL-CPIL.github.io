import numpy as np
import time
import random
from scipy.optimize import minimize, differential_evolution

N = 600
DEADLINE = 145

def construct():
    start_time = time.time()

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def compute_autoconv_max(h):
        """Compute max of autoconvolution of h (assumed sum=1)."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return np.max(conv)

    i = np.arange(N, dtype=float)
    center = (N - 1) / 2.0

    # --- GMM (Gaussian Mixture Model) parameterization ---
    # Represent h as a mixture of K Gaussians
    K = 5  # Increased from 4 to 5 for more flexibility

    def gmm_to_h(params):
        """Convert GMM parameters to discrete heights on N intervals."""
        h = np.zeros(N)
        for k in range(K):
            w_k = max(params[3*k], 0.01)
            mu_k = params[3*k+1]
            sigma_k = max(params[3*k+2], 1.0)
            h += w_k * np.exp(-((i - mu_k)**2) / (2 * sigma_k**2))
        h = np.maximum(h, 1e-10)
        h /= np.sum(h)
        return h

    def gmm_objective(params):
        """Objective: max autoconvolution of GMM-generated h."""
        h = gmm_to_h(params)
        return compute_autoconv_max(h)

    # DE bounds for GMM parameters
    bounds = []
    for k in range(K):
        bounds.append((0.1, 5.0))   # weight
        bounds.append((0, N-1))      # mean
        bounds.append((5.0, N/3.0))  # sigma

    # Initial best: uniform
    best_h = np.ones(N) / N
    best_val = compute_autoconv_max(best_h)

    # Add some structured initializations to best_h candidates
    # 1. Sinc-like (wide Gaussian)
    h_sinc = np.exp(-((i - center)**2) / (2 * (N/4)**2))
    h_sinc = np.maximum(h_sinc, 0)
    h_sinc /= np.sum(h_sinc)
    val_sinc = compute_autoconv_max(h_sinc)
    if val_sinc < best_val:
        best_val = val_sinc
        best_h = h_sinc.copy()

    # 2. Raised Cosine
    x = np.linspace(-1, 1, N)
    h_rc = 0.5 * (1 + np.cos(np.pi * x))
    h_rc = np.maximum(h_rc, 1e-10)
    h_rc /= np.sum(h_rc)
    val_rc = compute_autoconv_max(h_rc)
    if val_rc < best_val:
        best_val = val_rc
        best_h = h_rc.copy()

    # Stage 1: Differential Evolution on GMM parameters
    if time_left() > 60:
        try:
            de_result = differential_evolution(
                gmm_objective,
                bounds=bounds,
                seed=42,
                maxiter=30,
                popsize=15,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube',
                updating='deferred'
            )
            if de_result.fun < best_val:
                h_de = gmm_to_h(de_result.x)
                best_val = de_result.fun
                best_h = h_de.copy()
        except Exception:
            pass

    # Stage 2: L-BFGS-B polish with annealed beta
    # We will run multiple polish stages with increasing beta
    betas = [50.0, 200.0, 1000.0]
    
    def smooth_peak_obj_and_grad(h, beta):
        """Smooth max autoconvolution with gradient, using temperature beta."""
        h = np.maximum(h, 1e-10)
        h = h / np.sum(h)

        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real

        m = np.max(conv)
        exp_conv = np.exp((conv - m) * beta)
        weights = exp_conv / np.sum(exp_conv)

        w_full = np.zeros(size)
        w_full[:len(conv)] = weights
        W = np.fft.fft(w_full, n=size)
        grad_h = 2.0 * np.fft.ifft(H * W).real[:N]

        val = m + np.log(np.sum(exp_conv)) / beta
        return val, grad_h

    current_h = best_h.copy()
    
    for beta in betas:
        if time_left() < 10:
            break
        
        try:
            # Use a wrapper to pass beta to the objective and jac
            obj_wrapper = lambda h: smooth_peak_obj_and_grad(h, beta)[0]
            jac_wrapper = lambda h: smooth_peak_obj_and_grad(h, beta)[1]
            
            result = minimize(
                obj_wrapper,
                current_h,
                jac=jac_wrapper,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 200}
            )
            
            if result.success:
                final_h = np.maximum(result.x, 0)
                if np.sum(final_h) > 0:
                    final_h = final_h / np.sum(final_h)
                    final_val = compute_autoconv_max(final_h)
                    if final_val < best_val:
                        best_val = final_val
                        best_h = final_h.copy()
                        current_h = final_h.copy()
        except Exception:
            pass

    # Stage 3: Multi-restart with GMM perturbations + L-BFGS-B polish (high beta)
    if time_left() > 15:
        n_restarts = int(min(3, time_left() // 5))
        for r in range(n_restarts):
            if time_left() < 8:
                break

            # Perturb GMM parameters
            base_params = np.zeros(3*K)
            for k in range(K):
                base_params[3*k] = 1.0
                # Spread means across the interval
                base_params[3*k+1] = N * (0.1 + 0.8 * k / (K-1)) if K > 1 else N/2
                base_params[3*k+2] = N/10

            perturbation = np.random.normal(0, 0.2, size=3*K)
            params_perturbed = base_params + perturbation
            for k in range(K):
                params_perturbed[3*k] = max(0.1, min(5.0, params_perturbed[3*k]))
                params_perturbed[3*k+1] = max(0, min(N-1, params_perturbed[3*k+1]))
                params_perturbed[3*k+2] = max(5.0, min(N/3.0, params_perturbed[3*k+2]))

            h_init = gmm_to_h(params_perturbed)

            try:
                obj_wrapper = lambda h: smooth_peak_obj_and_grad(h, 1000.0)[0]
                jac_wrapper = lambda h: smooth_peak_obj_and_grad(h, 1000.0)[1]
                
                result = minimize(
                    obj_wrapper,
                    h_init,
                    jac=jac_wrapper,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 100}
                )
                if result.success:
                    final_h = np.maximum(result.x, 0)
                    if np.sum(final_h) > 0:
                        final_h = final_h / np.sum(final_h)
                        final_val = compute_autoconv_max(final_h)
                        if final_val < best_val:
                            best_val = final_val
                            best_h = final_h.copy()
            except Exception:
                pass

    # Stage 4: Coordinate descent fine-tuning
    if time_left() > 10:
        current_h = best_h.copy()
        current_val = best_val
        max_rounds = int(min(5, time_left() // 2))

        for round_num in range(max_rounds):
            if time_left() < 4:
                break

            improved = False
            order = np.random.permutation(N)
            for idx in order:
                if time_left() < 2:
                    break

                for direction in [1, -1]:
                    lo, hi = 0.0, 0.3
                    best_step = 0.0
                    best_new_val = current_val
                    for _ in range(5):
                        mid = (lo + hi) / 2
                        new_h = current_h.copy()
                        new_h[idx] = max(0, current_h[idx] * (1 + direction * mid))
                        s = np.sum(new_h)
                        if s < 1e-10:
                            break
                        new_h /= s
                        new_val = compute_autoconv_max(new_h)
                        if new_val < best_new_val:
                            best_new_val = new_val
                            best_step = mid
                            lo = mid
                        else:
                            hi = mid

                    if best_step > 0 and best_new_val < current_val - 1e-12:
                        new_h = current_h.copy()
                        new_h[idx] = max(0, current_h[idx] * (1 + direction * best_step))
                        s = np.sum(new_h)
                        new_h /= s
                        current_h = new_h
                        current_val = best_new_val
                        improved = True
                        break

            if not improved:
                break

        if current_val < best_val:
            best_val = current_val
            best_h = current_h

    return best_h.tolist()
