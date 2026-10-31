import numpy as np
from scipy import optimize
import time
import random

N = 600

def _fft_convolve(h):
    """Compute linear autoconvolution using FFT for speed."""
    fft_size = 2048
    H = np.fft.rfft(h, fft_size)
    P = H * np.conj(H)
    conv = np.fft.irfft(P, fft_size)
    return conv[:2*N-1]

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    conv = _fft_convolve(h)
    total = np.sum(h)
    if total == 0:
        return 0.0
    return np.max(conv) / (total * total)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    def time_left():
        return deadline - time.time()

    # --- GMM Representation ---
    # h(x) = sum_{j=1}^{K} w_j * exp(-((x - c_j)^2) / (2 * s_j^2))
    K = 20  # Number of Gaussian components
    i_grid = np.arange(N, dtype=float)
    
    def params_to_h(params):
        """Convert parameter vector to height array."""
        h = np.zeros(N)
        for j in range(K):
            w_j = params[3*j]
            c_j = params[3*j+1]
            s_j = params[3*j+2]
            if s_j < 1e-6:
                s_j = 1e-6
            h += w_j * np.exp(-((i_grid - c_j)**2) / (2 * s_j**2))
        return h

    def gmm_objective(params):
        """Objective function for GMM parameters: max autoconvolution of normalized h."""
        if time_left() < 2:
            return 1e10
        h = params_to_h(params)
        s = np.sum(h)
        if s < 1e-10:
            return 1e10
        h_norm = h / s
        return compute_autoconv_max(h_norm)

    # Bounds for GMM parameters
    bounds = []
    for j in range(K):
        bounds.append((0.0, 10.0))   # w_j
        bounds.append((0.0, N-1))    # c_j
        bounds.append((1.0, N/2))    # s_j

    # Initial best
    best_h = np.ones(N) / N
    best_val = compute_autoconv_max(best_h)
    
    # --- Phase 1: Differential Evolution on GMM Parameters ---
    if time_left() > 60:
        # Generate diverse initial population
        def generate_initial_pop(size):
            pop = np.zeros((size, 3*K))
            for idx in range(size):
                strategy = random.randint(0, 3)
                for j in range(K):
                    if strategy == 0:
                        # Bimodal edges
                        if j < K//2:
                            pop[idx, 3*j] = random.uniform(0.5, 2.0)
                            pop[idx, 3*j+1] = random.uniform(0, N*0.2)
                            pop[idx, 3*j+2] = random.uniform(N/10, N/4)
                        else:
                            pop[idx, 3*j] = random.uniform(0.5, 2.0)
                            pop[idx, 3*j+1] = random.uniform(N*0.8, N-1)
                            pop[idx, 3*j+2] = random.uniform(N/10, N/4)
                    elif strategy == 1:
                        # Spread
                        pop[idx, 3*j] = random.uniform(0.1, 1.0)
                        pop[idx, 3*j+1] = j * (N-1) / (K-1) + random.uniform(-N/10, N/10)
                        pop[idx, 3*j+2] = random.uniform(N/8, N/3)
                    elif strategy == 2:
                        # Central
                        pop[idx, 3*j] = random.uniform(0.1, 0.5)
                        pop[idx, 3*j+1] = random.uniform(N*0.3, N*0.7)
                        pop[idx, 3*j+2] = random.uniform(N/6, N/2)
                    else:
                        # Random
                        pop[idx, 3*j] = random.uniform(0.01, 5.0)
                        pop[idx, 3*j+1] = random.uniform(0, N-1)
                        pop[idx, 3*j+2] = random.uniform(2, N/2)
            return pop

        try:
            pop_size = 30
            max_iter = 50
            # Adjust based on time
            if time_left() > 90:
                pop_size = 50
                max_iter = 80
                
            result_de = optimize.differential_evolution(
                gmm_objective,
                bounds=bounds,
                maxiter=max_iter,
                popsize=pop_size,
                tol=1e-5,
                seed=42,
                polish=False,
                init=generate_initial_pop(pop_size)
            )
            
            if result_de.fun < best_val:
                h_de = params_to_h(result_de.x)
                s = np.sum(h_de)
                if s > 1e-10:
                    h_de = h_de / s
                    best_val = compute_autoconv_max(h_de)
                    best_h = h_de.copy()
        except Exception:
            pass

    # --- Phase 2: Local Refinement of GMM Parameters ---
    if time_left() > 30:
        # Start from DE result if available, else random
        start_params = None
        if 'result_de' in locals() and hasattr(result_de, 'x'):
            start_params = result_de.x.copy()
        else:
            # Fallback: simple spread
            start_params = np.zeros(3*K)
            for j in range(K):
                start_params[3*j] = 0.5
                start_params[3*j+1] = j * (N-1) / (K-1)
                start_params[3*j+2] = N/4

        try:
            result_lbfgs_gmm = optimize.minimize(
                gmm_objective,
                start_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result_lbfgs_gmm.fun < best_val:
                h_gmm = params_to_h(result_lbfgs_gmm.x)
                s = np.sum(h_gmm)
                if s > 1e-10:
                    h_gmm = h_gmm / s
                    val_gmm = compute_autoconv_max(h_gmm)
                    if val_gmm < best_val:
                        best_val = val_gmm
                        best_h = h_gmm.copy()
        except Exception:
            pass

    # --- Phase 3: Direct Vector Refinement ---
    # Use the smooth GMM result as a starting point for a direct optimization on the 600-dim vector.
    # We use a soft-max proxy to ensure smooth gradients for L-BFGS-B.
    
    def soft_objective_vector(x, T=0.01):
        h = np.abs(x)
        conv = _fft_convolve(h)
        total = np.sum(h)
        if total < 1e-10:
            return 1e10
        conv_max = np.max(conv)
        shifted = (conv - conv_max) / T
        shifted = np.clip(shifted, -50, 50)
        soft_max = conv_max + T * np.log(np.sum(np.exp(shifted)))
        return soft_max / (total * total)

    if time_left() > 15:
        # Start from current best_h
        # Ensure it's normalized to sum N for stability in optimization
        h_start = best_h.copy()
        h_start = h_start / np.sum(h_start) * N
        
        try:
            result_vec = optimize.minimize(
                soft_objective_vector,
                h_start,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 1000}
            )
            
            h_new = np.abs(result_vec.x)
            # Check with exact objective
            val_new = compute_autoconv_max(h_new)
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
                
        except Exception:
            pass
            
    # Final check and return
    return list(best_h)
