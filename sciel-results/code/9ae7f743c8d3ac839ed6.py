import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_genlaguerre, roots_genlaguerre
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin before 630s kill

    m = 12
    alpha = 1

    # Precompute the transformation matrix from monomial basis to Laguerre basis
    # L_k^(alpha)(x) = sum_{j=0}^k A[k,j] x^j
    # For alpha=1: A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k + alpha, j + 1) / math.factorial(j)
    
    # Precompute Vandermonde matrix V[i,k] = roots[i]^k and M = V @ diag(weights) @ V^T
    # so that norm_sq = p^T M p where p is the monomial coefficient vector.
    # This avoids np.poly and per-call matrix inversion.

    weights = np.arange(m + 1) + alpha  # (k + alpha) for k=0..m
    sqrt_weights = np.sqrt(weights)

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        roots: array of 12 positive numbers.
        """
        # Build polynomial coefficients in standard basis (ascending)
        # P(x) = prod(x - r_i)
        # np.poly returns descending coeffs: [1, c1, ..., cm]
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]  # ascending, length m+1
        
        # V[i,k] = roots[i]^k
        V = np.vander(roots, N=m+1, increasing=True)  # shape (12, 13)
        # M = V @ diag(sqrt_w) @ (V @ diag(sqrt_w))^T = (V * sqrt_w) @ (V * sqrt_w)^T
        Vw = V * sqrt_weights  # shape (12, 13)
        M = Vw.T @ Vw  # shape (13, 13)
        norm_sq = float(p @ M @ p)
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize the upper bound on C4 = 1/||P||^2.
        """
        roots = np.exp(log_roots)
        
        # Check for numerical issues
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Get the zeros of L_12^(1)(x) as a natural starting point
    try:
        laguerre_roots = roots_genlaguerre(m, alpha)[0]
        laguerre_roots = np.sort(laguerre_roots)
        # Ensure positive
        laguerre_roots = np.maximum(laguerre_roots, 1e-10)
        laguerre_log_roots = np.log(laguerre_roots)
    except Exception:
        # Fallback: use a geometric sequence
        laguerre_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)

    # Define bounds for log_roots
    # Roots in [1e-4, 1e4] -> logs in [-9.2, 9.2]
    log_low = -9.0
    log_high = 10.0
    bounds = [(log_low, log_high)] * 12

    # Generate diverse initial guesses
    rng = np.random.default_rng(42)
    
    initial_guesses = []
    
    # 1. Zeros of L_12^(1)(x) - the most natural starting point
    initial_guesses.append(laguerre_log_roots)
    
    # 2. Perturbations of the Laguerre zeros
    for scale in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        perturbed = laguerre_roots * scale
        perturbed = np.maximum(perturbed, 1e-8)
        logs = np.log(perturbed)
        if np.all(logs > log_low) and np.all(logs < log_high):
            initial_guesses.append(logs)
    
    # 3. Original geometric seed
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    initial_guesses.append(seed_log_roots)
    
    # 4. More moderate geometric sequences
    for base in [1.3, 1.5, 1.8, 2.0, 2.5, 3.0]:
        for start in [0.5, 1.0, 2.0, 4.0, 8.0]:
            logs = np.log(start) + np.arange(12) * np.log(base)
            if np.all(logs > log_low) and np.all(logs < log_high):
                initial_guesses.append(logs)
    
    # 5. Linear spacing in log space centered around Laguerre zeros
    center = np.mean(laguerre_log_roots)
    for width in [1.0, 2.0, 3.0, 4.0, 5.0]:
        logs = center + width * (np.arange(12) - 5.5) / 5.5
        if np.all(logs > log_low) and np.all(logs < log_high):
            initial_guesses.append(logs)
    
    # 6. Random log-uniform starts
    for _ in range(100):
        logs = rng.uniform(log_low, log_high, size=12)
        initial_guesses.append(logs)
    
    # 7. Clustered roots near Laguerre zeros
    for _ in range(30):
        center = rng.uniform(-1, 4)
        spread = rng.uniform(0.3, 3)
        logs = center + spread * rng.normal(0, 1, size=12)
        logs = np.clip(logs, log_low, log_high)
        initial_guesses.append(logs)

    best_val = 1e10
    best_log_roots = None
    best_roots = None

    # Run L-BFGS-B on each guess with time management
    for i, init in enumerate(initial_guesses):
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
            
        try:
            result = minimize(
                cost_func, init, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass

    # If we have time, try DE runs to escape potential local minima
    if time.time() - start_time < deadline - 90:
        de_seeds = [42, 123, 456, 789, 101, 202, 303, 404]
        for seed in de_seeds:
            if time.time() - start_time > deadline - 30:
                break
            try:
                result_de = differential_evolution(
                    cost_func, 
                    bounds, 
                    seed=seed, 
                    maxiter=150, 
                    popsize=20, 
                    tol=1e-12, 
                    polish=True,
                    init='sobol'
                )
                if result_de.fun < best_val:
                    best_val = result_de.fun
                    best_log_roots = result_de.x.copy()
                    best_roots = np.exp(best_log_roots)
            except Exception:
                pass

    # Final high-precision polish
    if time.time() - start_time < deadline - 5 and best_log_roots is not None:
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-18, 'maxls': 100}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Second polish pass with Nelder-Mead for potential escape from L-BFGS-B saddle
    if time.time() - start_time < deadline - 3 and best_log_roots is not None:
        try:
            nm_result = minimize(
                cost_func, best_log_roots, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
            )
            if nm_result.fun < best_val:
                best_val = nm_result.fun
                best_log_roots = nm_result.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Ensure roots are positive
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-10)
        return list(best_roots)
    
    # Absolute fallback: zeros of L_12^(1)(x)
    try:
        laguerre_roots = roots_genlaguerre(m, alpha)[0]
        laguerre_roots = np.sort(laguerre_roots)
        laguerre_roots = np.maximum(laguerre_roots, 1e-10)
        return list(laguerre_roots)
    except Exception:
        return [4.0 * (2.0 ** i) for i in range(12)]
