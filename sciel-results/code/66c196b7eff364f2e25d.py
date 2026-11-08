import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_genlaguerre
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    m = 12
    alpha = 1  # Laguerre parameter

    # Precompute transformation matrix from monomial to Laguerre basis
    # L_k^(alpha)(x) = sum_{j=0}^k A[k,j] x^j
    # A[k,j] = (-1)^j * C(k+alpha, k-j) / j!
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k + alpha, k - j) / math.factorial(j)
    
    # c = A^{-1} @ p where p are monomial coefficients (ascending)
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)

    def compute_norm_sq(roots):
        """Compute squared norm of P(x) = prod(x - r_i) in Laguerre basis."""
        # Build polynomial coefficients in ascending order
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]  # ascending
        
        # Convert to Laguerre basis
        c = A_inv @ p
        
        # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
        norm_sq = np.sum(c**2 * (np.arange(m + 1) + 1))
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """Minimize 1/norm_sq (equivalent to maximizing norm_sq)."""
        roots = np.exp(log_roots)
        
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        
        # 1. Geometric sequence: r_i = a * q^i
        for a in [1.0, 2.0, 4.0, 8.0]:
            for q in [1.5, 2.0, 2.5, 3.0]:
                roots = a * (q ** np.arange(m))
                guesses.append(np.log(roots))
        
        # 2. Arithmetic sequence
        for start in [1.0, 2.0, 5.0, 10.0]:
            for step in [1.0, 2.0, 5.0, 10.0]:
                roots = start + np.arange(m) * step
                if np.all(roots > 0):
                    guesses.append(np.log(roots))
        
        # 3. Zeros of Laguerre polynomial L_m^(alpha)
        try:
            x, w = roots_genlaguerre(m, alpha)
            # Scale to get good spread
            for scale in [0.5, 1.0, 2.0, 4.0]:
                roots = x * scale
                if np.all(roots > 0):
                    guesses.append(np.log(roots))
        except:
            pass
        
        # 4. Random log-uniform
        np.random.seed(42)
        for _ in range(20):
            log_roots = np.random.uniform(-1.0, 8.0, size=m)
            guesses.append(log_roots)
        
        # 5. Based on Cohn-Goncalves construction hints
        # Try roots near the zeros of L_{2m} or related
        for k in [10, 12, 14, 16, 20, 24]:
            try:
                x, w = roots_genlaguerre(k, alpha)
                # Take m roots from the larger set
                if len(x) >= m:
                    roots = x[:m]
                    if np.all(roots > 0):
                        guesses.append(np.log(roots))
                    roots = x[-m:]
                    if np.all(roots > 0):
                        guesses.append(np.log(roots))
            except:
                pass
        
        return guesses

    # Optimization strategy
    best_val = 1e10
    best_log_roots = None
    best_roots = None

    # Phase 1: Try all initial guesses with BFGS
    guesses = get_initial_guesses()
    
    for i, guess in enumerate(guesses):
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:  # Leave time for final polish
            break
        
        try:
            result = minimize(
                cost_func, guess, method='L-BFGS-B',
                bounds=[(-5.0, 12.0)] * m,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(best_log_roots)
                
        except Exception as e:
            pass

    # Phase 2: Differential Evolution for global search
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        try:
            # Use DE with the best found as a seed
            de_bounds = [(-4.0, 11.0)] * m
            
            if best_log_roots is not None:
                # Start DE near the best solution
                result = differential_evolution(
                    cost_func, de_bounds,
                    seed=42, maxiter=100, popsize=15,
                    tol=1e-10, polish=True, init='sobol'
                )
            else:
                result = differential_evolution(
                    cost_func, de_bounds,
                    seed=42, maxiter=100, popsize=15,
                    tol=1e-10, polish=True, init='sobol'
                )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(best_log_roots)
                
        except Exception as e:
            pass

    # Phase 3: Additional DE restarts if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 45:
        for seed in [123, 456, 789]:
            elapsed = time.time() - start_time
            if elapsed > deadline - 30:
                break
            try:
                result = differential_evolution(
                    cost_func, [(-4.0, 11.0)] * m,
                    seed=seed, maxiter=50, popsize=12,
                    tol=1e-10, polish=True, init='sobol'
                )
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_log_roots = result.x.copy()
                    best_roots = np.exp(best_log_roots)
                    
            except:
                pass

    # Phase 4: Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 10 and best_log_roots is not None:
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=[(-6.0, 13.0)] * m,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x.copy()
                best_roots = np.exp(best_log_roots)
                
        except:
            pass

    # Ensure roots are positive
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-8)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]

