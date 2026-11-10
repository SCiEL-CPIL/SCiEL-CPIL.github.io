# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt
import scipy.stats as stats

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    
    Maximizes D/M where D = H_{-1} (entropy of x-y distribution) and
    M = max(H_0, H_1, H_2) (max entropy of x, x+y, x+2y distributions).
    
    Strategy:
    1. Global search using Differential Evolution on a Mixture of Gaussians model.
    2. Local refinement using L-BFGS-B on the full 100-dimensional probability vector.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # 40s safety margin

    x_idx = np.arange(n)
    y_idx = np.arange(n)
    
    # Precompute value matrices for each slope
    # P[y, x] corresponds to X=x, Y=y
    diff_vals = x_idx[None, :] - y_idx[:, None]      # x - y
    x_vals = x_idx[None, :]                          # x
    sum_vals = x_idx[None, :] + y_idx[:, None]       # x + y
    sum2_vals = x_idx[None, :] + 2 * y_idx[:, None]  # x + 2*y

    def compute_entropies(P):
        """
        Compute H_{-1}, H_0, H_1, H_2 for a given 10x10 distribution P.
        """
        def shannon_entropy_from_values(P, vals):
            # Flatten P and vals to 1D
            P_flat = P.flatten()
            vals_flat = vals.flatten()
            
            # Get unique values and their indices
            unique_vals, inverse_indices = np.unique(vals_flat, return_inverse=True)
            
            # Sum probabilities for each unique value
            probs = np.bincount(inverse_indices, weights=P_flat)
            
            # Filter out zero probabilities
            probs = probs[probs > 1e-15]
            
            if len(probs) == 0:
                return 0.0
            
            return -np.sum(probs * np.log2(probs))
        
        H_neg1 = shannon_entropy_from_values(P, diff_vals)
        H_0 = shannon_entropy_from_values(P, x_vals)
        H_1 = shannon_entropy_from_values(P, sum_vals)
        H_2 = shannon_entropy_from_values(P, sum2_vals)
        
        return H_neg1, H_0, H_1, H_2

    def score_P(P):
        """
        Calculate the score D/M for a given distribution P.
        """
        P = np.clip(P, 1e-15, None) # Avoid log(0)
        P = P / P.sum()
        
        H_neg1, H_0, H_1, H_2 = compute_entropies(P)
        M = max(H_0, H_1, H_2)
        
        if M < 1e-15:
            return 0.0
        return H_neg1 / M

    def full_objective(params):
        """
        Objective for full 100-dim distribution optimization.
        Minimizes -score.
        """
        # Softmax to ensure valid probabilities
        params_clipped = np.clip(params, -50, 50)
        exp_params = np.exp(params_clipped - np.max(params_clipped))
        P = exp_params / exp_params.sum()
        P = P.reshape(n, n)
        
        score = score_P(P)
        return -score

    def full_objective_grad(params):
        """
        Objective and gradient for full 100-dim distribution.
        Note: Gradient calculation for the max() term is tricky. 
        We can approximate or use finite differences if needed, but L-BFGS-B can use numerical gradients if not provided.
        Given the complexity of the max() derivative, we will rely on numerical gradients in L-BFGS-B 
        or use a smooth approximation for the max term during optimization if we implement analytic grads.
        For simplicity and robustness, we let L-BFGS-B compute numerical gradients.
        """
        return full_objective(params)

    # --- Phase 1: Global Search with Mixture of Gaussians ---
    
    def gmm_params_to_P(params, K):
        """
        Convert GMM parameters to a 10x10 probability distribution.
        params: [w1, mx1, my1, sx1, sy1, r1, w2, ...]
        """
        P = np.zeros((n, n))
        X, Y = np.meshgrid(x_idx, y_idx)
        
        for k in range(K):
            base = k * 6
            w = params[base]
            mx = params[base + 1]
            my = params[base + 2]
            sx = params[base + 3]
            sy = params[base + 4]
            rho = params[base + 5]
            
            # Clamp rho to avoid singular covariance
            rho = np.clip(rho, -0.99, 0.99)
            
            # Covariance matrix
            cov = np.array([
                [sx**2, rho * sx * sy],
                [rho * sx * sy, sy**2]
            ])
            
            try:
                # Evaluate Gaussian PDF
                # stats.multivariate_normal.pdf expects (x, y) pairs
                points = np.column_stack((X.flatten(), Y.flatten()))
                pdf = stats.multivariate_normal.pdf(points, mean=[mx, my], cov=cov)
                P += w * pdf.reshape(n, n)
            except Exception:
                pass
        
        P = np.clip(P, 0, None)
        total = P.sum()
        if total > 1e-15:
            P = P / total
        else:
            P = np.ones((n, n)) / (n * n)
        return P

    def gmm_objective(params, K):
        """
        Objective for GMM optimization.
        """
        P = gmm_params_to_P(params, K)
        score = score_P(P)
        return -score

    best_P = None
    best_score = -np.inf
    
    rng = np.random.default_rng(42)

    # Try different numbers of components
    K_values = [2, 3, 4]
    
    for K in K_values:
        if time.time() > deadline:
            break
            
        n_params = K * 6
        bounds = []
        for _ in range(K):
            bounds.append((0.0, 1.0))      # Weight
            bounds.append((0.0, 9.0))      # Mean X
            bounds.append((0.0, 9.0))      # Mean Y
            bounds.append((0.5, 5.0))      # Std X
            bounds.append((0.5, 5.0))      # Std Y
            bounds.append((-0.9, 0.9))     # Correlation
        
        # Differential Evolution
        try:
            result = opt.differential_evolution(
                gmm_objective,
                bounds,
                args=(K,),
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-6,
                polish=False # We will polish locally
            )
            
            if result.success or result.fun < 0:
                P_gmm = gmm_params_to_P(result.x, K)
                score_gmm = score_P(P_gmm)
                
                if score_gmm > best_score:
                    best_score = score_gmm
                    best_P = P_gmm
                    
        except Exception:
            pass

    # Also try a few random GMM inits and local refine
    for _ in range(5):
        if time.time() > deadline:
            break
        K = rng.integers(2, 5)
        n_params = K * 6
        # Random init within bounds
        params = np.array([
            rng.uniform(0.1, 1.0) for _ in range(K) # Weights
        ] + [
            rng.uniform(0, 9) for _ in range(2*K)   # Means
        ] + [
            rng.uniform(1.0, 3.0) for _ in range(2*K) # Stds
        ] + [
            rng.uniform(-0.5, 0.5) for _ in range(K)  # Rhos
        ])
        
        P_gmm = gmm_params_to_P(params, K)
        score_gmm = score_P(P_gmm)
        
        if score_gmm > best_score:
            best_score = score_gmm
            best_P = P_gmm

    # --- Phase 2: Local Refinement on Full Grid ---
    
    if best_P is None:
        best_P = np.ones((n, n)) / (n * n)
    
    # Initialize logits from best_P
    P_init = best_P
    P_init = np.clip(P_init, 1e-15, None)
    P_init = P_init / P_init.sum()
    logits_init = np.log(P_init).flatten()
    
    # Run L-BFGS-B
    try:
        result_local = opt.minimize(
            full_objective,
            logits_init,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        
        if result_local.fun < 0:
            params_opt = result_local.x
            exp_params = np.exp(np.clip(params_opt, -50, 50) - np.max(np.clip(params_opt, -50, 50)))
            P_opt = (exp_params / exp_params.sum()).reshape(n, n)
            
            score_opt = score_P(P_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_P = P_opt
                
    except Exception:
        pass

    # If we have time, try one more local refine from a slightly perturbed uniform
    if time.time() < deadline - 60:
        P_uniform = np.ones((n, n)) / (n * n)
        logits_uniform = np.log(P_uniform).flatten()
        
        try:
            result_local2 = opt.minimize(
                full_objective,
                logits_uniform,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14}
            )
            if result_local2.fun < 0:
                params_opt2 = result_local2.x
                exp_params2 = np.exp(np.clip(params_opt2, -50, 50) - np.max(np.clip(params_opt2, -50, 50)))
                P_opt2 = (exp_params2 / exp_params2.sum()).reshape(n, n)
                
                score_opt2 = score_P(P_opt2)
                
                if score_opt2 > best_score:
                    best_score = score_opt2
                    best_P = P_opt2
        except Exception:
            pass

    # Final cleanup
    best_P = np.clip(best_P, 0, None)
    total = best_P.sum()
    if total > 1e-15:
        best_P = best_P / total
    else:
        best_P = np.ones((n, n), dtype=np.float64) / (n * n)
        
    return best_P

# EVOLVE-BLOCK-END
