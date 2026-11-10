import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    timeout = 630.0
    safety_margin = 10.0
    
    # Pre-compute projection indices for fast binning
    x = np.arange(n)
    y = np.arange(n)
    xx, yy = np.meshgrid(x, y, indexing='xy') # xx[y, x] = x, yy[y, x] = y
    
    proj_m1 = xx - yy      # x - y
    proj_0 = xx            # x
    proj_1 = xx + yy       # x + y
    proj_2 = xx + 2 * yy   # x + 2y
    
    # Flatten for bincount
    proj_m1_flat = proj_m1.flatten()
    proj_0_flat = proj_0.flatten()
    proj_1_flat = proj_1.flatten()
    proj_2_flat = proj_2.flatten()
    
    # Unique values and inverse indices
    _, inv_m1 = np.unique(proj_m1_flat, return_inverse=True)
    _, inv_0 = np.unique(proj_0_flat, return_inverse=True)
    _, inv_1 = np.unique(proj_1_flat, return_inverse=True)
    _, inv_2 = np.unique(proj_2_flat, return_inverse=True)
    
    n_m1 = len(np.unique(proj_m1_flat))
    n_0 = len(np.unique(proj_0_flat))
    n_1 = len(np.unique(proj_1_flat))
    n_2 = len(np.unique(proj_2_flat))
    
    def calc_entropy(inv, weights, n_bins):
        q = np.bincount(inv, weights=weights, minlength=n_bins)
        mask = q > 1e-15
        if not np.any(mask):
            return 0.0
        q_nonzero = q[mask]
        return -np.sum(q_nonzero * np.log2(q_nonzero))
    
    def calc_entropies(weights):
        H_m1 = calc_entropy(inv_m1, weights, n_m1)
        H_0 = calc_entropy(inv_0, weights, n_0)
        H_1 = calc_entropy(inv_1, weights, n_1)
        H_2 = calc_entropy(inv_2, weights, n_2)
        return H_m1, H_0, H_1, H_2
    
    def score_weights(weights):
        H_m1, H_0, H_1, H_2 = calc_entropies(weights)
        M = max(H_0, H_1, H_2)
        if M < 1e-9:
            return 0.0
        return H_m1 / M
    
    # Strategy: Low-rank factorization P = U V^T
    # U (10 x r), V (10 x r)
    # P[y, x] = sum_k U[y, k] * V[x, k]
    # We optimize U and V parameters.
    # To ensure P >= 0 and sum P = 1, we can use a parameterization that guarantees positivity,
    # or simply clip and normalize in the objective, but gradients become tricky.
    # A better approach: optimize log-parameters or use softmax-like outputs.
    # However, standard scipy minimize with bounds on U, V can work if we handle the normalization/clip carefully.
    # Since P must be non-negative, let's use U = exp(u), V = exp(v) ? No, that restricts to positive correlations only if we aren't careful.
    # Actually, P can have zeros.
    # Let's stick to optimizing the 100 weights directly but using the low-rank *initialization* and *search* ideas.
    # The directive suggests: "Represent the probability mass ... as a low-rank matrix factorization ... which allows for complex ... correlations".
    # If we just optimize 100 weights, we don't use the low-rank structure for *optimization*, only for init.
    # To truly use it, we should optimize U and V.
    
    # Let's define the objective in terms of U and V.
    # P = U @ V.T
    # We need P >= 0. This is hard to enforce with simple bounds on U, V if we want negative values in U or V.
    # If we restrict U, V to be non-negative, then P is non-negative.
    # Sum(P) = 1 constraint is handled by normalizing P inside the objective, but this creates a non-smooth boundary if P hits 0.
    # Alternatively, we can optimize U, V and then normalize P in the evaluation, but the gradient of the normalization is complex.
    
    # Let's try a hybrid:
    # 1. Generate diverse low-rank initializations.
    # 2. Use L-BFGS-B on the 100 weights, starting from these initializations.
    # 3. The low-rank structure helps jump out of local minima by providing structured starting points that capture correlations.
    
    best_P = np.ones((n, n)) / (n * n)
    best_score = score_weights(best_P.flatten())
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Differential Evolution on the full 100 weights (global search)
    # This is expensive but robust. We can run it for a short time.
    # Bounds: (0, 1) for each weight.
    # We need to handle normalization. The objective should normalize the input.
    
    def objective_de(p_flat):
        P = p_flat.reshape((n, n))
        # Normalize
        s = np.sum(P)
        if s < 1e-9:
            return 1e9 # Penalty
        P_norm = P / s
        # Ensure non-negative (bounds should handle this, but just in case)
        P_norm = np.clip(P_norm, 0, None)
        s2 = np.sum(P_norm)
        if s2 < 1e-9:
            return 1e9
        P_norm = P_norm / s2
        
        sc = score_weights(P_norm.flatten())
        return -sc # Minimize negative score
    
    # Run DE for a limited number of iterations
    bounds = [(0, 1)] * (n * n)
    
    try:
        # Use a smaller population and maxiter to fit in time
        de_res = differential_evolution(
            objective_de,
            bounds,
            seed=42,
            maxiter=50,
            popsize=10,
            tol=1e-4,
            polish=False # We will polish later
        )
        
        if de_res.fun < 1e8:
            P_de = de_res.x.reshape((n, n))
            P_de = np.clip(P_de, 0, None)
            s = np.sum(P_de)
            if s > 0:
                P_de /= s
            sc = score_weights(P_de.flatten())
            if sc > best_score:
                best_score = sc
                best_P = P_de.copy()
    except Exception:
        pass
    
    if time.time() - start_time > timeout - safety_margin:
        return best_P
    
    # Phase 2: Local Refinement using L-BFGS-B
    # Start from best_P found so far, and also from some low-rank structured inits.
    
    def objective_lbfgs(p_flat):
        P = p_flat.reshape((n, n))
        P = np.clip(P, 0, None)
        s = np.sum(P)
        if s < 1e-9:
            return 1e9
        P_norm = P / s
        sc = score_weights(P_norm.flatten())
        return -sc
    
    # Try refining the current best
    p0 = best_P.flatten()
    try:
        res = minimize(
            objective_lbfgs,
            p0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.fun < 1e8:
            P_opt = res.x.reshape((n, n))
            P_opt = np.clip(P_opt, 0, None)
            s = np.sum(P_opt)
            if s > 0:
                P_opt /= s
            sc = score_weights(P_opt.flatten())
            if sc > best_score:
                best_score = sc
                best_P = P_opt.copy()
    except Exception:
        pass
    
    if time.time() - start_time > timeout - safety_margin:
        return best_P
    
    # Phase 3: Low-Rank Structured Search
    # Generate random low-rank matrices, clip/normalize, and refine.
    ranks = [2, 3, 4, 5]
    
    for r in ranks:
        for _ in range(20): # 20 restarts per rank
            if time.time() - start_time > timeout - safety_margin:
                break
            
            # Random low-rank init
            U = rng.normal(size=(n, r))
            V = rng.normal(size=(n, r))
            P_init = U @ V.T
            # Clip and normalize
            P_init = np.clip(P_init, 0, None)
            s = np.sum(P_init)
            if s < 1e-9:
                continue
            P_init /= s
            
            p0 = P_init.flatten()
            try:
                res = minimize(
                    objective_lbfgs,
                    p0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.fun < 1e8:
                    P_opt = res.x.reshape((n, n))
                    P_opt = np.clip(P_opt, 0, None)
                    s = np.sum(P_opt)
                    if s > 0:
                        P_opt /= s
                    sc = score_weights(P_opt.flatten())
                    if sc > best_score:
                        best_score = sc
                        best_P = P_opt.copy()
            except Exception:
                pass
            
            if time.time() - start_time > timeout - safety_margin:
                break
    
    # Final cleanup
    best_P = np.clip(best_P, 0, None)
    s = np.sum(best_P)
    if s > 0:
        best_P /= s
    else:
        best_P = np.ones((n, n)) / (n * n)
        
    if not np.all(np.isfinite(best_P)):
        best_P = np.ones((n, n)) / (n * n)
        
    return best_P

if __name__ == "__main__":
    P = arithmetic_kakeya_distribution()
    print(P.shape)
    print(np.sum(P))
    print(np.all(P >= 0))
