# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximize the arithmetic Kakeya entropy ratio D/M on a 10x10 grid.
    D = H_{-1} (entropy of x - y), M = max(H_0, H_1, H_2).
    Uses a parametric ansatz P(x,y) ∝ exp(-α x² - β y² - γ xy) optimized via differential evolution.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin for 630s limit

    # Precompute grid indices
    # P[y, x], so row index is y, col index is x
    # x ranges 0..9, y ranges 0..9
    X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing='xy')
    # X[y, x] = x, Y[y, x] = y

    # Center the grid
    Xc = X - (n - 1) / 2.0
    Yc = Y - (n - 1) / 2.0

    # Precompute the value matrices for each slope
    # Slopes: -1, 0, 1, 2
    # Value for slope r: x + r*y
    vals_minus1 = X - Y       # x - y, range -9..9
    vals_0 = X                # x, range 0..9
    vals_1 = X + Y            # x + y, range 0..18
    vals_2 = X + 2 * Y        # x + 2y, range 0..27

    # Precompute unique values and mapping indices for fast pooling
    # For each slope, we need to sum P[y,x] where value == v
    
    # Let's use a helper to compute entropy from a probability matrix and a value matrix
    def entropy_from_matrix(P, vals):
        """
        Compute Shannon entropy of the distribution of vals, weighted by P.
        P is 10x10, vals is 10x10.
        """
        # Flatten
        p_flat = P.ravel()
        v_flat = vals.ravel()
        
        # Get unique values
        unique_vals, inverse = np.unique(v_flat, return_inverse=True)
        
        # Sum probabilities for each unique value
        # np.bincount is fast
        counts = np.bincount(inverse, weights=p_flat)
        
        # Filter out zeros
        counts = counts[counts > 1e-15]
        
        if len(counts) == 0:
            return 0.0
            
        # Shannon entropy in bits
        return -np.sum(counts * np.log2(counts))

    def compute_ratio_from_params(alpha, beta, gamma):
        """
        Compute D/M given parameters alpha, beta, gamma.
        P(y,x) ∝ exp(-alpha * Xc^2 - beta * Yc^2 - gamma * Xc * Yc)
        """
        logP = -alpha * Xc**2 - beta * Yc**2 - gamma * Xc * Yc
        
        # Stabilize
        logP -= np.max(logP)
        P = np.exp(logP)
        
        # Normalize
        total = P.sum()
        if total <= 1e-15:
            return 0.0
        P = P / total
        
        # Compute entropies
        D = entropy_from_matrix(P, vals_minus1)
        H0 = entropy_from_matrix(P, vals_0)
        H1 = entropy_from_matrix(P, vals_1)
        H2 = entropy_from_matrix(P, vals_2)
        
        M = max(H0, H1, H2)
        
        if M < 1e-9:
            return 0.0
            
        return D / M

    def objective(params):
        """
        Negative ratio for minimization.
        """
        alpha, beta, gamma = params
        ratio = compute_ratio_from_params(alpha, beta, gamma)
        if ratio <= 0:
            return 1e6
        return -ratio

    # Define bounds for differential evolution
    # We want to explore a reasonable range.
    # Based on previous analysis, negative alpha/beta (boosting corners) might help D.
    # Let's try a wide range.
    bounds = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
    
    best_score = -1.0
    best_params = None
    best_P = None

    # Run Differential Evolution
    try:
        # maxiter is the max number of iterations for the DE population
        # We need to be careful with time.
        # DE can be slow. Let's limit it.
        # If time is tight, we might not finish.
        # Let's use a timeout wrapper or just check time inside.
        # scipy's differential_evolution doesn't have a direct time limit, only maxiter.
        # We can estimate. 100 evaluations per iteration? 
        # Let's just run it and hope it's fast enough or use a small maxiter.
        
        # Actually, let's just run a few restarts with Nelder-Mead or CMA-ES if available?
        # No, let's stick to DE with a strict maxiter and check time.
        
        # To handle time safety, we can't easily interrupt DE.
        # So let's run a limited DE, then refine with Nelder-Mead.
        
        # Phase 1: Coarse search with DE
        # maxiter=50, popsize=10 -> ~500 evaluations. Each eval is ~1ms? So 0.5s.
        # Let's try maxiter=100.
        
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-4, 
            seed=42,
            polish=False  # We will polish manually
        )
        
        if time.time() < deadline - 10:
            if res.fun < 1e5:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x
                    # Compute P
                    alpha, beta, gamma = best_params
                    logP = -alpha * Xc**2 - beta * Yc**2 - gamma * Xc * Yc
                    logP -= np.max(logP)
                    best_P = np.exp(logP)
                    best_P /= best_P.sum()
            else:
                # DE failed or found bad point, try random inits
                pass
        else:
            # Time up, skip polish
            pass

    except Exception as e:
        print(f"DE failed: {e}")

    # Phase 2: Refine with Nelder-Mead from best point found or random starts
    if best_params is None:
        # If DE didn't give a good start, try some manual inits
        inits = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, -0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, -0.5],
        ]
    else:
        inits = [best_params]
        # Add a few random perturbations around best
        for _ in range(5):
            if time.time() > deadline - 20:
                break
            pert = np.random.normal(0, 0.5, 3)
            inits.append(best_params + pert)
            
        # Also add some diverse inits
        inits.extend([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
        ])

    for init in inits:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(objective, init, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < 1e5:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    alpha, beta, gamma = res.x
                    logP = -alpha * Xc**2 - beta * Yc**2 - gamma * Xc * Yc
                    logP -= np.max(logP)
                    P = np.exp(logP)
                    P /= P.sum()
                    best_P = P
        except Exception:
            pass

    # Fallback if everything failed
    if best_P is None:
        best_P = np.ones((n, n)) / (n * n)
    else:
        # Final validation and cleanup
        best_P = np.clip(best_P, 0, None)
        total = best_P.sum()
        if total <= 1e-9:
            best_P = np.ones((n, n)) / (n * n)
        else:
            best_P /= total

    return best_P

# EVOLVE-BLOCK-END
