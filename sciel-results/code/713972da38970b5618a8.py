# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the ratio D/M where:
    D = H(x - y)
    M = max(H(x), H(x + y), H(x + 2y))
    
    Uses numerical optimization on the log-probabilities to find a non-uniform distribution.
    """
    n = 10
    deadline = time.time() + 590.0  # Safety margin
    
    # Grid coordinates
    # P[y, x] -> y is row (0..9), x is col (0..9)
    ys, xs = np.mgrid[0:n, 0:n]
    
    # Linear forms
    # d_val = x - y
    # m0_val = x
    # m1_val = x + y
    # m2_val = x + 2y
    d_val = (xs - ys).flatten()
    m0_val = xs.flatten()
    m1_val = (xs + ys).flatten()
    m2_val = (xs + 2 * ys).flatten()
    
    # Unique values and inverse mapping (indices)
    # We need to sum probabilities for cells with the same value.
    # Using np.unique with return_inverse gives indices into the unique values.
    d_vals, d_inv = np.unique(d_val, return_inverse=True)
    m0_vals, m0_inv = np.unique(m0_val, return_inverse=True)
    m1_vals, m1_inv = np.unique(m1_val, return_inverse=True)
    m2_vals, m2_inv = np.unique(m2_val, return_inverse=True)
    
    # Precompute indices for fast binning
    # For each unique value index, we need the sum of exp(log_p) for cells mapping to it.
    # We can use np.bincount for this if we have the inverse indices.
    
    def compute_entropies(log_p_flat):
        """
        log_p_flat: shape (100,)
        Returns: D, M (max of H0, H1, H2)
        """
        p = np.exp(log_p_flat)
        
        # Calculate distributions for each projection
        # Sum p over cells with same value
        p_d = np.bincount(d_inv, weights=p, minlength=len(d_vals))
        p_m0 = np.bincount(m0_inv, weights=p, minlength=len(m0_vals))
        p_m1 = np.bincount(m1_inv, weights=p, minlength=len(m1_vals))
        p_m2 = np.bincount(m2_inv, weights=p, minlength=len(m2_vals))
        
        def shannon(p_dist):
            # Ignore zeros
            p_nonzero = p_dist[p_dist > 1e-18]
            if len(p_nonzero) == 0:
                return 0.0
            return -np.sum(p_nonzero * np.log2(p_nonzero))
        
        H_d = shannon(p_d)
        H_m0 = shannon(p_m0)
        H_m1 = shannon(p_m1)
        H_m2 = shannon(p_m2)
        
        M = max(H_m0, H_m1, H_m2)
        return H_d, M

    def objective(log_p_flat):
        """
        Minimize -D/M (since we want to maximize D/M)
        """
        # Normalize log_p to ensure sum(p) = 1
        # p = exp(log_p). sum(exp(log_p)) = 1
        # log_p is already on the simplex if we constrain it, but L-BFGS-B might drift.
        # Better to normalize inside objective to be safe, or use softmax parametrization.
        # Let's use softmax parametrization: p = exp(log_p - max(log_p)) / sum(exp(log_p - max(log_p)))
        # Actually, standard trick: optimize over log_p, and normalize p = exp(log_p) / sum(exp(log_p)).
        # This ensures p is always a valid distribution.
        
        # To avoid overflow in exp, subtract max
        log_p_shifted = log_p_flat - np.max(log_p_flat)
        p = np.exp(log_p_shifted)
        p = p / np.sum(p)
        
        # Now p is the actual probability distribution
        # We need entropies based on p
        # But wait, the objective function inside minimize expects to return a scalar.
        # We need to compute H_d and M from p.
        
        # Re-use bincount logic
        p_d = np.bincount(d_inv, weights=p, minlength=len(d_vals))
        p_m0 = np.bincount(m0_inv, weights=p, minlength=len(m0_vals))
        p_m1 = np.bincount(m1_inv, weights=p, minlength=len(m1_vals))
        p_m2 = np.bincount(m2_inv, weights=p, minlength=len(m2_vals))
        
        def shannon(p_dist):
            p_nonzero = p_dist[p_dist > 1e-18]
            if len(p_nonzero) == 0:
                return 0.0
            return -np.sum(p_nonzero * np.log2(p_nonzero))
        
        H_d = shannon(p_d)
        H_m0 = shannon(p_m0)
        H_m1 = shannon(p_m1)
        H_m2 = shannon(p_m2)
        
        M = max(H_m0, H_m1, H_m2)
        
        if M < 1e-9:
            return 1e9  # Invalid, return large value (minimizing -ratio, so large positive is bad)
            
        return -H_d / M

    # Initial guess: Uniform distribution
    # log_p = log(1/100)
    log_p0 = np.full(n * n, -np.log(n * n))
    
    # Run optimization
    best_p = np.ones(n * n) / (n * n)
    best_score = -objective(log_p0) # D/M for uniform
    
    # Try multiple restarts with different initializations
    # Uniform is a local max/min? 
    # Let's try a few random perturbations or structured ones.
    
    # Strategy: 
    # 1. Uniform
    # 2. Diagonal heavy (x=y) -> increases H(x-y)? No, x-y=0 becomes high prob, lowering H.
    #    We want H(x-y) to be HIGH. So we want x-y to be spread out.
    #    Uniform x-y on 0..18? No, x-y ranges -9..9.
    #    Uniform P gives x-y distribution: 1,2,...,10,9,...,1 (triangular).
    #    To maximize H(x-y), we want the triangular distribution to be flatter.
    #    This means increasing probabilities of extreme diagonals (x-y = +/-9) and decreasing center (x-y=0).
    #    
    #    What about M? 
    #    H(x) for uniform is 3.32 (10 values).
    #    H(x+y) for uniform: values 0..18, triangular. H is lower than 3.32?
    #    Let's check: 19 values. Max entropy is log2(19) ~ 4.25.
    #    Triangular entropy is approx 3.5-3.7?
    #    H(x+2y): values 0..27. 28 values. Triangular-ish.
    
    # Let's try an initialization that favors the "corners" of the grid to flatten x-y.
    # P[y, x] ~ exp(-alpha * (x-y)^2) ? No, that concentrates on x=y.
    # We want to INCREASE x-y variance.
    # P[y, x] ~ exp(alpha * (x-y)^2) ?
    
    # Let's try a few specific initializations:
    inits = []
    # 1. Uniform
    inits.append(np.full(n * n, -np.log(n * n)))
    
    # 2. Corner heavy (maximize |x-y|)
    # log_p = alpha * (x-y)^2
    alpha = 0.5
    log_p_corner = alpha * (xs - ys).flatten()**2
    log_p_corner -= np.max(log_p_corner) # normalize
    inits.append(log_p_corner)
    
    # 3. Anti-corner (minimize |x-y|, just to see)
    log_p_anti = -alpha * (xs - ys).flatten()**2
    log_p_anti -= np.max(log_p_anti)
    inits.append(log_p_anti)
    
    # 4. Random
    np.random.seed(42)
    for _ in range(3):
        log_p_rand = np.random.randn(n * n)
        log_p_rand -= np.max(log_p_rand)
        inits.append(log_p_rand)
        
    # Optimization loop
    for i, lp_init in enumerate(inits):
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                objective, 
                lp_init, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9}
            )
            
            # Extract best p from result
            lp_res = res.x
            lp_shifted = lp_res - np.max(lp_res)
            p_res = np.exp(lp_shifted)
            p_res = p_res / np.sum(p_res)
            
            # Evaluate
            score = -objective(lp_res)
            if score > best_score:
                best_score = score
                best_p = p_res
                
        except Exception:
            pass
            
    # Final check and return
    # Reshape to 10x10
    P = best_p.reshape((n, n))
    
    # Ensure valid
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)
        
    return P

# EVOLVE-BLOCK-END
