# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the ratio D/M where:
    D = H(x - y)
    M = max(H(x), H(x + y), H(x + 2y))
    
    Strategy:
    1. Precompute projection indices for x, y, x-y, x+y, x+2y.
    2. Use a structured parameterization: P[x, y] = exp(a*x + b*y + c*(x-y)^2) normalized.
       This family captures correlations that affect x-y spread vs x, x+y, x+2y entropies.
       Actually, a more flexible family: P[x, y] = exp(u_x[x] + u_y[y] + v[x,y]) is too large.
       Let's use a bilinear + quadratic model:
       log P[x, y] = A + B*x + C*y + D*x*y + E*x^2 + F*y^2
       This has 6 parameters. It allows for skewness and correlation.
       
       Alternative: Direct optimization of the 100 probabilities is hard.
       Let's try optimizing the 100 log-probs using Differential Evolution, 
       but DE on 100 dimensions is slow.
       
       Better Strategy: 
       The uniform distribution gives score ~1.0.
       The current best is ~1.98.
       
       Let's analyze the structure.
       H(x-y) is maximized when x-y is uniform over -9..9.
       H(x) is maximized when x is uniform over 0..9.
       
       If P is uniform, x-y is triangular (peak at 0). H(x-y) < log2(19).
       H(x) = log2(10).
       
       To increase D/M, we want to flatten x-y (increase D) without increasing M too much.
       Flattening x-y means putting mass on corners (x=0, y=9) and (x=9, y=0).
       This increases H(x) and H(y) potentially, but H(x) is already maxed at log2(10) if x is uniform.
       If we put mass on corners, x is still uniform (each x appears in 2 corners? No, x=0 has (0,0) and (0,9)).
       
       Let's use a parameterized family that allows "corner" weighting.
       P[x, y] proportional to exp( alpha * (x - 4.5) * (y - 4.5) ) ?
       This creates correlation.
       
       Let's try a more general approach: 
       Use Differential Evolution on the 100 probabilities directly, but with a tight time limit.
       DE is stochastic and global. 
       
       However, 100 dimensions is high for DE in 600s.
       
       Alternative: 
       Grid search over a low-dimensional manifold of distributions.
       Manifold: P[x, y] = exp( sum_{i,j} w_i w_j K(x,y) )?
       
       Let's stick to the 6-parameter quadratic model:
       log P[x, y] = c0 + c1*x + c2*y + c3*x*y + c4*x^2 + c5*y^2
       
       This allows:
       - c3: correlation (positive c3 favors diagonal x~y, negative favors anti-diagonal)
       - c4, c5: curvature in x and y (concentration vs spread)
       
       We will optimize these 6 parameters using Differential Evolution.
    """
    n = 10
    deadline = time.time() + 580.0  # Safety margin
    
    # Grid coordinates
    # P[y, x] -> y is row, x is col
    ys, xs = np.mgrid[0:n, 0:n]
    
    # Flatten for binning
    d_val = (xs - ys).flatten()
    m0_val = xs.flatten()
    m1_val = (xs + ys).flatten()
    m2_val = (xs + 2 * ys).flatten()
    
    # Unique values and inverse mapping
    d_vals, d_inv = np.unique(d_val, return_inverse=True)
    m0_vals, m0_inv = np.unique(m0_val, return_inverse=True)
    m1_vals, m1_inv = np.unique(m1_val, return_inverse=True)
    m2_vals, m2_inv = np.unique(m2_val, return_inverse=True)
    
    def compute_entropies_from_p(p_flat):
        """
        p_flat: shape (100,) normalized probabilities
        Returns: D, M
        """
        p_d = np.bincount(d_inv, weights=p_flat, minlength=len(d_vals))
        p_m0 = np.bincount(m0_inv, weights=p_flat, minlength=len(m0_vals))
        p_m1 = np.bincount(m1_inv, weights=p_flat, minlength=len(m1_vals))
        p_m2 = np.bincount(m2_inv, weights=p_flat, minlength=len(m2_vals))
        
        def shannon(p_dist):
            p_nonzero = p_dist[p_dist > 1e-15]
            if len(p_nonzero) == 0:
                return 0.0
            return -np.sum(p_nonzero * np.log2(p_nonzero))
        
        H_d = shannon(p_d)
        H_m0 = shannon(p_m0)
        H_m1 = shannon(p_m1)
        H_m2 = shannon(p_m2)
        
        M = max(H_m0, H_m1, H_m2)
        return H_d, M

    def objective(params):
        """
        params: [c0, c1, c2, c3, c4, c5]
        log P[x, y] = c0 + c1*x + c2*y + c3*x*y + c4*x^2 + c5*y^2
        """
        if time.time() > deadline:
            return 1e9
            
        c0, c1, c2, c3, c4, c5 = params
        
        # Calculate log probabilities
        log_p = c0 + c1 * xs + c2 * ys + c3 * xs * ys + c4 * xs**2 + c5 * ys**2
        log_p_flat = log_p.flatten()
        
        # Normalize to get probabilities
        # Subtract max for stability
        log_p_flat -= np.max(log_p_flat)
        p_flat = np.exp(log_p_flat)
        sum_p = np.sum(p_flat)
        
        if sum_p < 1e-15:
            return 1e9
            
        p_flat /= sum_p
        
        H_d, M = compute_entropies_from_p(p_flat)
        
        if M < 1e-9:
            return 1e9
            
        return -H_d / M

    # Define bounds for the parameters
    # c0 doesn't matter for normalization, but keep it small to avoid overflow in exp before normalization
    # c1, c2, c3, c4, c5 control the shape.
    # Range: +/- 5 should be enough to cover corner-heavy and center-heavy distributions.
    bounds = [
        (-10, 10),  # c0 (irrelevant but part of vector)
        (-5, 5),    # c1 (linear x)
        (-5, 5),    # c2 (linear y)
        (-5, 5),    # c3 (interaction x*y)
        (-5, 5),    # c4 (quadratic x)
        (-5, 5)     # c5 (quadratic y)
    ]
    
    # Initial best: Uniform distribution
    # Uniform corresponds to c1=c2=c3=c4=c5=0
    init_params = np.zeros(6)
    best_score = -objective(init_params)
    best_params = init_params.copy()
    
    # Run Differential Evolution
    # DE is global, so it should find the best corner-heavy distribution.
    try:
        # Use a smaller population to fit in time, but multiple restarts if possible
        # maxiter=100, popsize=15 (default is 15)
        # 100 * 15 * 6 dimensions evaluations.
        # Each eval is fast (100x100 ops).
        
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=200, 
            popsize=20,
            tol=1e-6,
            init='sobol'  # Better initial sampling
        )
        
        if res.success or res.fun < -best_score:
            best_score = -res.fun
            best_params = res.x
            
    except Exception as e:
        pass
        
    # If we have time left, try a second run with different seed or bounds
    if time.time() < deadline - 100:
        try:
            # Expand bounds slightly
            bounds2 = [
                (-10, 10),
                (-8, 8),
                (-8, 8),
                (-8, 8),
                (-8, 8),
                (-8, 8)
            ]
            res2 = differential_evolution(
                objective, 
                bounds2, 
                seed=123,
                maxiter=150, 
                popsize=20,
                tol=1e-6,
                init='latinhypercube'
            )
            if res2.fun < -best_score:
                best_score = -res2.fun
                best_params = res2.x
        except Exception:
            pass

    # Convert best parameters back to P
    c0, c1, c2, c3, c4, c5 = best_params
    log_p = c0 + c1 * xs + c2 * ys + c3 * xs * ys + c4 * xs**2 + c5 * ys**2
    log_p_flat = log_p.flatten()
    log_p_flat -= np.max(log_p_flat)
    p_flat = np.exp(log_p_flat)
    p_flat /= np.sum(p_flat)
    
    P = p_flat.reshape((n, n))
    
    # Final validation
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)
        
    return P

# EVOLVE-BLOCK-END
