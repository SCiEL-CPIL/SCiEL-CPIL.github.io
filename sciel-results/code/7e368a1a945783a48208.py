# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
import math

def rudin_problem10() -> np.ndarray:
    """
    Returns the coefficients for the 'flattest' degree-10 spherical harmonic.
    
    Strategy:
    We hypothesize that the flattest harmonic is a mixture of the maximum order 
    (m=10, equatorial ring) and potentially the zonal (m=0) or other modes.
    The m=10 mode is already quite flat. Adding a small amount of m=0 might 
    "fill in" the equator or smooth out variations, or conversely, the m=10 
    mode alone might be optimal. 
    
    Instead of generic gradient descent, we perform a structured search:
    1. Define a subspace spanned by specific harmonics (e.g., m=0, m=10, m=-10).
    2. Parametrize the coefficients in this subspace.
    3. Perform a fine-grained grid search over the mixing parameters to minimize 
       the L4/L2 ratio.
    4. Use the best result from the grid as a starting point for a local 
       refinement using Nelder-Mead or L-BFGS-B on the full 21-dim space 
       (or restricted subspace) to catch any fine structure.
    
    Given the symmetry, we can restrict to real coefficients.
    The ratio R = ||f||_4 / ||f||_2.
    Note that R is scale-invariant. We can fix the norm of f to 1 (L2 norm = 1) 
    and minimize ||f||_4.
    
    We will search over a combination of m=0, m=10, and m=-10.
    Let c_0 be coeff for m=0.
    Let c_10 be coeff for m=10.
    Let c_-10 be coeff for m=-10.
    Due to symmetry of the sphere and the real harmonics, mixing c_10 and c_-10 
    with equal magnitude creates a standing wave (cos(10phi) or sin(10phi)).
    Actually, Y_10 is proportional to cos(10phi) and Y_-10 to sin(10phi).
    A linear combination a*Y_10 + b*Y_-10 is equivalent to Y_10 rotated in phi.
    So the relative phase between m=10 and m=-10 doesn't change the distribution 
    on the sphere (just a rotation). We can set b=0 and vary a.
    So we search over (c_0, c_10) with c_-10=0 (WLOG due to rotational symmetry 
    around z-axis? No, the problem fixes the basis. But the ratio is invariant 
    under rotation of phi. So any mix of m=10 and m=-10 is equivalent to a single 
    m=10 term with a different amplitude/phase. So we can fix c_-10=0 and vary 
    c_10).
    
    Wait, the basis is fixed. But the value R depends on the function f.
    f(theta, phi) = c_0 Y_0 + c_10 Y_10 + c_-10 Y_-10.
    Y_10 ~ cos(10phi), Y_-10 ~ sin(10phi).
    c_10 Y_10 + c_-10 Y_-10 = R_10 cos(10phi - phi_0).
    This is just a rotation in phi. The integral over the sphere is invariant 
    under phi rotation. So we can set c_-10 = 0 and search over c_0, c_10.
    
    We will also consider adding m=5 or other modes if the 2D search (c0, c10) 
    doesn't yield a strong improvement, but let's start with the most likely candidates 
    for "flattening": the ring (m=10) and the pole (m=0).
    
    We perform a grid search on c_0 and c_10.
    We normalize the vector to have L2 norm 1.
    Then we compute L4 norm.
    
    To be safe and potentially find a better solution, we will also run a local 
    optimization (L-BFGS-B) on the full 21-dimensional space starting from the 
    best grid point.
    """
    
    start_time = time.time()
    deadline = 600.0
    time_margin = 10.0
    
    L = 10
    N_THETA = 70
    N_PHI = 130
    
    # Precompute quadrature
    # Gauss-Legendre nodes and weights for [-1, 1]
    u, w = np.polynomial.legendre.leggauss(N_THETA)
    
    # Phi nodes
    phi = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
    dphi = 2 * np.pi / N_PHI
    
    # Create meshgrid for u and phi
    U, PHI = np.meshgrid(u, phi, indexing='ij')
    
    # Weights for the sphere integral
    # dOmega = dphi * du
    # Integral = sum w_i * dphi * f(u_i, phi_j)
    W = np.outer(w, np.ones(N_PHI)) * dphi
    
    # Function to compute L4/L2 ratio for a given coefficient vector c
    def compute_ratio(c):
        # c is shape (21,)
        # Check validity
        if np.any(np.isnan(c)) or np.any(np.isinf(c)):
            return np.inf
        if np.max(np.abs(c)) < 1e-3 or np.max(np.abs(c)) > 1e5:
            return np.inf
        
        # Construct f on the grid
        f = np.zeros((N_THETA, N_PHI), dtype=np.float64)
        
        for i in range(21):
            m = i - 10
            if c[i] == 0.0:
                continue
                
            # Compute P_10^m(u)
            # lpmv(m, l, x) returns P_l^m(x) with Condon-Shortley phase
            # Note: scipy lpmv handles the phase.
            P = lpmv(abs(m), L, U)
            
            # Normalization N_m
            # N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
            # For m=0, factor is 1. For m!=0, factor is sqrt(2).
            # Wait, the definition in the prompt:
            # Y_0 = N_0 P^0
            # Y_m = sqrt(2) N_m P^m cos(m phi) for m > 0
            # Y_m = sqrt(2) N_|m| P^|m| sin(|m| phi) for m < 0
            # N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
            
            # Let's compute N_m
            # (10-m)! / (10+m)!
            # Using math.factorial
            num = math.factorial(10 - abs(m))
            den = math.factorial(10 + abs(m))
            N_m = math.sqrt(21.0 / (4.0 * np.pi) * (num / den))
            
            if m == 0:
                Y = N_m * P
            elif m > 0:
                Y = math.sqrt(2.0) * N_m * P * np.cos(m * PHI)
            else:
                Y = math.sqrt(2.0) * N_m * P * np.sin(abs(m) * PHI)
                
            f += c[i] * Y
            
        # Compute norms
        # L2 norm squared: sum W * f^2
        l2_sq = np.sum(W * f**2)
        if l2_sq <= 0:
            return np.inf
        l2 = np.sqrt(l2_sq)
        
        # L4 norm: (sum W * f^4)^(1/4)
        l4 = np.sum(W * f**4) ** 0.25
        
        return l4 / l2

    # Phase 1: Structured Grid Search
    # We search over c_0 (index 10) and c_10 (index 20).
    # We set other coefficients to 0 initially.
    # We normalize the vector to L2 norm 1 before evaluation? 
    # The ratio is scale invariant. 
    # Let's just search over angles or relative weights.
    # Let c_10 = 1.0. Search c_0 in [-1, 1].
    # Also search relative weight of m=10 vs m=-10? 
    # As argued, m=10 and m=-10 mix is rotationally invariant in phi, 
    # so we can fix c_-10 = 0 and vary c_10.
    
    best_ratio = np.inf
    best_c = np.zeros(21)
    
    # Grid for c_0
    c_0_vals = np.linspace(-1.0, 1.0, 41)
    # Grid for c_10
    c_10_vals = np.linspace(-1.0, 1.0, 41)
    
    # We also consider that maybe m=10 alone is best, or m=0 alone.
    # The grid covers these.
    
    # To speed up, we can vectorize the computation of f for multiple c vectors.
    # But 41*41 = 1681 evaluations. Each eval involves 21 loops and lpmv.
    # lpmv is fast. 1681 * 21 * (70*130) ops is small.
    # However, python loops are slow.
    # Let's precompute the basis functions Y_m on the grid.
    
    Y_basis = np.zeros((21, N_THETA, N_PHI), dtype=np.float64)
    
    for i in range(21):
        m = i - 10
        P = lpmv(abs(m), L, U)
        
        num = math.factorial(10 - abs(m))
        den = math.factorial(10 + abs(m))
        N_m = math.sqrt(21.0 / (4.0 * np.pi) * (num / den))
        
        if m == 0:
            Y_basis[i] = N_m * P
        elif m > 0:
            Y_basis[i] = math.sqrt(2.0) * N_m * P * np.cos(m * PHI)
        else:
            Y_basis[i] = math.sqrt(2.0) * N_m * P * np.sin(abs(m) * PHI)
            
    # Now compute ratio for a batch of coefficients
    def compute_ratio_batch(c_batch):
        # c_batch: (N, 21)
        # f: (N, N_THETA, N_PHI)
        # Using einsum: f = sum_k c_batch[:, k] * Y_basis[k]
        # f = np.einsum('nk,kij->nij', c_batch, Y_basis)
        f = np.einsum('nk,kij->nij', c_batch, Y_basis)
        
        l2_sq = np.sum(W[None, :, :] * f**2, axis=(1, 2))
        l2 = np.sqrt(np.maximum(l2_sq, 1e-30))
        
        l4 = np.sum(W[None, :, :] * f**4, axis=(1, 2)) ** 0.25
        
        ratios = l4 / l2
        return ratios

    # Generate grid points
    c0_grid, c10_grid = np.meshgrid(c_0_vals, c_10_vals, indexing='ij')
    
    # We will test two families:
    # 1. c_0, c_10, others 0
    # 2. c_0, c_10, c_5 (maybe m=5 helps?) - Let's stick to 2D first for speed.
    
    c_batch = np.zeros((len(c0_grid), 21))
    c_batch[:, 10] = c0_grid.flatten()
    c_batch[:, 20] = c10_grid.flatten()
    
    # Filter out zero vectors
    norms_sq = np.sum(c_batch**2, axis=1)
    valid_mask = norms_sq > 1e-12
    
    if np.any(valid_mask):
        c_valid = c_batch[valid_mask]
        ratios = compute_ratio_batch(c_valid)
        
        # Find best
        min_idx = np.argmin(ratios)
        best_ratio = ratios[min_idx]
        best_c = c_valid[min_idx]
        
        # Check time
        if time.time() > start_time + deadline - time_margin:
            # Return best found so far
            # Ensure validity constraints
            max_c = np.max(np.abs(best_c))
            if max_c < 1e-3:
                best_c *= (1e-3 / max_c)
            elif max_c > 1e5:
                best_c *= (1e5 / max_c)
            return best_c

    # Phase 2: Local Optimization
    # Use L-BFGS-B on the full 21-dim space starting from best_c.
    # Objective: R(c)
    # We can use scipy.optimize.minimize
    
    def objective(c):
        if time.time() > start_time + deadline - time_margin:
            return best_ratio # Return a value that doesn't trigger further search
        return compute_ratio(c)
    
    # Gradient is expensive to compute analytically for L4/L2.
    # We can use numerical gradient or just let L-BFGS-B estimate it.
    # Given the small dimension (21), numerical gradient is fine.
    
    # Initial guess
    x0 = best_c.copy()
    
    # Constraints:
    # max|c| between 1e-3 and 1e5.
    # We can handle this by clipping or using bounds.
    # Bounds: [-1e5, 1e5] for each c.
    # But we also need max|c| >= 1e-3. This is a non-convex constraint.
    # However, if we start with a valid point and the optimizer stays in a reasonable 
    # range, it should be fine.
    # To be safe, we can normalize c to unit L2 norm during optimization?
    # No, L-BFGS-B works on the vector c.
    # Let's just use bounds [-1e5, 1e5].
    
    bounds = [(-1e5, 1e5) for _ in range(21)]
    
    try:
        result = scipy.optimize.minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success:
            c_opt = result.x
            r_opt = compute_ratio(c_opt)
            
            if r_opt < best_ratio:
                best_ratio = r_opt
                best_c = c_opt
    except Exception as e:
        pass # Keep best from grid search
        
    # Final validity check
    max_c = np.max(np.abs(best_c))
    if max_c < 1e-3:
        if max_c > 0:
            best_c *= (1e-3 / max_c)
        else:
            # Fallback to m=10
            best_c = np.zeros(21)
            best_c[20] = 1.0
    elif max_c > 1e5:
        best_c *= (1e5 / max_c)
        
    return best_c

# EVOLVE-BLOCK-END
