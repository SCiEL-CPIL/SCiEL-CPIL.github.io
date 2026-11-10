# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
import time

def _precompute_basis():
    """
    Precompute the real orthonormal spherical harmonics Y_m for degree 10, m=-10..10
    on a 70x130 quadrature grid.
    Returns:
        B: Matrix of shape (n_pts, 21) where B[i, j] = Y_j(theta_i, phi_i)
        weights: Array of shape (n_pts,) with quadrature weights
    """
    k = 10
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights in [-1, 1]
    x, w = np.polynomial.legendre.leggauss(n_theta)
    
    # Phi points
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    # Total points
    n_pts = n_theta * n_phi
    dim = 2 * k + 1
    
    # Precompute P_10^m(x) for m=0..10
    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = lpmv(m, k, x)
        
    # Normalization constants N_m
    N = np.zeros(11)
    for m in range(11):
        # N_m = sqrt((2k+1)/(4*pi) * (k-m)!/(k+m)!)
        # Calculate factorials directly for small k
        num = 1.0
        for i in range(1, k - m + 1):
            num *= i
        den = 1.0
        for i in range(1, k + m + 1):
            den *= i
        N[m] = np.sqrt((2 * k + 1) / (4 * np.pi) * (num / den))
        
    # Build B matrix
    # B[i, j] corresponds to point i and harmonic index j (m = j - 10)
    B = np.zeros((n_pts, dim))
    
    # Weights for quadrature: sum w_i * (2*pi/n_phi)
    weights = np.repeat(w, n_phi) * (2.0 * np.pi / n_phi)
    
    sqrt2 = np.sqrt(2.0)
    
    # m = 0 is at index 10
    B[:, 10] = np.repeat(N[0] * P[0, :], n_phi)
    
    for m in range(1, 11):
        idx_pos = m + 10  # index for +m
        idx_neg = -m + 10 # index for -m
        
        cos_term = np.cos(m * phi)
        sin_term = np.sin(m * phi)
        
        val_theta = sqrt2 * N[m] * P[m, :]
        
        # B shape is (n_theta, n_phi, dim) essentially, but we fill column by column
        # Point index i corresponds to theta_idx = i // n_phi, phi_idx = i % n_phi
        # It's more efficient to construct the full grid values
        
        # Create meshgrid for theta and phi
        # theta_idx = 0..n_theta-1, phi_idx = 0..n_phi-1
        # B_flat[i] where i = theta_idx * n_phi + phi_idx
        
        # For m > 0:
        # Y_m = sqrt(2) N_m P_m(cos theta) cos(m phi)
        # Y_-m = sqrt(2) N_m P_m(cos theta) sin(m phi)
        
        # Vectorized fill
        # val_theta is shape (n_theta,)
        # cos_term is shape (n_phi,)
        # We need to broadcast to (n_theta, n_phi) and ravel
        
        vals_pos = (val_theta[:, None] * cos_term[None, :]).ravel()
        vals_neg = (val_theta[:, None] * sin_term[None, :]).ravel()
        
        B[:, idx_pos] = vals_pos
        B[:, idx_neg] = vals_neg
        
    return B, weights

# Precompute at module load time
B_GLOBAL, W_GLOBAL = _precompute_basis()

def _compute_ratio(c):
    """
    Compute R = ||f||_4 / ||f||_2 for coefficients c.
    c has shape (21,).
    """
    f = B_GLOBAL @ c
    f2 = f * f
    norm2_sq = np.sum(W_GLOBAL * f2)
    
    if norm2_sq < 1e-30:
        return np.inf
        
    norm2 = np.sqrt(norm2_sq)
    norm4_4 = np.sum(W_GLOBAL * f2 * f2)
    norm4 = norm4_4 ** 0.25
    
    return norm4 / norm2

def _nelder_mead_minimize(func, x0, max_iter=2000, tol=1e-10, deadline=None):
    """
    Simple Nelder-Mead minimization.
    """
    n = len(x0)
    # Initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        xi[i] += 0.1 if xi[i] != 0 else 0.1
        simplex.append(xi)
    
    # Evaluate initial simplex
    fvals = [func(v) for v in simplex]
    
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5    # contraction
    sigma = 0.5  # shrink
    
    best_idx = np.argmin(fvals)
    best_val = fvals[best_idx]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 30:
            break
        
        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[-1] - fvals[0]) < tol * (1 + abs(fvals[0])):
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = func(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                # Outside contraction
                xc = centroid + rho * (simplex[-1] - centroid)
            else:
                # Inside contraction
                xc = centroid - rho * (simplex[-1] - centroid)
            fc = func(xc)
            
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = func(simplex[i])
        
        # Track best
        current_best = min(fvals)
        if current_best < best_val:
            best_val = current_best
            best_idx = fvals.index(current_best)
    
    return best_val, simplex[np.argmin(fvals)]

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c_m for m = -10..10 minimizing L4/L2 ratio of degree-10 spherical harmonic.
    Strategy:
    1. Test the pure zonal harmonic (m=0) which is physically expected to be the flattest.
    2. Test pure harmonics for m=1..10 to confirm m=0 is best among single modes.
    3. If m=0 is not the absolute best among singles (unlikely), use the best single as seed.
    4. Run a local optimization (Nelder-Mead) starting from the best candidate found in step 1-2.
    5. This exploits the symmetry/reduction: the search is effectively 1D (choice of mode) + local refinement.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    k = 10
    dim = 2 * k + 1
    
    # Candidate 1: Pure Zonal (m=0)
    c_zonal = np.zeros(dim)
    c_zonal[10] = 1.0  # m=0 corresponds to index 10
    
    # Candidate 2..11: Pure m=1..10
    best_val = np.inf
    best_coeffs = c_zonal.copy()
    
    # Evaluate zonal
    val_zonal = _compute_ratio(c_zonal)
    if val_zonal < best_val:
        best_val = val_zonal
        best_coeffs = c_zonal.copy()
        
    for m in range(1, 11):
        c_test = np.zeros(dim)
        c_test[m + 10] = 1.0 # +m
        val_test = _compute_ratio(c_test)
        if val_test < best_val:
            best_val = val_test
            best_coeffs = c_test.copy()
            
        # Also check -m (should be same ratio, but good to be sure)
        c_test_neg = np.zeros(dim)
        c_test_neg[-m + 10] = 1.0 # -m
        val_test_neg = _compute_ratio(c_test_neg)
        if val_test_neg < best_val:
            best_val = val_test_neg
            best_coeffs = c_test_neg.copy()

    # Now, refine from the best single-mode candidate using Nelder-Mead
    # This allows for small linear combinations if they yield a slightly lower ratio
    # due to interference effects (though unlikely for L4/L2 minimization)
    
    # We use a restricted Nelder-Mead to ensure we don't drift too far or waste time
    # If the zonal harmonic is indeed the global minimum, this step will quickly converge or stagnate.
    
    def objective(c):
        if not np.all(np.isfinite(c)):
            return np.inf
        return _compute_ratio(c)
    
    # Run Nelder-Mead
    # max_iter kept reasonable to ensure speed
    val_opt, c_opt = _nelder_mead_minimize(objective, best_coeffs, max_iter=1000, tol=1e-12, deadline=deadline)
    
    if val_opt < best_val:
        best_val = val_opt
        best_coeffs = c_opt.copy()

    # Final validation
    coeffs = best_coeffs.astype(np.float64)
    
    # Check constraints
    if len(coeffs) != 21:
        coeffs = c_zonal.copy()
    if not np.all(np.isfinite(coeffs)):
        coeffs = c_zonal.copy()
        
    max_c = np.max(np.abs(coeffs))
    if max_c < 1e-3 or max_c > 1e5:
        if max_c > 0:
            # Scale to 1.0 which is safely within [1e-3, 1e5]
            coeffs = coeffs / max_c * 1.0
        else:
            coeffs = c_zonal.copy()
            
    return coeffs

# EVOLVE-BLOCK-END
