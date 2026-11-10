import numpy as np
import scipy.optimize as opt
import time

def rudin_problem10() -> np.ndarray:
    k = 10
    n_m = 2 * k + 1
    m_vals = np.arange(-k, k + 1)
    
    # Quadrature grid
    # 70 points in cos(theta) -> theta in [0, pi]
    # 130 points in phi -> phi in [0, 2*pi)
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre quadrature for cos(theta)
    # We need nodes x and weights w for integral from -1 to 1
    x, w_gl = np.polynomial.legendre.leggauss(n_theta)
    # theta = arccos(x)
    theta = np.arccos(x)
    # Weights for dOmega = sin(theta) dtheta dphi = dx dphi
    # So the weight for each theta point is w_gl[i]
    # The weight for each phi point is 2*pi / n_phi
    w_phi = 2 * np.pi / n_phi
    
    # Create grid
    theta_grid, phi_grid = np.meshgrid(theta, np.linspace(0, 2 * np.pi, n_phi, endpoint=False))
    theta_flat = theta_grid.flatten()
    phi_flat = phi_grid.flatten()
    
    # Weights
    # w_gl is for integral over x in [-1, 1]. 
    # The surface measure is dOmega = sin(theta) dtheta dphi.
    # x = cos(theta) => dx = -sin(theta) dtheta.
    # So dOmega = -dx dphi. The integral over sphere is \int_0^{2pi} \int_{-1}^1 ... dx dphi.
    # So weight for each point (i, j) is w_gl[i] * w_phi.
    weights = np.tile(w_gl, (n_phi, 1)).flatten() * w_phi
    
    # Precompute basis functions Y_m(theta, phi)
    # Y_0 = N_0 P_10^0(cos theta)
    # Y_m = sqrt(2) N_m P_10^m(cos theta) cos(m phi) for m > 0
    # Y_m = sqrt(2) N_|m| P_10^|m|(cos theta) sin(|m| phi) for m < 0
    # N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
    
    # We need associated Legendre polynomials P_l^m(x) with Condon-Shortley phase.
    # scipy.special.lpmv(m, l, x) gives P_l^m(x) with (-1)^m factor.
    from scipy.special import lpmv, factorial
    
    cos_theta = np.cos(theta_flat)
    
    Basis = np.zeros((len(theta_flat), n_m))
    
    for idx, m in enumerate(m_vals):
        abs_m = abs(m)
        # Normalization constant
        # N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
        # Note: (10-m)!/(10+m)! is the same for m and -m.
        num = factorial(10 - abs_m, exact=False)
        den = factorial(10 + abs_m, exact=False)
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * (num / den))
        
        P = lpmv(abs_m, 10, cos_theta)
        
        if m == 0:
            Basis[:, idx] = N_m * P
        elif m > 0:
            Basis[:, idx] = np.sqrt(2.0) * N_m * P * np.cos(m * phi_flat)
        else: # m < 0
            Basis[:, idx] = np.sqrt(2.0) * N_m * P * np.sin(abs_m * phi_flat)
    
    # Objective function and gradient
    def objective_and_grad(c):
        f = Basis @ c
        f2 = f * f
        f4 = f2 * f2
        
        A = np.sum(weights * f4)
        B = np.sum(weights * f2)
        
        if B < 1e-12:
            return 1e10, np.zeros_like(c)
            
        Q = A / B
        
        # Gradients
        # dA/dc = 4 * Basis.T @ (weights * f^3)
        # dB/dc = 2 * Basis.T @ (weights * f)
        
        df3 = f * f2
        df = f
        
        grad_A = 4.0 * (Basis.T @ (weights * df3))
        grad_B = 2.0 * (Basis.T @ (weights * df))
        
        grad_Q = (grad_A * B - A * grad_B) / (B * B)
        
        return Q, grad_Q

    def objective(c):
        val, _ = objective_and_grad(c)
        return val

    def grad(c):
        _, g = objective_and_grad(c)
        return g

    # Optimization
    # Use L-BFGS-B
    # Multiple restarts to avoid local minima
    best_c = None
    best_val = np.inf
    
    # Time limit check
    start_time = time.time()
    time_limit = 600.0 # seconds
    
    # Try a few random starts
    # Also try the seed (all ones)
    starts = [np.ones(n_m)]
    rng = np.random.default_rng(42)
    for _ in range(5):
        starts.append(rng.normal(size=n_m))
    
    for c0 in starts:
        if time.time() - start_time > time_limit - 10:
            break
            
        res = opt.minimize(objective, c0, jac=grad, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success or res.fun < best_val:
            best_val = res.fun
            best_c = res.x
            
    # Normalize? The problem doesn't require normalization, but scale invariance means 
    # the ratio is the same. However, the validity check requires max|c| in [1e-3, 1e5].
    # L-BFGS-B might find a solution with very small or very large norm.
    # Let's rescale to have max|c| = 1.0 or something reasonable, e.g., norm 1.
    if best_c is not None:
        # Scale to ensure validity
        max_abs = np.max(np.abs(best_c))
        if max_abs < 1e-3:
            best_c = best_c / max_abs * 1.0
        elif max_abs > 1e5:
            best_c = best_c / max_abs * 1.0
        else:
            # Optional: normalize to unit L2 norm for consistency? 
            # The ratio is invariant, so any scale is fine as long as valid.
            # Let's just leave it, or normalize to unit vector for cleanliness.
            norm = np.linalg.norm(best_c)
            if norm > 0:
                best_c = best_c / norm
    
    # Final check for validity
    if best_c is None:
        best_c = np.ones(n_m)
        
    # Ensure no NaN or Inf
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(n_m)
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c * (1.0 / max_abs)
    elif max_abs > 1e5:
        best_c = best_c * (1.0 / max_abs)
        
    return best_c
