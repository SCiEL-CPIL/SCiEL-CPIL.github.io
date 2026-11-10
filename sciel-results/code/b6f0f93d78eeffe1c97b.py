# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv, factorial

def rudin_problem10() -> np.ndarray:
    """
    Finds the real coefficients c for m = -10..10 of a real degree-10 spherical harmonic
    f = sum_m c_m Y_m that minimize the ratio ||f||_4 / ||f||_2.
    
    Returns:
        coeffs: np.ndarray of shape (21,)
    """
    k = 10
    
    # 1. Setup Quadrature Grid
    # 70-point Gauss-Legendre in cos(theta)
    # 130 equally spaced phi
    N_theta = 70
    N_phi = 130
    
    # Gauss-Legendre nodes and weights on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(N_theta)
    
    # Map to theta: cos(theta) = x, so theta = arccos(x)
    # The integral over sphere is ∫_0^π ∫_0^{2π} f(θ,φ) sin(θ) dφ dθ
    # Change variable u = cos(θ), du = -sin(θ) dθ
    # ∫_{-1}^1 ∫_0^{2π} f(arccos(u), φ) dφ du
    # So weights for u are w_i, and we integrate over φ with step dφ = 2π/N_phi
    # Total weight for point (i, j) is w_i * (2π/N_phi)
    
    u = x
    theta = np.arccos(u)
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    
    # Create meshgrid
    # We want arrays of shape (N_theta, N_phi)
    U, P = np.meshgrid(u, phi, indexing='ij')
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    # Weights
    dphi = 2 * np.pi / N_phi
    W_grid = np.outer(w, np.full(N_phi, dphi))
    W_flat = W_grid.flatten()
    
    # 2. Precompute Spherical Harmonics Y_m(θ, φ)
    # Y_m definitions:
    # Y_0 = N_0 P_10^0(cos θ)
    # Y_m = sqrt(2) N_m P_10^m(cos θ) cos(m φ) for m > 0
    # Y_m = sqrt(2) N_|m| P_10^|m|(cos θ) sin(|m| φ) for m < 0
    # N_m = sqrt( (2k+1)/(4π) * (k-m)!/(k+m)! )
    
    m_vals = np.arange(-k, k + 1)
    Y_flat = np.zeros((N_theta * N_phi, 2 * k + 1))
    
    for idx, m in enumerate(m_vals):
        abs_m = abs(m)
        # Associated Legendre P_k^m(x) with Condon-Shortley phase
        # scipy.special.lpmv(m, n, x) includes (-1)^m
        P_val = lpmv(abs_m, k, U)
        
        # Normalization N_m
        # Factorial for large numbers might overflow, use gammaln or log
        # N_m = sqrt( (2k+1)/(4π) * (k-m)!/(k+m)! )
        log_num = np.log(2 * k + 1) - np.log(4 * np.pi) + np.log(factorial(k - abs_m)) - np.log(factorial(k + abs_m))
        N_m = np.sqrt(np.exp(log_num))
        
        if m == 0:
            Y_val = N_m * P_val
        elif m > 0:
            Y_val = np.sqrt(2) * N_m * P_val * np.cos(m * PHI)
        else:
            Y_val = np.sqrt(2) * N_m * P_val * np.sin(abs_m * PHI)
        
        Y_flat[:, idx] = Y_val.flatten()
    
    # 3. Define Objective and Gradient
    # Minimize J(c) = sum_k W_k (Y_k^T c)^4 subject to ||c||^2 = 1
    # We use the constraint c^T c = 1.
    
    def objective(c):
        f = Y_flat @ c
        # f^4 weighted sum
        return np.sum(W_flat * (f ** 4))
    
    def gradient(c):
        f = Y_flat @ c
        # d/dc sum W f^4 = 4 sum W f^3 Y = 4 Y^T (W * f^3)
        return 4.0 * (Y_flat.T @ (W_flat * (f ** 3)))
    
    def hessian(c):
        f = Y_flat @ c
        # H = 12 Y^T (W * f^2) Y
        # This is 21x21, Y is 9100x21.
        # (Y^T diag(W*f^2) Y)
        # Efficiently: (Y * (W*f^2)[:, None]).T @ Y
        w_f2 = W_flat * (f ** 2)
        # Multiply columns of Y by w_f2
        Y_weighted = Y_flat * w_f2[:, None]
        return 12.0 * (Y_weighted.T @ Y_flat)
    
    # 4. Optimization
    # Try several initial points
    best_c = None
    best_val = np.inf
    
    # Initial guesses
    inits = []
    # 1. All ones (normalized)
    c0 = np.ones(2 * k + 1)
    c0 /= np.linalg.norm(c0)
    inits.append(c0)
    
    # 2. Only m=0
    c1 = np.zeros(2 * k + 1)
    c1[k] = 1.0  # index for m=0 is k
    inits.append(c1)
    
    # 3. Only m=k
    c2 = np.zeros(2 * k + 1)
    c2[-1] = 1.0
    inits.append(c2)
    
    # 4. Only m=-k
    c3 = np.zeros(2 * k + 1)
    c3[0] = 1.0
    inits.append(c3)
    
    # 5. Random unit vectors
    rng = np.random.default_rng(42)
    for _ in range(5):
        cr = rng.standard_normal(2 * k + 1)
        cr /= np.linalg.norm(cr)
        inits.append(cr)
        
    # 6. Alternating signs for even m? 
    # The "flattest" harmonic often has specific parity.
    # Let's try a vector that is symmetric in m.
    c4 = np.zeros(2 * k + 1)
    for m in range(-k, k + 1, 2):
        c4[m + k] = 1.0
    c4 /= np.linalg.norm(c4)
    inits.append(c4)

    for c_init in inits:
        try:
            # Use SLSQP or Trust-Constr
            # SLSQP is good for small problems
            res = minimize(
                objective, 
                c_init, 
                method='SLSQP',
                jac=gradient,
                hess= hessian, # SLSQP accepts hess? No, SLSQP uses finite diff or BFGS approx. 
                # Let's use 'trust-constr' which accepts analytic hess
                # But trust-constr can be slow. 
                # Let's stick to SLSQP with analytic gradient. It's robust.
                constraints=[{'type': 'eq', 'fun': lambda c: np.dot(c, c) - 1.0, 'jac': lambda c: 2 * c}],
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success:
                val = res.fun
                if val < best_val:
                    best_val = val
                    best_c = res.x
        except Exception:
            pass
            
    # If optimization failed for all, return seed
    if best_c is None:
        best_c = np.ones(2 * k + 1)
        
    # Normalize to ensure validity (though constraint should handle it)
    best_c = best_c / np.linalg.norm(best_c)
    
    return best_c

# EVOLVE-BLOCK-END
