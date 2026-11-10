# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv, factorial
import time

def rudin_problem10() -> np.ndarray:
    """
    Finds the real coefficients for a degree-10 spherical harmonic that minimizes
    the L4/L2 norm ratio.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s safety margin
    
    k = 10
    n_coeffs = 2 * k + 1
    ms = np.arange(-k, k + 1)
    
    # 1. Setup Quadrature
    # 70-point Gauss-Legendre in cos(theta)
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    # x_gl are cos(theta), w_gl are weights for integral over [-1, 1]
    
    # 130 equally spaced phi
    n_phi = 130
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    
    # Create meshgrid for evaluation
    # We want to evaluate at all (x_i, phi_j)
    # x is cos(theta)
    X, Phi = np.meshgrid(x_gl, phi, indexing='ij')
    X_flat = X.flatten()
    Phi_flat = Phi.flatten()
    W_flat = np.repeat(w_gl, n_phi) * (2 * np.pi / n_phi)
    
    # 2. Compute Spherical Harmonics Y_m at all points
    # Y_m shape: (n_points, 21)
    Y = np.zeros((X_flat.size, n_coeffs))
    
    # Precompute factorials for normalization
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    # Note: lpmv includes Condon-Shortley phase (-1)^m
    for idx, m in enumerate(ms):
        abs_m = abs(m)
        # Normalization constant
        # (10-abs_m)! / (10+abs_m)!
        # Using gamma or factorial
        num = factorial(10 - abs_m, exact=False)
        den = factorial(10 + abs_m, exact=False)
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * (num / den))
        
        # Associated Legendre Polynomial
        # lpmv(m, n, x) computes P_n^m(x) with Condon-Shortley phase
        P = lpmv(abs_m, 10, X_flat)
        
        if m == 0:
            Y[:, idx] = N_m * P
        elif m > 0:
            Y[:, idx] = np.sqrt(2.0) * N_m * P * np.cos(m * Phi_flat)
        else:
            Y[:, idx] = np.sqrt(2.0) * N_m * P * np.sin(abs_m * Phi_flat)
    
    # 3. Define Objective and Gradient
    # Minimize f(c) = (Integral f^4) / (Integral f^2)^2
    # This is equivalent to minimizing R^4.
    
    def objective_and_grad(c):
        # f = Y @ c
        f = Y @ c
        
        # Integrals
        # Int_f2 = sum(W * f^2)
        f2 = f * f
        Int_f2 = np.dot(W_flat, f2)
        
        # Int_f4 = sum(W * f^4)
        f4 = f2 * f2
        Int_f4 = np.dot(W_flat, f4)
        
        # Avoid division by zero
        if Int_f2 < 1e-12:
            return 1e10, np.zeros_like(c)
        
        val = Int_f4 / (Int_f2 * Int_f2)
        
        # Gradient
        # grad_val_k = 4 * (Int_f2 * Int_f2_k - Int_f4 * Int_f2_k) / Int_f2^4  <-- Wait, let's re-derive
        # val = N / D, N = Int_f4, D = Int_f2^2
        # dval/dck = (D * dN/dck - N * dD/dck) / D^2
        # dN/dck = 4 * sum(W * f^2 * Y_k)
        # d(Int_f2)/dck = 2 * sum(W * f * Y_k) = 2 * (Y.T @ (W * f))_k
        # dD/dck = 2 * Int_f2 * d(Int_f2)/dck = 4 * Int_f2 * (Y.T @ (W * f))_k
        
        # Let A_k = sum(W * f^2 * Y_k)
        # Let B_k = sum(W * f * Y_k)
        
        # Vectorized:
        # W_f2 = W_flat * f2
        # W_f = W_flat * f
        # A = Y.T @ (W_flat * f2)
        # B = Y.T @ (W_flat * f)
        
        A = Y.T @ (W_flat * f2)
        B = Y.T @ (W_flat * f)
        
        dN = 4.0 * A
        dD = 4.0 * Int_f2 * B
        
        grad = (dN * Int_f2 * Int_f2 - N_val_placeholder * dD) / (Int_f2 ** 4)
        # Wait, N_val is Int_f4
        grad = (dN * Int_f2**2 - Int_f4 * dD) / (Int_f2**4)
        
        return val, grad

    # Helper to run one optimization
    def run_optimization(c0):
        # Normalize c0
        norm = np.linalg.norm(c0)
        if norm < 1e-12:
            c0 = np.random.randn(n_coeffs)
            norm = np.linalg.norm(c0)
        c0 = c0 / norm
        
        res = minimize(
            objective_and_grad, 
            c0, 
            jac=True, 
            method='L-BFGS-B', 
            options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10}
        )
        return res.x, res.fun

    best_c = None
    best_val = np.inf
    
    # Restart loop
    n_restarts = 0
    while time.time() < deadline:
        # Random initialization
        # Mix of random and structured inits
        if n_restarts % 2 == 0:
            c0 = np.random.randn(n_coeffs)
        else:
            # Try to initialize with some symmetry or specific patterns
            # e.g. only m=0, or m=10, or alternating signs
            c0 = np.zeros(n_coeffs)
            c0[np.random.randint(0, n_coeffs)] = 1.0
            if np.random.rand() > 0.5:
                c0 = np.random.randn(n_coeffs)
        
        c_opt, val_opt = run_optimization(c0)
        
        if val_opt < best_val:
            best_val = val_opt
            best_c = c_opt.copy()
            
        n_restarts += 1
        
        # Check time more frequently after some restarts
        if n_restarts > 50 and (n_restarts % 5 == 0):
             if time.time() > deadline - 30:
                 break

    # If no optimization succeeded, return ones
    if best_c is None:
        return np.ones(n_coeffs)
    
    # Final check: Ensure validity
    # max|c| < 1e-3 or > 1e5 is invalid.
    # The optimization is scale invariant in objective, but L-BFGS might drift.
    # We should normalize the result to a reasonable scale.
    # The problem doesn't specify a scale, but the evaluator checks max|c|.
    # "max|c| < 1e-3 or max|c| > 1e5"
    # Let's normalize such that max|c| is around 1.0.
    max_c = np.max(np.abs(best_c))
    if max_c > 0:
        best_c = best_c / max_c * 1.0
        
    # Check if valid
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        return np.ones(n_coeffs)
    
    return best_c

# EVOLVE-BLOCK-END
