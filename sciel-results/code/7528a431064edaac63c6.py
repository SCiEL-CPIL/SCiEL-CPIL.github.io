# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.special as sp
from scipy.optimize import minimize

def compute_Y_matrix():
    """
    Precompute the matrix of real orthonormal spherical harmonics Y_m
    for l=10, m=-10..10 on the specified quadrature grid.
    Returns Y (N_theta * N_phi, 21) and weights w (N_theta * N_phi,).
    """
    l = 10
    N_theta = 70
    N_phi = 130
    
    # Gauss-Legendre nodes and weights for cos(theta)
    # x = cos(theta), dx = -sin(theta) dtheta
    # Integral over sphere: int_0^{2pi} dphi int_0^pi sin(theta) f(theta, phi) dtheta
    # = int_{-1}^1 dx int_0^{2pi} f(acos(x), phi) dphi
    x, wx = np.polynomial.legendre.leggauss(N_theta)
    # wx are weights for integral over x in [-1, 1]
    
    # Phi nodes
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    dphi = 2 * np.pi / N_phi
    
    # Total weights for the double integral
    # w_i = wx_i * dphi
    # We need to map this to the 2D grid
    # Let's create the full grid
    X, Phi = np.meshgrid(x, phi, indexing='ij')
    # Flatten
    x_flat = X.flatten()
    phi_flat = Phi.flatten()
    w_flat = np.tile(wx * dphi, N_phi)
    
    # Compute Y_m for each point
    # Y is a matrix of size (N_theta * N_phi, 21)
    N = N_theta * N_phi
    Y = np.zeros((N, 21))
    
    for idx_m, m in enumerate(range(-10, 11)):
        abs_m = abs(m)
        # Normalization factor N_m
        # N_m = sqrt( (2l+1)/(4pi) * (l-m)!/(l+m)! )
        # Note: The problem statement says N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
        # This is consistent with standard orthonormal harmonics.
        # Using log-gamma for stability
        log_norm = 0.5 * (np.log(21) - np.log(4 * np.pi) + sp.gammaln(10 - abs_m + 1) - sp.gammaln(10 + abs_m + 1))
        norm = np.exp(log_norm)
        
        # Associated Legendre polynomial P_l^m(x)
        # scipy.special.lpmv(m, l, x) includes Condon-Shortley phase (-1)^m
        P_lm = sp.lpmv(abs_m, l, x_flat)
        
        if m == 0:
            Y[:, idx_m] = norm * P_lm
        elif m > 0:
            Y[:, idx_m] = np.sqrt(2) * norm * P_lm * np.cos(m * phi_flat)
        else:
            Y[:, idx_m] = np.sqrt(2) * norm * P_lm * np.sin(abs_m * phi_flat)
            
    return Y, w_flat

def objective_and_grad(c, Y, w):
    """
    Compute the objective function g(c) = ||f||_4^4 / ||f||_2^4 and its gradient.
    Minimizing this is equivalent to minimizing ||f||_4 / ||f||_2.
    """
    f = Y @ c
    
    # S_2 = sum w_i f_i^2
    # S_4 = sum w_i f_i^4
    f2 = f * f
    f4 = f2 * f2
    
    S2 = np.dot(w, f2)
    S4 = np.dot(w, f4)
    
    # Avoid division by zero, though c=0 is not a valid solution (max|c| < 1e-3 check)
    # We assume S2 > 0
    g = S4 / (S2 * S2)
    
    # Gradient
    # dg/dc = (4 * Y.T @ (w * f^3) * S2^2 - S4 * 2 * S2 * Y.T @ (w * f) * 2) / S2^4
    # dg/dc = 4 * Y.T @ (w * f^3) / S2^2 - 8 * S4 * Y.T @ (w * f) / S2^3
    # dg/dc = (4 / S2^3) * ( S2 * Y.T @ (w * f^3) - 2 * S4 * Y.T @ (w * f) )
    
    term1 = Y.T @ (w * f * f2)  # Y.T @ (w * f^3)
    term2 = Y.T @ (w * f)       # Y.T @ (w * f)
    
    grad = (4.0 / (S2 * S2 * S2)) * (S2 * term1 - 2.0 * S4 * term2)
    
    return g, grad

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    # Precompute harmonics
    Y, w = compute_Y_matrix()
    
    # Initial guess: all ones
    c0 = np.ones(21)
    
    best_c = c0
    best_val = np.inf
    
    # We can try a few restarts if time permits
    # But first, let's do a solid optimization from the seed
    # The seed is all ones.
    
    # Option 1: Optimize from all ones
    res = minimize(
        lambda c: objective_and_grad(c, Y, w)[0],
        c0,
        jac=lambda c: objective_and_grad(c, Y, w)[1],
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
    )
    
    if res.success and res.fun < best_val:
        best_val = res.fun
        best_c = res.x
    
    # Check time
    if time.time() > deadline:
        return best_c

    # Option 2: Try random restarts to find better local minima
    # The landscape might have multiple local minima.
    # However, for this specific problem, the "flattest" harmonic is likely unique up to symmetry.
    # Let's try a few random starts to be safe.
    
    rng = np.random.default_rng(42)
    for i in range(5):
        if time.time() > deadline - 60.0:
            break
            
        c_rand = rng.normal(0, 1, 21)
        # Normalize to similar scale as seed to help convergence?
        # The objective is scale invariant, so scale doesn't matter for the final result,
        # but it can affect the path.
        
        res = minimize(
            lambda c: objective_and_grad(c, Y, w)[0],
            c_rand,
            jac=lambda c: objective_and_grad(c, Y, w)[1],
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_c = res.x
            
    return best_c

# EVOLVE-BLOCK-END
