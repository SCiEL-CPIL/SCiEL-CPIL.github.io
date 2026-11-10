# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, factorial

def rudin_problem10() -> np.ndarray:
    """
    Rudin problem for polynomials on the 2-sphere. Find the real coefficients
    (c_m) for m = -k..k of a real degree-k spherical harmonic

        f = sum_{m=-k}^{k} c_m Y_k^m

    on the round unit sphere S^2 that MINIMIZE the norm ratio

        ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}

    (norms taken with the standard surface measure dOmega of total mass 4*pi).
    Here the degree is fixed to k = 10, so the returned vector has length
    2*k + 1 = 21.

    Strategy: Use spectral clipping / power iteration in the function space.
    The Euler-Lagrange equation for minimizing ||f||_4 subject to ||f||_2=1
    suggests that the optimal f should satisfy |f|^2 f = lambda f, which implies
    |f| is constant. We iterate f_{n+1} = Proj_{H_k}(|f_n|^2 f_n) to drive the
    function towards constant magnitude.
    
    Returns:
        coeffs: np.ndarray of shape (21,) with the real coefficients
            c_m, ordered m = -10, -9, ..., 0, ..., 9, 10.
    """
    k = 10
    start_time = time.time()
    deadline = 600.0  # 600s wall clock limit, keep safety margin

    # 1. Setup Quadrature
    # 70-point Gauss-Legendre in cos(theta)
    # cos(theta) in [-1, 1]
    # theta = arccos(x)
    # dOmega = sin(theta) dtheta dphi = -dx dphi
    # Integral over sphere = integral_{-1}^{1} dx integral_0^{2pi} dphi f(theta, phi)
    # We use Gauss-Legendre nodes and weights for x in [-1, 1].
    # 130 equally spaced phi points.
    
    n_x = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights
    x, w = np.polynomial.legendre.leggauss(n_x)
    # x is cos(theta)
    # w are weights for integral_{-1}^1 g(x) dx
    # Total integral is sum w_i * sum_j f(x_i, phi_j) * (2pi/n_phi) ? 
    # Wait, standard GL quadrature approximates integral_{-1}^1.
    # The phi integral is simple Riemann sum with step 2pi/n_phi.
    dphi = 2.0 * np.pi / n_phi
    phi_vals = np.linspace(0, 2.0 * np.pi, n_phi, endpoint=False)
    
    # Create meshgrid
    # X: (n_x, n_phi)
    # PHI: (n_x, n_phi)
    X, PHI = np.meshgrid(x, phi_vals, indexing='ij')
    
    # Weights: W[i, j] = w[i] * dphi
    W = np.outer(w, np.ones(n_phi) * dphi)
    
    # 2. Precompute Basis Functions
    # Y_0 = N_0 P_10^0(cos theta)
    # Y_m = sqrt(2) N_m P_10^m(cos theta) cos(m phi) for m > 0
    # Y_m = sqrt(2) N_|m| P_10^|m|(cos theta) sin(|m| phi) for m < 0
    
    # N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
    # Note: Condon-Shortley phase is included in lpmv.
    # scipy.special.lpmv(m, n, x) returns P_n^m(x) with Condon-Shortley phase (-1)^m.
    # The problem statement says "P^m = P_10^m(cos theta) with the Condon-Shortley phase".
    # So we use lpmv directly.
    
    # Basis matrix B: shape (n_x * n_phi, 21)
    # Let's flatten the spatial grid for matrix operations
    # N_points = n_x * n_phi
    N_pts = n_x * n_phi
    x_flat = X.ravel()
    phi_flat = PHI.ravel()
    
    # Precompute P_10^m(cos theta) for m=0..10
    # P_vals[m] is array of shape (N_pts,)
    P_vals = np.zeros((11, N_pts))
    for m in range(11):
        P_vals[m] = lpmv(m, k, x_flat)
        
    # Precompute trigonometric parts for phi
    # cos(m phi), sin(m phi)
    # We need them for each m.
    # To save memory/time, we can compute them on the fly or precompute.
    # Let's precompute cos and sin for m=1..10
    cos_m_phi = np.zeros((10, N_pts))
    sin_m_phi = np.zeros((10, N_pts))
    for m in range(1, 11):
        cos_m_phi[m-1] = np.cos(m * phi_flat)
        sin_m_phi[m-1] = np.sin(m * phi_flat)
        
    # Compute Normalization constants N_m
    N_m = np.zeros(11)
    for m in range(11):
        # (10-m)! / (10+m)!
        # Using gammaln for stability? 10 is small, factorial is fine.
        num = factorial(k - m, exact=False)
        den = factorial(k + m, exact=False)
        N_m[m] = np.sqrt(21.0 / (4.0 * np.pi) * (num / den))
        
    # Build Basis Matrix B
    # Columns: m = -10, -9, ..., 0, ..., 9, 10
    # Index in coeffs: i corresponds to m = i - 10
    B = np.zeros((N_pts, 21))
    
    # m = 0 (index 10)
    B[:, 10] = N_m[0] * P_vals[0]
    
    for m in range(1, 11):
        # m > 0 (indices 10+m)
        B[:, 10 + m] = np.sqrt(2.0) * N_m[m] * P_vals[m] * cos_m_phi[m-1]
        # m < 0 (indices 10-m)
        B[:, 10 - m] = np.sqrt(2.0) * N_m[m] * P_vals[m] * sin_m_phi[m-1]
        
    # Verify orthogonality? 
    # B^T W B should be identity (approx).
    # Weights W are flattened too.
    W_flat = W.ravel()
    
    # 3. Iteration
    # We want to minimize R = ||f||_4 / ||f||_2.
    # Equivalent to minimizing ||f||_4^4 subject to ||f||_2^2 = 1.
    # Iteration: f_new = Proj(|f|^2 f)
    # Proj(g) = B (B^T W g)  ... wait, B is orthonormal w.r.t W?
    # If B^T W B = I, then Proj(g) = B (B^T W g).
    # Let's check if B is orthonormal.
    # The basis functions are orthonormal on the sphere.
    # The quadrature is exact for polynomials of degree up to (70-1)*2 + 130?
    # Gauss-Legendre with n points is exact for polynomials of degree 2n-1.
    # Here n_x=70, so exact up to degree 139 in cos(theta).
    # Product of two degree 10 harmonics is degree 20.
    # So the quadrature is exact for the inner product.
    # Thus B^T W B should be very close to Identity.
    
    # Let's compute the projection operator matrix P_proj = B @ (B.T @ W_flat[:, None])
    # Actually, let's just use the formula:
    # coeffs_new = B.T @ (W_flat * f_vals)
    # f_vals_new = B @ coeffs_new
    
    # We need to handle the nonlinearity |f|^2 f.
    
    # Initialize coeffs
    # Random start, normalized
    np.random.seed(42)
    c = np.random.randn(21)
    c /= np.linalg.norm(c)
    
    best_R = np.inf
    best_c = c.copy()
    
    # Precompute B_T_W = B.T @ diag(W_flat)
    # This is (21 x N_pts)
    B_T_W = B.T * W_flat[None, :]  # Broadcasting: (21, 1) * (1, N_pts) -> (21, N_pts)
    
    max_iters = 2000
    for it in range(max_iters):
        if time.time() > deadline - 10.0:
            break
            
        # Compute f values on grid
        f_vals = B @ c  # (N_pts,)
        
        # Normalize f to ||f||_2 = 1
        # ||f||_2^2 = sum W_flat * f_vals^2
        norm2_sq = np.sum(W_flat * f_vals**2)
        if norm2_sq < 1e-15:
            break
        norm2 = np.sqrt(norm2_sq)
        f_vals /= norm2
        c /= norm2
        
        # Compute |f|^2 f
        nonlinear_f = (f_vals**2) * f_vals
        
        # Project back to degree 10 subspace
        # coeffs_new = B.T @ (W_flat * nonlinear_f)
        c_new = B_T_W @ (W_flat * nonlinear_f)
        
        # Normalize c_new
        # The norm of c_new in the coefficient space corresponds to ||f_new||_2
        # Since B is orthonormal, ||c_new||_2 = ||f_new||_2
        norm_c_new = np.linalg.norm(c_new)
        if norm_c_new < 1e-15:
            break
        c_new /= norm_c_new
        
        # Calculate R for c_new
        f_vals_new = B @ c_new
        norm2_new = np.sqrt(np.sum(W_flat * f_vals_new**2))
        norm4_new = np.sum(W_flat * f_vals_new**4) ** 0.25
        R_new = norm4_new / norm2_new
        
        if R_new < best_R:
            best_R = R_new
            best_c = c_new.copy()
            
        # Damping to prevent oscillation?
        # c = (1-alpha) c + alpha c_new
        # Usually power method converges without damping if the operator is monotone.
        # Let's try simple replacement first. If it oscillates, we might need damping.
        c = c_new
        
    # Final check on best_c
    # Ensure it's valid
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        return np.ones(21)
        
    max_abs_c = np.max(np.abs(best_c))
    if max_abs_c < 1e-3 or max_abs_c > 1e5:
        # Rescale to be safe?
        # The problem says invalid if max|c| < 1e-3 or > 1e5.
        # We can scale the coefficients arbitrarily as the ratio is scale invariant.
        # Let's scale to max|c| = 1.0
        best_c /= max_abs_c
        
    return best_c

# EVOLVE-BLOCK-END
