"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

def _hermite_polynomials(x):
    """
    Computes H0, H4, H8, H12 for input x.
    Uses explicit formulas for speed and accuracy.
    H0(x) = 1
    H4(x) = x^4 - 6x^2 + 3
    H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    H12(x) = x^12 - 54x^10 + 1080x^8 - 7560x^6 + 15120x^4 - 7560x^2 + 945
    """
    x2 = x**2
    x4 = x2**2
    x6 = x4 * x2
    x8 = x4**2
    x10 = x8 * x2
    x12 = x8 * x4
    
    H0 = np.ones_like(x)
    H4 = x4 - 6*x2 + 3
    H8 = x8 - 28*x6 + 210*x4 - 420*x2 + 105
    H12 = x12 - 54*x10 + 1080*x8 - 7560*x6 + 15120*x4 - 7560*x2 + 945
    
    return H0, H4, H8, H12

def _compute_objective(coeffs):
    """
    Computes the uncertainty product bound for the given coefficients.
    coeffs: [a0, a4, a8]
    Returns the value of J = (||g||_4^4 * ||g_hat||_4^4) / (||g||_2^8)
    """
    a0, a4, a8 = coeffs
    
    # Enforce g(0) = 0 to find a12
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=945
    # a0*1 + a4*3 + a8*105 + a12*945 = 0
    a12 = -(a0 + 3*a4 + 105*a8) / 945.0
    
    # Coefficients for g(x) = P(x) e^{-x^2/2}
    # P(x) = a0 H0 + a4 H4 + a8 H8 + a12 H12
    # Coefficients for g_hat(x) = Q(x) e^{-x^2/2}
    # Q(x) = a0 i^0 H0 + a4 i^4 H4 + a8 i^8 H8 + a12 i^12 H12
    # i^0 = 1, i^4 = 1, i^8 = 1, i^12 = i^8 * i^4 = 1 * 1 = 1
    # So Q(x) = P(x). Wait.
    # Let's check the phase.
    # F[H_n e^{-x^2/2}] = i^n H_n e^{-x^2/2}
    # n=0: 1
    # n=4: i^4 = 1
    # n=8: i^8 = 1
    # n=12: i^12 = 1
    # So g_hat(x) = g(x) in terms of polynomial coefficients?
    # This implies ||g||_4 = ||g_hat||_4.
    # If so, J = ||g||_4^8 / ||g||_2^8 = (||g||_4 / ||g||_2)^8.
    # Minimizing J is equivalent to minimizing ||g||_4 / ||g||_2.
    
    # Let's verify the Fourier transform convention.
    # If the transform is unitary, the L4 norm is preserved only if the function is an eigenfunction.
    # Here g is a sum of eigenfunctions.
    # g(x) = sum c_n phi_n(x).
    # g_hat(x) = sum c_n i^n phi_n(x).
    # If all i^n are 1, then g_hat = g.
    # This happens for n divisible by 4.
    # So for this specific combination of H0, H4, H8, H12, the function is invariant under Fourier transform (up to sign/phase which is 1 here).
    # Thus ||g||_4 = ||g_hat||_4 and ||g||_2 = ||g_hat||_2.
    # The objective simplifies to (||g||_4^4 / ||g||_2^4)^2?
    # The inequality is A(f) A(F) >= C.
    # A(f) = ||f||_4 / ||f||_2^{3/2}.
    # Product = (||f||_4 / ||f||_2^{3/2}) (||F||_4 / ||F||_2^{3/2}).
    # If f is FT-invariant, Product = (||f||_4^2 / ||f||_2^3)^2 = (||f||_4^4 / ||f||_2^6).
    # Let's minimize this quantity.
    
    # Setup Gauss-Hermite Quadrature
    # We need to integrate P(x)^4 e^{-2x^2} and P(x)^2 e^{-x^2}.
    # Using scaled GH:
    # Integrand for L4: |P(x)|^4 e^{-2x^2}.
    # Let x = t/sqrt(2). dx = dt/sqrt(2).
    # Integral = 1/sqrt(2) * Int |P(t/sqrt(2))|^4 e^{-t^2} dt.
    # Integrand for L2: |P(x)|^2 e^{-x^2}.
    # Integral = Int |P(x)|^2 e^{-x^2} dx.
    
    # We use N=50 nodes for safety (degree 48 needs N>=25).
    N = 50
    nodes, weights = np.polynomial.hermite.hermgauss(N)
    
    # Evaluate P(x) at nodes for L2 (x = nodes)
    H0, H4, H8, H12 = _hermite_polynomials(nodes)
    P_vals = a0*H0 + a4*H4 + a8*H8 + a12*H12
    
    L2_sq = np.sum(weights * P_vals**2)
    
    # Evaluate P(x) at nodes for L4 (x = nodes/sqrt(2))
    x_scaled = nodes / np.sqrt(2.0)
    H0_s, H4_s, H8_s, H12_s = _hermite_polynomials(x_scaled)
    P_vals_s = a0*H0_s + a4*H4_s + a8*H8_s + a12*H12_s
    
    # Integral of |P|^4 e^{-2x^2} dx
    # = 1/sqrt(2) * sum w_i |P(x_i)|^4
    L4_4 = (1.0/np.sqrt(2.0)) * np.sum(weights * P_vals_s**4)
    
    # Avoid division by zero
    if L2_sq < 1e-15:
        return 1e15
        
    # Objective: (L4_4 / L2_sq^3)^2  ? 
    # A(g) = ||g||_4 / ||g||_2^{3/2}
    # Product = A(g)^2 (since g ~ g_hat)
    # A(g)^2 = ||g||_4^2 / ||g||_2^3
    # Wait. ||g||_4 is (Int |g|^4)^{1/4}.
    # So A(g) = (L4_4)^{1/4} / (L2_sq)^{3/4}.
    # Product = A(g)^2 = (L4_4)^{1/2} / (L2_sq)^{3/2} = sqrt(L4_4 / L2_sq^3).
    # We want to minimize this.
    # Minimizing sqrt(L4_4 / L2_sq^3) is same as minimizing L4_4 / L2_sq^3.
    
    return L4_4 / (L2_sq**3)

def construct():
    start_time = time.time()
    deadline = start_time + 170.0 # 180s limit, keep margin for safety
    
    # Seed solution
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _compute_objective(seed)
    best_x = seed.copy()
    
    # Strategy:
    # 1. Differential Evolution for global search
    # 2. L-BFGS-B for local refinement from DE result and seed
    # 3. High-precision Nelder-Mead polish
    
    # Define bounds for coefficients. 
    # Based on seed and typical scales, -5 to 5 should cover it.
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-0.5, 0.5)]
    
    # 1. Differential Evolution
    if time.time() < deadline - 30.0:
        try:
            # maxiter=100, popsize=15 (default is 15)
            # tol=1e-8
            res_de = opt.differential_evolution(
                _compute_objective, 
                bounds, 
                seed=42, 
                maxiter=100, 
                tol=1e-8,
                polish=False # We will do our own polish
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_x = res_de.x
        except Exception:
            pass

    # 2. Local Refinement from best_x and seed
    candidates = [best_x, seed]
    # Also add a few random points if time permits? 
    # For now, stick to DE result and seed.
    
    for x0 in candidates:
        if time.time() > deadline - 15.0:
            break
        try:
            res_lbfgs = opt.minimize(_compute_objective, x0, method='L-BFGS-B', 
                                     options={'maxiter': 1000, 'ftol': 1e-12})
            if res_lbfgs.fun < best_val:
                best_val = res_lbfgs.fun
                best_x = res_lbfgs.x
        except Exception:
            pass

    # 3. Final Polish with Nelder-Mead
    if time.time() < deadline - 5.0:
        try:
            res_nm = opt.minimize(_compute_objective, best_x, method='Nelder-Mead',
                                  options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 2000, 'adaptive': True})
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_x = res_nm.x
        except Exception:
            pass

    return best_x

# EVOLVE-BLOCK-END
