"""Optimized search for Hermite-Gaussian test function coefficients."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, linprog

def construct():
    """
    Find optimal coefficients (c0, c4, c8) for the Hermite-Gaussian test function
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is determined by g(0)=0.
    
    Minimizes (sup |x*g(x)|)^2.
    """
    start_time = time.time()
    deadline = 190.0  # Safety margin

    # Constants
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    # Grid setup
    # We need to capture the peak of |x*g(x)|. The Gaussian decays fast.
    # Range [-8, 8] is sufficient for high precision.
    N = 65536
    x = np.linspace(-8.0, 8.0, N)
    
    # Pre-compute polynomials
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    x10 = x8 * x2
    x12 = x6 * x6
    
    H0 = np.ones_like(x)
    H4 = 16.0*x4 - 48.0*x2 + 12.0
    H8 = 256.0*x8 - 4480.0*x6 + 26880.0*x4 - 6720.0*x2 + 1680.0
    H12 = (4096.0*x12 - 129024.0*x10 + 1761280.0*x8 - 11942400.0*x6 
           + 37158400.0*x4 - 54432000.0*x2 + 47900160.0)
    
    # Basis functions for the free coefficients
    # g(x) = exp(-x^2/2) * [ c0*(H0 - (H0_0/H12_0)H12) + ... ]
    basis0 = (H0 - (H0_0/H12_0)*H12) * np.exp(-x2 / 2.0)
    basis4 = (H4 - (H4_0/H12_0)*H12) * np.exp(-x2 / 2.0)
    basis8 = (H8 - (H8_0/H12_0)*H12) * np.exp(-x2 / 2.0)
    
    # Vectors x * basis for the objective
    xb0 = x * basis0
    xb4 = x * basis4
    xb8 = x * basis8

    def objective(coeffs):
        c0, c4, c8 = coeffs
        gx = c0*xb0 + c4*xb4 + c8*xb8
        max_val = np.max(np.abs(gx))
        return max_val * max_val

    # Initial guess
    x0 = np.array([-1.13, 0.04, 0.0003])
    best_val = objective(x0)
    best_x = x0.copy()

    # Global Search: Differential Evolution
    # Using a tighter box based on the seed and typical scales
    bounds = [(-3.0, 3.0), (-0.5, 0.5), (-0.05, 0.05)]
    
    try:
        if time.time() < deadline - 20:
            de_result = differential_evolution(
                objective, bounds, 
                maxiter=200, 
                popsize=20, 
                seed=123,
                tol=1e-10,
                mutation=(0.3, 1.0),
                crossover=0.8,
                polish=False  # We do our own polish
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_x = de_result.x.copy()
    except Exception:
        pass

    # Local Refinement: High-Precision Nelder-Mead
    # The objective is convex-ish in the region but has a kink at 0.
    # NM is robust for this.
    try:
        if time.time() < deadline - 5:
            # First pass: standard NM
            res = minimize(objective, best_x, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
                
            # Second pass: Powell, which handles the kink well sometimes
            res_p = minimize(objective, best_x, method='Powell', 
                          options={'maxiter': 500, 'ftol': 1e-15, 'xtol': 1e-15})
            if res_p.fun < best_val:
                best_val = res_p.fun
                best_x = res_p.x.copy()

            # Final Polish: Linear Programming Relaxation
            # We want to minimize M s.t. -M <= x*g(x) <= M
            # This is a LP in (c0, c4, c8, M) with 2N constraints.
            # Might be too slow for N=65536. Let's subsample for the LP check.
            if time.time() < deadline - 1:
                # Subsample grid for LP
                step = 10
                x_sub = x[::step]
                b0_sub = xb0[::step]
                b4_sub = xb4[::step]
                b8_sub = xb8[::step]
                
                n_vars = 4 # c0, c4, c8, M
                n_constraints = 2 * len(x_sub)
                
                A_ub = np.zeros((n_constraints, n_vars))
                b_ub = np.zeros(n_constraints)
                
                # Constraints: c0*b0 + c4*b4 + c8*b8 - M <= 0
                A_ub[:len(x_sub), 0] = b0_sub
                A_ub[:len(x_sub), 1] = b4_sub
                A_ub[:len(x_sub), 2] = b8_sub
                A_ub[:len(x_sub), 3] = -1.0
                b_ub[:len(x_sub)] = 0.0
                
                # Constraints: -(c0*b0 + c4*b4 + c8*b8) - M <= 0
                A_ub[len(x_sub):, 0] = -b0_sub
                A_ub[len(x_sub):, 1] = -b4_sub
                A_ub[len(x_sub):, 2] = -b8_sub
                A_ub[len(x_sub):, 3] = -1.0
                b_ub[len(x_sub):] = 0.0
                
                # Objective: minimize M
                c_obj = np.array([0.0, 0.0, 0.0, 1.0])
                
                res_lp = linprog(c_obj, A_ub=A_ub, b_ub=b_ub)
                if res_lp.success:
                    # res_lp.x = [c0, c4, c8, M]
                    # Note: LP minimizes M, which is sup |xg|. We want M^2.
                    # If LP gives a better M, use its c's.
                    if res_lp.x[3] < np.sqrt(best_val):
                         best_x = res_lp.x[:3]
                         best_val = res_lp.x[3] ** 2
                         
    except Exception:
        pass

    return best_x

# EVOLVE-BLOCK-END
