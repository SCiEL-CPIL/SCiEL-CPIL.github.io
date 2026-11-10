import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def rudin_problem10() -> np.ndarray:
    K = 10
    N_THETA = 70
    N_PHI = 130
    
    # Deadline guard
    start_time = time.time()
    DEADLINE = 600.0
    SAFETY_MARGIN = 10.0
    
    # 1. Setup Quadrature
    # Gauss-Legendre nodes and weights for cos(theta)
    x, w_theta = np.polynomial.legendre.leggauss(N_THETA)
    # x is in [-1, 1], representing cos(theta)
    # The integral over theta is int_0^pi f(theta) sin(theta) dtheta = int_{-1}^1 f(arccos(x)) dx
    # So the weight for the cos(theta) grid is just w_theta.
    # Wait, the measure is dOmega = sin(theta) dtheta dphi = dx dphi.
    # So the weights for the sum over theta are w_theta.
    
    phi = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
    
    # Total number of points
    N_POINTS = N_THETA * N_PHI
    
    # 2. Precompute Spherical Harmonics Y_m at quadrature points
    # Y_0 = N_0 P_10^0(cos theta)
    # Y_m = sqrt(2) N_m P_10^m(cos theta) cos(m phi) for m > 0
    # Y_m = sqrt(2) N_|m| P_10^|m|(cos theta) sin(|m| phi) for m < 0
    # N_m = sqrt( (2K+1)/(4 pi) * (K-m)! / (K+m)! )
    
    Y = np.zeros((N_POINTS, 2 * K + 1))
    
    # Flatten grids
    # x is (N_THETA,), phi is (N_PHI,)
    # We need to evaluate for each (i, j)
    
    # Compute N_m
    N_m = np.zeros(K + 1)
    for m in range(K + 1):
        # Factorials can be large, use log or direct float
        # (20)! is ~ 2e18, fits in float64
        num = (2 * K + 1) / (4 * np.pi) * factorial(K - m) / factorial(K + m)
        N_m[m] = np.sqrt(num)
    
    # Evaluate P_10^m(x)
    # lpmv(m, l, x) includes Condon-Shortley phase
    P = np.zeros((K + 1, N_THETA))
    for m in range(K + 1):
        P[m] = lpmv(m, K, x)
        
    # Fill Y matrix
    # Index mapping: m = -10..10 -> index 0..20
    # m_idx = m + 10
    
    # Reshape for broadcasting
    # x is (N_THETA, 1), phi is (1, N_PHI)
    x_grid = x[:, np.newaxis]
    phi_grid = phi[np.newaxis, :]
    
    for m in range(K + 1):
        idx_pos = m + K
        idx_neg = -m + K
        
        N_val = N_m[m]
        P_val = P[m, :, np.newaxis] # (N_THETA, 1)
        
        if m == 0:
            # Y_0 = N_0 P_10^0(cos theta)
            # Shape (N_THETA, 1) needs to be broadcast to (N_THETA, N_PHI)
            Y_m = N_val * P_val * np.ones((1, N_PHI))
            Y[:, idx_pos] = Y_m.flatten()
        else:
            cos_m_phi = np.cos(m * phi_grid) # (1, N_PHI)
            sin_m_phi = np.sin(m * phi_grid) # (1, N_PHI)
            
            # Y_pos = sqrt(2) * N_m * P_m * cos(m phi)
            Y_pos = np.sqrt(2.0) * N_val * P_val * cos_m_phi # (N_THETA, N_PHI)
            Y_neg = np.sqrt(2.0) * N_val * P_val * sin_m_phi # (N_THETA, N_PHI)
            
            Y[:, idx_pos] = Y_pos.flatten()
            Y[:, idx_neg] = Y_neg.flatten()
            
    # Weights for the double sum
    # Sum_i,j w_theta[i] * (1/N_PHI) * Y[i,j]^p
    # The total measure is 4 pi.
    # int dOmega = int_{-1}^1 dx int_0^{2pi} dphi = 2 * 2pi = 4pi.
    # Quadrature: sum_i w_theta[i] * sum_j (2pi / N_PHI) * f(x_i, phi_j)
    w_phi = 2 * np.pi / N_PHI
    weights = np.kron(w_theta, np.ones(N_PHI) * w_phi) # (N_POINTS,)
    
    # 3. Define Objective and Gradient
    def objective_and_grad(c):
        # c is shape (21,)
        f = Y.dot(c) # (N_POINTS,)
        
        f2 = f * f
        f4 = f2 * f2
        
        A = np.dot(weights, f4)
        B = np.dot(weights, f2)
        
        # Avoid division by zero
        if B < 1e-12:
            return 1e10, np.zeros_like(c)
            
        val = A / (B * B)
        
        # Gradient
        # dA/dc = 4 * Y^T (w * f^3)
        # dB/dc = 2 * Y^T (w * f)
        
        w_f3 = weights * (f * f2)
        w_f = weights * f
        
        dA = 4.0 * Y.T.dot(w_f3)
        dB = 2.0 * Y.T.dot(w_f)
        
        grad = dA / (B * B) - 2.0 * A * dB / (B ** 3)
        
        return val, grad

    # 4. Optimization
    # The objective is scale invariant and non-convex. Use differential evolution
    # (global search) to locate a good basin, then a local L-BFGS-B polish.
    from scipy.optimize import differential_evolution

    # Good initial guess: m=0 only (index K)
    c_init = np.zeros(2 * K + 1, dtype=np.float64)
    c_init[K] = 1.0

    best_c = c_init.copy()
    best_val = objective_and_grad(best_c)[0]

    def obj_only(c):
        return objective_and_grad(c)[0]

    rng = np.random.default_rng(42)

    # Global search over a bounded box. Scale-invariant objective: box size
    # only affects where DE explores, so use a symmetric box around zero.
    bounds = [(-10.0, 10.0)] * (2 * K + 1)

    if time.time() < DEADLINE - SAFETY_MARGIN:
        try:
            de_res = differential_evolution(
                obj_only,
                bounds,
                seed=42,
                maxiter=400,
                popsize=15,
                tol=1e-8,
                mutation=(0.4, 1.0),
                recombination=0.7,
                polish=False,
                init='latinhypercube',
                updating='deferred',
            )
            c_de = np.asarray(de_res.x, dtype=np.float64)
            if np.all(np.isfinite(c_de)):
                v_de = objective_and_grad(c_de)[0]
                if v_de < best_val:
                    best_val = v_de
                    best_c = c_de.copy()
        except Exception:
            pass

    # Local polish from the best global point plus a few extra starts.
    starts = [best_c.copy(), c_init.copy()]
    for _ in range(8):
        starts.append(rng.normal(size=2 * K + 1))

    for c0 in starts:
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        if np.allclose(c0, 0):
            continue
        try:
            res = minimize(
                obj_only,
                c0,
                jac=lambda c: objective_and_grad(c)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-12},
            )
            c_opt = res.x
            val_opt = objective_and_grad(c_opt)[0]
            if val_opt < best_val:
                best_val = val_opt
                best_c = c_opt
        except Exception:
            continue

    # Final check
    # Ensure validity: no NaN, inf, max|c| in [1e-3, 1e5]
    if not np.all(np.isfinite(best_c)):
        return np.ones(2 * K + 1, dtype=np.float64)

    max_abs_c = np.max(np.abs(best_c))
    if max_abs_c < 1e-3:
        best_c = best_c * (1.0 / max_abs_c)
    elif max_abs_c > 1e5:
        best_c = best_c * (1e5 / max_abs_c)

    return best_c
