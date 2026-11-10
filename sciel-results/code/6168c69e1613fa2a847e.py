import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize, differential_evolution
import time

def rudin_problem10() -> np.ndarray:
    K = 10
    N_THETA = 70
    N_PHI = 130
    
    start_time = time.time()
    DEADLINE = 600.0
    SAFETY_MARGIN = 10.0
    
    # 1. Setup Quadrature
    x, w_theta = np.polynomial.legendre.leggauss(N_THETA)
    phi = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
    N_POINTS = N_THETA * N_PHI
    
    # 2. Precompute Spherical Harmonics Y_m at quadrature points
    Y = np.zeros((N_POINTS, 2 * K + 1))
    
    N_m = np.zeros(K + 1)
    for m in range(K + 1):
        num = (2 * K + 1) / (4 * np.pi) * factorial(K - m) / factorial(K + m)
        N_m[m] = np.sqrt(num)
    
    P = np.zeros((K + 1, N_THETA))
    for m in range(K + 1):
        P[m] = lpmv(m, K, x)
    
    x_grid = x[:, np.newaxis]
    phi_grid = phi[np.newaxis, :]
    
    for m in range(K + 1):
        idx_pos = m + K
        idx_neg = -m + K
        N_val = N_m[m]
        P_val = P[m, :, np.newaxis]
        
        if m == 0:
            Y_m = N_val * P_val * np.ones((1, N_PHI))
            Y[:, idx_pos] = Y_m.flatten()
        else:
            cos_m_phi = np.cos(m * phi_grid)
            sin_m_phi = np.sin(m * phi_grid)
            Y_pos = np.sqrt(2.0) * N_val * P_val * cos_m_phi
            Y_neg = np.sqrt(2.0) * N_val * P_val * sin_m_phi
            Y[:, idx_pos] = Y_pos.flatten()
            Y[:, idx_neg] = Y_neg.flatten()
    
    w_phi = 2 * np.pi / N_PHI
    weights = np.kron(w_theta, np.ones(N_PHI) * w_phi)
    
    # 3. Define Objective and Gradient
    def objective_and_grad(c):
        f = Y.dot(c)
        f2 = f * f
        f4 = f2 * f2
        A = np.dot(weights, f4)
        B = np.dot(weights, f2)
        if B < 1e-12:
            return 1e10, np.zeros_like(c)
        val = A / (B * B)
        w_f3 = weights * (f * f2)
        w_f = weights * f
        dA = 4.0 * Y.T.dot(w_f3)
        dB = 2.0 * Y.T.dot(w_f)
        grad = dA / (B * B) - 2.0 * A * dB / (B ** 3)
        return val, grad
    
    def obj_only(c):
        return objective_and_grad(c)[0]
    
    # 4. Optimization with smart initialization
    # The optimal flat harmonic should be dominated by m=0
    # Start with m=0 only, then add other components
    
    c_init = np.zeros(2 * K + 1, dtype=np.float64)
    c_init[K] = 1.0  # m=0
    
    best_c = c_init.copy()
    best_val = objective_and_grad(best_c)[0]
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Try m=0 only (should be optimal or near-optimal)
    # Phase 2: Add small perturbations in other m values
    # Phase 3: Full optimization from multiple starts
    
    # Start list: m=0, then m=0 + small noise in other components
    starts = [c_init.copy()]
    
    # Add starts with m=0 dominant + small components in other m
    for trial in range(30):
        c_trial = c_init.copy()
        # Add small random components in other m values
        mask = np.ones(2 * K + 1, dtype=bool)
        mask[K] = False  # keep m=0 fixed at 1.0
        noise = rng.normal(0, 0.1, size=2 * K + 1)
        noise[K] = 0
        c_trial += noise
        starts.append(c_trial)
    
    # Also add fully random starts
    for trial in range(20):
        c_rand = rng.normal(size=2 * K + 1)
        starts.append(c_rand)
    
    # Local optimization from each start
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
                options={'maxiter': 1000, 'gtol': 1e-12}
            )
            c_opt = res.x
            val_opt = objective_and_grad(c_opt)[0]
            if val_opt < best_val:
                best_val = val_opt
                best_c = c_opt.copy()
        except Exception:
            continue
    
    # Phase 2: Differential evolution for global search if time permits
    if time.time() < DEADLINE - 120:
        bounds = [(-5.0, 5.0)] * (2 * K + 1)
        try:
            de_res = differential_evolution(
                obj_only,
                bounds,
                seed=42,
                maxiter=300,
                popsize=12,
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
    
    # Phase 3: Final polish from best found
    if time.time() < DEADLINE - SAFETY_MARGIN:
        try:
            res = minimize(
                obj_only,
                best_c,
                jac=lambda c: objective_and_grad(c)[1],
                method='L-BFGS-B',
                options={'maxiter': 2000, 'gtol': 1e-14}
            )
            c_opt = res.x
            val_opt = objective_and_grad(c_opt)[0]
            if val_opt < best_val:
                best_val = val_opt
                best_c = c_opt.copy()
        except Exception:
            pass
    
    # Final validity check
    if not np.all(np.isfinite(best_c)):
        return np.ones(2 * K + 1, dtype=np.float64)
    
    max_abs_c = np.max(np.abs(best_c))
    if max_abs_c < 1e-3:
        best_c = best_c * (1.0 / max_abs_c)
    elif max_abs_c > 1e5:
        best_c = best_c * (1e5 / max_abs_c)
    
    return best_c
