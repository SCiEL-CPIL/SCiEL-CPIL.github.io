import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 200.0

def construct():
    """
    Finds coefficients c0, c4, c8 for the Hermite combination
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    such that g(0)=0 and the upper bound on A(g)*A(hat{g}) is minimized.
    Returns np.array([c0, c4, c8]).
    """
    
    def fact(n):
        r = 1.0
        for k in range(2, n + 1):
            r *= k
        return r
    
    # Precompute Q0 and Q4 matrices for the Hermite basis
    # Q0[d,d] = sqrt(pi) * 2^d * d!
    # Q4[d_i, d_j] = int x^4 H_{d_i} H_{d_j} e^{-x^2} dx
    
    def x4_coeffs(n):
        d = {}
        d[n + 4] = 1.0 / 16.0
        d[n + 2] = (n + 5.0 / 4.0)
        d[n] = (7.0 * n * n / 4.0) + (5.0 * n / 2.0) + 1.0
        if n >= 4:
            d[n - 4] = float(n * (n - 1) * (n - 2) * (n - 3)) / 16.0
        return d
    
    degrees = [0, 4, 8, 12]
    nd = len(degrees)
    deg_set = {d: i for i, d in enumerate(degrees)}
    
    Q0 = np.zeros((nd, nd))
    for i, d in enumerate(degrees):
        Q0[i, i] = np.sqrt(np.pi) * (2.0 ** d) * fact(d)
    
    Q4 = np.zeros((nd, nd))
    for i, di in enumerate(degrees):
        for m, a in x4_coeffs(di).items():
            if m in deg_set:
                j = deg_set[m]
                Q4[i, j] = a * Q0[j, j]
    Q4 = 0.5 * (Q4 + Q4.T)
    
    # Constraint: c12 = -(c0 + 3*c4 + 105*c8) / 945
    def c12_of(c0, c4, c8):
        return -(c0 + 3.0 * c4 + 105.0 * c8) / 945.0
    
    # Objective: S = J / N^2 where N = c^T Q0 c, J = c^T Q4 c
    # c = [c0, c4, c8, c12]^T
    def obj_and_grad(x):
        if time.time() > TIME_LIMIT - 5:
            return 1e30, np.zeros(3)
        c0, c4, c8 = x
        c12 = c12_of(c0, c4, c8)
        c = np.array([c0, c4, c8, c12])
        N = float(c @ Q0 @ c)
        if N <= 1e-300:
            return 1e30, np.zeros(3)
        J = float(c @ Q4 @ c)
        S = J / (N * N)
        
        dN = 2.0 * (Q0 @ c)
        dJ = 2.0 * (Q4 @ c)
        dS = dJ / (N * N) - 2.0 * J * dN / (N * N * N)
        
        g = np.empty(3)
        g[0] = dS[0] + dS[3] * (-1.0 / 945.0)
        g[1] = dS[1] + dS[3] * (-3.0 / 945.0)
        g[2] = dS[2] + dS[3] * (-105.0 / 945.0)
        return S, g
    
    # Seed
    x_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val, _ = obj_and_grad(x_seed)
    best_x = x_seed.copy()
    
    rng = np.random.default_rng(12345)
    
    # Generate diverse starting points
    starts = [x_seed.copy()]
    for scale in [1e-3, 1e-2, 5e-2, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        for _ in range(2):
            x0 = x_seed + scale * rng.standard_normal(3)
            x0[0] = np.clip(x0[0], -50.0, 50.0)
            x0[1] = np.clip(x0[1], -5.0, 5.0)
            x0[2] = np.clip(x0[2], -0.5, 0.5)
            starts.append(x0)
    for _ in range(4):
        x0 = np.array([
            rng.uniform(-50.0, 50.0),
            rng.uniform(-5.0, 5.0),
            rng.uniform(-0.5, 0.5),
        ])
        starts.append(x0)
    
    bounds = [(-50.0, 50.0), (-5.0, 5.0), (-0.5, 0.5)]
    
    # Pass 1: Multi-start with L-BFGS-B
    for x0 in starts:
        if time.time() > TIME_LIMIT - 15:
            break
        try:
            res = minimize(
                lambda z: obj_and_grad(z)[0],
                x0,
                jac=lambda z: obj_and_grad(z)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 40}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Pass 2: SLSQP refinement from current best
    if time.time() < TIME_LIMIT - 10:
        try:
            res = minimize(
                lambda z: obj_and_grad(z)[0],
                best_x,
                jac=lambda z: obj_and_grad(z)[1],
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Pass 3: Tight L-BFGS-B polish
    if time.time() < TIME_LIMIT - 10:
        try:
            res = minimize(
                lambda z: obj_and_grad(z)[0],
                best_x,
                jac=lambda z: obj_and_grad(z)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Pass 4: Exact Newton polish using analytical Hessian
    if time.time() < TIME_LIMIT - 15:
        try:
            for _ in range(100):
                if time.time() > TIME_LIMIT - 12:
                    break
                c0, c4, c8 = best_x
                c12 = c12_of(c0, c4, c8)
                c = np.array([c0, c4, c8, c12])
                N = float(c @ Q0 @ c)
                J = float(c @ Q4 @ c)
                S = J / (N * N)
                
                # Gradient in 4-d
                g4 = 2.0 * (Q4 @ c) / (N * N) - 4.0 * J * (Q0 @ c) / (N * N * N)
                # Hessian in 4-d
                H4 = 2.0 * Q4 / (N * N) - 8.0 * J * (Q0 @ Q0) / (N * N * N) + 8.0 * J * np.outer(Q0 @ c, Q0 @ c) / (N * N * N * N)
                
                # Project to 3-d
                dc12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
                g3 = g4[:3] + g4[3] * dc12
                H3 = H4[:3,:3].copy()
                for i in range(3):
                    for j in range(3):
                        H3[i,j] += H4[i,3]*dc12[j] + H4[3,j]*dc12[i] + H4[3,3]*dc12[i]*dc12[j]
                H3 = 0.5*(H3 + H3.T)
                
                try:
                    delta = np.linalg.solve(H3, -g3)
                except np.linalg.LinAlgError:
                    delta = np.linalg.lstsq(H3, -g3, rcond=None)[0]
                
                step = 1.0
                while step > 1e-12:
                    x_trial = best_x + step * delta
                    x_trial[0] = np.clip(x_trial[0], -50.0, 50.0)
                    x_trial[1] = np.clip(x_trial[1], -5.0, 5.0)
                    x_trial[2] = np.clip(x_trial[2], -0.5, 0.5)
                    v_trial, _ = obj_and_grad(x_trial)
                    if v_trial < S:
                        best_x = x_trial
                        S = v_trial
                        break
                    step *= 0.5
                if step <= 1e-12:
                    break
                if np.linalg.norm(delta) * step < 1e-16:
                    break
        except Exception:
            pass
    
    # Final safety check
    try:
        v, _ = obj_and_grad(best_x)
        if v < best_val:
            best_val = v
    except Exception:
        pass
    
    return best_x
