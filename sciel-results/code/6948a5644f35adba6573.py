import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 210.0


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

    def c12_of(c0, c4, c8):
        return -(c0 + 3.0 * c4 + 105.0 * c8) / 945.0

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

    x_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val, _ = obj_and_grad(x_seed)
    best_x = x_seed.copy()

    rng = np.random.default_rng(42)

    starts = [x_seed.copy()]
    
    # More diverse starting points with varied scales
    for scale in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
        for _ in range(3):
            x0 = x_seed + scale * rng.standard_normal(3)
            x0[0] = np.clip(x0[0], -100.0, 100.0)
            x0[1] = np.clip(x0[1], -10.0, 10.0)
            x0[2] = np.clip(x0[2], -1.0, 1.0)
            starts.append(x0)
    
    # Uniform random starts
    for _ in range(8):
        x0 = np.array([
            rng.uniform(-100.0, 100.0),
            rng.uniform(-10.0, 10.0),
            rng.uniform(-1.0, 1.0),
        ])
        starts.append(x0)
    
    # Scaled seed variations
    for scale in [0.1, 0.5, 2.0, 5.0]:
        starts.append(scale * x_seed)
        starts.append(-scale * x_seed)

    bounds = [(-100.0, 100.0), (-10.0, 10.0), (-1.0, 1.0)]

    for x0 in starts:
        if time.time() > TIME_LIMIT - 20:
            break
        try:
            res = minimize(
                lambda z: obj_and_grad(z)[0],
                x0,
                jac=lambda z: obj_and_grad(z)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # SLSQP polish
    if time.time() < TIME_LIMIT - 15:
        try:
            res = minimize(
                lambda z: obj_and_grad(z)[0],
                best_x,
                jac=lambda z: obj_and_grad(z)[1],
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'disp': False}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Final L-BFGS-B with very tight tolerance
    if time.time() < TIME_LIMIT - 15:
        try:
            res = minimize(
                lambda z: obj_and_grad(z)[0],
                best_x,
                jac=lambda z: obj_and_grad(z)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Newton's method polish with careful line search
    if time.time() < TIME_LIMIT - 20:
        try:
            for _ in range(200):
                if time.time() > TIME_LIMIT - 15:
                    break
                c0, c4, c8 = best_x
                c12 = c12_of(c0, c4, c8)
                c = np.array([c0, c4, c8, c12])
                N = float(c @ Q0 @ c)
                J = float(c @ Q4 @ c)
                S = J / (N * N)

                # Compute gradient and Hessian in 3D space
                g4 = 2.0 * (Q4 @ c) / (N * N) - 4.0 * J * (Q0 @ c) / (N * N * N)
                H4 = 2.0 * Q4 / (N * N) - 8.0 * J * (Q0 @ Q0) / (N * N * N) + 8.0 * J * np.outer(Q0 @ c, Q0 @ c) / (N * N * N * N)

                dc12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
                g3 = g4[:3] + g4[3] * dc12
                H3 = H4[:3,:3].copy()
                for i in range(3):
                    for j in range(3):
                        H3[i,j] += H4[i,3]*dc12[j] + H4[3,j]*dc12[i] + H4[3,3]*dc12[i]*dc12[j]
                H3 = 0.5*(H3 + H3.T)

                # Add small regularization for numerical stability
                reg = 1e-10 * np.eye(3) * np.trace(H3) / 3.0
                H3_reg = H3 + reg

                try:
                    delta = np.linalg.solve(H3_reg, -g3)
                except np.linalg.LinAlgError:
                    try:
                        delta = np.linalg.lstsq(H3_reg, -g3, rcond=None)[0]
                    except Exception:
                        break

                # Line search with backtracking
                step = 1.0
                improved = False
                for _ in range(30):
                    x_trial = best_x + step * delta
                    x_trial[0] = np.clip(x_trial[0], -100.0, 100.0)
                    x_trial[1] = np.clip(x_trial[1], -10.0, 10.0)
                    x_trial[2] = np.clip(x_trial[2], -1.0, 1.0)
                    v_trial, _ = obj_and_grad(x_trial)
                    if v_trial < S - 1e-15:
                        best_x = x_trial
                        S = v_trial
                        improved = True
                        break
                    step *= 0.5
                
                if not improved:
                    # Try smaller steps
                    for step in [1e-3, 1e-4, 1e-5, 1e-6]:
                        x_trial = best_x + step * delta
                        x_trial[0] = np.clip(x_trial[0], -100.0, 100.0)
                        x_trial[1] = np.clip(x_trial[1], -10.0, 10.0)
                        x_trial[2] = np.clip(x_trial[2], -1.0, 1.0)
                        v_trial, _ = obj_and_grad(x_trial)
                        if v_trial < S - 1e-15:
                            best_x = x_trial
                            S = v_trial
                            improved = True
                            break
                    
                    if not improved:
                        break
                
                if np.linalg.norm(delta) * step < 1e-18:
                    break
        except Exception:
            pass

    # Final verification
    try:
        v, _ = obj_and_grad(best_x)
        if v < best_val:
            best_val = v
    except Exception:
        pass

    return best_x
