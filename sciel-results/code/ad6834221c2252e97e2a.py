import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    return list(_best_coeffs())

def _best_coeffs():
    t0 = time.time()
    deadline = t0 + 85.0
    c = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    try:
        c = _fit(deadline)
    except Exception:
        pass
    return c

def _fit(deadline):
    # ---- Broad, dense reference grid ----
    om0_vals = np.array([0.20, 0.24, 0.27, 0.30, 0.33, 0.36, 0.40])
    w0_vals  = np.array([-1.0, -0.7, -0.4, -0.1, 0.2])
    wa_vals  = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    a_vals   = np.linspace(0.10, 1.00, 40)
    
    models = []
    for o in om0_vals:
        for w in w0_vals:
            for wa in wa_vals:
                models.append((o, w, wa))
    
    M = len(models)
    N = len(a_vals)
    
    # Fast RK4 ODE solver for reference
    # D'' + (3/2)(1/a) D' - (3/2) Om(a)/a^2 D = 0
    # In a as independent variable:
    # Let y = [D, D'], then:
    # y0' = y1
    # y1' = -1.5/a * y1 + 1.5*Om(a)/a^2 * y0
    
    def _ode_rhs(a, y, o, w, wa):
        D, Dp = y
        u = 1.0 + w + wa * (1.0 - a)
        # Om(a) = Om0 * a^3 / (Om0*a^3 + (1-Om0)*a^(3u))
        m = o * a**3
        d = (1.0 - o) * a**(3.0 * u)
        denom = m + d
        if denom < 1e-30:
            Om_a = 0.0
        else:
            Om_a = m / denom
        dD = Dp
        dDp = -1.5 / a * Dp + 1.5 * Om_a / (a * a) * D
        return [dD, dDp]
    
    def _rk4_solve(o, w, wa, a_grid):
        """Fast RK4 integration from a_grid[0] to a_grid[-1]."""
        n = len(a_grid)
        D = np.zeros(n)
        Dp = np.zeros(n)
        # Initial conditions: matter-dominated at small a
        D[0] = a_grid[0]
        Dp[0] = 1.0
        
        for i in range(n - 1):
            h = a_grid[i + 1] - a_grid[i]
            a = a_grid[i]
            y = [D[i], Dp[i]]
            
            k1 = _ode_rhs(a, y, o, w, wa)
            y2 = [y[0] + 0.5*h*k1[0], y[1] + 0.5*h*k1[1]]
            k2 = _ode_rhs(a + 0.5*h, y2, o, w, wa)
            y3 = [y[0] + 0.5*h*k2[0], y[1] + 0.5*h*k2[1]]
            k3 = _ode_rhs(a + 0.5*h, y3, o, w, wa)
            y4 = [y[0] + h*k3[0], y[1] + h*k3[1]]
            k4 = _ode_rhs(a + h, y4, o, w, wa)
            
            D[i+1] = y[0] + h/6.0 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
            Dp[i+1] = y[1] + h/6.0 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        
        # Normalize so D(1) = 1
        norm = D[-1]
        if abs(norm) < 1e-20:
            return a_grid**0.55, 0.55 * np.ones(n)
        D = D / norm
        Dp = Dp / norm
        
        # f = a * D' / D
        with np.errstate(divide='ignore', invalid='ignore'):
            f = a_grid * Dp / D
        f = np.nan_to_num(f, nan=0.55)
        return D, f
    
    # Precompute references
    ref_D = np.zeros((M, N))
    ref_f = np.zeros((M, N))
    Om0_g = np.zeros(M * N)
    W0_g = np.zeros(M * N)
    Wa_g = np.zeros(M * N)
    A_g = np.zeros(M * N)
    
    idx = 0
    for m_idx, (o, w, wa) in enumerate(models):
        for n_idx, a in enumerate(a_vals):
            Om0_g[idx] = o
            W0_g[idx] = w
            Wa_g[idx] = wa
            A_g[idx] = a
            idx += 1
        
        D_ref, f_ref = _rk4_solve(o, w, wa, a_vals)
        ref_D[m_idx, :] = D_ref
        ref_f[m_idx, :] = f_ref
    
    # Precompute ansatz features
    def _compute_om_a():
        u = 1.0 + W0_g + Wa_g * (1.0 - A_g)
        num = Om0_g * A_g**3
        den = num + (1.0 - Om0_g) * A_g**(3.0 * u)
        with np.errstate(divide='ignore', invalid='ignore'):
            om_a = num / den
        return np.nan_to_num(om_a, nan=1.0)
    
    Om_a_global = _compute_om_a()
    u_global = 1.0 + W0_g + Wa_g * (1.0 - A_g)
    q_global = 1.0 - Om_a_global
    
    # Least-squares initial guess
    def _lsq_initial():
        # Fit gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
        # to ln(f_ref) / ln(Om_a)
        with np.errstate(divide='ignore', invalid='ignore'):
            target = np.log(ref_f.reshape(M, N)) / np.log(np.clip(Om_a_global, 1e-10, 1-1e-10)).reshape(M, N)
        target = np.nan_to_num(target, nan=0.55)
        
        # Build design matrix
        X = np.column_stack([
            np.ones(M*N),
            q_global,
            q_global**2,
            u_global,
            Wa_g * A_g,
            u_global * q_global
        ])
        
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, target.ravel(), rcond=None)
            return np.clip(coeffs, -10.0, 10.0)
        except Exception:
            return np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    def _gamma(c):
        g = (c[0] + c[1]*q_global + c[2]*q_global**2 + c[3]*u_global
             + c[4]*Wa_g*A_g + c[5]*u_global*q_global)
        return np.clip(g, 0.1, 1.2)
    
    def _f_ansatz(c):
        g = _gamma(c)
        return Om_a_global ** g
    
    # Precompute x-grid for D integration
    x_vals = np.log(a_vals)
    dx = np.diff(x_vals)
    
    def _compute_D_from_f(f_ana):
        """Compute D(a)/D(1) from f(a) via cumulative integral."""
        # f_ana is (M, N)
        # D(a_i)/D(1) = exp(-int_{x_i}^{x_{N-1}} f(exp(x)) dx)
        # S_i = int_{x_i}^{x_{N-1}} f dx
        # S_{N-1} = 0
        # S_i = 0.5*(f_i + f_{i+1})*(x_{i+1}-x_i) + S_{i+1}
        
        S = np.zeros((M, N))
        for i in range(N-2, -1, -1):
            trap = 0.5 * (f_ana[:, i] + f_ana[:, i+1]) * dx[i]
            S[:, i] = S[:, i+1] + trap
        
        return np.exp(-S)
    
    def objective(c):
        f_ana = _f_ansatz(c).reshape(M, N)
        
        # Error in f
        with np.errstate(divide='ignore', invalid='ignore'):
            err_f = np.abs(f_ana - ref_f) / np.abs(ref_f)
        err_f = np.nan_to_num(err_f, nan=1.0)
        
        # Error in D
        D_ana = _compute_D_from_f(f_ana)
        with np.errstate(divide='ignore', invalid='ignore'):
            err_D = np.abs(D_ana - ref_D) / np.abs(ref_D)
        err_D = np.nan_to_num(err_D, nan=1.0)
        
        # Smooth max via log-sum-exp
        vals = np.concatenate([err_f.ravel(), err_D.ravel()])
        if vals.max() < 1e-15:
            return 0.0
        beta = 200.0
        m = vals.max()
        s = np.logaddexp.reduce((vals - m) * beta) / beta + m
        return float(s)
    
    # Optimization
    x0 = _lsq_initial()
    best = x0.copy()
    best_f = objective(best)
    
    rng = np.random.default_rng(42)
    
    # Main optimization loop
    while time.time() < deadline - 10.0:
        res = minimize(objective, best, method='Nelder-Mead', 
                       options=dict(xatol=1e-7, fatol=1e-9, maxiter=300, adaptive=True))
        if res.fun < best_f:
            best_f = res.fun
            best = np.clip(res.x, -10.0, 10.0)
        else:
            # Perturbation
            improved = False
            for _ in range(5):
                if time.time() > deadline - 10.0:
                    break
                cand = best + 0.03 * rng.standard_normal(6)
                cand = np.clip(cand, -10.0, 10.0)
                fc = objective(cand)
                if fc < best_f:
                    best_f = fc
                    best = cand
                    improved = True
            if not improved:
                # Try a larger perturbation
                cand = best + 0.1 * rng.standard_normal(6)
                cand = np.clip(cand, -10.0, 10.0)
                fc = objective(cand)
                if fc < best_f:
                    best_f = fc
                    best = cand
                else:
                    break
    
    # Polish pass with tighter tolerances
    if time.time() < deadline - 3.0:
        res = minimize(objective, best, method='Nelder-Mead',
                       options=dict(xatol=1e-10, fatol=1e-12, maxiter=500, adaptive=True))
        if res.fun < best_f:
            best_f = res.fun
            best = np.clip(res.x, -10.0, 10.0)
    
    # Final clip
    best = np.clip(best, -10.0, 10.0)
    return list(best)
# EVOLVE-BLOCK-END
