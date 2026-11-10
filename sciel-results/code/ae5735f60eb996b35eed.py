import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin

    # --- 1. Define a small, diverse set of training cosmologies ---
    # Focus on extreme cases that likely drive the worst-case error.
    z_vals = np.array([0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    w0_vals = np.array([-1.5, -0.5, 0.0, 0.5])
    w1_vals = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
    
    cases = []
    for w0 in w0_vals:
        for w1 in w1_vals:
            for z in z_vals:
                cases.append((w0, w1, z))
                
    n = len(cases)
    w0_arr = np.array([c[0] for c in cases])
    w1_arr = np.array([c[1] for c in cases])
    z_arr = np.array([c[2] for c in cases])

    # --- 2. Compute reference integrals ---
    def integrand(u, w0, w1, z):
        t = z * u
        # E(z)^2 = (1+z)^3 + (1+z)^(3+3w0) * exp(3w1 * z/(1+z))
        # We integrate 1/E(z u) du from 0 to 1.
        # Note: The formula in the prompt is dL/(c/H0) = (1+z)*z*sum...
        # The integral is I = \int_0^1 du / E(z u).
        # E(z u) = H(z u)/H0.
        # E^2(x) = (1+x)^3 + (1+x)^(3+3w0) * exp(3w1 * x/(1+x))
        
        if t < 1e-15:
            return 1.0
            
        log1 = 3.0 * np.log1p(t)
        log2 = (3.0 + 3.0 * w0) * np.log1p(t) + 3.0 * w1 * (t / (1.0 + t))
        
        # Stable log-sum-exp for exp(log1) + exp(log2)
        m = np.maximum(log1, log2)
        s = np.exp(log1 - m) + np.exp(log2 - m)
        log_Esq = m + np.log(s)
        return np.exp(-0.5 * log_Esq)

    ref = np.zeros(n)
    for i in range(n):
        try:
            ref[i], _ = scipy.integrate.quad(
                lambda u: integrand(u, w0_arr[i], w1_arr[i], z_arr[i]),
                0.0, 1.0, limit=100, epsabs=1e-10, epsrel=1e-10)
        except Exception:
            # Fallback
            uu = np.linspace(0, 1, 1000)
            vals = np.array([integrand(u, w0_arr[i], w1_arr[i], z_arr[i]) for u in uu])
            ref[i] = np.trapz(vals, uu)
            
    ref = np.maximum(ref, 1e-30)

    if time.time() > deadline - 30.0:
        gl, gw = np.polynomial.legendre.leggauss(8)
        return {"nodes": 0.5 + 0.5 * gl, "weights": 0.5 * gw}

    # --- 3. Objective Function ---
    # Parameters: p[0:8] are logit(nodes), p[8:16] are log(weights)
    
    def eval_params(p):
        # Unconstrain nodes
        u = 1.0 / (1.0 + np.exp(-np.clip(p[:8], -30, 30)))
        w_raw = np.exp(np.clip(p[8:16], -30, 30))
        w = w_raw / w_raw.sum()
        
        # Sort nodes and weights to ensure strictly increasing nodes
        idx = np.argsort(u)
        u = u[idx]
        w = w[idx]
        
        # Calculate errors for all cases
        # t[i,k] = z_arr[i] * u[k]
        t = z_arr[:, None] * u[None, :]
        log1 = 3.0 * np.log1p(t)
        log2 = (3.0 + 3.0 * w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (t / (1.0 + t))
        m = np.maximum(log1, log2)
        s = np.exp(log1 - m) + np.exp(log2 - m)
        log_Esq = m + np.log(s)
        f = np.exp(-0.5 * log_Esq)
        
        quad = f @ w
        rel_err = np.abs(quad - ref) / ref
        
        # Minimize a combination of max and mean error
        # Use log-sum-exp for smooth max
        beta = 20.0
        scaled = rel_err * beta
        max_s = np.max(scaled)
        lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
        smooth_max = lse / beta
        
        mean_err = np.mean(rel_err)
        
        return smooth_max + 0.1 * mean_err

    def get_u_w(p):
        u = 1.0 / (1.0 + np.exp(-np.clip(p[:8], -30, 30)))
        w_raw = np.exp(np.clip(p[8:16], -30, 30))
        w = w_raw / w_raw.sum()
        idx = np.argsort(u)
        u = u[idx]
        w = w[idx]
        return u, w

    # --- 4. Optimization ---
    # Initialize with Gauss-Legendre
    gl, gw = np.polynomial.legendre.leggauss(8)
    u_init = 0.5 + 0.5 * gl
    w_init = 0.5 * gw
    
    p_init = np.empty(16)
    p_init[:8] = np.log(u_init / (1.0 - u_init))
    p_init[8:16] = np.log(w_init)
    
    best_p = p_init.copy()
    best_score = eval_params(p_init)
    best_u, best_w = get_u_w(p_init)

    # Use Differential Evolution for global optimization
    # Bounds for logit(nodes): [-20, 20] is sufficient for [0,1]
    # Bounds for log(weights): [-20, 20] is sufficient
    bounds = [(-20, 20)] * 16
    
    if time.time() < deadline - 10.0:
        try:
            # DE is stochastic and can be slow, so limit maxiter
            # popsize=15, maxiter=50 should be fast enough for 16 params
            res = scipy.optimize.differential_evolution(
                eval_params, bounds, maxiter=50, popsize=15, tol=1e-4, 
                seed=42, polish=True
            )
            if res.fun < best_score:
                best_p = res.x.copy()
                best_score = res.fun
                best_u, best_w = get_u_w(best_p)
        except Exception:
            pass

    # Local polish with L-BFGS-B if time permits
    if time.time() < deadline - 2.0:
        try:
            res2 = scipy.optimize.minimize(
                eval_params, best_p, method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            if res2.fun < best_score:
                best_p = res2.x.copy()
                best_score = res2.fun
                best_u, best_w = get_u_w(best_p)
        except Exception:
            pass

    # --- 5. Final Processing ---
    u_final = best_u
    w_final = best_w
    
    # Ensure strictly increasing and within [0,1]
    idx = np.argsort(u_final)
    u_final = u_final[idx]
    w_final = w_final[idx]
    
    # Clip nodes
    u_final = np.clip(u_final, 1e-10, 1.0 - 1e-10)
    
    # Normalize weights
    w_final = w_final / w_final.sum()
    
    # Check against GL baseline
    u_gl = 0.5 + 0.5 * gl
    w_gl = 0.5 * gw
    
    # Quick error check on training set
    def calc_err(u, w):
        t = z_arr[:, None] * u[None, :]
        log1 = 3.0 * np.log1p(t)
        log2 = (3.0 + 3.0 * w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (t / (1.0 + t))
        m = np.maximum(log1, log2)
        s = np.exp(log1 - m) + np.exp(log2 - m)
        f = np.exp(-0.5 * (m + np.log(s)))
        quad = f @ w
        return np.max(np.abs(quad - ref) / ref)
        
    err_opt = calc_err(u_final, w_final)
    err_gl = calc_err(u_gl, w_gl)
    
    if err_opt > err_gl:
        u_final = u_gl
        w_final = w_gl

    return {"nodes": u_final, "weights": w_final}
