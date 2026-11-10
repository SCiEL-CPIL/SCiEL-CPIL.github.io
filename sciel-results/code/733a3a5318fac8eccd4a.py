import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    t0 = time.time()
    deadline = t0 + 82.0  # safety margin before 90s limit

    # --- 1. Define a rich set of training/validation cosmologies ---
    # CPL parameters: E(z)^2 = (1+z)^3 + (1+z)^(3+3w0) * exp(3w1 * z/(1+z))
    # We want a broad but representative grid.
    w0_vals = np.linspace(-1.5, 0.5, 12)
    w1_vals = np.linspace(-1.5, 1.5, 12)
    # Redshifts: log-spaced for low z, linear for high z
    z_low = np.logspace(np.log10(0.005), np.log10(1.0), 12)
    z_high = np.linspace(1.0, 8.0, 12)
    z_vals = np.unique(np.concatenate([z_low, z_high]))
    
    cases = []
    for w0 in w0_vals:
        for w1 in w1_vals:
            for z in z_vals:
                cases.append((w0, w1, z))
    
    n = len(cases)
    w0_arr = np.array([c[0] for c in cases])
    w1_arr = np.array([c[1] for c in cases])
    z_arr = np.array([c[2] for c in cases])

    # --- 2. Compute reference integrals with high-accuracy quadrature ---
    def integrand(u, w0, w1, z):
        t = z * u
        if t < 1e-15:
            return 1.0
        log1 = 3.0 * np.log1p(t)
        log2 = (3.0 + 3.0 * w0) * np.log1p(t) + 3.0 * w1 * (t / (1.0 + t))
        m = max(log1, log2)
        s = np.exp(log1 - m) + np.exp(log2 - m)
        log_Esq = m + np.log(s)
        return np.exp(-0.5 * log_Esq)

    ref = np.zeros(n)
    for i in range(n):
        try:
            ref[i], _ = scipy.integrate.quad(
                lambda u: integrand(u, w0_arr[i], w1_arr[i], z_arr[i]),
                0.0, 1.0, limit=200, epsabs=1e-12, epsrel=1e-12)
        except Exception:
            # fallback: dense trapezoid
            uu = np.linspace(0, 1, 2000)
            vals = np.array([integrand(u, w0_arr[i], w1_arr[i], z_arr[i]) for u in uu])
            ref[i] = np.trapz(vals, uu)
    
    ref = np.maximum(ref, 1e-30)

    if time.time() > deadline - 25.0:
        gl, gw = np.polynomial.legendre.leggauss(8)
        return {"nodes": 0.5 + 0.5 * gl, "weights": 0.5 * gw}

    # --- 3. Objective functions ---
    # Parameterize nodes via logit and weights via softmax to maintain constraints.
    # p[0:8] = logit(nodes), p[8:16] = log-weights
    
    def eval_quad(p):
        # Returns: (max_rel_err, mean_rel_err, u, w)
        u = 1.0 / (1.0 + np.exp(-np.clip(p[:8], -30, 30)))
        w_raw = np.exp(np.clip(p[8:16], -30, 30))
        w = w_raw / w_raw.sum()
        
        # Sort nodes and corresponding weights to ensure strictly increasing
        idx = np.argsort(u)
        u = u[idx]
        w = w[idx]
        
        # Compute f values for all cases and nodes
        # t[i,k] = z_arr[i] * u[k]
        t = z_arr[:, None] * u[None, :]
        log1 = 3.0 * np.log1p(t)
        log2 = (3.0 + 3.0 * w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (t / (1.0 + t))
        m = np.maximum(log1, log2)
        s = np.exp(log1 - m) + np.exp(log2 - m)
        log_Esq = m + np.log(s)
        f = np.exp(-0.5 * log_Esq)
        
        # Quadrature sum: sum_k w[k] * f[i,k]
        quad = f @ w
        
        rel_err = np.abs(quad - ref) / ref
        max_err = np.max(rel_err)
        mean_err = np.mean(rel_err)
        
        return max_err, mean_err, u, w

    def obj_smooth(p):
        # Smooth approximation to max error using log-sum-exp
        max_err, mean_err, _, _ = eval_quad(p)
        # Combine: primarily minimize max, but use mean as secondary
        # Use a soft-max with high beta for max error
        # Actually, let's just minimize a weighted combination
        # Or better: use log-sum-exp on the individual errors
        # But eval_quad returns aggregated. Let's modify to return raw errors.
        # For simplicity in the smooth objective, let's just use max_err + 0.1 * mean_err
        # This is differentiable enough for L-BFGS-B if we handle the max carefully.
        # However, max is not differentiable at the maximum.
        # Let's implement a proper smooth max in the objective.
        return obj_lse(p)

    def obj_lse(p):
        # Compute raw relative errors and apply log-sum-exp
        # We need to access the raw errors, so let's duplicate the computation logic here
        # or modify eval_quad to return raw errors when requested.
        # To avoid code duplication and keep it fast, let's just use the current eval_quad
        # but for the smooth objective, we'll approximate the max by a high-beta log-sum-exp.
        # We need the vector of errors. Let's create a helper.
        
        u = 1.0 / (1.0 + np.exp(-np.clip(p[:8], -30, 30)))
        w_raw = np.exp(np.clip(p[8:16], -30, 30))
        w = w_raw / w_raw.sum()
        idx = np.argsort(u)
        u = u[idx]
        w = w[idx]
        
        t = z_arr[:, None] * u[None, :]
        log1 = 3.0 * np.log1p(t)
        log2 = (3.0 + 3.0 * w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (t / (1.0 + t))
        m = np.maximum(log1, log2)
        s = np.exp(log1 - m) + np.exp(log2 - m)
        log_Esq = m + np.log(s)
        f = np.exp(-0.5 * log_Esq)
        quad = f @ w
        rel_err = np.abs(quad - ref) / ref
        
        # Log-sum-exp with beta=50 for smooth max
        beta = 50.0
        scaled = rel_err * beta
        max_s = np.max(scaled)
        lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
        return lse / beta + 0.05 * np.mean(rel_err)

    # --- 4. Optimization ---
    # Initialize with Gauss-Legendre nodes/weights mapped to [0,1]
    gl, gw = np.polynomial.legendre.leggauss(8)
    u_init = 0.5 + 0.5 * gl
    w_init = 0.5 * gw
    
    # Inverse transform for initialization
    p_init = np.empty(16)
    p_init[:8] = np.log(u_init / (1.0 - u_init))
    p_init[8:16] = np.log(w_init)
    
    best_p = p_init.copy()
    best_max_err, _, best_u, best_w = eval_quad(p_init)
    
    # Phase 1: L-BFGS-B on smooth objective
    try:
        res = scipy.optimize.minimize(
            obj_lse, p_init, method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if res.success or res.fun < obj_lse(p_init):
            m, mn, u, w = eval_quad(res.x)
            if m < best_max_err:
                best_p = res.x.copy()
                best_max_err = m
                best_u = u.copy()
                best_w = w.copy()
    except Exception:
        pass

    if time.time() > deadline - 8.0:
        pass # Skip polish

    # Phase 2: Nelder-Mead polish on true max error
    try:
        res2 = scipy.optimize.minimize(
            lambda p: eval_quad(p)[0], best_p, method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True}
        )
        if res2.fun < best_max_err:
            best_p = res2.x.copy()
            best_max_err, _, best_u, best_w = eval_quad(best_p)
    except Exception:
        pass

    # Phase 3: Coordinate descent / local search if time permits
    if time.time() < deadline - 1.0:
        # Try small perturbations in node positions
        for _ in range(200):
            if time.time() > deadline - 1.0:
                break
            p_try = best_p.copy()
            # Perturb one node
            i = np.random.randint(0, 8)
            p_try[i] += np.random.normal(0, 1e-4)
            m_try, _, u_try, w_try = eval_quad(p_try)
            if m_try < best_max_err:
                best_p = p_try
                best_max_err = m_try
                best_u = u_try
                best_w = w_try

    # --- 5. Final processing ---
    u_final = best_u
    w_final = best_w
    
    # Ensure strictly increasing
    idx = np.argsort(u_final)
    u_final = u_final[idx]
    w_final = w_final[idx]
    
    # Clip nodes to [eps, 1-eps] to be safe
    u_final = np.clip(u_final, 1e-10, 1.0 - 1e-10)
    
    # Normalize weights to sum to 1
    w_final = w_final / w_final.sum()
    
    # Final validation check
    # Re-evaluate to ensure no numerical drift
    max_e, _, _, _ = eval_quad(np.concatenate([
        np.log(u_final / (1 - u_final)), 
        np.log(w_final)
    ]))
    
    # Compare with GL baseline
    u_gl = 0.5 + 0.5 * gl
    w_gl = 0.5 * gw
    t_gl = z_arr[:, None] * u_gl[None, :]
    log1 = 3.0 * np.log1p(t_gl)
    log2 = (3.0 + 3.0 * w0_arr)[:, None] * np.log1p(t_gl) + 3.0 * w1_arr[:, None] * (t_gl / (1.0 + t_gl))
    m_gl = np.maximum(log1, log2)
    s_gl = np.exp(log1 - m_gl) + np.exp(log2 - m_gl)
    f_gl = np.exp(-0.5 * (m_gl + np.log(s_gl)))
    quad_gl = f_gl @ w_gl
    err_gl = np.max(np.abs(quad_gl - ref) / ref)
    
    if max_e > err_gl:
        u_final = u_gl
        w_final = w_gl

    return {"nodes": u_final, "weights": w_final}
