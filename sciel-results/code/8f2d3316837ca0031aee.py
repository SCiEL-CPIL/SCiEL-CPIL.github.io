"""Universal 8-node positive quadrature for dimensionless luminosity distance."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _moment_residuals(params):
    """
    Residuals for moment matching.
    params: [u_1..u_8, v_1..v_8]
    x_i = sigmoid(u_i)
    w_i = softmax(v_i)
    Basis: phi_k(x) = x * (1-x)^k for k=0..7
    Target: I_k = 1/((k+1)(k+2))
    """
    u = params[:8]
    v = params[8:]
    
    x = 1.0 / (1.0 + np.exp(-u))
    x = np.sort(x)
    
    v_shifted = v - np.max(v)
    exp_v = np.exp(v_shifted)
    w = exp_v / np.sum(exp_v)
    
    K = 7
    ks = np.arange(K + 1)
    I_target = 1.0 / ((ks + 1) * (ks + 2))
    
    # Compute phi_k(x_i)
    # x shape (8,1), ks shape (1,8)
    phi = x[:, None] * np.power(1.0 - x[:, None], ks[None, :])
    
    Q = w @ phi
    return Q - I_target

def _solve():
    # Step 1: Moment Matching Initialization
    x0_gl, w0_gl = np.polynomial.legendre.leggauss(8)
    x0_gl_clipped = np.clip(x0_gl, 1e-6, 1-1e-6)
    u0 = np.log(x0_gl_clipped / (1.0 - x0_gl_clipped))
    v0 = np.log(w0_gl)
    params0 = np.concatenate([u0, v0])
    
    res_moment = minimize(_moment_residuals, params0, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
    params_opt = res_moment.x if res_moment.success else params0
    
    # Step 2: Refinement using Worst Relative Error
    w0_grid = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    wa_grid = np.array([-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0])
    
    # Pre-compute true integrals
    true_integrals = np.zeros((len(w0_grid), len(wa_grid)))
    from scipy.integrate import quad
    for i, w0 in enumerate(w0_grid):
        for j, wa in enumerate(wa_grid):
            def integrand(x):
                a = 1.0 - x
                if a < 1e-12:
                    return 0.0
                exponent = 3.0 * (1.0 + w0 + wa / a)
                return x * np.exp(exponent * np.log(a))
            val, _ = quad(integrand, 0, 1, limit=100)
            true_integrals[i, j] = val
    
    def objective_polish(params):
        u = params[:8]
        v = params[8:]
        x = np.sort(1.0 / (1.0 + np.exp(-u)))
        v_shifted = v - np.max(v)
        w = np.exp(v_shifted) / np.sum(np.exp(v_shifted))
        
        max_err = 0.0
        for i, w0 in enumerate(w0_grid):
            for j, wa in enumerate(wa_grid):
                a = 1.0 - x
                a_safe = np.clip(a, 1e-12, 1.0)
                exponent = 3.0 * (1.0 + w0 + wa / a_safe)
                g = np.exp(exponent * np.log(a_safe))
                f = x * g
                Q = np.sum(w * f)
                I_true = true_integrals[i, j]
                if I_true < 1e-10:
                    continue
                rel_err = np.abs(Q - I_true) / I_true
                if rel_err > max_err:
                    max_err = rel_err
        return max_err
    
    res_polish = minimize(objective_polish, params_opt, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
    params_final = res_polish.x if res_polish.success else params_opt
    
    u_final = params_final[:8]
    v_final = params_final[8:]
    nodes = np.sort(1.0 / (1.0 + np.exp(-u_final)))
    v_shifted = v_final - np.max(v_final)
    weights = np.exp(v_shifted) / np.sum(np.exp(v_shifted))
    
    weights = np.maximum(weights, 1e-12)
    weights = weights / np.sum(weights)
    nodes = np.clip(nodes, 1e-6, 1 - 1e-6)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1] + 1e-6:
            nodes[i] = nodes[i-1] + 1e-6
    nodes[-1] = min(nodes[-1], 1 - 1e-6)
    
    return {"nodes": nodes, "weights": weights}

def construct():
    try:
        rule = _solve()
    except Exception:
        x, w = np.polynomial.legendre.leggauss(8)
        rule = {"nodes": x, "weights": w}
        
    nodes = np.asarray(rule["nodes"], dtype=float)
    weights = np.asarray(rule["weights"], dtype=float)
    nodes = np.clip(np.sort(nodes), 1e-9, 1 - 1e-9)
    for i in range(1, 8):
        if nodes[i] <= nodes[i - 1] + 1e-9:
            nodes[i] = nodes[i - 1] + 1e-9
    weights = np.maximum(weights, 1e-12)
    weights = weights / weights.sum()
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
