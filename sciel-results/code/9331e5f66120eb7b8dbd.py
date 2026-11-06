"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def _E_of_z_vec(z, Om0, Ol0, w0, w1):
    """Vectorized E(z) = H(z)/H0 for flat CPL model."""
    a = 1.0 / (1.0 + z)
    a3 = a ** 3
    exponent = -3.0 * w1 * (1.0 - a)
    de_factor = a ** (-3.0 * (1.0 + w0)) * np.exp(exponent)
    E2 = Om0 / a3 + Ol0 * de_factor
    return np.sqrt(np.maximum(E2, 1e-30))

def _integrand_vec(z, Om0, Ol0, w0, w1):
    """Vectorized f(z) = z/(1+z) / E(z)"""
    mask = z > 1e-15
    result = np.zeros_like(z, dtype=float)
    if np.any(mask):
        E = _E_of_z_vec(z[mask], Om0, Ol0, w0, w1)
        result[mask] = z[mask] / (1.0 + z[mask]) / E
    return result

def _true_integral_fast(z_max, Om0, Ol0, w0, w1, n_pts=400):
    """High-accuracy integration using dense fixed quadrature (faster than adaptive quad)."""
    # Use Gauss-Legendre with many points for high accuracy
    x, w = np.polynomial.legendre.leggauss(n_pts)
    # Map from [-1,1] to [0, z_max]
    z_pts = 0.5 * z_max * (x + 1.0)
    w_pts = 0.5 * z_max * w
    f_vals = _integrand_vec(z_pts, Om0, Ol0, w0, w1)
    return np.sum(w_pts * f_vals)

def _approx_integral_vec(z_max, nodes, weights, Om0, Ol0, w0, w1):
    """Vectorized quadrature approximation."""
    z_eval = z_max * nodes
    f_vals = _integrand_vec(z_eval, Om0, Ol0, w0, w1)
    return z_max * np.dot(weights, f_vals)

def construct():
    t_start = time.time()
    deadline = t_start + 78.0  # Safety margin before 90s limit
    
    # Start with Gauss-Legendre 8-point nodes mapped to [0,1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    best_nodes = 0.5 * (gl_nodes + 1.0)
    best_weights = 0.5 * gl_weights
    
    # Design cosmologies: diverse CPL parameter space
    design_cosmos = [
        (0.30, 0.70, -1.0, 0.0),    # Lambda CDM
        (0.27, 0.73, -1.0, 0.0),    # Planck-like
        (0.30, 0.70, -1.0, 1.0),    # CPL positive w1
        (0.30, 0.70, -1.0, -1.0),   # CPL negative w1
        (0.20, 0.80, -1.2, 0.5),    # More DE
        (0.35, 0.65, -0.8, 0.3),    # Less DE
        (0.25, 0.75, -1.5, 1.0),    # Phantom w0
        (0.30, 0.70, -0.5, -0.5),   # Mildly varying
        (0.30, 0.70, -1.0, 0.5),    # Moderate CPL
        (0.30, 0.70, -1.5, -0.5),   # Phantom, negative w1
        (0.28, 0.72, -0.9, 0.2),    # Near-Planck CPL
        (0.32, 0.68, -1.1, -0.3),   # Slightly different
        (0.22, 0.78, -0.7, 0.8),    # Low Om0, high w1
        (0.38, 0.62, -1.3, 0.4),    # High Om0
        (0.30, 0.70, -1.0, 2.0),    # Strong positive w1
        (0.30, 0.70, -1.0, -2.0),   # Strong negative w1
    ]
    z_max_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
    
    # Precompute true integrals
    print("Precomputing reference integrals...")
    true_vals = []
    for (Om0, Ol0, w0, w1) in design_cosmos:
        for z_max in z_max_list:
            tv = _true_integral_fast(z_max, Om0, Ol0, w0, w1)
            true_vals.append((Om0, Ol0, w0, w1, z_max, tv))
    print(f"Precomputation done at t={time.time()-t_start:.1f}s")
    
    def _fast_objective(params):
        """Minimize worst relative error. params: [node_params(8), log_weights(8)]"""
        n = 8
        node_params = params[:n]
        log_w = params[n:]
        
        # Nodes via cumulative sigmoid to ensure strict increase
        # Use a different approach: map through arctanh then enforce spacing
        nodes = 0.5 + 0.5 * np.tanh(node_params)
        
        # Enforce strictly increasing by adding minimum spacing
        # If not sorted, return penalty
        diffs = np.diff(nodes)
        if np.any(diffs <= 1e-14):
            return 1e6
        
        weights = np.exp(log_w)
        w_sum = np.sum(weights)
        if w_sum < 1e-20:
            return 1e6
        weights = weights / w_sum
        
        max_err = 0.0
        for i in range(len(true_vals)):
            Om0, Ol0, w0, w1, z_max, tv = true_vals[i]
            if abs(tv) < 1e-15:
                continue
            approx = _approx_integral_vec(z_max, nodes, weights, Om0, Ol0, w0, w1)
            rel_err = abs(approx - tv) / abs(tv) * 100.0
            if rel_err > max_err:
                max_err = rel_err
        return max_err
    
    # Initial params from GL
    init_node_params = np.arctanh(2.0 * best_nodes - 1.0)
    init_log_w = np.log(best_weights)
    x0 = np.concatenate([init_node_params, init_log_w])
    
    best_score = _fast_objective(x0)
    print(f"Initial score: {best_score:.6f} at t={time.time()-t_start:.1f}s")
    
    # Stage 1: Nelder-Mead coarse optimization
    if time.time() < deadline - 20:
        try:
            result = minimize(_fast_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if result.fun < best_score:
                best_score = result.fun
                best_nodes = 0.5 + 0.5 * np.tanh(result.x[:8])
                best_weights = np.exp(result.x[8:])
                best_weights = best_weights / np.sum(best_weights)
                print(f"After NM1: {best_score:.6f} at t={time.time()-t_start:.1f}s")
        except Exception as e:
            print(f"NM1 failed: {e}")
    
    # Stage 2: L-BFGS-B fine optimization
    if time.time() < deadline - 10:
        try:
            x0_lbfgs = np.concatenate([np.arctanh(2.0 * best_nodes - 1.0), np.log(best_weights)])
            result2 = minimize(_fast_objective, x0_lbfgs, method='L-BFGS-B',
                              options={'maxiter': 500, 'gtol': 1e-12, 'ftol': 1e-12})
            if result2.fun < best_score:
                best_score = result2.fun
                best_nodes = 0.5 + 0.5 * np.tanh(result2.x[:8])
                best_weights = np.exp(result2.x[8:])
                best_weights = best_weights / np.sum(best_weights)
                print(f"After L-BFGS: {best_score:.6f} at t={time.time()-t_start:.1f}s")
        except Exception as e:
            print(f"L-BFGS failed: {e}")
    
    # Stage 3: Multi-start with random perturbations
    np.random.seed(12345)
    for restart in range(8):
        if time.time() > deadline - 5:
            break
        x0_r = x0.copy()
        x0_r[:8] += np.random.randn(8) * 0.5
        x0_r[8:] += np.random.randn(8) * 0.2
        
        try:
            result_r = minimize(_fast_objective, x0_r, method='Nelder-Mead',
                              options={'maxiter': 400, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True})
            if result_r.fun < best_score:
                best_score = result_r.fun
                best_nodes = 0.5 + 0.5 * np.tanh(result_r.x[:8])
                best_weights = np.exp(result_r.x[8:])
                best_weights = best_weights / np.sum(best_weights)
                print(f"After restart {restart}: {best_score:.6f} at t={time.time()-t_start:.1f}s")
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 3:
        try:
            node_params_final = np.arctanh(2.0 * best_nodes - 1.0)
            log_w_final = np.log(best_weights)
            x0_final = np.concatenate([node_params_final, log_w_final])
            result_f = minimize(_fast_objective, x0_final, method='Nelder-Mead',
                              options={'maxiter': 600, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True})
            if result_f.fun < best_score:
                best_score = result_f.fun
                best_nodes = 0.5 + 0.5 * np.tanh(result_f.x[:8])
                best_weights = np.exp(result_f.x[8:])
                best_weights = best_weights / np.sum(best_weights)
                print(f"After final polish: {best_score:.6f} at t={time.time()-t_start:.1f}s")
        except Exception:
            pass
    
    # Ensure nodes are strictly in (0,1) and increasing
    best_nodes = np.clip(best_nodes, 1e-10, 1.0 - 1e-10)
    order = np.argsort(best_nodes)
    best_nodes = best_nodes[order]
    best_weights = best_weights[order]
    
    # Normalize weights to sum to exactly 1
    best_weights = best_weights / np.sum(best_weights)
    
    # Ensure positive
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    print(f"Final score: {best_score:.6f} at t={time.time()-t_start:.1f}s")
    print(f"Nodes: {best_nodes}")
    print(f"Weights: {best_weights}")
    print(f"Weight sum: {np.sum(best_weights):.15f}")
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
