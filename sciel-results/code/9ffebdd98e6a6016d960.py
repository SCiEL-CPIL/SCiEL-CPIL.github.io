"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # E(a) for a generic CPL model: w0, wa
    # E(a) = (a^{-3(1+w0+wa)})^{1/3} * a^{-wa} for flat universe
    # More precisely: E(a)^2 = a^{-6(1+w0)} * (a-1)^{-3*wa} ... 
    # Actually: E(a) = (a^{-(3+3w0+3wa)} * (a^{-1}-1)^{-3wa})^{1/2} ... 
    # Standard form: E(a) = a^{-3(1+w0+wa)/2} * (a^{-1}-1)^{-3wa/2} ... 
    # Let me use the standard: E(a) = (a^{-6(1+w0)} * (a^{-1}-1)^{-3wa})^{1/2}
    # which simplifies to: E(a) = a^{-3(1+w0)} * (a^{-1}-1)^{-3wa/2}
    
    def _E(a, w0, wa):
        # Handle a=0 edge case (shouldn't happen with nodes in (0,1))
        a = np.maximum(a, 1e-10)
        # E(a) = (1+z)^{3(1+w0+wa)/2} * (1 - (1+z)^{-(1+wa)})^{-3wa/2}
        # where z = 1/a - 1
        z = 1.0 / a - 1.0
        z = np.maximum(z, 1e-10)
        term1 = z ** (1.5 * (1.0 + w0 + wa))
        term2 = (1.0 - z ** (-(1.0 + wa))) ** (-1.5 * wa)
        return term1 * term2
    
    def _integrand(z_node, w0, wa):
        # Integrand: (1+z) * z / E(a) where a = 1/(1+z)
        a = 1.0 / (1.0 + z_node)
        return (1.0 + z_node) * z_node / _E(a, w0, wa)
    
    def _quadrature(nodes, weights, w0, wa):
        """Compute dL/(c/H0) using the quadrature rule."""
        vals = _integrand(nodes, w0, wa)
        return np.sum(weights * vals)
    
    def _reference(w0, wa):
        """Adaptive reference integral using scipy."""
        from scipy.integrate import quad
        def f(z):
            a = 1.0 / (1.0 + z)
            return (1.0 + z) * z / _E(a, w0, wa)
        val, _ = quad(f, 0, 1, limit=200, epsabs=1e-12, epsrel=1e-12)
        return val
    
    # Library of CPL models for training
    model_library = [
        (0.0, 0.0),      # LCDM
        (-1.0, 0.0),     # Cosmological constant
        (-0.5, 0.0),     # Quintessence
        (-0.9, 0.3),     # CPL with positive wa
        (-1.2, 0.5),     # Phantom-like
        (-0.3, -0.5),    # Negative wa
        (-1.0, 0.5),     # CPL
        (-0.7, -0.3),    # Another CPL
        (0.5, 0.0),      # Matter-dominated-like
        (-2.0, 1.0),     # Exotic
    ]
    
    # Precompute reference values
    refs = np.array([_reference(w0, wa) for w0, wa in model_library])
    if not np.all(np.isfinite(refs)) or np.any(refs <= 0):
        # Fallback if reference computation fails
        n = 8
        return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}
    
    def _objective(params):
        """Maximize worst-case relative error (minimize this)."""
        if time.time() > deadline - 5.0:
            return 1e10
        
        t = params[:8]
        u = params[8:]
        
        # Transform to nodes and weights
        nodes = (np.tanh(t) + 1.0) / 2.0
        weights = np.exp(u)
        weights = weights / np.sum(weights)
        
        # Sort by nodes
        sort_idx = np.argsort(nodes)
        nodes = nodes[sort_idx]
        weights = weights[sort_idx]
        
        # Check strictly increasing
        if np.any(np.diff(nodes) <= 1e-12):
            return 1e10
        
        # Compute worst relative error
        max_rel_err = 0.0
        for i, (w0, wa) in enumerate(model_library):
            quad_val = _quadrature(nodes, weights, w0, wa)
            if refs[i] <= 0:
                continue
            rel_err = abs(quad_val - refs[i]) / refs[i]
            if rel_err > max_rel_err:
                max_rel_err = rel_err
        
        return max_rel_err * 100.0  # In percent
    
    # Differential evolution for global optimization
    bounds = [(-5, 5)] * 8 + [(-5, 5)] * 8
    
    best_params = None
    best_obj = np.inf
    
    # Try DE with limited population and iterations
    try:
        if time.time() < deadline - 30.0:
            result = differential_evolution(
                _objective, bounds,
                maxiter=50, popsize=20, tol=1e-6,
                seed=42, polish=False,
                init='latinhypercube'
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
    except Exception:
        pass
    
    # Local refinement with Nelder-Mead from best DE result
    if best_params is not None:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Multi-start Nelder-Mead if time permits
    if best_params is None or time.time() < deadline - 15.0:
        init_nodes = (np.arange(8) + 0.5) / 8.0
        init_t = np.arctanh(np.clip(2.0 * init_nodes - 1.0, -0.99, 0.99))
        init_u = np.zeros(8)
        init_params = np.concatenate([init_t, init_u])
        
        n_starts = 5
        for i in range(n_starts):
            if time.time() > deadline - 5.0:
                break
            if i == 0:
                x0 = init_params
            else:
                x0 = init_params + 0.3 * np.random.randn(16)
            
            try:
                res = minimize(_objective, x0, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_params = res.x
            except Exception:
                continue
    
    # Construct final rule
    if best_params is not None:
        t = best_params[:8]
        u = best_params[8:]
        nodes = (np.tanh(t) + 1.0) / 2.0
        weights = np.exp(u)
        weights = weights / np.sum(weights)
        
        sort_idx = np.argsort(nodes)
        nodes = nodes[sort_idx]
        weights = weights[sort_idx]
        
        nodes = np.clip(nodes, 1e-9, 1.0 - 1e-9)
        weights = weights / np.sum(weights)
        
        # Final validation
        if np.all(np.diff(nodes) > 1e-12) and np.all(weights > 0) and abs(np.sum(weights) - 1.0) < 1e-12:
            return {"nodes": nodes, "weights": weights}
    
    # Fallback: Midpoint rule
    n = 8
    return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}

# EVOLVE-BLOCK-END
