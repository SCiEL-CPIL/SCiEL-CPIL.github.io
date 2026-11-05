"""Cosmology: universal luminosity-distance quadrature.
STRATEGY: Random Sampling of Nodes with LP-Optimized Weights and Local Refinement.
Instead of using Differential Evolution (which is slow and stochastic), we:
1. Generate a large number of random node configurations.
   - We use a Beta distribution (alpha=0.5, beta=0.5) for nodes to cluster them 
     near the endpoints, where the integrand 1/E(a) often has higher curvature 
     in cosmological models.
   - We ensure nodes are strictly increasing.
2. For each node configuration, we solve a Linear Programming (LP) problem to find 
   the weights that minimize the worst-case relative error (L_infinity) over a 
   dense grid of representative cosmologies (w_a) and redshifts (z).
   - The LP is fast and convex, providing a global optimum for the weights given the nodes.
3. We track the best node configuration (lowest worst-case error).
4. After the random search, we perform a local refinement of the best nodes using 
   Nelder-Mead, where the objective function is the worst-case error with LP-optimized 
   weights. This fine-tunes the nodes to escape shallow local minima found by random sampling.
5. We return the best nodes and their corresponding LP-optimized weights.

This approach is robust because:
- LP ensures optimal weights for any given nodes.
- Random sampling with endpoint clustering explores the node space more effectively 
  than uniform DE for this specific problem structure.
- Local refinement polishes the result.
"""

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog, minimize

START = time.time()
DEADLINE = 85.0  # Safety margin before 90s limit

def _fallback():
    n = 8
    return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}

def _E_cpl(a, w0, wa):
    """CPL E(a) for flat universe."""
    a = np.clip(a, 1e-8, 1.0)
    Omega_m = 0.3
    Omega_L0 = 0.7
    # w(a) = w0 + wa(1-a)
    # Omega_L(a) = Omega_L0 * a^(-3(1+w0+wa)) * exp(3wa(a-1))
    exp_term = np.exp(3.0 * wa * (a - 1.0))
    pow_term = a ** (-3.0 * wa)
    Omega_L = Omega_L0 * pow_term * exp_term
    E2 = a ** (-3.0) + Omega_L
    return np.sqrt(E2)

def _ref_integrate(z, Efun):
    """High-accuracy reference integration using Gauss-Legendre."""
    n = 64
    xi, wi = np.polynomial.legendre.leggauss(n)
    u = 0.5 * (xi + 1.0)
    a = 1.0 / (1.0 + z * u)
    E = Efun(a)
    val = z * np.sum(wi / E)
    return (1.0 + z) * val

def _optimize_weights(nodes, zs, wa_vals):
    """Optimize weights to minimize L_infinity error for fixed nodes."""
    n_nodes = len(nodes)
    # Variables: w_0, ..., w_7, t
    # Minimize t
    c = np.zeros(n_nodes + 1)
    c[-1] = 1.0
    
    A_ub = []
    b_ub = []
    
    for wa in wa_vals:
        Efun = lambda a: _E_cpl(a, -1.0, wa)
        for z in zs:
            a = 1.0 / (1.0 + z * nodes)
            E = Efun(a)
            A_row = (1.0 + z) * z / E
            
            ref = _ref_integrate(z, Efun)
            if ref == 0:
                continue
            
            # app = sum(A_row * w)
            # |app - ref| <= t * ref
            # app - t*ref <= ref  =>  sum(A_row*w) - t*ref <= ref
            row1 = np.zeros(n_nodes + 1)
            row1[:n_nodes] = A_row
            row1[-1] = -ref
            A_ub.append(row1)
            b_ub.append(ref)
            
            # -app + t*ref <= -ref => -sum(A_row*w) - t*ref <= -ref
            row2 = np.zeros(n_nodes + 1)
            row2[:n_nodes] = -A_row
            row2[-1] = -ref
            A_ub.append(row2)
            b_ub.append(-ref)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(1e-12, None)] * n_nodes + [(0, None)]
    
    A_eq = np.zeros((1, n_nodes + 1))
    A_eq[0, :n_nodes] = 1.0
    b_eq = np.array([1.0])
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.status == 0:
        weights = res.x[:n_nodes]
        weights = np.clip(weights, 1e-12, None)
        weights = weights / np.sum(weights)
        return weights
    else:
        return np.ones(n_nodes) / n_nodes

def _get_error(nodes, weights, zs, wa_vals):
    """Compute worst-case relative error."""
    max_err = 0.0
    for wa in wa_vals:
        Efun = lambda a: _E_cpl(a, -1.0, wa)
        for z in zs:
            a = 1.0 / (1.0 + z * nodes)
            E = Efun(a)
            app = (1.0 + z) * z * np.sum(weights / E)
            ref = _ref_integrate(z, Efun)
            if ref == 0:
                continue
            err = abs(app - ref) / ref
            if err > max_err:
                max_err = err
    return max_err

def construct():
    fallback = _fallback()
    
    # Training set: 
    # wa in [-1, 1] (covers quintessence to phantom)
    # z in [0.1, 5.0] (low to high redshift)
    wa_vals = np.linspace(-1.0, 1.0, 9)
    zs = np.linspace(0.1, 5.0, 20)
    
    best_nodes = fallback["nodes"].copy()
    best_weights = fallback["weights"].copy()
    best_err = _get_error(best_nodes, best_weights, zs, wa_vals)
    
    # Strategy: Random Sampling of Nodes
    # We sample nodes from a Beta(0.5, 0.5) distribution mapped to [0,1]
    # This clusters nodes near 0 and 1.
    # We then sort them to ensure they are increasing.
    
    n_samples = 2000
    rng = np.random.default_rng(42)
    
    for i in range(n_samples):
        if time.time() > DEADLINE - 10.0:
            break
            
        # Sample nodes
        # Beta(0.5, 0.5) has density proportional to 1/sqrt(x(1-x))
        # This puts more weight near endpoints.
        raw_nodes = rng.beta(0.5, 0.5, 8)
        nodes = np.sort(raw_nodes)
        
        # Ensure nodes are strictly inside (0,1) with small margin
        if nodes[0] < 1e-4 or nodes[-1] > 1 - 1e-4:
            continue
        if np.any(np.diff(nodes) < 1e-4):
            continue
            
        # Optimize weights for these nodes
        weights = _optimize_weights(nodes, zs, wa_vals)
        
        # Evaluate error
        err = _get_error(nodes, weights, zs, wa_vals)
        
        if err < best_err:
            best_err = err
            best_nodes = nodes.copy()
            best_weights = weights.copy()
            
    # Local Refinement using Nelder-Mead
    # Objective: Error with LP-optimized weights
    def objective(nodes):
        if time.time() > DEADLINE:
            return 1e10
        if nodes[0] < 1e-4 or nodes[-1] > 1 - 1e-4:
            return 1e10
        if np.any(np.diff(nodes) < 1e-4):
            return 1e10
        weights = _optimize_weights(nodes, zs, wa_vals)
        err = _get_error(nodes, weights, zs, wa_vals)
        return err

    try:
        res = minimize(
            objective, 
            best_nodes, 
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if res.fun < best_err:
            best_nodes = res.x
            best_weights = _optimize_weights(best_nodes, zs, wa_vals)
            best_err = res.fun
    except Exception:
        pass

    # Final validation and normalization
    if np.all(np.diff(best_nodes) > 0) and np.all(best_weights > 0):
        best_weights = best_weights / np.sum(best_weights)
        return {"nodes": best_nodes, "weights": best_weights}
    
    return fallback
# EVOLVE-BLOCK-END
