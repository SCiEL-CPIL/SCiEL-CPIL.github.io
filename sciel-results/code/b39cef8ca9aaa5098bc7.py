"""Cosmology: universal luminosity-distance quadrature.
STRATEGY: Moment-matching with global node refinement.
Instead of alternating local optimizations against specific models, we:
1. Select a set of basis functions (Chebyshev polynomials) that approximate the 
   worst-case curvature of the integrand 1/E(a) over the domain a in [1/(1+z_max), 1].
   However, the quadrature is defined on z' in [0, z] scaled by z.
   The formula is dL/(c/H0) = (1+z)*z*sum_i w_i / E(a_i) where a_i = 1/(1+z*x_i).
   This is effectively approximating \int_0^1 f(x) dx where f(x) = 1/E(1/(1+z*x)).
   Since f depends on z, we cannot have a single rule for all z unless the rule is 
   designed to be robust.
   
   Actually, the prompt says: "The same nodes and weights must work across all models."
   And "dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)".
   This implies the integral \int_0^z dz'/E(z') is approximated by z * \sum w_i / E(a_i).
   Let u = z'/z, then \int_0^z dz'/E(z') = z \int_0^1 du / E(1/(1+z*u)).
   So we are approximating I(z) = \int_0^1 g(u, z) du where g(u,z) = 1/E(1/(1+z*u)).
   The quadrature is \sum w_i g(x_i, z).
   
   We want to find x_i, w_i such that \sum w_i g(x_i, z) \approx \int_0^1 g(u, z) du 
   for a wide range of z and cosmological models.
   
   Since g(u,z) varies with z, a fixed quadrature rule is an approximation.
   We can view this as finding a rule that minimizes the worst-case error over a set of 
   representative functions g_k(u) derived from different models and redshifts.
   
   A robust approach is to use the nodes/weights that exactly integrate a set of 
   basis functions \phi_j(u) that span the variation of g(u,z).
   We can choose \phi_j(u) to be Chebyshev polynomials of the first kind on [0,1].
   If we choose 8 nodes, we can exactly integrate polynomials up to degree 15.
   However, g(u,z) is not a polynomial.
   
   Alternative: Use the nodes and weights from a Gaussian quadrature rule that is 
   optimal for a specific weight function, or use a minimax approach on a grid of 
   (model, z) pairs.
   
   The current program uses alternating optimization. Let's try a different structural change:
   Use Differential Evolution to optimize the nodes, and for each set of nodes, 
   solve for the weights that minimize the L_infinity error on a dense grid of 
   (model, z) pairs. This is a "global node search" strategy.
   The key difference is using a global optimizer (DE) for nodes instead of Nelder-Mead, 
   which is more robust to local minima in the 8-dimensional space.
"""

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, linprog

START = time.time()
DEADLINE = 85.0  # Safety margin

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
    best_nodes = fallback["nodes"].copy()
    best_weights = fallback["weights"].copy()
    best_err = float('inf')
    
    # Training set: 
    # wa in [-1, 1] (covers quintessence to phantom)
    # z in [0.1, 5.0] (low to high redshift)
    wa_vals = np.linspace(-1.0, 1.0, 9)
    zs = np.linspace(0.1, 5.0, 20)
    
    # Initial nodes: Gauss-Legendre
    xi, _ = np.polynomial.legendre.leggauss(8)
    init_nodes = 0.5 * (xi + 1.0)
    
    # Strategy: Use Differential Evolution to optimize nodes.
    # For each candidate set of nodes, we optimize weights via LP.
    # This is expensive, so we use a coarse grid first, then refine.
    
    def objective(nodes):
        if time.time() > DEADLINE:
            return 1e10
        # Ensure nodes are strictly increasing and in (0,1)
        if nodes[0] < 1e-6 or nodes[-1] > 1 - 1e-6:
            return 1e10
        if np.any(np.diff(nodes) < 1e-6):
            return 1e10
        
        weights = _optimize_weights(nodes, zs, wa_vals)
        err = _get_error(nodes, weights, zs, wa_vals)
        return err
    
    # Differential Evolution bounds: nodes in [0.01, 0.99]
    bounds = [(0.01, 0.99)] * 8
    
    # Run DE with limited iterations to stay within time budget
    # maxiter=50, popsize=10 -> ~500 evaluations
    # Each evaluation involves LP + error calc.
    # LP is fast, error calc is 9*20=180 integrals.
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=50, 
            popsize=10, 
            tol=1e-6, 
            seed=42,
            polish=False
        )
        
        if res.success:
            cand_nodes = res.x
            # Ensure strict ordering
            if np.all(np.diff(cand_nodes) > 1e-6) and cand_nodes[0] > 1e-6 and cand_nodes[-1] < 1 - 1e-6:
                cand_weights = _optimize_weights(cand_nodes, zs, wa_vals)
                cand_err = _get_error(cand_nodes, cand_weights, zs, wa_vals)
                
                if cand_err < best_err:
                    best_err = cand_err
                    best_nodes = cand_nodes
                    best_weights = cand_weights
    except Exception:
        pass
    
    # If DE didn't improve or failed, try a few random restarts with local refinement?
    # Or just stick with the best found.
    
    # Final validation and normalization
    if np.all(np.diff(best_nodes) > 0) and np.all(best_weights > 0):
        best_weights = best_weights / np.sum(best_weights)
        return {"nodes": best_nodes, "weights": best_weights}
    
    return fallback
# EVOLVE-BLOCK-END
