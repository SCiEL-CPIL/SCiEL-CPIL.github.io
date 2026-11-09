"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.special import roots_genlaguerre

# EVOLVE-BLOCK-START
def _c4_upper_bound(roots, m=12, n=1):
    """
    Compute the C4 upper bound for the given root locations.
    The Cohn-Goncalves construction gives an uncertainty inequality where
    the constant C4 depends on the root configuration.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != m:
        return 1e10
    if np.any(roots <= 0):
        return 1e10
    if not np.all(np.diff(roots) > 0):
        return 1e10
    
    # The C4 bound for the refined uncertainty inequality with double roots
    # is related to the product of root spacings and the Laguerre weight.
    # For the Cohn-Goncalves construction, C4 involves:
    # 1. The Vandermonde determinant of the roots
    # 2. The Laguerre polynomial values
    # 3. The weight function e^{-x} x^n
    
    # A key quantity is the product of all pairwise differences (Vandermonde)
    # weighted by the Laguerre weight at each root
    gaps = np.diff(roots)
    if np.any(gaps <= 0):
        return 1e10
    
    # The upper bound on C4 is proportional to the spread of roots
    # normalized by the weight function decay
    # C4 ~ max(roots) / min(roots) * product of gaps
    weight_decay = np.exp(-roots) * roots**n
    obj = (roots[-1] / roots[0]) * np.prod(gaps) / np.prod(weight_decay)
    
    return obj

def construct():
    start_time = time.time()
    deadline = 600.0
    safety_margin = 30.0
    
    m = 12
    n = 1
    
    # Strategy: Use generalized Gauss-Laguerre quadrature nodes with tunable alpha
    # The nodes of L_m^(alpha) provide optimal root configurations
    # Search over alpha to find the configuration that minimizes C4
    
    best_roots = None
    best_obj = 1e10
    
    # Try a range of alpha values
    alpha_values = np.linspace(-1.0, 5.0, 100)
    for alpha in alpha_values:
        elapsed = time.time() - start_time
        if elapsed + safety_margin > deadline:
            break
        
        try:
            # Get the zeros of the generalized Laguerre polynomial L_m^(alpha)
            nodes, weights = roots_genlaguerre(m, alpha)
            
            # Verify nodes are positive and sorted
            if np.any(nodes <= 0) or not np.all(np.diff(nodes) > 0):
                continue
            
            obj = _c4_upper_bound(nodes, m, n)
            if obj < best_obj:
                best_obj = obj
                best_roots = nodes.copy()
        except Exception:
            continue
    
    # If no valid roots found, fall back to standard Laguerre nodes
    if best_roots is None:
        nodes, _ = roots_genlaguerre(m, n)
        best_roots = nodes
    
    # Ensure roots are positive and sorted
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 0.01)
    
    return [float(x) for x in best_roots]
# EVOLVE-BLOCK-END
