import numpy as np
from scipy.special import roots_genlaguerre
from scipy.optimize import minimize_scalar, minimize
import time

def construct():
    """
    Returns a list of 12 positive double-root locations for the Laguerre combination.
    Uses Gauss-Laguerre quadrature nodes with optimized shape parameter beta,
    then optimizes the overall scale to minimize the C4 upper bound.
    """
    t0 = time.time()
    deadline = 580.0  # Safety margin

    m = 12

    def get_nodes(beta):
        """Get Gauss-Laguerre nodes for weight e^{-x} x^beta."""
        try:
            zeros = roots_genlaguerre(m, beta)[0]
            zeros = np.sort(zeros)
            if np.any(zeros <= 0):
                zeros = np.abs(zeros) + 1e-10
            return zeros
        except Exception:
            return None

    def c4_proxy(nodes):
        """
        Proxy for the C4 upper bound.
        The C4 bound involves products of root locations and their separations.
        We use a combination of:
        - Product of all roots (favors larger roots)
        - Minimum pairwise separation (favors well-spread roots)
        - Variance of log-roots (favors uniform log-spacing)
        """
        n = len(nodes)
        # Product of roots
        log_prod = np.sum(np.log(nodes))
        # Min pairwise separation
        min_sep = np.inf
        for i in range(n):
            for j in range(i+1, n):
                sep = abs(nodes[j] - nodes[i])
                if sep < min_sep:
                    min_sep = sep
        # Variance of log nodes
        log_nodes = np.log(nodes)
        log_var = np.var(log_nodes)
        # Combine: we want large product, large min_sep, moderate log_var
        # The C4 bound roughly scales as (product)^(1/n) / (min_sep)^(something)
        # Use a geometric mean type combination
        score = (10 ** (log_prod / n)) / (min_sep ** 0.5)
        return score

    def neg_c4_proxy(nodes):
        return -c4_proxy(nodes)

    def optimize_scale(nodes):
        """Find optimal scale factor for given node pattern."""
        base_nodes = np.array(nodes, dtype=float)
        max_val = base_nodes[-1]
        
        def neg_score(log_scale):
            scale = np.exp(log_scale)
            scaled = base_nodes * scale
            return -c4_proxy(scaled)
        
        # Search over log scale
        # Try a range of scales
        best_result = minimize_scalar(neg_score, bounds=(-5, 10), method='bounded')
        return best_result

    # Step 1: Optimize beta (shape parameter)
    best_beta = 11.0
    best_nodes = get_nodes(11.0)
    best_score = c4_proxy(best_nodes)

    # Grid search over beta
    betas = np.linspace(0, 20, 41)
    for beta in betas:
        if time.time() - t0 > deadline - 30:
            break
        nodes = get_nodes(beta)
        if nodes is None:
            continue
        score = c4_proxy(nodes)
        if score > best_score:
            best_score = score
            best_beta = beta
            best_nodes = nodes

    # Refine beta around best
    if time.time() - t0 < deadline - 30:
        beta_lo = max(0, best_beta - 1.0)
        beta_hi = min(20, best_beta + 1.0)
        
        def neg_beta_score(beta_val):
            nodes = get_nodes(beta_val)
            if nodes is None:
                return 1e10
            return -c4_proxy(nodes)
        
        beta_result = minimize_scalar(neg_beta_score, bounds=(beta_lo, beta_hi), method='bounded')
        if beta_result.success:
            refined_nodes = get_nodes(beta_result.x)
            if refined_nodes is not None:
                refined_score = c4_proxy(refined_nodes)
                if refined_score > best_score:
                    best_score = refined_score
                    best_beta = beta_result.x
                    best_nodes = refined_nodes

    # Step 2: Optimize scale for the best node pattern
    if time.time() - t0 < deadline - 10:
        scale_result = optimize_scale(best_nodes)
        if scale_result.success:
            optimal_scale = np.exp(scale_result.x)
        else:
            # Fallback: try some heuristic scales
            optimal_scale = 1.0
            best_scale_score = c4_proxy(best_nodes)
            for test_scale in [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
                scaled = best_nodes * test_scale
                s = c4_proxy(scaled)
                if s > best_scale_score:
                    best_scale_score = s
                    optimal_scale = test_scale
    else:
        optimal_scale = 1.0

    # Final: also try the known-good arithmetic progression as fallback
    if time.time() - t0 < deadline - 5:
        # Arithmetic progression scaled to match best
        arith_nodes = np.array([4.0 * (i + 1) for i in range(12)], dtype=float)
        arith_score = c4_proxy(arith_nodes)
        # Scale arithmetic to match
        arith_scale_result = optimize_scale(arith_nodes)
        if arith_scale_result.success:
            arith_optimal_scale = np.exp(arith_scale_result.x)
            arith_scaled = arith_nodes * arith_optimal_scale
            arith_final_score = c4_proxy(arith_scaled)
        else:
            arith_scaled = arith_nodes
            arith_final_score = arith_score
        
        laguerre_scaled = best_nodes * optimal_scale
        laguerre_final_score = c4_proxy(laguerre_scaled)
        
        if arith_final_score > laguerre_final_score:
            final_nodes = arith_scaled
        else:
            final_nodes = laguerre_scaled
    else:
        final_nodes = best_nodes * optimal_scale

    # Ensure all positive
    final_nodes = np.array(final_nodes, dtype=float)
    final_nodes = np.abs(final_nodes)
    final_nodes[final_nodes <= 0] = 1e-10
    final_nodes = np.sort(final_nodes)

    return list(final_nodes)
