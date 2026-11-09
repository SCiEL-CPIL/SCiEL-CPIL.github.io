"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

def _validate_z_list(z_list):
    """Check that z_list is valid: finite, > 0, pairwise distinct."""
    if len(z_list) != 12:
        return False
    for z in z_list:
        if not math.isfinite(z) or z <= 0:
            return False
    for i in range(12):
        for j in range(i+1, 12):
            if abs(z_list[i] - z_list[j]) < 1e-10:
                return False
    return True


def _compute_score_fast(z_list):
    """
    Fast numerical approximation of B/(2*pi) for given z_list.
    
    Strategy:
    1. Build the 26 Laguerre polynomials L_{2k}^{(-1/2)}(x) numerically
    2. Solve the 26x26 linear system for coefficients c_k
    3. Compute q(x) = g(x) / (x * prod_i (x - z_i)^2) numerically
    4. Find the largest real root of q that has a sign change
    5. Return B/(2*pi)
    """
    import numpy as np
    from scipy.special import eval_genlaguerre
    from scipy.optimize import brentq
    
    z = sorted([float(zi) for zi in z_list])
    n = len(z)  # should be 12
    
    # Degrees: 0, 2, 4, ..., 50
    degrees = list(range(0, 51, 2))
    alpha = -0.5
    
    # Build the 26 Laguerre polynomials evaluated at various points
    # We need to solve: g(0)=0, g'(0)=1, g(z_i)=0, g'(z_i)=0
    
    # First, evaluate Laguerre polynomials and their derivatives at x=0
    # L_n^{(alpha)}(0) = binom(n+alpha, n)
    # For numerical stability, we'll evaluate directly
    
    # Build matrix A and vector b for the linear system
    # Row 0: g(0) = 0
    # Row 1: g'(0) = 1
    # Rows 2..25: g(z_i) = 0, g'(z_i) = 0 for i=0..11
    
    A = np.zeros((26, 26))
    b = np.zeros(26)
    
    # Evaluate Laguerre polynomials and their derivatives
    # Using the recurrence relation for derivatives
    # L_n'(x) = -L_{n-1}^{(alpha+1)}(x)
    
    # Row 0: g(0) = sum_k c_k * L_{2k}(0) = 0
    for k, d in enumerate(degrees):
        A[0, k] = eval_genlaguerre(d, alpha, 0.0)
    b[0] = 0.0
    
    # Row 1: g'(0) = sum_k c_k * L'_{2k}(0) = 1
    # L_n'(x) = -L_{n-1}^{(alpha+1)}(x)
    for k, d in enumerate(degrees):
        if d == 0:
            A[1, k] = 0.0
        else:
            A[1, k] = -eval_genlaguerre(d-1, alpha+1, 0.0)
    b[1] = 1.0
    
    # Rows 2..25: for each z_i, g(z_i)=0 and g'(z_i)=0
    for i in range(n):
        zi = z[i]
        row_g = 2 + 2*i
        row_gp = 3 + 2*i
        
        for k, d in enumerate(degrees):
            A[row_g, k] = eval_genlaguerre(d, alpha, zi)
            if d == 0:
                A[row_gp, k] = 0.0
            else:
                A[row_gp, k] = -eval_genlaguerre(d-1, alpha+1, zi)
        b[row_g] = 0.0
        b[row_gp] = 0.0
    
    # Solve the linear system
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    
    # Now compute q(x) = g(x) / (x * prod_i (x - z_i)^2)
    # We need to find real roots of q(x)
    
    # g(x) = sum_k c_k * L_{2k}^{(alpha)}(x)
    # The denominator is x * prod_i (x - z_i)^2, which has degree 1 + 24 = 25
    # g(x) has degree 50, so q(x) has degree 50 - 25 = 25
    
    # To find roots of q(x), we can:
    # 1. Evaluate q(x) on a grid
    # 2. Find sign changes
    # 3. Refine with brentq
    
    def g_func(x_val):
        """Evaluate g(x) = sum_k c_k * L_{2k}^{(alpha)}(x)"""
        result = 0.0
        for k, d in enumerate(degrees):
            result += c[k] * eval_genlaguerre(d, alpha, x_val)
        return result
    
    def denom_func(x_val):
        """Evaluate x * prod_i (x - z_i)^2"""
        if abs(x_val) < 1e-15:
            return 1e-15  # avoid division by zero
        result = x_val
        for zi in z:
            result *= (x_val - zi)**2
        return result
    
    def q_func(x_val):
        """Evaluate q(x) = g(x) / denom(x)"""
        return g_func(x_val) / denom_func(x_val)
    
    # Find roots by scanning for sign changes
    # The roots of q are where g has roots that are NOT at z_i or 0
    # Since g has double roots at each z_i and a single root at 0 (from g(0)=0, g'(0)=1),
    # q should be smooth away from 0 and z_i
    
    # Scan a reasonable range
    # The largest root of interest is likely in a moderate range
    x_min = 0.001
    x_max = 200.0
    n_points = 2000
    
    xs = np.linspace(x_min, x_max, n_points)
    qs = np.array([q_func(xi) for xi in xs])
    
    # Find sign changes
    roots = []
    for i in range(len(xs)-1):
        if np.isfinite(qs[i]) and np.isfinite(qs[i+1]):
            if qs[i] * qs[i+1] < 0:
                # Sign change between xs[i] and xs[i+1]
                try:
                    root = brentq(q_func, xs[i], xs[i+1], xtol=1e-12)
                    roots.append(root)
                except:
                    pass
    
    if not roots:
        return 0.0
    
    # B = max(0, largest counted root)
    # A root "counts" if q has strictly opposite signs at r-eps and r+eps
    # For simplicity, assume all found roots count
    B = max(0.0, max(roots))
    
    return B / (2.0 * math.pi)


def _local_search(z_init, deadline, max_evals=50):
    """
    Local search around z_init to minimize the score.
    Uses coordinate descent with adaptive step sizes.
    """
    best_z = list(z_init)
    best_score = _compute_score_fast(best_z)
    
    if best_score is None:
        return best_z, float('inf')
    
    eval_count = 0
    step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    
    improved = True
    while improved and eval_count < max_evals:
        if time.time() > deadline - 30:
            break
        
        improved = False
        for i in range(12):
            if time.time() > deadline - 30:
                break
            if eval_count >= max_evals:
                break
            
            for step in step_sizes:
                if eval_count >= max_evals:
                    break
                if time.time() > deadline - 30:
                    break
                
                # Try increasing
                candidate = list(best_z)
                candidate[i] = best_z[i] * (1 + step)
                if candidate[i] > 0 and _validate_z_list(candidate):
                    score = _compute_score_fast(candidate)
                    eval_count += 1
                    if score is not None and score < best_score:
                        best_score = score
                        best_z = candidate
                        improved = True
                        break
                
                # Try decreasing
                candidate = list(best_z)
                candidate[i] = best_z[i] * (1 - step)
                if candidate[i] > 0 and _validate_z_list(candidate):
                    score = _compute_score_fast(candidate)
                    eval_count += 1
                    if score is not None and score < best_score:
                        best_score = score
                        best_z = candidate
                        improved = True
                        break
    
    return best_z, best_score


def _generate_candidates():
    """Generate multiple candidate initial configurations."""
    candidates = []
    
    # 1. Bessel J_0 zeros squared
    def bessel_j0_zero(k):
        x = (k - 0.25) * math.pi
        if x < 0.1:
            return 2.4048255576957728
        return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)
    
    z_j0 = [bessel_j0_zero(i)**2 for i in range(1, 13)]
    if _validate_z_list(z_j0):
        candidates.append(('j0_sq', z_j0))
    
    # 2. Bessel J_1 zeros squared
    def bessel_j1_zero(k):
        x = (k + 0.25) * math.pi
        if x < 0.1:
            return 3.8317059702075125
        return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)
    
    z_j1 = [bessel_j1_zero(i)**2 for i in range(1, 13)]
    if _validate_z_list(z_j1):
        candidates.append(('j1_sq', z_j1))
    
    # 3. Geometric progression (various ratios)
    for ratio in [1.3, 1.5, 1.8, 2.0]:
        z_geo = [2.0 * ratio**i for i in range(12)]
        if _validate_z_list(z_geo):
            candidates.append((f'geo_{ratio}', z_geo))
    
    # 4. Arithmetic progression
    z_arith = [2.0 + 3.0*i for i in range(12)]
    if _validate_z_list(z_arith):
        candidates.append(('arith', z_arith))
    
    # 5. Clustered near small values
    z_cluster = [0.5 * (i+1)**0.8 for i in range(12)]
    if _validate_z_list(z_cluster):
        candidates.append(('cluster', z_cluster))
    
    # 6. Spread out more
    z_spread = [1.0 + i*5.0 for i in range(12)]
    if _validate_z_list(z_spread):
        candidates.append(('spread', z_spread))
    
    # 7. Optimal-like: based on Laguerre root distribution
    # The roots of L_n^{(alpha)} are roughly distributed as ~ (i/(n+1))^2 * something
    # For alpha=-1/2, try a power law
    for power in [0.5, 0.7, 1.0, 1.2]:
        z_power = [1.0 + i**power for i in range(1, 13)]
        if _validate_z_list(z_power):
            candidates.append((f'power_{power}', z_power))
    
    # 8. Logarithmic spacing
    z_log = [math.exp(0.3 * i) for i in range(12)]
    if _validate_z_list(z_log):
        candidates.append(('log', z_log))
    
    return candidates


def construct():
    """Return 12 double-root locations z_1..z_12 that minimize B/(2*pi)."""
    # 630s kill. Set deadline to 580s to allow for cleanup and return.
    deadline = time.time() + 580.0
    
    # Generate candidates
    candidates = _generate_candidates()
    
    if not candidates:
        # Fallback
        default_z = [4.0 * (i+1) for i in range(12)]
        return default_z
    
    # Evaluate all candidates quickly
    best_score = float('inf')
    best_z = candidates[0][1]
    
    for name, z in candidates:
        if time.time() > deadline - 60:
            break
        try:
            score = _compute_score_fast(z)
            if score is not None and score < best_score:
                best_score = score
                best_z = z
        except Exception:
            pass
    
    # If we have time, do local search on the best candidate
    if time.time() < deadline - 120 and best_score < float('inf'):
        try:
            refined_z, refined_score = _local_search(best_z, deadline, max_evals=80)
            if refined_score < best_score:
                best_score = refined_score
                best_z = refined_z
        except Exception:
            pass
    
    # Final validation
    if not _validate_z_list(best_z):
        best_z = [4.0 * (i+1) for i in range(12)]
    
    return best_z

# EVOLVE-BLOCK-END
