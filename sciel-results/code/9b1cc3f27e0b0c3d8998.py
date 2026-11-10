# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _logsumexp(x, axis=None, keepdims=False):
    """Stable log-sum-exp implementation."""
    x_max = np.max(x, axis=axis, keepdims=True)
    if np.any(np.isinf(x_max)):
        return x_max
    temp = x - x_max
    res = np.log(np.sum(np.exp(temp), axis=axis, keepdims=keepdims))
    if keepdims:
        return res + x_max
    return res + np.squeeze(x_max, axis=axis)

def _objective_and_grad(x, beta):
    """
    Objective: Minimize log(d_max^2) - log(d_min^2)
    Approximated by:
    logsumexp(dist^2 / beta) - logsumexp(1/dist^2 / beta)
    
    This is a smooth approximation of log(d_max^2) - log(d_min^2).
    Minimizing this maximizes log(d_min^2) - log(d_max^2) = log((d_min/d_max)^2).
    """
    # Reshape to (14, 3)
    P = x.reshape(14, 3)
    
    # Center to remove translation invariance (helps optimization stability)
    # We don't strictly need to center for the objective if it's translation invariant,
    # but it keeps numbers small. The objective is translation invariant.
    # Let's enforce centering in the constraint or just rely on symmetry.
    # Actually, L-BFGS-B doesn't handle constraints well without penalty.
    # We will just optimize the raw coordinates. The objective is translation invariant.
    
    # Compute pairwise squared distances
    # dist^2_{ij} = ||p_i - p_j||^2
    # Use pdist for efficiency
    d_sq = pdist(P, 'sqeuclidean')
    
    # Avoid division by zero in 1/d_sq
    d_sq_safe = np.maximum(d_sq, 1e-12)
    
    # Term 1: Approximate log(d_max^2)
    # logsumexp(d_sq / beta) ~ log(sum(exp(d_sq/beta)))
    # If beta is small, this is dominated by the max term.
    term_max = _logsumexp(d_sq / beta)
    
    # Term 2: Approximate log(1/d_min^2) = -log(d_min^2)
    # We want to minimize -log(d_min^2), which is equivalent to minimizing log(1/d_min^2).
    # 1/d_sq is large when d_sq is small.
    # logsumexp(1/d_sq / beta) ~ log(sum(exp(1/(beta * d_sq))))
    # This is dominated by the term with the smallest d_sq (largest 1/d_sq).
    term_min_inv = _logsumexp(1.0 / (d_sq_safe * beta))
    
    # Objective to minimize:
    # We want to maximize log(d_min^2) - log(d_max^2)
    # Minimize: log(d_max^2) - log(d_min^2)
    # Approx: term_max - term_min_inv ?
    # Wait. 
    # Maximize log(d_min^2) - log(d_max^2)
    # Minimize log(d_max^2) - log(d_min^2)
    # log(d_max^2) approx term_max
    # log(d_min^2) is approx -logsumexp(1/d_sq^2 / beta) ? No.
    # Let's look at limits.
    # As beta -> 0:
    # term_max -> log(exp(d_max^2/beta)) = d_max^2/beta
    # term_min_inv -> log(exp(1/(d_min^2 * beta))) = 1/(d_min^2 * beta)
    
    # We want to minimize: log(d_max^2) - log(d_min^2)
    # This is NOT linearly related to term_max and term_min_inv in a simple subtraction way
    # because of the 1/beta scaling differences.
    
    # Better approach:
    # Maximize f = log(d_min^2) - log(d_max^2)
    # Approx f ~ -_logsumexp(1/d_sq^2 / beta) - _logsumexp(d_sq^2 / beta)
    # Wait, log(d_min^2) = min(log(d_ij^2)). 
    # min(x_i) approx -1/beta * log(sum(exp(-beta * x_i)))
    # So log(d_min^2) approx -1/beta * logsumexp(-beta * d_sq)
    # And -log(d_max^2) approx 1/beta * logsumexp(-beta * d_sq) ? No.
    # max(x_i) approx 1/beta * logsumexp(beta * x_i)
    # So log(d_max^2) approx 1/beta * logsumexp(beta * d_sq)
    
    # So f approx: -1/beta * logsumexp(-beta * d_sq) - 1/beta * logsumexp(beta * d_sq)
    # We want to MAXIMIZE f.
    # So we MINIMIZE -f = 1/beta * logsumexp(-beta * d_sq) + 1/beta * logsumexp(beta * d_sq)
    
    # Let's implement this.
    # Note: d_sq can be 0. exp(-beta * 0) = 1. exp(beta * 0) = 1.
    # If two points coincide, d_sq=0.
    # logsumexp(-beta * d_sq) will have a term exp(0)=1.
    # logsumexp(beta * d_sq) will have a term exp(0)=1.
    # This is stable.
    
    obj_val = (1.0 / beta) * (_logsumexp(-beta * d_sq) + _logsumexp(beta * d_sq))
    
    # Gradient calculation
    # Let S1 = sum(exp(-beta * d_ij^2)), L1 = log(S1)
    # Let S2 = sum(exp(beta * d_ij^2)), L2 = log(S2)
    # Obj = (L1 + L2) / beta
    # dObj/dx = (1/beta) * (dL1/dx + dL2/dx)
    
    # dL1/dx_k = (1/S1) * sum_j d(exp(-beta * d_jk^2))/dx_k
    # d(exp(-beta * d_jk^2))/dx_k = exp(-beta * d_jk^2) * (-beta) * d(d_jk^2)/dx_k
    # d(d_jk^2)/dx_k = 2 * (p_k - p_j)
    
    # So dL1/dx_k = (1/S1) * sum_j exp(-beta * d_jk^2) * (-beta) * 2 * (p_k - p_j)
    #               = -2 * sum_j w1_jk * (p_k - p_j)
    # where w1_jk = exp(-beta * d_jk^2) / S1
    
    # Similarly for L2:
    # dL2/dx_k = (1/S2) * sum_j exp(beta * d_jk^2) * beta * 2 * (p_k - p_j)
    #           = 2 * sum_j w2_jk * (p_k - p_j)
    # where w2_jk = exp(beta * d_jk^2) / S2
    
    # Total Grad = (1/beta) * (-2 * sum_j w1_jk (p_k - p_j) + 2 * sum_j w2_jk (p_k - p_j))
    #            = (2/beta) * sum_j (w2_jk - w1_jk) * (p_k - p_j)
    
    # We need to compute this efficiently.
    # d_sq is (91,). We need to map back to indices.
    # Let's compute gradients using the vectorized form if possible, or loop over points (14 is small).
    
    P_grad = np.zeros_like(P)
    
    # Compute weights
    # d_sq corresponds to pairs (i, j) with i < j.
    # Let's create a matrix of squared distances D (14x14)
    D = np.sum(P[:, np.newaxis, :]**2, axis=2) + np.sum(P[np.newaxis, :, :]**2, axis=2) - 2 * (P @ P.T)
    # Ensure non-negative due to floating point
    D = np.maximum(D, 0.0)
    
    # Weights for L1 (min approximation)
    # w1_ij = exp(-beta * D_ij)
    # Note: diagonal D_ii = 0. exp(0)=1.
    # The sum S1 should probably exclude self-distances?
    # In pdist, we don't include self. 
    # In the logsumexp derivation for min/max over pairs, we sum over i<j.
    # My D includes i=j.
    # Let's mask the diagonal.
    mask = ~np.eye(14, dtype=bool)
    
    W1 = np.zeros_like(D)
    W1[mask] = np.exp(-beta * D[mask])
    S1 = np.sum(W1)
    if S1 > 0:
        W1 = W1 / S1
    else:
        W1 = 0.0
        
    W2 = np.zeros_like(D)
    W2[mask] = np.exp(beta * D[mask])
    S2 = np.sum(W2)
    if S2 > 0:
        W2 = W2 / S2
    else:
        W2 = 0.0
        
    # Gradient for point k:
    # grad_k = (2/beta) * sum_j (W2_kj - W1_kj) * (P_k - P_j)
    
    # Vectorized:
    # Term = (W2 - W1) * (P_k - P_j)
    # Sum over j
    
    # diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]  # (14, 14, 3)
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    
    # weights_diff = W2 - W1  # (14, 14)
    weights_diff = W2 - W1
    
    # grad_k = (2/beta) * sum_j weights_diff[k, j] * diff[k, j]
    # This is a matrix multiplication of weights_diff and diff?
    # sum_j W[k,j] * (P_k - P_j)
    # = P_k * sum_j W[k,j] - sum_j W[k,j] P_j
    
    sum_W = np.sum(weights_diff, axis=1, keepdims=True) # (14, 1)
    # sum_j W[k,j] P_j is W @ P
    weighted_P = weights_diff @ P # (14, 3)
    
    grad_P = 2.0 / beta * (P * sum_W - weighted_P)
    
    return obj_val, grad_P.flatten()

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes the min/max distance ratio for 14 points in 3D.
    Strategy: Deterministic gradient-based optimization using L-BFGS-B
    on a smooth log-sum-exp approximation of the min/max ratio objective.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    n = 14
    best_score = 0.0
    best_points = None
    
    # Initial guess: Fibonacci sphere
    def fibonacci_sphere(n):
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - 2 * (i + 0.5) / n
            r = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / golden_ratio
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)
    
    # Try multiple initializations and betas
    # Beta controls the sharpness of the approximation.
    # Small beta = closer to true max/min. Large beta = smoother.
    # We can start with larger beta and anneal to smaller beta.
    
    rng = np.random.default_rng(42)
    
    # List of (initial_points, beta_start, beta_end)
    # We'll run a few restarts with different random perturbations of the fib sphere
    
    for restart in range(5):
        if time.time() > deadline:
            break
            
        # Initialize
        if restart == 0:
            init_pts = fibonacci_sphere(n)
        else:
            init_pts = fibonacci_sphere(n)
            # Perturb
            noise = rng.normal(scale=0.1, size=init_pts.shape)
            init_pts += noise
            # Re-center and scale
            init_pts -= np.mean(init_pts, axis=0)
            scale = np.linalg.norm(init_pts)
            if scale > 1e-6:
                init_pts /= scale
            else:
                init_pts = fibonacci_sphere(n)
                
        x0 = init_pts.flatten()
        
        # Beta schedule: start smooth, end sharp
        # The objective function changes with beta, so we can't just minimize one function.
        # We will run L-BFGS-B for a few iterations with a specific beta,
        # then update the result and run with a smaller beta.
        
        current_x = x0.copy()
        
        betas = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        
        for beta in betas:
            if time.time() > deadline:
                break
                
            # Optimize with current beta
            # Max iterations per beta stage
            max_iter = 200
            
            res = minimize(
                _objective_and_grad,
                current_x,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            current_x = res.x
            
            # Evaluate true score
            pts = current_x.reshape(14, 3)
            score = _score(pts)
            
            if score > best_score:
                best_score = score
                best_points = pts.copy()
                
            # If score is 0 or very low, something went wrong (e.g. points collapsed)
            if score < 1e-6:
                break
                
    # If no valid points found (should not happen), return fib sphere
    if best_points is None:
        best_points = fibonacci_sphere(n)
        
    return best_points

# EVOLVE-BLOCK-END
