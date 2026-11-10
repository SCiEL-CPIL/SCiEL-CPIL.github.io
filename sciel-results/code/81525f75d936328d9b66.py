# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _evaluate(points: np.ndarray) -> float:
    """Compute the metric (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    # Check for non-finite values
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def _reconstruct_points_from_gram(gram: np.ndarray) -> np.ndarray:
    """
    Reconstruct points from a Gram matrix.
    The Gram matrix G = P P^T.
    We want P such that P P^T = G.
    Since we are in 3D, G should have rank <= 3.
    We perform eigendecomposition G = V D V^T.
    If G is PSD, D has non-negative eigenvalues.
    P = V_k D_k^{1/2} where V_k are eigenvectors corresponding to the largest 3 eigenvalues.
    """
    # Symmetrize to remove numerical noise
    gram = (gram + gram.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(gram)
    
    # Sort eigenvalues in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Take the top 3 eigenvalues and eigenvectors
    # Ensure eigenvalues are non-negative
    top_vals = np.maximum(eigvals[:3], 0)
    top_vecs = eigvecs[:, :3]
    
    # P = top_vecs * sqrt(top_vals)
    # top_vecs is (14, 3), top_vals is (3,)
    # We need to scale columns of top_vecs by sqrt(top_vals)
    sqrt_vals = np.sqrt(top_vals)
    points = top_vecs * sqrt_vals
    
    return points

def _distances_to_gram(dist_sq: np.ndarray) -> np.ndarray:
    """
    Convert pairwise squared distances to a Gram matrix.
    Given dist_sq[i,j] = ||p_i - p_j||^2 = G_ii + G_jj - 2 G_ij
    We can recover G up to a translation. 
    To fix the translation, we set the centroid to 0.
    If centroid is 0, sum(p_i) = 0 => sum_i G_ij = 0 for all j? No.
    Let's use the formula: G_ij = -0.5 * (d_ij^2 - d_i0^2 - d_j0^2 + d_00^2)
    This fixes point 0 as origin. But we want centered points for best rank properties?
    Actually, any translation preserves distances. 
    However, the rank of the Gram matrix of centered points is <= 3.
    The Gram matrix of uncentered points is also rank <= 4 (due to the 1 in the constant term).
    To ensure we can embed in 3D, we should center the points.
    
    Standard double centering:
    G = -1/2 * H * D^2 * H
    where H = I - (1/n) * ones
    D^2 is the matrix of squared distances.
    """
    n = dist_sq.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    # dist_sq is the matrix of squared distances
    # G = -0.5 * H @ dist_sq @ H
    G = -0.5 * (H @ dist_sq @ H)
    return G

def _optimize_distances(dist_sq_init, time_limit):
    """
    Optimize the squared distances to maximize (d_min / d_max)^2.
    This is equivalent to maximizing log(d_min) - log(d_max).
    We minimize - (log(d_min) - log(d_max)).
    Variables are the squared distances d_ij^2.
    Constraints:
    1. The Gram matrix derived from these distances must be PSD (to be embeddable in Euclidean space).
       Actually, we can just check PSD and penalize if not, or use a parameterization that ensures PSD?
       Directly optimizing distances is hard because of the PSD constraint on the derived Gram matrix.
       
    Alternative Strategy:
    Optimize the Gram matrix G directly, but constrain it to be PSD and rank <= 3.
    But G has 14x14 = 196 variables. Rank <= 3 is a non-convex constraint.
    
    Let's stick to optimizing the points directly but use a smarter parameterization?
    No, the directive says "MATERIALLY DIFFERENT".
    
    Let's try optimizing the **log-ratio** using a smooth approximation on the **points**, 
    but using **SLSQP** with **numerical gradients** and **multiple random restarts** with 
    **simulated annealing** on the objective landscape?
    
    Actually, the previous attempt used L-BFGS-B on a soft objective.
    Let's try **Differential Evolution** (global optimizer) on the points, 
    followed by **SLSQP** refinement on the exact metric (using numerical gradients).
    DE is robust to local minima and handles the 42-dimensional space (14*3) well.
    """
    pass

def _objective_soft(pts, beta):
    """Smooth objective for global search."""
    dists = pdist(pts)
    if np.any(dists <= 0):
        return 1e6
    
    # Soft min and max
    # We want to maximize d_min / d_max
    # log(d_min) - log(d_max)
    # Approximate min with -logsumexp(-beta * d) / beta
    # Approximate max with logsumexp(beta * d) / beta
    
    x_min = -beta * dists
    max_x_min = np.max(x_min)
    log_sum_exp_min = max_x_min + np.log(np.sum(np.exp(x_min - max_x_min)))
    d_min_soft = -log_sum_exp_min / beta
    
    x_max = beta * dists
    max_x_max = np.max(x_max)
    log_sum_exp_max = max_x_max + np.log(np.sum(np.exp(x_max - max_x_max)))
    d_max_soft = log_sum_exp_max / beta
    
    if d_min_soft < 1e-12 or d_max_soft < 1e-12:
        return 1e6
        
    val = np.log(d_min_soft) - np.log(d_max_soft)
    return -val

def _objective_exact(pts):
    """Exact objective for local refinement."""
    dists = pdist(pts)
    if np.any(dists <= 0):
        return 1e6
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 1e6
    val = np.log(d_min) - np.log(d_max)
    return -val

def _refine_slsqp(points, time_limit):
    """Refine using SLSQP with numerical gradients."""
    start_time = time.time()
    best_points = points.copy()
    best_score = _evaluate(points)
    
    def obj_func(x):
        return _objective_exact(x.reshape(14, 3))
        
    try:
        result = minimize(
            obj_func,
            points.flatten(),
            method='SLSQP',
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        refined_pts = result.x.reshape(14, 3)
        refined_score = _evaluate(refined_pts)
        
        if refined_score > best_score:
            best_points = refined_pts
            best_score = refined_score
            
    except Exception:
        pass
        
    return best_points, best_score

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Strategy: Global search using Differential Evolution on a smooth surrogate,
    followed by local refinement using SLSQP on the exact metric.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit
    
    rng = np.random.RandomState(42)
    
    # Initial population for DE
    # We can seed with icosahedron + poles and random points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico = ico / norms
    
    poles = np.array([[0, 0, 1], [0, 0, -1]])
    initial_set = np.vstack([ico, poles])
    
    # Generate random initial points
    random_points = rng.randn(14, 3)
    random_points /= np.linalg.norm(random_points, axis=1, keepdims=True)
    
    # Combine for DE population
    # DE requires a population of size > 5. Let's use 50.
    pop_size = 50
    x0 = np.zeros((pop_size, 42))
    x0[0] = initial_set.flatten()
    x0[1] = random_points.flatten()
    for i in range(2, pop_size):
        p = rng.randn(14, 3)
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        x0[i] = p.flatten()
    
    # Run Differential Evolution
    # We use the soft objective for DE to provide a smooth landscape
    beta_de = 50.0
    
    def de_obj(x):
        return _objective_soft(x.reshape(14, 3), beta_de)
        
    try:
        res_de = minimize(
            de_obj,
            x0,
            method='differential_evolution',
            options={
                'maxiter': 50, 
                'polish': False, # We will polish separately
                'seed': 42,
                'init': 'sobol' # Better coverage
            }
        )
        best_de_pts = res_de.x.reshape(14, 3)
        best_de_score = _evaluate(best_de_pts)
        
    except Exception:
        best_de_pts = initial_set
        best_de_score = _evaluate(best_de_pts)
        
    # Keep the best few candidates from DE and the initial ones
    candidates = [best_de_pts, initial_set, random_points]
    
    # Add some random restarts if time permits
    if time.time() < deadline - 100:
        for _ in range(5):
            p = rng.randn(14, 3)
            p /= np.linalg.norm(p, axis=1, keepdims=True)
            candidates.append(p)
            
    # Refine each candidate using SLSQP
    best_points = candidates[0]
    best_score = _evaluate(best_points)
    
    for cand in candidates:
        if time.time() > deadline - 10:
            break
            
        time_per_refine = max(10, (deadline - 10 - time.time()) / len(candidates))
        
        refined_pts, refined_score = _refine_slsqp(cand, time_per_refine)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts
            
    # Final polish with Nelder-Mead to escape any SLSQP local minima
    if time.time() < deadline - 5:
        def nm_obj(x):
            return -_evaluate(x.reshape(14, 3))
            
        try:
            res_nm = minimize(
                nm_obj,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            polished_pts = res_nm.x.reshape(14, 3)
            polished_score = _evaluate(polished_pts)
            
            if polished_score > best_score:
                best_points = polished_pts
                best_score = polished_score
        except Exception:
            pass

    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        best_points = initial_set

    return best_points

# EVOLVE-BLOCK-END
