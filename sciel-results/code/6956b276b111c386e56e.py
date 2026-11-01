import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.linalg import sqrtm, cholesky
import sys

N = 14
D = 3
DEADLINE = 145.0
_t0 = None

def _true_score(pts):
    """
    Calculate the squared ratio of max to min pairwise distance.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _gram_to_points(G):
    """
    Convert a Gram matrix G (N x N) to points in R^D.
    We enforce rank <= D by taking the top D eigenvectors.
    If G is PSD, G = V^T V. Points are rows of V.
    """
    # Ensure symmetry
    G = (G + G.T) / 2.0
    
    # Eigen decomposition
    # Since G is PSD, eigenvalues should be >= 0. Clip small negatives.
    w, v = np.linalg.eigh(G)
    w = np.maximum(w, 0)
    
    # Take top D eigenvectors
    # v is N x N, columns are eigenvectors.
    # We want N x D matrix X such that X X^T = G (approximately)
    # X = v_top * sqrt(w_top)
    idx = np.argsort(w)[::-1][:D]
    v_top = v[:, idx]
    w_top = w[idx]
    
    X = v_top * np.sqrt(w_top)
    
    # Center the points (remove mean) to fix translational invariance
    X = X - X.mean(axis=0)
    
    return X

def _points_to_gram(X):
    """
    Convert points X (N x D) to Gram matrix G = X X^T.
    """
    G = X @ X.T
    return (G + G.T) / 2.0

def _soft_ratio_obj_vec(vec):
    """
    Objective function for optimization.
    Input: vec is a flattened symmetric matrix G (N*(N+1)/2 elements).
    Output: Approximate squared ratio (max_d2 / min_d2) + penalty for non-PSD.
    
    We use soft-min and soft-max on the squared distances derived from G.
    d_ij^2 = G_ii + G_jj - 2 G_ij.
    """
    # Reconstruct symmetric G from vec
    n = N
    G = np.zeros((n, n))
    k = 0
    for i in range(n):
        for j in range(i, n):
            G[i, j] = vec[k]
            G[j, i] = vec[k]
            k += 1
            
    # Penalty for non-PSD: sum of negative eigenvalues squared
    w = np.linalg.eigvalsh(G)
    neg_w = w[w < 0]
    psd_penalty = 100.0 * np.sum(neg_w**2) if len(neg_w) > 0 else 0.0
    
    # Calculate squared distances
    diag_G = np.diag(G)
    d2 = diag_G[:, None] + diag_G[None, :] - 2 * G
    np.fill_diagonal(d2, 0)
    
    # Get upper triangle
    iu = np.triu_indices(n, k=1)
    d2_flat = d2[iu]
    
    if len(d2_flat) == 0 or np.any(d2_flat <= 0):
        return 1e10 + psd_penalty
        
    # Soft min and max
    # k controls sharpness. 
    # For ratio minimization, we want to push min up and max down.
    # Objective: (soft_max / soft_min)
    # This is differentiable.
    
    # To avoid numerical issues, we can optimize log(ratio) or just ratio.
    # Let's use a temperature parameter k.
    k_temp = 0.05
    
    m_min = np.min(d2_flat)
    m_max = np.max(d2_flat)
    
    # Soft min: -k log(sum(exp(-d2/k)))
    # Shift by min for stability
    exp_min = np.exp(-(d2_flat - m_min) / k_temp)
    sum_exp_min = np.sum(exp_min)
    soft_min = m_min + k_temp * np.log(sum_exp_min)
    
    # Soft max: k log(sum(exp(d2/k)))
    exp_max = np.exp((d2_flat - m_max) / k_temp)
    sum_exp_max = np.sum(exp_max)
    soft_max = m_max + k_temp * np.log(sum_exp_max)
    
    if soft_min < 1e-6:
        return 1e10 + psd_penalty
        
    ratio = soft_max / soft_min
    
    # We want to minimize ratio.
    # Add a small regularization to encourage distinct points?
    # The PSD penalty handles rank/degeneracy somewhat.
    
    return ratio + psd_penalty

def _construct_from_gram(G_init, time_limit):
    """
    Optimize the Gram matrix starting from G_init.
    """
    # Flatten symmetric matrix
    vec_init = []
    for i in range(N):
        for j in range(i, N):
            vec_init.append(G_init[i, j])
    vec_init = np.array(vec_init)
    
    # Use L-BFGS-B
    try:
        res = minimize(
            _soft_ratio_obj_vec,
            vec_init,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        vec_opt = res.x
        G_opt = np.zeros((N, N))
        k = 0
        for i in range(N):
            for j in range(i, N):
                G_opt[i, j] = vec_opt[k]
                G_opt[j, i] = vec_opt[k]
                k += 1
        return G_opt
    except Exception as e:
        print(f"Optimization failed: {e}", file=sys.stderr)
        return G_init

def _get_initial_grams():
    """
    Generate a few diverse initial Gram matrices.
    1. Icosahedron-based (12 points) + 2 extra
    2. Fibonacci sphere
    3. Random
    """
    inits = []
    
    # 1. Fibonacci Sphere
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    pts_fib = np.column_stack([x, y, z])
    pts_fib = pts_fib - pts_fib.mean(axis=0)
    inits.append(_points_to_gram(pts_fib))
    
    # 2. Icosahedron vertices (12) + 2 points (e.g., near poles or random)
    # Icosahedron vertices
    phi_gold = (1 + np.sqrt(5)) / 2
    ico_pts = np.array([
        [0, 1, phi_gold], [0, -1, phi_gold],
        [0, 1, -phi_gold], [0, -1, -phi_gold],
        [1, phi_gold, 0], [-1, phi_gold, 0],
        [1, -phi_gold, 0], [-1, -phi_gold, 0],
        [phi_gold, 0, 1], [-phi_gold, 0, 1],
        [phi_gold, 0, -1], [-phi_gold, 0, -1]
    ])
    ico_pts = ico_pts / np.linalg.norm(ico_pts, axis=1)[:, None]
    
    # Add 2 points. Let's try adding them at the "holes" or just random on sphere.
    # For simplicity, add 2 random points on sphere, then optimize.
    rng = np.random.default_rng(42)
    extra = rng.normal(size=(2, 3))
    extra = extra / np.linalg.norm(extra, axis=1)[:, None]
    pts_ico = np.vstack([ico_pts, extra])
    pts_ico = pts_ico - pts_ico.mean(axis=0)
    inits.append(_points_to_gram(pts_ico))
    
    # 3. Random
    rng2 = np.random.default_rng(123)
    pts_rand = rng2.normal(size=(N, 3))
    pts_rand = pts_rand - pts_rand.mean(axis=0)
    inits.append(_points_to_gram(pts_rand))
    
    return inits

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    best_pts = None
    best_score = np.inf
    
    inits = _get_initial_grams()
    
    for G_init in inits:
        if time.time() - _t0 > DEADLINE - 20:
            break
            
        # Optimize Gram
        G_opt = _construct_from_gram(G_init, time.time() - _t0)
        
        # Convert back to points
        pts_opt = _gram_to_points(G_opt)
        
        # Calculate true score
        score_opt = _true_score(pts_opt)
        
        if score_opt < best_score:
            best_score = score_opt
            best_pts = pts_opt.copy()
            
        # If we have time, do a second pass with tighter tolerance or different start
        if time.time() - _t0 < DEADLINE - 10:
            # Perturb the optimized G slightly and re-optimize
            rng = np.random.default_rng(int(time.time() * 1000) % 10000)
            G_pert = G_opt + rng.normal(scale=0.01, size=G_opt.shape)
            G_pert = (G_pert + G_pert.T) / 2.0
            G_opt2 = _construct_from_gram(G_pert, time.time() - _t0)
            pts_opt2 = _gram_to_points(G_opt2)
            score_opt2 = _true_score(pts_opt2)
            if score_opt2 < best_score:
                best_score = score_opt2
                best_pts = pts_opt2.copy()

    # Fallback
    if best_pts is None:
        # Return Fibonacci sphere
        i = np.arange(N)
        phi = np.arccos(1 - 2 * (i + 0.5) / N)
        theta = np.pi * (1 + 5 ** 0.5) * i
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        best_pts = np.column_stack([x, y, z])
        
    return best_pts
