import time
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist

N = 14
D = 3
DEADLINE = 140.0
_t0 = None

def _true_score(pts):
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min(); mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx/mn)**2

def _gram_to_points(G, rank=D):
    """Extract points from Gram matrix via eigendecomposition."""
    G = (G + G.T) / 2.0
    vals, vecs = eigh(G)
    # Keep top 'rank' eigenvalues
    idx = np.argsort(vals)[::-1]
    idx = idx[:rank]
    vals_sel = np.maximum(vals[idx], 0)
    vecs_sel = vecs[:, idx]
    pts = vecs_sel @ np.diag(np.sqrt(vals_sel))
    return pts

def _fib_sphere(n, r=1.0):
    i = np.arange(n)
    phi = np.arccos(1 - 2*(i+0.5)/n)
    theta = np.pi*(1+5**0.5)*i
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)
    return np.column_stack([x,y,z])

def _icosa14():
    t = (1+5**0.5)/2
    v = [(-1,t,0),(1,t,0),(-1,-t,0),(1,-t,0),(0,-1,t),(0,1,t),(0,-1,-t),(0,1,-t),(t,0,-1),(t,0,1),(-t,0,-1),(-t,0,1)]
    v = np.array(v,float)
    v /= np.linalg.norm(v[0])
    extra = np.array([[0,0,1],[0,0,-1]],float)
    return np.vstack([v, extra])

def _gram_objective(G):
    """
    Objective on Gram matrix G (n x n):
    - G should be PSD (penalize negative eigenvalues)
    - G should be low-rank (penalize spread of eigenvalues)
    - Off-diagonal squared distances should be tightly clustered (minimize variance)
    - Trace fixed for scale invariance
    """
    G = (G + G.T) / 2.0
    vals = np.linalg.eigvalsh(G)
    # PSD penalty
    neg_pen = np.sum(np.maximum(-vals, 0)**2) * 10.0
    # Rank penalty: encourage only top D eigenvalues to be significant
    vals_sorted = np.sort(vals)[::-1]
    if len(vals_sorted) > D:
        rank_pen = np.sum(vals_sorted[D:]**2) * 1.0
    else:
        rank_pen = 0.0
    # Distance clustering: off-diagonal entries of G represent squared distances (up to scale)
    off_diag = G[~np.eye(N, dtype=bool)]
    mean_d = off_diag.mean()
    var_d = off_diag.var()
    # Normalize by mean to make scale-invariant
    if abs(mean_d) < 1e-12:
        return 1e10
    cv = np.sqrt(var_d) / abs(mean_d)
    # Also penalize if min distance is too small relative to mean
    min_d = off_diag.min()
    if min_d < 1e-6:
        return 1e10
    ratio_pen = (abs(mean_d) / min_d) ** 2
    
    # Trace normalization penalty: want trace ~ N (since sum of squared norms = N for unit-scaled)
    tr = np.trace(G)
    trace_pen = (tr - N)**2 * 0.01
    
    return cv * 10.0 + rank_pen + neg_pen + trace_pen + ratio_pen * 0.1

def _project_gram(G):
    """Project G to be PSD and centered (zero row sums for centering)."""
    G = (G + G.T) / 2.0
    # Center: G = P P^T where P has zero mean => G @ 1 = 0
    # Enforce by projecting onto subspace orthogonal to all-ones vector
    ones = np.ones((N, 1))
    proj = np.eye(N) - ones @ ones.T / N
    G = proj @ G @ proj
    # PSD projection
    vals, vecs = eigh(G)
    vals = np.maximum(vals, 0)
    G = vecs @ np.diag(vals) @ vecs.T
    return G

def _optimize_gram(G0, max_iter=2000, deadline=None):
    """
    Iterative optimization on Gram matrix using gradient-free Nelder-Mead
    on the vectorized Gram matrix, with periodic projection.
    """
    G = G0.copy()
    G = _project_gram(G)
    # Vectorize lower triangle for optimization
    def vec_G(G):
        return G[np.tril_indices(N, -1)]
    
    def obj_vec(x):
        G_mat = np.zeros((N, N))
        G_mat[np.tril_indices(N, -1)] = x
        G_mat = G_mat + G_mat.T
        G_mat = _project_gram(G_mat)
        return _gram_objective(G_mat)
    
    from scipy.optimize import minimize
    x0 = vec_G(G)
    try:
        res = minimize(obj_vec, x0, method='Nelder-Mead', 
                       options=dict(maxfev=5000, maxiter=5000, xatol=1e-8, fatol=1e-10))
        G_opt = np.zeros((N, N))
        G_opt[np.tril_indices(N, -1)] = res.x
        G_opt = G_opt + G_opt.T
        G_opt = _project_gram(G_opt)
        return G_opt
    except Exception:
        return G

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    rng = np.random.default_rng(42)
    
    # Generate initial point configurations
    starts = []
    starts.append(_fib_sphere(N))
    starts.append(_icosa14())
    for k in range(8):
        starts.append(rng.uniform(-1, 1, size=(N, D)))
    
    best_pts = None
    best_score = np.inf
    
    def consider(pts):
        nonlocal best_pts, best_score
        if pts is None:
            return
        pts = pts - pts.mean(axis=0)
        s = _true_score(pts)
        if s < best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Evaluate raw starts
    for s in starts:
        consider(s)
    
    # For each start, compute Gram matrix and optimize in Gram space
    for s in starts:
        if time.time() - _t0 > DEADLINE - 10:
            break
        
        pts = s - s.mean(axis=0)
        # Normalize trace to N
        tr = np.sum(pts**2)
        if tr > 1e-12:
            pts = pts * np.sqrt(N / tr)
        G = pts @ pts.T
        G = _project_gram(G)
        
        # Optimize Gram matrix
        G_opt = _optimize_gram(G, max_iter=3000)
        pts_opt = _gram_to_points(G_opt, rank=D)
        consider(pts_opt)
        
        # Also try a random perturbation of the Gram matrix
        if time.time() - _t0 > DEADLINE - 10:
            break
        G_pert = G + rng.normal(0, 0.05, size=(N, N))
        G_pert = _project_gram(G_pert)
        G_opt2 = _optimize_gram(G_pert, max_iter=2000)
        pts_opt2 = _gram_to_points(G_opt2, rank=D)
        consider(pts_opt2)
    
    # Final polish: local search in coordinate space on best found
    if best_pts is not None:
        from scipy.optimize import minimize
        def coord_obj(x):
            pts = x.reshape(N, D)
            pts = pts - pts.mean(axis=0)
            d = cdist(pts, pts)
            np.fill_diagonal(d, np.inf)
            mn = d.min(); mx = d.max()
            if mn < 1e-9:
                return 1e10
            return (np.log(mx) - np.log(mn))
        
        x0 = best_pts.flatten()
        try:
            res = minimize(coord_obj, x0, method='Nelder-Mead',
                          options=dict(maxfev=3000, maxiter=3000))
            pts_final = res.x.reshape(N, D)
            consider(pts_final)
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = _fib_sphere(N)
    
    return best_pts
