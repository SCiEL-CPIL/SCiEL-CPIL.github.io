"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    n = points.shape[0]
    dists = pdist(points)
    if len(dists) == 0:
        return 1e10
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d < 1e-10:
        return 1e10
    return (max_d / min_d) ** 2

def construct():
    n = N
    d = D
    
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(pairs)
    
    def gram_to_dist_sq(G):
        """Compute squared distances from Gram matrix G (n x n)."""
        diag = np.diag(G)
        d_sq = np.zeros(num_pairs)
        for k, (i, j) in enumerate(pairs):
            d_sq[k] = diag[i] + diag[j] - 2.0 * G[i, j]
        return d_sq
    
    def smooth_max_min_ratio(d_sq, T_max=0.05, T_min=0.05):
        """Smooth approximation of max/min ratio using log-sum-exp."""
        d_sq_clipped = np.maximum(d_sq, 1e-14)
        max_approx = T_max * np.log(np.sum(np.exp(d_sq_clipped / T_max)))
        min_approx = -T_min * np.log(np.sum(np.exp(-d_sq_clipped / T_min)))
        if min_approx < 1e-14:
            return 1e10
        return max_approx / min_approx
    
    def objective_gram(G_flat, T_max=0.05, T_min=0.05, w_rank=50.0, w_psd=50.0, w_scale=5.0):
        G = G_flat.reshape(n, n)
        G = 0.5 * (G + G.T)
        
        eigvals = np.linalg.eigvalsh(G)
        
        neg_eigs = eigvals[eigvals < 0]
        psd_penalty = np.sum(neg_eigs ** 2)
        
        rank_penalty = np.sum(eigvals[d:] ** 2)
        
        d_sq = gram_to_dist_sq(G)
        
        if np.any(d_sq < -1e-5):
            return 1e10 + psd_penalty + rank_penalty
        
        d_sq_clipped = np.maximum(d_sq, 1e-14)
        
        ratio = smooth_max_min_ratio(d_sq_clipped, T_max, T_min)
        
        trace_G = np.trace(G)
        scale_penalty = (trace_G - 1.0) ** 2
        
        return ratio + w_rank * rank_penalty + w_psd * psd_penalty + w_scale * scale_penalty
    
    def objective_coords(x, T_max=0.05, T_min=0.05):
        V = x.reshape(n, d)
        d_sq = pdist(V) ** 2
        if np.any(d_sq < 1e-12):
            return 1e10
        d_sq_clipped = np.maximum(d_sq, 1e-14)
        return smooth_max_min_ratio(d_sq_clipped, T_max, T_min)
    
    np.random.seed(42)
    
    initializations = []
    
    # 1. Random rank-d Gram
    V_rand = np.random.randn(n, d)
    G_rand = V_rand @ V_rand.T
    initializations.append(G_rand.flatten())
    
    # 2. Icosahedron + 2 extra points
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(ico_verts, axis=1, keepdims=True)
    ico_verts = ico_verts / norms
    
    extra1 = np.array([1.0, 0, 0])
    extra2 = np.array([0, 0, 1.0])
    
    V_ico = np.vstack([ico_verts, extra1, extra2])
    V_ico = V_ico / np.linalg.norm(V_ico, axis=1, keepdims=True)
    G_ico = V_ico @ V_ico.T
    initializations.append(G_ico.flatten())
    
    # 3. Cube + octahedron hybrid
    cube_verts = np.array([
        [1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],
        [-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]
    ]) / np.sqrt(3)
    octa_verts = np.array([
        [1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]
    ])
    V_hybrid = np.vstack([cube_verts, octa_verts])
    G_hybrid = V_hybrid @ V_hybrid.T
    initializations.append(G_hybrid.flatten())
    
    # 4. Random structured
    V_struct = np.random.randn(n, d)
    V_struct = V_struct / np.linalg.norm(V_struct, axis=1, keepdims=True)
    G_struct = V_struct @ V_struct.T
    initializations.append(G_struct.flatten())
    
    best_G_flat = None
    best_val = 1e10
    
    for x0 in initializations:
        try:
            res = minimize(objective_gram, x0, method='L-BFGS-B',
                           options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_G_flat = res.x
        except Exception:
            pass
    
    if best_G_flat is None:
        return np.random.randn(n, d)
    
    G_opt = best_G_flat.reshape(n, n)
    G_opt = 0.5 * (G_opt + G_opt.T)
    
    eigvals, eigvecs = np.linalg.eigh(G_opt)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    V = eigvecs[:, :d] * np.sqrt(np.maximum(eigvals[:d], 0))
    points = V
    
    # Phase 2: Refine in coordinate space with annealed temperature
    current_points = points
    for T in [0.1, 0.05, 0.02, 0.01]:
        try:
            res = minimize(objective_coords, current_points.flatten(), 
                          args=(T, T),
                          method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _compute_score(current_points):
                current_points = res.x.reshape(n, d)
        except Exception:
            pass
    
    score = _compute_score(current_points)
    if score > 1e8:
        return points
    
    return current_points

# EVOLVE-BLOCK-END
