import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import itertools

N = 14
D = 3
DEADLINE = 115.0  # Safety margin from 150s limit

def _pairwise_distances(pts):
    """Compute pairwise distances between all points."""
    d = cdist(pts, pts)
    mask = ~np.eye(N, dtype=bool)
    return d[mask]

def _true_score(pts):
    """Calculate the actual score: (max/min)^2."""
    d = _pairwise_distances(pts)
    if len(d) == 0:
        return np.inf
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _soft_log_ratio(pts, temp=0.1):
    """
    Smooth approximation of log(max/min) using soft min/max.
    This is differentiable and directly targets the ratio.
    """
    d = _pairwise_distances(pts)
    if np.any(d < 1e-12):
        return 1e6
    
    log_d = np.log(d)
    
    # Soft max: log(sum(exp(d/temp))) / temp
    # Use stable version
    log_max = temp * np.log(np.sum(np.exp(log_d / temp)))
    log_min = -temp * np.log(np.sum(np.exp(-log_d / temp)))
    
    return log_max - log_min

def _objective_coords(x, temp=0.1):
    """
    Coordinate-space objective: minimize soft log ratio.
    """
    pts = x.reshape(N, D)
    pts = pts - np.mean(pts, axis=0)
    return _soft_log_ratio(pts, temp)

def _objective_gram(g_vec, temp=0.1):
    """
    Gram matrix objective: g_vec is upper-triangular entries of NxN Gram matrix.
    Gram matrix G must be PSD with rank <= D.
    """
    n = N
    # Reconstruct symmetric Gram matrix from upper triangular
    G = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i, n):
            G[i, j] = g_vec[idx]
            G[j, i] = g_vec[idx]
            idx += 1
    
    # Ensure diagonal is positive (squared norms)
    diag_idx = np.diag_indices(n)
    G[diag_idx] = np.maximum(G[diag_idx], 1e-8)
    
    # Get eigenvalues to check PSD and compute distances
    eigvals, eigvecs = np.linalg.eigh(G)
    
    # If not PSD, penalize
    min_eig = eigvals.min()
    if min_eig < -1e-6:
        return 1e6 + abs(min_eig) * 1e3
    
    # Clamp small negative eigenvalues to zero
    eigvals_clamped = np.maximum(eigvals, 0)
    
    # Compute pairwise squared distances from Gram matrix
    # d_ij^2 = G_ii + G_jj - 2*G_ij
    diag = np.diag(G)
    dist_sq = diag[:, None] + diag[None, :] - 2 * G
    dist_sq = np.maximum(dist_sq, 1e-12)
    dist = np.sqrt(dist_sq)
    
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    d = dist[mask]
    
    if np.any(d < 1e-12):
        return 1e6
    
    log_d = np.log(d)
    log_max = temp * np.log(np.sum(np.exp(log_d / temp)))
    log_min = -temp * np.log(np.sum(np.exp(-log_d / temp)))
    
    return log_max - log_min

def _generate_icosahedron_plus_2():
    """Generate icosahedron vertices (12) + 2 center-ish points."""
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = np.array(verts, dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Add 2 points near center but offset
    p13 = np.array([0.3, 0.3, 0.3])
    p14 = np.array([-0.3, -0.3, -0.3])
    pts = np.vstack([verts, p13, p14])
    return pts

def _generate_fib_sphere_points():
    """Generate points on a sphere using Fibonacci distribution."""
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack((x, y, z))

def _generate_ring_config():
    """Generate points in two rings on parallel circles."""
    n_ring1 = 7
    n_ring2 = 7
    pts = []
    # Ring 1 at z = +0.5
    for i in range(n_ring1):
        angle = 2 * np.pi * i / n_ring1
        pts.append([np.cos(angle), np.sin(angle), 0.5])
    # Ring 2 at z = -0.5, rotated
    for i in range(n_ring2):
        angle = 2 * np.pi * (i + 0.5) / n_ring2
        pts.append([np.cos(angle), np.sin(angle), -0.5])
    return np.array(pts, dtype=float)

def _generate_random_points():
    """Generate random points."""
    return np.random.randn(N, D)

def construct():
    t_start = time.time()
    
    best_score = np.inf
    best_pts = None
    
    # Generate diverse initial candidates
    candidates = [
        _generate_icosahedron_plus_2(),
        _generate_fib_sphere_points(),
        _generate_ring_config(),
        np.random.randn(N, D),
        np.random.randn(N, D) * 0.5,
    ]
    
    # Add more random restarts if time permits
    max_restarts = 30
    current_restarts = len(candidates)
    
    while current_restarts < max_restarts:
        if time.time() - t_start > DEADLINE:
            break
        candidates.append(np.random.randn(N, D))
        current_restarts += 1
    
    # Optimize each candidate using coordinate-space optimization
    for i, init_pts in enumerate(candidates):
        if time.time() - t_start > DEADLINE:
            break
            
        # Center initial points
        init_pts = init_pts - np.mean(init_pts, axis=0)
        
        try:
            # Use L-BFGS-B for local optimization
            res = minimize(
                _objective_coords, 
                init_pts.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            opt_pts = res.x.reshape(N, D)
            opt_pts = opt_pts - np.mean(opt_pts, axis=0)
            
            # Check for distinct points
            d = _pairwise_distances(opt_pts)
            if d.min() < 1e-6:
                continue
                
            score = _true_score(opt_pts)
            
            if score < best_score:
                best_score = score
                best_pts = opt_pts.copy()
                
        except Exception:
            continue
    
    # Second pass: try Gram matrix optimization from best coordinate solution
    if best_pts is not None and time.time() - t_start < DEADLINE - 5:
        try:
            # Build Gram matrix from best points
            centered = best_pts - np.mean(best_pts, axis=0)
            G = centered @ centered.T
            
            # Extract upper triangular
            g_vec = []
            for i in range(N):
                for j in range(i, N):
                    g_vec.append(G[i, j])
            g_vec = np.array(g_vec)
            
            res_g = minimize(
                _objective_gram, 
                g_vec, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            
            # Reconstruct points from optimized Gram
            g_opt = res_g.x
            G_opt = np.zeros((N, N))
            idx = 0
            for i in range(N):
                for j in range(i, N):
                    G_opt[i, j] = g_opt[idx]
                    G_opt[j, i] = g_opt[idx]
                    idx += 1
            
            # Ensure PSD
            eigvals, eigvecs = np.linalg.eigh(G_opt)
            eigvals_clamped = np.maximum(eigvals, 0)
            G_opt = (eigvecs * eigvals_clamped) @ eigvecs.T
            
            # Extract points via Cholesky or eigen decomposition
            # Use eigen decomposition: G = V diag(lambda) V^T, so X = diag(sqrt(lambda)) V^T
            eigvals_final, eigvecs_final = np.linalg.eigh(G_opt)
            eigvals_final = np.maximum(eigvals_final, 0)
            
            # Keep only top D eigenvalues for 3D embedding
            eigvals_top = eigvals_final[D-1::-1] if D <= N else eigvals_final
            eigvecs_top = eigvecs_final[:, D-1::-1] if D <= N else eigvecs_final
            
            # Actually, let's just take top 3
            idx_sort = np.argsort(eigvals_final)[::-1]
            eigvals_3 = eigvals_final[idx_sort[:D]]
            eigvecs_3 = eigvecs_final[:, idx_sort[:D]]
            
            opt_pts_gram = eigvecs_3 * np.sqrt(np.maximum(eigvals_3, 0))
            opt_pts_gram = opt_pts_gram - np.mean(opt_pts_gram, axis=0)
            
            d = _pairwise_distances(opt_pts_gram)
            if d.min() > 1e-6:
                score = _true_score(opt_pts_gram)
                if score < best_score:
                    best_score = score
                    best_pts = opt_pts_gram.copy()
                    
        except Exception:
            pass
    
    # If no valid points were found, fallback
    if best_pts is None:
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_pts = np.array(grid, dtype=float)
        
    return best_pts
