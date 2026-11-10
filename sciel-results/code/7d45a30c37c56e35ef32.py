# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses Gram matrix optimization with PSD/Rank-3 projection and smooth surrogates.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580.0  # 20s safety margin

    def get_score(pts):
        """Calculate the actual metric: (d_min / d_max)^2"""
        if not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def make_icosahedron_plus_poles():
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        norms_ico = np.linalg.norm(ico_verts, axis=1)
        ico_verts_norm = ico_verts / norms_ico[:, np.newaxis]
        
        # Add two poles
        poles = np.array([[0, 0, 1.0], [0, 0, -1.0]])
        points_init = np.vstack([ico_verts_norm, poles])
        
        # Center the points to remove translation invariance
        points_init = points_init - np.mean(points_init, axis=0)
        return points_init

    def make_fibonacci_sphere(n_points):
        points = np.zeros((n_points, 3))
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n_points):
            y = 1 - (i / (n_points - 1)) * 2
            radius = np.sqrt(1 - y**2)
            theta = golden_angle * i
            x = radius * np.cos(theta)
            z = radius * np.sin(theta)
            points[i] = [x, y, z]
        points = points - np.mean(points, axis=0)
        return points

    # Initial points
    init_pts = make_icosahedron_plus_poles()
    best_points = init_pts
    best_score = get_score(best_points)
    
    # Gram Matrix Initialization
    # Centered points have G = X X^T
    G_init = init_pts @ init_pts.T
    
    # Symmetrize
    G_init = (G_init + G_init.T) / 2.0

    def project_gram(G, rank=3):
        """Project G to PSD cone and enforce rank <= 3"""
        try:
            # Ensure symmetry
            G = (G + G.T) / 2.0
            eigvals, eigvecs = np.linalg.eigh(G)
            # Clip negative eigenvalues
            eigvals[eigvals < 0] = 0
            # Enforce rank constraint
            if rank < len(eigvals):
                # Keep largest eigenvalues
                # Note: eigh returns in ascending order
                idx = np.argsort(eigvals)[::-1][:rank]
                eigvals_new = np.zeros_like(eigvals)
                eigvals_new[idx] = eigvals[idx]
                eigvals = eigvals_new
            # Reconstruct
            G_proj = (eigvecs * eigvals) @ eigvecs.T
            G_proj = (G_proj + G_proj.T) / 2.0
            return G_proj
        except Exception:
            return G

    def gram_objective(flat_G, beta=0.05):
        """
        Objective for Gram matrix optimization.
        Minimizes (soft_max(d^2) / soft_min(d^2))^2
        """
        G = flat_G.reshape((n, n))
        G = (G + G.T) / 2.0
        
        # Compute squared distances from Gram matrix
        # d_ij^2 = G_ii + G_jj - 2 G_ij
        diag = np.diag(G)
        # dist_sq matrix
        dist_sq = diag[:, np.newaxis] + diag[np.newaxis, :] - 2 * G
        
        # Ensure non-negative distances (numerical stability)
        dist_sq = np.maximum(dist_sq, 1e-12)
        
        # Extract upper triangle for pdist-like vector
        iu = np.triu_indices(n, k=1)
        dists_sq = dist_sq[iu]
        
        if not np.all(np.isfinite(dists_sq)):
            return 1e10
            
        # Soft min/max on squared distances
        # We want to minimize the ratio, so we minimize (max/min)^2
        
        # Soft min
        a_neg = -dists_sq / beta
        max_a_neg = np.max(a_neg)
        if max_a_neg > 700:
            return 1e10
        log_sum_neg = max_a_neg + np.log(np.sum(np.exp(a_neg - max_a_neg)))
        soft_min_sq = -beta * log_sum_neg
        
        # Soft max
        a_pos = dists_sq / beta
        max_a_pos = np.max(a_pos)
        if max_a_pos > 700:
            return 1e10
        log_sum_pos = max_a_pos + np.log(np.sum(np.exp(a_pos - max_a_pos)))
        soft_max_sq = beta * log_sum_pos
        
        if soft_min_sq <= 1e-9 or soft_max_sq <= 1e-9:
            return 1e10
            
        ratio_sq = (soft_max_sq / soft_min_sq) ** 2
        return ratio_sq

    # Optimization Loop on Gram Matrix
    # We use L-BFGS-B on the flattened Gram matrix.
    # We project after each evaluation? No, L-BFGS-B assumes a smooth function.
    # Instead, we optimize a "penalized" or "projected" version.
    # A common trick: Optimize G, but in the objective, project G first.
    # This makes the objective non-smooth if projection changes the active set.
    # Alternative: Use a penalty for non-PSD or rank > 3.
    # Let's try optimizing G directly with a smooth penalty for eigenvalues < 0 and rank > 3.
    
    def gram_objective_penalized(flat_G, beta=0.05, lambda_psd=100.0, lambda_rank=100.0):
        G = flat_G.reshape((n, n))
        G = (G + G.T) / 2.0
        
        # Eigenvalues for penalties
        try:
            eigvals = np.linalg.eigvalsh(G)
        except Exception:
            return 1e10
            
        # Penalty for negative eigenvalues (PSD violation)
        neg_eigs = eigvals[eigvals < 0]
        if len(neg_eigs) > 0:
            pen_psd = lambda_psd * np.sum(neg_eigs**2)
        else:
            pen_psd = 0.0
            
        # Penalty for rank > 3 (sum of squares of eigenvalues beyond top 3)
        # Sort eigenvalues descending
        eigvals_sorted = np.sort(eigvals)[::-1]
        if n > 3:
            tail_eigs = eigvals_sorted[3:]
            if len(tail_eigs) > 0:
                pen_rank = lambda_rank * np.sum(tail_eigs**2)
            else:
                pen_rank = 0.0
        else:
            pen_rank = 0.0
            
        # Main objective
        base_obj = gram_objective(flat_G, beta)
        
        return base_obj + pen_psd + pen_rank

    # Run L-BFGS-B on Gram matrix
    if time.time() < deadline - 100.0:
        try:
            res_gram = minimize(
                gram_objective_penalized,
                G_init.ravel(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res_gram.fun < 1e9:
                G_opt = res_gram.x.reshape((n, n))
                G_opt = project_gram(G_opt, rank=3)
                
                # Extract points from G_opt
                try:
                    eigvals, eigvecs = np.linalg.eigh(G_opt)
                    # Take top 3
                    eigvals_top = eigvals[-3:]
                    eigvecs_top = eigvecs[:, -3:]
                    pts_gram = eigvecs_top * np.sqrt(np.maximum(eigvals_top, 0))
                    pts_gram = pts_gram.T # shape (3, n) -> (n, 3)
                    
                    # Center
                    pts_gram = pts_gram - np.mean(pts_gram, axis=0)
                    
                    cand_score = get_score(pts_gram)
                    if cand_score > best_score:
                        best_points = pts_gram
                        best_score = cand_score
                except Exception:
                    pass
        except Exception:
            pass

    # Coordinate Space Polish
    # Use the best points found so far (either initial or from Gram optimization)
    
    def smooth_objective(x, beta=0.05):
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10
        dists = pdist(p, metric='euclidean')
        a_neg = -dists / beta
        max_a_neg = np.max(a_neg)
        if max_a_neg > 700:
            return 1e10
        log_sum_neg = max_a_neg + np.log(np.sum(np.exp(a_neg - max_a_neg)))
        soft_min = -beta * log_sum_neg
        a_pos = dists / beta
        max_a_pos = np.max(a_pos)
        if max_a_pos > 700:
            return 1e10
        log_sum_pos = max_a_pos + np.log(np.sum(np.exp(a_pos - max_a_pos)))
        soft_max = beta * log_sum_pos
        if soft_min <= 1e-9 or soft_max <= 1e-9:
            return 1e10
        return (soft_max / soft_min) ** 2

    def exact_objective(x):
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10
        dists = pdist(p, metric='euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min <= 1e-9 or d_max <= 1e-9:
            return 1e10
        return (d_max / d_min) ** 2

    candidates_to_polish = [best_points.copy(), make_icosahedron_plus_poles(), make_fibonacci_sphere(n)]
    
    # Add random perturbations
    np.random.seed(123)
    for _ in range(5):
        perturb = best_points + np.random.randn(n, d) * 0.1
        candidates_to_polish.append(perturb)
        
    for init_pts in candidates_to_polish:
        if time.time() > deadline - 40.0:
            break
            
        try:
            # Stage 1: Polish with smooth objective
            res_local = minimize(
                lambda x: smooth_objective(x, beta=0.05), 
                init_pts.ravel(), 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res_local.fun < 1e9:
                opt_points = res_local.x.reshape((n, d))
                opt_score = get_score(opt_points)
                
                if opt_score > best_score:
                    best_points = opt_points
                    best_score = opt_score
                    
                    # Stage 2: Tighter smooth objective
                    if time.time() < deadline - 20.0:
                        res_local2 = minimize(
                            lambda x: smooth_objective(x, beta=0.01), 
                            opt_points.ravel(), 
                            method='L-BFGS-B', 
                            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                        )
                        if res_local2.fun < 1e9:
                            opt_points2 = res_local2.x.reshape((n, d))
                            opt_score2 = get_score(opt_points2)
                            if opt_score2 > best_score:
                                best_points = opt_points2
                                best_score = opt_score2
        except Exception:
            pass

    # Final high-precision polish with exact objective
    if time.time() < deadline - 5.0:
        try:
            res_final = minimize(
                exact_objective, 
                best_points.ravel(), 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res_final.fun < 1e9:
                final_points = res_final.x.reshape((n, d))
                final_score = get_score(final_points)
                if final_score > best_score:
                    best_points = final_points
                    best_score = final_score
        except Exception:
            pass

    # Final sanity check
    if not np.all(np.isfinite(best_points)):
        best_points = make_fibonacci_sphere(n)

    return best_points

# EVOLVE-BLOCK-END
