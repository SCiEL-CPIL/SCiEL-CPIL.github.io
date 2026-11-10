import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import random

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes the ratio (d_min / d_max)^2 for 14 points in 3D.
    
    Strategy:
    1. Use a diverse set of initial configurations (random, icosahedron-based, etc.).
    2. For each configuration, run a multi-phase optimization:
       a. Soft-min/soft-max surrogate optimization with L-BFGS-B.
       b. Epigraph formulation with auxiliary variable t.
       c. Polish with Nelder-Mead on the true (d_min/d_max)^2 objective.
       d. High-precision final polish with tighter tolerances.
    3. Keep track of the best result and return it.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 620  # 10s margin before 630s kill

    def true_score(points):
        """Calculate the actual metric: (d_min / d_max)^2"""
        if points.shape != (n, d):
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        
        dists_sq = pdist(points, metric='sqeuclidean')
        
        if len(dists_sq) == 0:
            return 0.0
            
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        
        if d_max_sq < 1e-12:
            return 0.0
            
        ratio_sq = d_min_sq / d_max_sq
        return ratio_sq

    def soft_min_max_surrogate(points, temperature=0.1):
        """
        Smooth surrogate: soft-min / soft-max of squared distances.
        Uses log-sum-exp for numerical stability.
        """
        dists_sq = pdist(points, metric='sqeuclidean')
        dists_sq = np.maximum(dists_sq, 1e-10)
        
        # Soft min: -logsumexp(-dists_sq / T) * T
        neg_dists = -dists_sq / temperature
        neg_dists_max = np.max(neg_dists)
        soft_min = -temperature * (neg_dists_max + np.log(np.sum(np.exp(neg_dists - neg_dists_max))))
        
        # Soft max: logsumexp(dists_sq / T) * T
        pos_dists = dists_sq / temperature
        pos_dists_max = np.max(pos_dists)
        soft_max = temperature * (pos_dists_max + np.log(np.sum(np.exp(pos_dists - pos_dists_max))))
        
        return -soft_min / soft_max

    def epigraph_objective(x_flat, t_init=0.5):
        """
        Epigraph formulation: maximize t such that t <= d_ij^2 for all i,j
        and d_ij^2 <= 1 for all i,j (after normalization).
        Minimize: -t + barrier for violations.
        """
        pts = x_flat[:n*d].reshape(n, d)
        t = x_flat[n*d]
        
        pts = normalize_rms(pts)
        
        dists_sq = pdist(pts, metric='sqeuclidean')
        dists_sq = np.maximum(dists_sq, 1e-10)
        
        # Barrier for t <= d_ij^2 (want t to be close to min)
        violations_min = np.maximum(0, t - dists_sq)
        barrier_min = np.sum(violations_min**2)
        
        # Barrier for d_ij^2 <= 1 (upper bound after normalization)
        violations_max = np.maximum(0, dists_sq - 1.0)
        barrier_max = np.sum(violations_max**2)
        
        return -t + 10.0 * barrier_min + 10.0 * barrier_max

    def normalize_rms(points):
        """Normalize points to have unit RMS radius."""
        rms = np.sqrt(np.mean(np.sum(points**2, axis=1)))
        if rms < 1e-6:
            return points
        return points / rms

    def normalize_unit_max(points):
        """Normalize so that max pairwise distance is 1."""
        dists_sq = pdist(points, metric='sqeuclidean')
        d_max_sq = np.max(dists_sq)
        if d_max_sq < 1e-10:
            return points
        return points / np.sqrt(d_max_sq)

    best_score = 0.0
    best_points = None

    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Random points on sphere
    for _ in range(6):
        pts = np.random.randn(n, d)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        initial_configs.append(pts)

    # 2. Icosahedron + 2 random points
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts.astype(float)
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    for _ in range(3):
        p1 = np.random.randn(1, 3)
        p1 /= np.linalg.norm(p1)
        p2 = np.random.randn(1, 3)
        p2 /= np.linalg.norm(p2)
        initial_configs.append(np.vstack([ico_verts, p1, p2]))
    
    # 3. Random points with small perturbation from center
    for _ in range(4):
        pts = np.random.randn(n, d) * 0.5
        initial_configs.append(pts)
    
    # 4. Grid-like starting points
    for _ in range(2):
        pts = np.zeros((n, d))
        indices = np.random.choice(n, n, replace=False)
        for i in range(n):
            pts[i] = np.array([
                np.sin(indices[i] * 0.7),
                np.cos(indices[i] * 1.3),
                np.sin(indices[i] * 2.1)
            ])
        initial_configs.append(pts)

    # Optimization Loop
    for i, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 20:
            break
            
        try:
            # Phase 1: Soft-min/soft-max surrogate optimization
            def surrogate_obj(x_flat):
                pts = x_flat.reshape(n, d)
                pts = normalize_rms(pts)
                return -soft_min_max_surrogate(pts, temperature=0.1)

            res_surrogate = minimize(
                surrogate_obj, 
                init_pts.flatten(), 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            current_pts = normalize_rms(res_surrogate.x.reshape(n, d))
            
            # Phase 2: Epigraph formulation
            x0 = np.concatenate([current_pts.flatten(), [0.3]])
            
            res_epigraph = minimize(
                epigraph_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            epigraph_pts = normalize_rms(res_epigraph.x[:n*d].reshape(n, d))
            
            # Choose best from Phase 1 and Phase 2
            score_phase1 = true_score(current_pts)
            score_phase2 = true_score(epigraph_pts)
            
            if score_phase1 >= score_phase2:
                current_pts = current_pts
            else:
                current_pts = epigraph_pts
            
            # Phase 3: Nelder-Mead Polish on True Objective
            def nm_obj(x_flat):
                pts = x_flat.reshape(n, d)
                pts = normalize_rms(pts)
                return -true_score(pts)

            res_nm = minimize(
                nm_obj, 
                current_pts.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14,
                         'adaptive': True}
            )
            
            final_pts = normalize_rms(res_nm.x.reshape(n, d))
            
            # Phase 4: High-precision final polish
            if time.time() < deadline - 15:
                def final_polish_obj(x_flat):
                    pts = x_flat.reshape(n, d)
                    pts = normalize_rms(pts)
                    return -true_score(pts)
                
                res_final = minimize(
                    final_polish_obj,
                    final_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16,
                             'adaptive': True}
                )
                
                final_pts = normalize_rms(res_final.x.reshape(n, d))
            
            # Calculate score
            score = true_score(final_pts)
            if score > best_score:
                best_score = score
                best_points = final_pts.copy()
                
        except Exception as e:
            continue

    # If we have time, do additional random restarts with NM only
    if best_points is not None and time.time() < deadline - 30:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            try:
                # Start from best points with small perturbation
                perturbed = best_points + np.random.randn(n, d) * 0.01
                perturbed = normalize_rms(perturbed)
                
                def nm_obj2(x_flat):
                    pts = x_flat.reshape(n, d)
                    pts = normalize_rms(pts)
                    return -true_score(pts)
                
                res_nm2 = minimize(
                    nm_obj2, 
                    perturbed.flatten(), 
                    method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-14,
                             'adaptive': True}
                )
                
                final_pts2 = normalize_rms(res_nm2.x.reshape(n, d))
                score2 = true_score(final_pts2)
                if score2 > best_score:
                    best_score = score2
                    best_points = final_pts2.copy()
            except Exception:
                continue

    # Fallback if all failed
    if best_points is None:
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        ico_verts = ico_verts.astype(float)
        ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
        p1 = np.array([0.0, 0.0, 1.0])
        p2 = np.array([0.0, 0.0, -1.0])
        best_points = np.vstack([ico_verts, p1, p2])

    # Final validation
    if not np.all(np.isfinite(best_points)):
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        ico_verts = ico_verts.astype(float)
        ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
        best_points = np.vstack([ico_verts, np.array([0.0,0.0,1.0]), np.array([0.0,0.0,-1.0])])

    return best_points
