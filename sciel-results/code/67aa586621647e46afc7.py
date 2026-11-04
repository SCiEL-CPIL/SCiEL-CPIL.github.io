import numpy as np
import time
from scipy.spatial import cKDTree

N = 300

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    def _compute_radius(pts):
        """Compute the packing radius for given points."""
        if len(pts) < 2:
            return np.min(1.0 - np.linalg.norm(pts, axis=1))
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
        rad = np.linalg.norm(pts, axis=1)
        bound_min = np.min(1.0 - rad)
        return min(pair_min, bound_min)
    
    def _hex_init(s_factor=1.0):
        """Generate hexagonal lattice initialization."""
        s = np.sqrt(2 * np.pi / (np.sqrt(3) * N)) * s_factor
        y_step = s * np.sqrt(3) / 2
        max_j = int(1.0 / y_step) + 3
        max_i = int(1.0 / s) + 3
        
        points_list = []
        for j in range(-max_j, max_j + 1):
            y = j * y_step
            offset = 0 if j % 2 == 0 else s / 2
            for i in range(-max_i, max_i + 1):
                x = i * s + offset
                if np.hypot(x, y) < 1.0:
                    points_list.append([x, y])
        
        points = np.array(points_list)
        if len(points) > N:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:N]
            points = points[idx]
        elif len(points) < N:
            while len(points) < N:
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.sqrt(np.random.uniform(0, 0.95**2))
                points = np.vstack([points, [r * np.cos(angle), r * np.sin(angle)]])
            points = points[:N]
        
        return points.copy()
    
    def _soft_min_logsumexp(vals, k):
        """
        Compute soft-min using log-sum-exp for numerical stability.
        Returns (soft_min_value, weights)
        """
        vals = np.asarray(vals)
        if len(vals) == 0:
            return 0.0, np.array([])
        
        v_min = np.min(vals)
        shifted = vals - v_min
        
        # For numerical stability, only consider values within threshold
        threshold = 50.0 / k
        mask = shifted < threshold
        
        if not np.any(mask):
            weights = np.ones_like(vals) / len(vals)
            return v_min, weights
        
        close_shifted = shifted[mask]
        # log-sum-exp: log(sum(exp(-k * x_i))) = -k * x_max + log(sum(exp(-k * (x_i - x_max))))
        x_max = np.max(close_shifted)
        exp_terms = np.exp(-k * (close_shifted - x_max))
        sum_exp = np.sum(exp_terms)
        
        if sum_exp <= 0:
            weights = np.ones_like(vals) / len(vals)
            return v_min, weights
        
        # soft_min = v_min - (1/k) * log(sum(exp(-k * shifted)))
        #         = v_min - (1/k) * (-k * x_max + log(sum_exp))
        #         = v_min + x_max - (1/k) * log(sum_exp)
        soft_min_val = v_min + x_max - (1.0 / k) * np.log(sum_exp)
        
        weights = np.zeros_like(vals)
        weights[mask] = exp_terms / sum_exp
        
        return soft_min_val, weights
    
    def _build_edges(pts, search_radius):
        """Build list of close pairs using KDTree."""
        tree = cKDTree(pts)
        pairs = tree.query_pairs(search_radius, output_type='ndarray')
        if len(pairs) == 0:
            return np.empty((0, 2), dtype=int)
        return pairs
    
    def _force_relax(pts, max_iter=500, time_budget=30.0):
        """
        Force-based relaxation using soft-min objective.
        Returns improved points.
        """
        pts = pts.copy()
        start_t = time.time()
        
        # Adaptive k schedule
        k_start = 5.0
        k_end = 200.0
        alpha_start = 0.01
        alpha_end = 0.0001
        cooling = 0.999
        
        k = k_start
        alpha = alpha_start
        best_pts = pts.copy()
        best_score = _compute_radius(pts)
        stall_count = 0
        stall_threshold = 100
        
        for it in range(max_iter):
            if time.time() - start_t > time_budget:
                break
            
            # Compute pairwise distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Get all pairs
            i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
            pair_vals = dists[i_idx, j_idx] / 2.0
            rad = np.linalg.norm(pts, axis=1)
            bound_vals = 1.0 - rad
            all_vals = np.concatenate([pair_vals, bound_vals])
            
            current_min = np.min(all_vals)
            if current_min > best_score:
                best_score = current_min
                best_pts = pts.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            # Soft-min weights
            _, weights = _soft_min_logsumexp(all_vals, k)
            
            # Compute gradients
            grads = np.zeros_like(pts)
            
            # Pair forces
            pair_weights = weights[:len(pair_vals)]
            d_pair = dists[i_idx, j_idx]
            mask = d_pair > 1e-10
            if np.any(mask):
                w = pair_weights[mask]
                d = d_pair[mask]
                vi = i_idx[mask]
                vj = j_idx[mask]
                vecs = (diff[vi, vj] / d[:, np.newaxis]) * w[:, np.newaxis]
                np.add.at(grads, vi, vecs)
                np.add.at(grads, vj, -vecs)
            
            # Boundary forces
            bound_weights = weights[len(pair_vals):]
            rad_safe = np.where(rad > 1e-10, rad, 1.0)
            grads += -(pts / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
            
            # Normalize and update
            grad_norm = np.linalg.norm(grads)
            if grad_norm > 1e-10:
                grads /= grad_norm
                pts += alpha * grads
            
            # Project inside circle
            rad_new = np.linalg.norm(pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts[mask_out] /= rad_new[mask_out, np.newaxis]
            
            # Schedule updates
            progress = it / max_iter
            k = k_start + (k_end - k_start) * (progress ** 1.5)
            alpha = alpha_start * (alpha_end / alpha_start) ** progress
            
            # Perturbation if stalled
            if stall_count > stall_threshold:
                noise = np.random.normal(0, alpha * 3.0, size=pts.shape)
                pts += noise
                rad_new = np.linalg.norm(pts, axis=1)
                mask_out = rad_new > 1.0
                if np.any(mask_out):
                    pts[mask_out] /= rad_new[mask_out, np.newaxis]
                stall_count = 0
                alpha = max(alpha, 0.005)
        
        return best_pts, best_score
    
    def _local_polish(pts, time_budget=20.0):
        """
        High-precision local polish focusing on the minimum constraint.
        """
        pts = pts.copy()
        start_t = time.time()
        
        k = 500.0
        alpha = 0.0002
        best_pts = pts.copy()
        best_score = _compute_radius(pts)
        
        max_iter = 2000
        for it in range(max_iter):
            if time.time() - start_t > time_budget:
                break
            
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
            pair_vals = dists[i_idx, j_idx] / 2.0
            rad = np.linalg.norm(pts, axis=1)
            bound_vals = 1.0 - rad
            all_vals = np.concatenate([pair_vals, bound_vals])
            
            current_min = np.min(all_vals)
            if current_min > best_score:
                best_score = current_min
                best_pts = pts.copy()
            
            # Very sharp soft-min
            _, weights = _soft_min_logsumexp(all_vals, k)
            
            grads = np.zeros_like(pts)
            
            pair_weights = weights[:len(pair_vals)]
            d_pair = dists[i_idx, j_idx]
            mask = d_pair > 1e-10
            if np.any(mask):
                w = pair_weights[mask]
                d = d_pair[mask]
                vi = i_idx[mask]
                vj = j_idx[mask]
                vecs = (diff[vi, vj] / d[:, np.newaxis]) * w[:, np.newaxis]
                np.add.at(grads, vi, vecs)
                np.add.at(grads, vj, -vecs)
            
            bound_weights = weights[len(pair_vals):]
            rad_safe = np.where(rad > 1e-10, rad, 1.0)
            grads += -(pts / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
            
            grad_norm = np.linalg.norm(grads)
            if grad_norm > 1e-12:
                grads /= grad_norm
                pts += alpha * grads
            
            rad_new = np.linalg.norm(pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts[mask_out] /= rad_new[mask_out, np.newaxis]
            
            # Slightly decrease alpha
            alpha *= 0.9995
        
        return best_pts, best_score
    
    # Main optimization with multiple restarts
    best_overall = None
    best_score_overall = 0.0
    
    # Initial hex lattice
    points = _hex_init(1.0)
    score = _compute_radius(points)
    best_overall = points.copy()
    best_score_overall = score
    
    # Phase 1: Force relaxation with main time budget
    phase1_budget = time_limit * 0.55
    points, score = _force_relax(points, max_iter=8000, time_budget=phase1_budget)
    if score > best_score_overall:
        best_score_overall = score
        best_overall = points.copy()
    
    # Check time
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    if remaining > 10:
        # Phase 2: Local polish
        phase2_budget = min(remaining * 0.6, 25.0)
        points, score = _local_polish(points, time_budget=phase2_budget)
        if score > best_score_overall:
            best_score_overall = score
            best_overall = points.copy()
    
    # Check time again
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    if remaining > 15:
        # Phase 3: Try a different initialization and quick optimize
        points2 = _hex_init(0.92)
        score2 = _compute_radius(points2)
        
        if score2 > best_score_overall * 0.98:  # Only if competitive
            phase3_budget = remaining * 0.5
            points2, score2 = _force_relax(points2, max_iter=3000, time_budget=phase3_budget)
            if score2 > best_score_overall:
                best_score_overall = score2
                best_overall = points2.copy()
            
            # Quick polish
            elapsed = time.time() - start_time
            remaining = time_limit - elapsed
            if remaining > 8:
                points2, score2 = _local_polish(points2, time_budget=min(remaining * 0.5, 12.0))
                if score2 > best_score_overall:
                    best_score_overall = score2
                    best_overall = points2.copy()
    
    # Final safety check
    rad_final = np.linalg.norm(best_overall, axis=1)
    mask_out = rad_final > 1.0
    if np.any(mask_out):
        best_overall[mask_out] /= rad_final[mask_out, np.newaxis]
    
    # Verify final score
    final_score = _compute_radius(best_overall)
    if final_score < best_score_overall:
        # If projection reduced score significantly, we might have an issue
        # But we must return valid points
        pass
    
    return best_overall
