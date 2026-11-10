import numpy as np
import time
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 600.0
    SAFETY_MARGIN = 20.0
    N = 16

    def get_ratio(pts):
        """Calculate d_min / d_max. Higher is better."""
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max <= 1e-12:
            return 0.0
        return d_min / d_max

    def get_ratio_sq(pts):
        """Calculate (d_min / d_max)^2. Higher is better. This is the actual metric R."""
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max <= 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    # Precompute pair indices for fast distance computation
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def fast_ratio(pts):
        """Fast ratio computation using precomputed indices."""
        diff = pts[i_idx] - pts[j_idx]
        d = np.sqrt(np.sum(diff * diff, axis=1))
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max <= 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def ring_init(n_inner, n_outer, r_inner, r_outer):
        """Points on two concentric rings."""
        pts = []
        if n_inner > 0:
            for i in range(n_inner):
                angle = 2 * np.pi * i / n_inner
                pts.append([r_inner * np.cos(angle), r_inner * np.sin(angle)])
        if n_outer > 0:
            for i in range(n_outer):
                angle = 2 * np.pi * i / n_outer + np.pi / n_outer
                pts.append([r_outer * np.cos(angle), r_outer * np.sin(angle)])
        return np.array(pts)

    def grid_init():
        """4x4 square grid."""
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append([i, j])
        pts = np.array(pts, dtype=float)
        pts = pts - np.mean(pts, axis=0)
        return pts

    def hex_init():
        """Hexagonal arrangement."""
        pts = []
        a = 1.0
        b = np.sqrt(3) / 2
        pts.append([0, 0])
        for i in range(6):
            angle = 2 * np.pi * i / 6
            pts.append([a * np.cos(angle), a * np.sin(angle)])
        ring2_pts = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * a + j * 0.5 * a
                y = j * b
                r = np.sqrt(x*x + y*y)
                if 1.5 * a - 0.1 < r < 2.5 * a + 0.1:
                    ring2_pts.append([x, y])
        ring2_pts = np.array(ring2_pts)
        if len(ring2_pts) >= 9:
            r_target = 2.0 * a
            dists_to_target = np.abs(np.linalg.norm(ring2_pts, axis=1) - r_target)
            idx = np.argsort(dists_to_target)[:9]
            ring2_pts = ring2_pts[idx]
        pts = np.array(pts)
        pts = np.vstack([pts, ring2_pts])
        pts = pts - np.mean(pts, axis=0)
        return pts

    def circle_init(n=16):
        """Points on a circle."""
        pts = []
        for i in range(n):
            angle = 2 * np.pi * i / n
            pts.append([np.cos(angle), np.sin(angle)])
        return np.array(pts)

    def square_ring_init():
        """Points on boundary of a square with some interior."""
        pts = []
        for i in range(4):
            pts.append([-1 + 2*i/3, 1.0])
        for i in range(1, 4):
            pts.append([1.0, 1 - 2*i/3])
        for i in range(1, 4):
            pts.append([-1 + 2*i/3, -1.0])
        for i in range(1, 3):
            pts.append([-1.0, 1 - 2*i/3])
        pts.append([0, 0])
        pts.append([0.5, 0.5])
        pts.append([-0.5, 0.5])
        pts.append([0.5, -0.5])
        pts = np.array(pts[:16], dtype=float)
        pts = pts - np.mean(pts, axis=0)
        return pts

    def generate_candidates():
        candidates = []
        candidates.append(ring_init(1, 15, 0, 1.0))
        candidates.append(ring_init(4, 12, 0.5, 1.0))
        candidates.append(ring_init(7, 9, 0.4, 1.0))
        candidates.append(ring_init(9, 7, 0.4, 1.0))
        candidates.append(ring_init(1, 3, 0, 0.5))
        for i, c in enumerate(candidates):
            c = c - np.mean(c, axis=0)
            max_r = np.max(np.linalg.norm(c, axis=1))
            if max_r > 0:
                c = c / max_r
            candidates[i] = c
        candidates.append(grid_init())
        candidates.append(hex_init())
        candidates.append(circle_init(16))
        candidates.append(square_ring_init())
        return candidates

    best_pts = None
    best_ratio = 0.0

    for c in generate_candidates():
        if len(c) != 16:
            continue
        ratio = fast_ratio(c)
        if ratio > best_ratio:
            best_ratio = ratio
            best_pts = c.copy()

    # New structural approach: Iterative local search with adaptive step size
    # and multiple restarts from diverse starting points
    def local_search(pts, time_budget=200.0):
        """Improve points using iterative local search with adaptive step size."""
        pts = pts.copy()
        t_start = time.time()
        
        current_ratio = fast_ratio(pts)
        best_ratio = current_ratio
        best_pts = pts.copy()
        
        step_size = 0.1
        min_step = 1e-6
        no_improve_count = 0
        max_no_improve = 100
        
        while time.time() - t_start < time_budget:
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            improved = False
            for _ in range(10):  # Try 10 random perturbations per step
                idx = np.random.randint(N)
                old_pos = pts[idx].copy()
                delta = np.random.randn(2) * step_size
                pts[idx] += delta
                
                new_ratio = fast_ratio(pts)
                
                if new_ratio > current_ratio:
                    current_ratio = new_ratio
                    improved = True
                    if new_ratio > best_ratio:
                        best_ratio = new_ratio
                        best_pts = pts.copy()
                    no_improve_count = 0
                else:
                    pts[idx] = old_pos
            
            if not improved:
                no_improve_count += 1
                step_size *= 0.9
                if step_size < min_step:
                    break
            
            if no_improve_count > max_no_improve:
                break
        
        return best_pts, best_ratio

    # Run local search from best candidate
    time_elapsed = time.time() - start_time
    ls_budget = min(300.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN)
    if ls_budget > 10:
        ls_pts, ls_ratio = local_search(best_pts, time_budget=ls_budget)
        if ls_ratio > best_ratio:
            best_pts = ls_pts
            best_ratio = ls_ratio

    # Multiple restarts from different initializations
    time_elapsed = time.time() - start_time
    remaining = TIME_LIMIT - time_elapsed - SAFETY_MARGIN
    if remaining > 30:
        candidates = generate_candidates()
        for c in candidates:
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            if len(c) != 16:
                continue
            
            # Add random perturbation to create diversity
            perturbed = c.copy()
            perturbed += np.random.randn(16, 2) * 0.1
            perturbed -= np.mean(perturbed, axis=0)
            
            r = fast_ratio(perturbed)
            if r > best_ratio:
                best_ratio = r
                best_pts = perturbed.copy()
            
            # Try local search from this perturbed point if time allows
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN - 20:
                continue
            
            ls_pts, ls_ratio = local_search(perturbed, time_budget=min(15.0, TIME_LIMIT - time.time() - SAFETY_MARGIN))
            if ls_ratio > best_ratio:
                best_ratio = ls_ratio
                best_pts = ls_pts

    # Final polish with finer step size
    time_elapsed = time.time() - start_time
    if TIME_LIMIT - time_elapsed - SAFETY_MARGIN > 10:
        final_pts, final_ratio = local_search(best_pts, time_budget=min(30.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN))
        if final_ratio > best_ratio:
            best_pts = final_pts
            best_ratio = final_ratio

    # Ensure finite and correct shape
    final_pts = best_pts
    if final_pts is None or not np.all(np.isfinite(final_pts)) or final_pts.shape != (16, 2):
        final_pts = grid_init()

    return final_pts
