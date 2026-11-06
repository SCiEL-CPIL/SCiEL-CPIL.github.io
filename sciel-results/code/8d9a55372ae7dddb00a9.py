import numpy as np
import time
import itertools
from scipy.optimize import minimize, newton
from scipy.spatial import Delaunay

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    def compute_score(pts):
        """Compute the score: Delta / 8^8 where Delta is product of ordered pair distances after scaling to diameter 2."""
        # 1. Scale to diameter 2
        d_mat = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2))
        d_max = np.max(d_mat)
        if d_max < 1e-9:
            return 0.0, pts
        pts_scaled = pts * (2.0 / d_max)
        
        # 2. Compute product of ordered pair distances
        d_mat2 = np.sqrt(np.sum((pts_scaled[:, np.newaxis, :] - pts_scaled[np.newaxis, :, :])**2, axis=2))
        mask = ~np.eye(8, dtype=bool)
        d_safe = np.where(mask, np.maximum(d_mat2, 1e-15), 1.0)
        log_delta = np.sum(np.log(d_safe) * mask)
        delta = np.exp(log_delta)
        score = delta / (8.0 ** 8)
        return score, pts_scaled

    def log_product_and_grad(pts):
        """
        Returns log product of all ordered pair distances and its gradient.
        """
        d_mat = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2))
        mask = ~np.eye(8, dtype=bool)
        d_safe = np.where(mask, np.maximum(d_mat, 1e-15), 1.0)
        log_prod = np.sum(np.log(d_safe) * mask)
        
        grad = np.zeros_like(pts)
        # Gradient of sum_{i!=j} log|p_i - p_j|
        # d/dp_i = sum_{j!=i} (p_i - p_j) / |p_i - p_j|^2
        for i in range(8):
            for j in range(8):
                if i == j: continue
                diff = pts[i] - pts[j]
                dist2 = np.dot(diff, diff)
                if dist2 < 1e-12: continue
                term = diff / dist2
                grad[i] += term
        return log_prod, grad

    def refine_local(pts0, max_iters=50):
        """
        Perform local refinement using projected gradient ascent.
        We maximize log product subject to diameter <= 2.
        We use a penalty method or simple projection.
        """
        pts = pts0.copy()
        # Normalize to diameter 2 initially
        d_mat = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2))
        d_max = np.max(d_mat)
        if d_max > 1e-9:
            pts = pts * (2.0 / d_max)
        
        best_pts = pts.copy()
        best_score, _ = compute_score(best_pts)
        
        step_size = 0.01
        for _ in range(max_iters):
            if time.time() > deadline: break
            log_prod, grad = log_product_and_grad(pts)
            
            # Check diameter constraint
            d_mat = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2))
            d_max = np.max(d_mat)
            
            # If diameter > 2, we need to scale down or move points inward.
            # Simple strategy: If d_max > 2, scale points towards centroid? 
            # Or just use a penalty in the objective.
            # Let's use a soft penalty for diameter > 2.
            penalty = 0.0
            grad_pen = np.zeros_like(pts)
            if d_max > 2.0:
                # Find pairs exceeding 2
                mask = ~np.eye(8, dtype=bool)
                violations = np.maximum(d_mat - 2.0, 0.0) * mask
                # Gradient of sum of violations
                for i in range(8):
                    for j in range(8):
                        if i < j and d_mat[i,j] > 2.0:
                            diff = pts[i] - pts[j]
                            dist = d_mat[i,j]
                            # d/dp_i (d_ij - 2) = diff / dist
                            term = (diff / dist) * 10.0 # Weight
                            grad_pen[i] += term
                            grad_pen[j] -= term
                penalty = np.sum(violations)
            
            total_grad = grad - 100.0 * grad_pen # Maximize log_prod - 100 * penalty
            
            # Gradient ascent step
            pts_new = pts + step_size * total_grad
            
            # Project to diameter 2 if necessary
            d_mat_new = np.sqrt(np.sum((pts_new[:, np.newaxis, :] - pts_new[np.newaxis, :, :])**2, axis=2))
            d_max_new = np.max(d_mat_new)
            if d_max_new > 2.0:
                pts_new = pts_new * (2.0 / d_max_new)
            
            # Center the points to prevent drift
            pts_new -= np.mean(pts_new, axis=0)
            
            score_new, _ = compute_score(pts_new)
            if score_new > best_score:
                best_score = score_new
                best_pts = pts_new.copy()
                step_size = min(step_size * 1.1, 0.1)
            else:
                step_size = max(step_size * 0.8, 0.001)
                pts = pts_new # Still update pts to allow escape? Or revert?
                # Let's revert if it got worse significantly
                if score_new < best_score * 0.9:
                    pts = best_pts.copy()
        
        return best_pts

    def solve_topology(edges, initial_pts):
        """
        Given a set of edges (topology), solve for the geometry that maximizes
        the product of distances for that fixed topology.
        This is a non-convex problem. We use Newton's method on the gradient of
        the log-product with respect to the angles/lengths?
        Actually, a simpler approach: Use the given edges as a hint for initial
        positions, then run the local refiner. The topology search is mainly
        to provide diverse, non-symmetric starting points.
        """
        # This function is a placeholder for a more complex solver.
        # For now, we just use the initial_pts provided by the combinatorial search.
        return initial_pts

    # --- Combinatorial Search for Topologies ---
    
    best_score_global = 0.0
    best_pts_global = None
    
    # 1. Regular Octagon
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    pts_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    sc, pts_oct_scaled = compute_score(pts_oct)
    if sc > best_score_global:
        best_score_global = sc
        best_pts_global = pts_oct_scaled.copy()
    
    # 2. Generate diverse topologies using random graphs or specific structures
    # We will generate random sets of points, compute their Delaunay triangulation,
    # and use that as a guide for "cage" structures.
    
    # Strategy: 
    # A. Try "2 concentric squares" with various rotations and radii.
    # B. Try "1 square + 4 points" (one on each side/corner).
    # C. Random points on a circle with perturbed radii.
    # D. Random points in the unit disk.
    
    starts = []
    
    # A. Concentric squares
    for r1 in np.linspace(0.2, 0.8, 5):
        for rot1 in np.linspace(0, np.pi/4, 5):
            for rot2 in np.linspace(0, np.pi/4, 5):
                p1 = np.column_stack([np.cos(np.linspace(0, 2*np.pi, 4, endpoint=False)+rot1), 
                                      np.sin(np.linspace(0, 2*np.pi, 4, endpoint=False)+rot1)])
                p2 = r1 * np.column_stack([np.cos(np.linspace(0, 2*np.pi, 4, endpoint=False)+rot2), 
                                           np.sin(np.linspace(0, 2*np.pi, 4, endpoint=False)+rot2)])
                starts.append(np.vstack([p1, p2]))
                
    # B. Square with 4 extra points
    base_sq = np.array([[1,1], [1,-1], [-1,-1], [-1,1]])
    for r_extra in [0.5, 0.8, 1.0]:
        for rot_extra in [0, np.pi/4, np.pi/2]:
            p_extra = r_extra * np.column_stack([np.cos(np.linspace(0, 2*np.pi, 4, endpoint=False)+rot_extra),
                                                 np.sin(np.linspace(0, 2*np.pi, 4, endpoint=False)+rot_extra)])
            starts.append(np.vstack([base_sq, p_extra]))

    # C. Perturbed Octagon
    np.random.seed(42)
    for _ in range(20):
        radii = 1.0 + np.random.normal(0, 0.2, 8)
        radii = np.abs(radii) + 0.1
        angles_p = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.random.normal(0, 0.1, 8)
        pts = np.column_stack([radii * np.cos(angles_p), radii * np.sin(angles_p)])
        starts.append(pts)

    # D. Random in disk
    for _ in range(20):
        r = np.sqrt(np.random.rand(8))
        theta = np.random.rand(8) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        starts.append(pts)

    # E. "Butterfly" or "Bowtie" shapes (two triangles sharing a vertex or edge)
    # Just random convex hulls with 8 points
    for _ in range(20):
        # Generate 8 points, ensure they are roughly convex or have specific structure
        # For now, just random
        pts = np.random.rand(8, 2) * 2 - 1
        starts.append(pts)

    # Evaluate and refine each start
    for i, s in enumerate(starts):
        if time.time() > deadline: break
        
        # Quick score check
        sc_init, _ = compute_score(s)
        
        # Refine
        refined_pts = refine_local(s, max_iters=30)
        sc_ref, refined_pts_scaled = compute_score(refined_pts)
        
        if sc_ref > best_score_global:
            best_score_global = sc_ref
            best_pts_global = refined_pts_scaled.copy()
            
            # Try a few more refinement steps from the best found so far
            # This helps polish the result
            if time.time() < deadline - 10:
                for _ in range(5):
                    refined_pts2 = refine_local(best_pts_global, max_iters=10)
                    sc_ref2, refined_pts2_scaled = compute_score(refined_pts2)
                    if sc_ref2 > best_score_global:
                        best_score_global = sc_ref2
                        best_pts_global = refined_pts2_scaled.copy()
                    else:
                        break

    # Fallback
    if best_pts_global is None:
        best_pts_global = pts_oct_scaled

    # Final scaling and clipping
    d_mat = np.sqrt(np.sum((best_pts_global[:, np.newaxis, :] - best_pts_global[np.newaxis, :, :])**2, axis=2))
    d_max = np.max(d_mat)
    if d_max > 1e-9:
        best_pts_global = best_pts_global * (2.0 / d_max)
    
    best_pts_global = np.clip(best_pts_global, -1e6, 1e6)
    
    return best_pts_global.tolist()
