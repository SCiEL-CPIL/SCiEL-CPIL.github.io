"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_score = -1.0
    best_points = None
    
    def calculate_score(points):
        if points is None:
            return 0.0
        pts = points.reshape(N, 3)
        if np.any(pts < -0.5 - 1e-9) or np.any(pts > 0.5 + 1e-9):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
            
        tree = cKDTree(pts)
        dists, _ = tree.pairs(), None
        # Use pairwise distances via query
        pairs = tree.query_pairs(1.0)  # All pairs within distance 1 (all of them)
        if len(pairs) == 0:
            min_pair = np.inf
        else:
            pair_dists = []
            for i, j in pairs:
                d = np.linalg.norm(pts[i] - pts[j])
                pair_dists.append(d)
            min_pair = min(pair_dists)
        
        boundary_dists = 0.5 - np.abs(pts)
        min_boundary = np.min(boundary_dists)
        
        r = min(min_pair / 2.0, min_boundary)
        return r

    def calc_min_dist_and_boundary(pts):
        """Return (min_pair_dist, min_boundary_dist, min_pair_indices, min_boundary_idx)"""
        tree = cKDTree(pts)
        pairs = tree.query_pairs(1.0)
        if len(pairs) == 0:
            min_pair = np.inf
            min_pair_idx = (0, 0)
        else:
            min_pair = np.inf
            min_pair_idx = (0, 0)
            for i, j in pairs:
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_pair:
                    min_pair = d
                    min_pair_idx = (i, j)
        
        boundary_dists = 0.5 - np.abs(pts)
        flat_idx = np.argmin(boundary_dists)
        min_boundary = boundary_dists.flat[flat_idx]
        min_boundary_idx = (flat_idx // 3, flat_idx % 3)
        
        return min_pair, min_pair_idx, min_boundary, min_boundary_idx

    def local_optimize(pts, max_iters=5000):
        """Local optimization using steepest ascent on the soft-min objective."""
        pts = pts.copy()
        step_size = 0.005
        best_local = -1.0
        best_local_pts = pts.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            min_pair, (i, j), min_boundary, (bi, bk) = calc_min_dist_and_boundary(pts)
            current_r = min(min_pair / 2.0, min_boundary)
            
            if current_r > best_local:
                best_local = current_r
                best_local_pts = pts.copy()
            
            # Compute gradient direction (ascent)
            grad = np.zeros_like(pts)
            
            # Pairwise contribution
            if min_pair < 1e-10:
                direction = np.random.randn(3)
                grad[i] += direction
                grad[j] -= direction
            else:
                diff = pts[i] - pts[j]
                norm = np.linalg.norm(diff)
                if norm > 1e-10:
                    diff = diff / norm
                    grad[i] += diff
                    grad[j] -= diff
            
            # Boundary contribution
            if min_boundary < min_pair / 2.0:
                s = np.sign(pts[bi, bk])
                if s == 0:
                    s = 1.0
                grad[bi, bk] += s
            
            # Normalize gradient
            gnorm = np.linalg.norm(grad)
            if gnorm > 1e-10:
                grad = grad / gnorm
            
            # Adaptive step size
            new_pts = pts + step_size * grad
            new_pts = np.clip(new_pts, -0.4999, 0.4999)
            
            new_min_pair, _, new_min_boundary, _ = calc_min_dist_and_boundary(new_pts)
            new_r = min(new_min_pair / 2.0, new_min_boundary)
            
            if new_r > current_r:
                pts = new_pts
                step_size = min(step_size * 1.1, 0.05)
            else:
                step_size *= 0.7
                if step_size < 1e-6:
                    break
        
        return best_local_pts, best_local

    def init_fcc():
        """FCC lattice initialization scaled to fit 100 points in unit cube."""
        # FCC lattice: points at corners and face centers
        # Generate FCC points and select 100
        pts_list = []
        # Use a 5x5x5 grid of FCC unit cells
        # Each FCC unit cell has 4 atoms: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
        n_cells = 5
        spacing = 0.9 / n_cells  # Leave margin
        
        for ix in range(n_cells):
            for iy in range(n_cells):
                for iz in range(n_cells):
                    base_x = -0.45 + ix * spacing
                    base_y = -0.45 + iy * spacing
                    base_z = -0.45 + iz * spacing
                    # FCC positions
                    pts_list.append([base_x, base_y, base_z])
                    pts_list.append([base_x + spacing/2, base_y + spacing/2, base_z])
                    pts_list.append([base_x + spacing/2, base_y, base_z + spacing/2])
                    pts_list.append([base_x, base_y + spacing/2, base_z + spacing/2])
        
        pts = np.array(pts_list[:N])
        # Scale to fit
        max_abs = np.max(np.abs(pts))
        if max_abs > 0.45:
            pts = pts * (0.45 / max_abs)
        
        return pts

    def init_grid(n1, n2, n3):
        tx = (np.arange(n1) + 0.5) / n1 - 0.5
        ty = (np.arange(n2) + 0.5) / n2 - 0.5
        tz = (np.arange(n3) + 0.5) / n3 - 0.5
        grid = np.array([(x, y, z) for z in tz for y in ty for x in tx])
        return grid[:N]

    def init_random():
        return np.random.uniform(-0.45, 0.45, size=(N, 3))

    starts = [
        init_fcc(),
        init_grid(5, 5, 4),
        init_grid(4, 5, 5),
        init_grid(5, 4, 5),
        init_fcc() + np.random.normal(0, 0.02, size=(N, 3)),
        init_grid(5, 5, 4) + np.random.normal(0, 0.02, size=(N, 3)),
        init_random(),
    ]
    
    # Clip all starts
    for i in range(len(starts)):
        starts[i] = np.clip(starts[i], -0.49, 0.49)
    
    # Run local optimization from each start
    for start_idx, x0 in enumerate(starts):
        if time.time() > deadline - 20:
            break
        
        try:
            # First pass: moderate iterations
            opt_pts, opt_score = local_optimize(x0, max_iters=3000)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            continue
    
    # Refinement passes on best result
    if best_points is not None:
        for refine_pass in range(3):
            if time.time() > deadline - 10:
                break
            try:
                opt_pts, opt_score = local_optimize(best_points, max_iters=5000)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_pts.copy()
                else:
                    break
            except Exception:
                break
    
    # Final fine-tuning: try small random perturbations and optimize
    if best_points is not None and time.time() < deadline - 15:
        for perturb_attempt in range(20):
            if time.time() > deadline - 8:
                break
            try:
                perturbed = best_points + np.random.normal(0, 0.01, size=(N, 3))
                perturbed = np.clip(perturbed, -0.49, 0.49)
                opt_pts, opt_score = local_optimize(perturbed, max_iters=2000)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_pts.copy()
            except Exception:
                continue
    
    if best_points is None:
        best_points = init_grid(5, 5, 4)
        
    # Ensure valid output
    best_points = np.clip(best_points, -0.5, 0.5)
    return best_points
# EVOLVE-BLOCK-END
