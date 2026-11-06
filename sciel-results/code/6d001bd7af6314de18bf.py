"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def construct():
    """
    Place 100 equal non-overlapping spheres in the unit cube [-0.5, 0.5]^3
    to maximize the common radius.
    
    Strategy:
    1. Jittered grid initialization (5x5x4 = 100 points).
    2. Two-phase soft-min gradient ascent with momentum (coarse→fine alpha).
    3. Vectorized gradient calculation using log-sum-exp.
    4. Simulated annealing acceptance for escaping local minima.
    5. High-precision polish on critical pairs/boundaries.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # --- Initialization: Jittered 5x5x4 grid ---
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    
    base_points = np.array([[x, y, z] for z in zs for y in ys for x in xs])
    
    np.random.seed(42)
    current_pts = base_points + np.random.uniform(-0.02, 0.02, (N, 3))
    current_pts = np.clip(current_pts, -0.5, 0.5)
    
    def compute_radius(pts):
        """Compute the maximum valid radius for given points (vectorized)."""
        if not np.all(np.abs(pts) <= 0.5):
            return 0.0
        
        n = len(pts)
        # Use cdist for efficient pairwise distances
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        boundary_dists = 0.5 - np.abs(pts)
        min_boundary = np.min(boundary_dists)
        
        return min(min_pair_dist / 2.0, min_boundary)
    
    def compute_radius_and_crit(pts):
        """Compute radius and return the critical pair/boundary indices."""
        if not np.all(np.abs(pts) <= 0.5):
            return 0.0, None, None
        
        n = len(pts)
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        
        # Find minimum pair
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_pair_dist = dists[min_pair_idx]
        
        boundary_dists = 0.5 - np.abs(pts)
        min_boundary_idx = np.unravel_index(np.argmin(boundary_dists), boundary_dists.shape)
        min_boundary = boundary_dists[min_boundary_idx]
        
        if min_pair_dist / 2.0 <= min_boundary:
            r = min_pair_dist / 2.0
            i_idx, j_idx = min_pair_idx
            if i_idx > j_idx:
                i_idx, j_idx = j_idx, i_idx
            crit_pair = (i_idx, j_idx)
            crit_boundary = None
        else:
            r = min_boundary
            crit_pair = None
            crit_boundary = min_boundary_idx
        
        return r, crit_pair, crit_boundary
    
    current_radius = compute_radius(current_pts)
    best_radius = current_radius
    best_pts = current_pts.copy()
    
    def soft_min_grad(pts, alpha):
        """
        Compute gradient of soft-min radius w.r.t. point positions.
        Uses log-sum-exp for numerical stability.
        """
        n = len(pts)
        # Pairwise distances
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        
        # Get upper triangle indices
        iu = np.triu_indices(n, k=1)
        pair_dists = dists[iu] / 2.0
        pair_diffs = pts[iu[0]] - pts[iu[1]]
        
        # Boundary terms
        boundary_terms = 0.5 - np.abs(pts)
        boundary_flat = boundary_terms.flatten()
        
        all_terms = np.concatenate([pair_dists, boundary_flat])
        
        # Log-sum-exp for numerical stability
        neg_alpha_x = -alpha * all_terms
        max_val = np.max(neg_alpha_x)
        exp_vals = np.exp(neg_alpha_x - max_val)
        weights = exp_vals / np.sum(exp_vals)
        
        grad = np.zeros_like(pts)
        
        # Pairwise term gradients
        pair_weights = weights[:len(pair_dists)]
        safe_dists = np.maximum(dists[iu], 1e-10)
        unit_vecs = pair_diffs / safe_dists[:, np.newaxis]
        
        i_idx, j_idx = iu
        np.add.at(grad, i_idx, pair_weights[:, np.newaxis] * unit_vecs)
        np.add.at(grad, j_idx, -pair_weights[:, np.newaxis] * unit_vecs)
        
        # Boundary term gradients
        boundary_weights = weights[len(pair_dists):]
        bw = boundary_weights.reshape(n, 3)
        signs = np.sign(pts)
        grad -= bw * signs
        
        return grad
    
    def polish_critical(pts, r, max_iter=5000):
        """
        High-precision polish: iteratively adjust the critical pair and boundary
        to equalize all constraints, using exact Newton-like steps.
        """
        pts = pts.copy()
        r = r
        for it in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            
            _, crit_pair, crit_boundary = compute_radius_and_crit(pts)
            
            if crit_pair is not None:
                i, j = crit_pair
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-12:
                    break
                # Move i and j apart by half the deficit
                target_r = r
                deficit = 2 * target_r - dist
                if deficit > 0:
                    unit = d / dist
                    step = 0.5 * deficit * 0.95  # Damped step
                    pts[i] += step * unit
                    pts[j] -= step * unit
            elif crit_boundary is not None:
                i, k = crit_boundary
                # Push point away from boundary
                sign = np.sign(pts[i, k])
                if sign == 0:
                    sign = 1
                pts[i, k] += sign * 0.95 * (0.5 - np.abs(pts[i, k]) - r)
            
            # Clip to cube
            pts = np.clip(pts, -0.5 + 1e-12, 0.5 - 1e-12)
            
            new_r = compute_radius(pts)
            if new_r <= r:
                # Try smaller step
                break
            r = new_r
        
        return pts, r
    
    # Phase 1: Coarse optimization (low alpha for smooth landscape)
    T = 0.005
    lr = 0.03
    alpha = 30.0
    momentum = np.zeros_like(current_pts)
    beta = 0.9
    no_improve = 0
    
    phase1_deadline = start_time + 50.0
    
    while time.time() < min(phase1_deadline, deadline - 30.0):
        grad = soft_min_grad(current_pts, alpha)
        
        # Momentum update
        momentum = beta * momentum + (1 - beta) * grad
        new_pts = current_pts + lr * momentum
        new_pts = np.clip(new_pts, -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_radius = compute_radius(new_pts)
        
        if new_radius >= current_radius:
            current_radius = new_radius
            current_pts = new_pts
            no_improve = 0
            T *= 0.998
            lr *= 0.9995
            alpha *= 1.001  # Slowly increase alpha
            if lr < 0.005:
                lr = 0.015
        else:
            delta = new_radius - current_radius
            if np.random.random() < np.exp(delta / max(T, 1e-8)):
                current_radius = new_radius
                current_pts = new_pts
            no_improve += 1
            T *= 0.97
            lr *= 0.93
            if lr < 0.005:
                if no_improve > 80:
                    # Perturb to escape local minimum
                    idx = np.random.choice(N, 8, replace=False)
                    current_pts[idx] = np.random.uniform(-0.4, 0.4, (8, 3))
                    current_pts = np.clip(current_pts, -0.5, 0.5)
                    current_radius = compute_radius(current_pts)
                    no_improve = 0
                    lr = 0.03
                    T = 0.005
                    alpha = 30.0
                    momentum = np.zeros_like(current_pts)
                else:
                    lr = 0.03
                    T = 0.005
                    alpha = 30.0
                    momentum = np.zeros_like(current_pts)
        
        # Track best
        if current_radius > best_radius:
            best_radius = current_radius
            best_pts = current_pts.copy()
    
    # Phase 2: Fine optimization (high alpha for sharper min)
    alpha = 80.0
    lr = 0.005
    momentum = np.zeros_like(current_pts)
    beta = 0.9
    no_improve = 0
    T = 0.001
    
    phase2_deadline = start_time + 90.0
    
    while time.time() < min(phase2_deadline, deadline - 15.0):
        grad = soft_min_grad(current_pts, alpha)
        
        # Momentum update
        momentum = beta * momentum + (1 - beta) * grad
        new_pts = current_pts + lr * momentum
        new_pts = np.clip(new_pts, -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_radius = compute_radius(new_pts)
        
        if new_radius >= current_radius:
            current_radius = new_radius
            current_pts = new_pts
            no_improve = 0
            lr *= 0.999
            alpha *= 1.0005
            if lr < 0.001:
                lr = 0.003
        else:
            delta = new_radius - current_radius
            if np.random.random() < np.exp(delta / max(T, 1e-8)):
                current_radius = new_radius
                current_pts = new_pts
            no_improve += 1
            T *= 0.95
            lr *= 0.95
            if lr < 0.001:
                if no_improve > 50:
                    # Small perturbation
                    idx = np.random.choice(N, 3, replace=False)
                    current_pts[idx] += np.random.uniform(-0.01, 0.01, (3, 3))
                    current_pts = np.clip(current_pts, -0.5, 0.5)
                    current_radius = compute_radius(current_pts)
                    no_improve = 0
                    lr = 0.005
                    T = 0.001
                    alpha = 80.0
                    momentum = np.zeros_like(current_pts)
                else:
                    lr = 0.005
                    T = 0.001
                    alpha = 80.0
                    momentum = np.zeros_like(current_pts)
        
        # Track best
        if current_radius > best_radius:
            best_radius = current_radius
            best_pts = current_pts.copy()
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 5.0:
        polish_pts, polish_r = polish_critical(best_pts, best_radius, max_iter=10000)
        if polish_r > best_radius:
            best_radius = polish_r
            best_pts = polish_pts
    
    # Final verification and clipping
    best_pts = np.clip(best_pts, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_pts
# EVOLVE-BLOCK-END
