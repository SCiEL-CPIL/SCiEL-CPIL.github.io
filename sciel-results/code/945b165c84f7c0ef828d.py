"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Place 100 equal non-overlapping spheres in the unit cube [-0.5, 0.5]^3
    to maximize the common radius.
    
    Strategy:
    1. Jittered grid initialization.
    2. Two-phase soft-min gradient ascent (coarse→fine alpha).
    3. Vectorized gradient calculation.
    4. Adaptive learning rate with momentum.
    5. High-precision polish on critical pairs.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
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
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        
        iu = np.triu_indices(n, k=1)
        min_pair_dist = np.min(dists[iu])
        
        boundary_dists = 0.5 - np.abs(pts)
        min_boundary = np.min(boundary_dists)
        
        return min(min_pair_dist / 2.0, min_boundary)
    
    def compute_radius_and_crit(pts):
        """Compute radius and return the critical pair/boundary indices."""
        if not np.all(np.abs(pts) <= 0.5):
            return 0.0, None, None
        
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        
        iu = np.triu_indices(n, k=1)
        pair_dists = dists[iu]
        min_pair_idx = np.argmin(pair_dists)
        min_pair_dist = pair_dists[min_pair_idx]
        
        boundary_dists = 0.5 - np.abs(pts)
        min_boundary_idx = np.argmin(boundary_dists.flatten())
        min_boundary = boundary_dists.flatten()[min_boundary_idx]
        
        if min_pair_dist / 2.0 <= min_boundary:
            r = min_pair_dist / 2.0
            i_idx, j_idx = iu
            crit_pair = (i_idx[min_pair_idx], j_idx[min_pair_idx])
            crit_boundary = None
        else:
            r = min_boundary
            crit_pair = None
            crit_boundary = (min_boundary_idx // 3, min_boundary_idx % 3)
        
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
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        iu = np.triu_indices(n, k=1)
        
        pair_dists = dists[iu] / 2.0
        pair_diffs = diff[iu]
        
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
    
    def polish_critical(pts, r, max_iter=2000):
        """
        High-precision polish: iteratively adjust the critical pair and boundary
        to equalize all constraints, using exact Newton-like steps.
        """
        pts = pts.copy()
        r = r
        for it in range(max_iter):
            if time.time() > deadline - 1.0:
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
                    step = 0.5 * deficit * 0.9  # Damped step
                    pts[i] += step * unit
                    pts[j] -= step * unit
            elif crit_boundary is not None:
                i, k = crit_boundary
                # Push point away from boundary
                sign = np.sign(pts[i, k])
                if sign == 0:
                    sign = 1
                pts[i, k] += sign * 0.9 * (0.5 - np.abs(pts[i, k]) - r)
            
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
    
    phase1_deadline = start_time + 60.0
    
    while time.time() < min(phase1_deadline, deadline - 20.0):
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
                    current_radius = compute_radius(current_pts)
                    no_improve = 0
                    lr = 0.03
                    T = 0.005
                    alpha = 30.0
                    momentum = np.zeros_like(current_pts)
                else:
                    lr = 0.03
                    T = 0.005
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_pts = current_pts.copy()
        
        # Gradually increase alpha for finer resolution
        if np.random.random() < 0.02:
            alpha = min(alpha * 1.05, 150.0)
    
    # Phase 2: Fine optimization (high alpha)
    T = 0.001
    lr = 0.008
    alpha = 100.0
    momentum = np.zeros_like(current_pts)
    no_improve = 0
    
    while time.time() < deadline - 15.0:
        grad = soft_min_grad(current_pts, alpha)
        
        momentum = beta * momentum + (1 - beta) * grad
        new_pts = current_pts + lr * momentum
        new_pts = np.clip(new_pts, -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_radius = compute_radius(new_pts)
        
        if new_radius >= current_radius:
            current_radius = new_radius
            current_pts = new_pts
            no_improve = 0
            T *= 0.999
            lr *= 0.9998
            if lr < 0.002:
                lr = 0.005
        else:
            delta = new_radius - current_radius
            if np.random.random() < np.exp(delta / max(T, 1e-8)):
                current_radius = new_radius
                current_pts = new_pts
            no_improve += 1
            T *= 0.96
            lr *= 0.92
            if lr < 0.002:
                if no_improve > 60:
                    idx = np.random.choice(N, 5, replace=False)
                    current_pts[idx] = np.random.uniform(-0.4, 0.4, (5, 3))
                    current_radius = compute_radius(current_pts)
                    no_improve = 0
                    lr = 0.008
                    T = 0.001
                    alpha = 100.0
                    momentum = np.zeros_like(current_pts)
                else:
                    lr = 0.008
                    T = 0.001
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_pts = current_pts.copy()
        
        if np.random.random() < 0.03:
            alpha = min(alpha * 1.03, 200.0)
    
    # Phase 3: High-precision polish on critical constraints
    if time.time() < deadline - 5.0:
        polish_pts = best_pts.copy()
        polish_r = best_radius
        polish_deadline = time.time() + 3.0
        
        # Multiple polish passes with increasing alpha
        for pass_num in range(3):
            if time.time() > polish_deadline:
                break
            polish_pts, polish_r = polish_critical(polish_pts, polish_r, max_iter=1000)
            
            # Also do a few gradient steps with very high alpha
            for _ in range(50):
                if time.time() > polish_deadline:
                    break
                grad = soft_min_grad(polish_pts, 200.0)
                new_pts = polish_pts + 0.001 * grad
                new_pts = np.clip(new_pts, -0.5 + 1e-12, 0.5 - 1e-12)
                new_r = compute_radius(new_pts)
                if new_r > polish_r:
                    polish_pts = new_pts
                    polish_r = new_r
        
        if polish_r > best_radius:
            best_radius = polish_r
            best_pts = polish_pts
    
    return best_pts
# EVOLVE-BLOCK-END
