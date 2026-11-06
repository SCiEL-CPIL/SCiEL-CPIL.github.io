"""Improved circle packing for N=300 in unit circle using soft-penalty force relaxation."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    def fibonacci_disk(n, radius=0.98):
        """Generate points using Fibonacci disk for uniform distribution."""
        points = np.zeros((n, 2))
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n):
            r = radius * np.sqrt((i + 0.5) / n)
            theta = i * golden_angle
            points[i, 0] = r * np.cos(theta)
            points[i, 1] = r * np.sin(theta)
        return points
    
    def compute_score(pts):
        """Compute the packing score r for given points."""
        if not np.all(np.isfinite(pts)):
            return 0.0
        radii = np.hypot(pts[:, 0], pts[:, 1])
        if np.any(radii >= 1.0):
            return 0.0
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        min_boundary = np.min(1.0 - radii)
        return min(min_pair, min_boundary)
    
    def soft_relax(pts, deadline):
        """
        Soft-penalty force relaxation to maximize minimum distance.
        Uses a smooth potential that penalizes overlaps and boundary violations.
        """
        current_pts = pts.copy()
        current_score = compute_score(current_pts)
        best_pts = current_pts.copy()
        best_score = current_score
        
        # Force parameters
        base_lr = 0.002
        decay = 0.99995
        no_improve = 0
        max_no_improve = 20000
        
        # Soft penalty parameters
        overlap_threshold = 0.005  # Start allowing some overlap to escape local minima
        boundary_threshold = 0.005
        
        it = 0
        while time.time() < deadline:
            it += 1
            n = len(current_pts)
            forces = np.zeros_like(current_pts)
            
            # Compute pairwise forces (repulsion when too close)
            diff = current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Soft repulsion: stronger when distance < target
            # Target distance is roughly 2 * expected radius
            target_dist = 0.11  # ~2 * 0.055
            # Use a smooth potential: F ~ 1/d^2 when d < target, else 0
            mask = dists < target_dist
            if np.any(mask):
                # Inverse square repulsion, softened
                inv_d2 = np.where(dists > 1e-10, 1.0 / (dists**2 + 1e-10), 0.0)
                # Scale force
                force_mag = np.where(mask, inv_d2 * (target_dist - dists), 0.0)
                # Normalize direction
                norm_d = np.where(dists > 1e-10, 1.0 / dists, 0.0)
                # Force = direction * magnitude
                force_x = diff[:, :, 0] * norm_d * force_mag
                force_y = diff[:, :, 1] * norm_d * force_mag
                forces[:, 0] += np.sum(force_x, axis=1)
                forces[:, 1] += np.sum(force_y, axis=1)
            
            # Boundary force: push inward if too close to boundary
            radii = np.hypot(current_pts[:, 0], current_pts[:, 1])
            boundary_dist = 1.0 - radii
            boundary_mask = boundary_dist < boundary_threshold
            if np.any(boundary_mask):
                inward_force_mag = (boundary_threshold - boundary_dist[boundary_mask]) ** 2
                # Inward direction is -pts / |pts|
                norm_r = np.where(radii > 1e-10, 1.0 / radii, 0.0)
                forces[boundary_mask, 0] -= current_pts[boundary_mask, 0] * norm_r[boundary_mask] * inward_force_mag * 0.5
                forces[boundary_mask, 1] -= current_pts[boundary_mask, 1] * norm_r[boundary_mask] * inward_force_mag * 0.5
            
            # Apply forces with adaptive learning rate
            new_pts = current_pts + base_lr * forces
            
            # Project inside unit circle
            new_radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
            mask_out = new_radii > 0.999
            if np.any(mask_out):
                new_pts[mask_out] = new_pts[mask_out] * (0.999 / new_radii[mask_out])
            
            new_score = compute_score(new_pts)
            
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
                no_improve = 0
                if new_score > best_score:
                    best_score = new_score
                    best_pts = current_pts.copy()
                    # Tighten thresholds as we improve
                    overlap_threshold = max(0.001, overlap_threshold * 0.999)
                    boundary_threshold = max(0.001, boundary_threshold * 0.999)
            else:
                no_improve += 1
                base_lr *= 0.999
                if base_lr < 0.0001:
                    base_lr = 0.0001
            
            # If stuck, try a small random perturbation
            if no_improve > max_no_improve:
                n_perturb = np.random.randint(3, 10)
                perturb_idx = np.random.choice(n, n_perturb, replace=False)
                for idx in perturb_idx:
                    angle = np.random.uniform(0, 2 * np.pi)
                    r = np.random.uniform(0, 0.95)
                    current_pts[idx] = [r * np.cos(angle), r * np.sin(angle)]
                current_score = compute_score(current_pts)
                no_improve = 0
                base_lr = 0.002
                overlap_threshold = 0.005
                boundary_threshold = 0.005
            
            # Cool down learning rate
            base_lr *= decay
            if base_lr < 0.0001:
                base_lr = 0.0001
        
        return best_pts, best_score
    
    # Main execution
    best_pts = None
    best_score = 0.0
    
    # Strategy 1: Fibonacci disk + soft relaxation
    pts1 = fibonacci_disk(N)
    pts1, score1 = soft_relax(pts1, deadline - 20)
    if score1 > best_score:
        best_score = score1
        best_pts = pts1
    
    # Strategy 2: Different initialization + soft relaxation
    if time.time() < deadline - 20:
        # Use a slightly different Fibonacci radius
        pts2 = fibonacci_disk(N, radius=0.96)
        pts2, score2 = soft_relax(pts2, deadline - 10)
        if score2 > best_score:
            best_score = score2
            best_pts = pts2
    
    # Final local refinement
    if best_pts is not None and time.time() < deadline - 2:
        current_pts = best_pts.copy()
        current_score = best_score
        
        for _ in range(5000):
            if time.time() > deadline - 1:
                break
            
            idx = np.random.randint(N)
            orig_pt = current_pts[idx].copy()
            
            move_size = 0.003
            step = np.random.randn(2) * move_size
            new_pt = orig_pt + step
            
            r = np.hypot(new_pt[0], new_pt[1])
            if r > 0.999:
                new_pt = new_pt * (0.999 / r)
            
            current_pts[idx] = new_pt
            new_score = compute_score(current_pts)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_pts = current_pts.copy()
            else:
                current_pts[idx] = orig_pt
    
    # Ensure validity
    if best_pts is None:
        best_pts = fibonacci_disk(N)
    
    radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
    if np.any(radii >= 1.0) or not np.all(np.isfinite(best_pts)):
        scale = 0.99 / np.max(radii) if np.max(radii) > 0 else 1.0
        best_pts = best_pts * scale
    
    return best_pts
# EVOLVE-BLOCK-END
