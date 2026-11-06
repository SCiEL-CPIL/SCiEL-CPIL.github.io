"""Circle packing for N=300 in unit circle using Fibonacci disk initialization and vectorized force relaxation."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    def compute_score(pts):
        """Compute the packing score r for given points."""
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        radii = np.hypot(pts[:, 0], pts[:, 1])
        if np.any(radii >= 1.0):
            return 0.0
        min_boundary = np.min(1.0 - radii)
        # Pairwise distances using broadcasting
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        return min(min_pair, min_boundary)

    def generate_fibonacci_disk():
        """Generate initial configuration using Fibonacci disk distribution."""
        # Golden angle
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        
        points = np.zeros((N, 2))
        for i in range(N):
            # Radial coordinate: sqrt distribution for uniform area density
            r = np.sqrt((i + 0.5) / N)
            # Angular coordinate: golden angle spiral
            theta = i * golden_angle
            points[i, 0] = r * np.cos(theta)
            points[i, 1] = r * np.sin(theta)
        
        # Scale to fit inside unit circle with margin
        max_r = np.max(np.hypot(points[:, 0], points[:, 1]))
        if max_r > 0.95:
            points *= 0.95 / max_r
        
        return points

    def optimize_vectorized(pts, max_iters=20000, deadline_check=None):
        """Vectorized force-based relaxation using soft penalty for overlaps and boundary."""
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        # Parameters
        step_size = 0.01
        decay = 0.9998
        softness = 0.05  # Softness parameter for penalty
        
        for it in range(max_iters):
            if deadline_check is not None and time.time() > deadline_check:
                break
            
            # Compute all pairwise distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            dists = np.sqrt(dists_sq)
            np.fill_diagonal(dists, np.inf)
            
            # Current score
            radii = np.hypot(pts[:, 0], pts[:, 1])
            min_boundary = np.min(1.0 - radii)
            min_pair = np.min(dists) / 2.0
            current_score = min(min_pair, min_boundary)
            
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
            
            # Target radius: slightly above current best to encourage expansion
            target_r = best_score * 1.005 + 0.002
            
            # Compute forces (vectorized)
            forces = np.zeros_like(pts)
            
            # Pairwise repulsion: for each pair, if distance < 2*target_r, push apart
            # Use soft penalty: force = (2*target_r - d) / d for d < 2*target_r
            mask = dists < 2.0 * target_r
            # Avoid division by zero
            safe_dists = np.where(dists > 1e-10, dists, 1e-10)
            force_mag = np.where(mask, (2.0 * target_r - safe_dists) / safe_dists, 0.0)
            force_mag[~mask] = 0.0
            np.fill_diagonal(force_mag, 0.0)
            
            # Force vector: direction from j to i, magnitude force_mag
            # diff[i,j] = pts[i] - pts[j]
            # force on i from j: force_mag[i,j] * diff[i,j] / dists[i,j]
            force_vec = diff * (force_mag / safe_dists)[:, :, np.newaxis]
            # Sum over j for each i
            forces += np.sum(force_vec, axis=1)
            
            # Boundary constraint: if point is too close to boundary, push inward
            boundary_mask = radii > 1.0 - target_r
            if np.any(boundary_mask):
                violations = np.maximum(radii - (1.0 - target_r), 0.0)
                # Direction: inward (negative radial direction)
                radial_dir = pts / np.maximum(radii, 1e-10)[:, np.newaxis]
                forces[boundary_mask] -= radial_dir[boundary_mask] * violations[boundary_mask, np.newaxis] * 20.0
            
            # Update positions
            pts += forces * step_size
            
            # Project back inside unit circle
            new_radii = np.hypot(pts[:, 0], pts[:, 1])
            outside = new_radii > 1.0
            if np.any(outside):
                pts[outside] = pts[outside] / new_radii[outside, np.newaxis]
            
            # Decay step size
            step_size *= decay
            
            # Occasionally do a small random perturbation to escape local minima
            if it % 500 == 0 and it > 0:
                perturb_mask = np.random.rand(N) < 0.1
                if np.any(perturb_mask):
                    pts[perturb_mask] += np.random.randn(np.sum(perturb_mask), 2) * 0.005
                    new_radii = np.hypot(pts[:, 0], pts[:, 1])
                    outside = new_radii > 1.0
                    if np.any(outside):
                        pts[outside] = pts[outside] / new_radii[outside, np.newaxis]
        
        return best_pts, best_score

    def optimize_annealing(pts, max_iters=5000, deadline_check=None):
        """Simulated annealing for final refinement."""
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        temp = 0.005
        cool_rate = 0.9999
        
        for it in range(max_iters):
            if deadline_check is not None and time.time() > deadline_check:
                break
            
            # Perturb a random point
            idx = np.random.randint(N)
            perturbation = np.random.randn(2) * temp
            new_pts = pts.copy()
            new_pts[idx] += perturbation
            
            # Project inside
            new_radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
            outside = new_radii > 1.0
            if np.any(outside):
                new_pts[outside] = new_pts[outside] / new_radii[outside, np.newaxis]
            
            new_score = compute_score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
            
            # Accept with probability
            if new_score >= best_score or np.random.rand() < np.exp((new_score - best_score) / (temp + 1e-10)):
                pts = new_pts
            
            temp *= cool_rate
        
        return best_pts, best_score

    # Main optimization
    np.random.seed(42)
    
    # Generate initial configuration using Fibonacci disk
    pts = generate_fibonacci_disk()
    best_pts = pts.copy()
    best_score = compute_score(pts)
    
    # Phase 1: Vectorized force-based relaxation
    pts, score = optimize_vectorized(pts, max_iters=15000, deadline_check=deadline)
    if score > best_score:
        best_score = score
        best_pts = pts
    
    # Phase 2: More force-based with smaller steps for refinement
    if time.time() < deadline - 20:
        pts, score = optimize_vectorized(best_pts, max_iters=10000, deadline_check=deadline)
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # Phase 3: Simulated annealing for final tweaks
    if time.time() < deadline - 10:
        pts, score = optimize_annealing(best_pts, max_iters=3000, deadline_check=deadline)
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # Final validation and return
    final_score = compute_score(best_pts)
    if final_score > best_score:
        best_score = final_score
    
    # Ensure valid
    radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
    if np.any(radii >= 1.0) or not np.all(np.isfinite(best_pts)):
        # Fallback: shrink to ensure validity
        scale = 0.99 / np.max(radii) if np.max(radii) > 0 else 1.0
        best_pts = best_pts * scale
        final_score = compute_score(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
