"""Improved circle packing for N=300 in unit circle using vectorized force relaxation."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s limit
    
    def compute_score(pts):
        """Compute the packing score r for given points."""
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        # Check all points are inside unit circle
        radii = np.hypot(pts[:, 0], pts[:, 1])
        if np.any(radii >= 1.0):
            return 0.0
        min_boundary = np.min(1.0 - radii)
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        return min(min_pair, min_boundary)

    def generate_hex_lattice():
        """Generate a hexagonal lattice configuration clipped to the unit circle."""
        # Estimate optimal spacing
        # Area per circle ~ pi / N * (1/density). Density ~ 0.9.
        # r ~ 0.053. d = 2r ~ 0.106.
        # Let's start with a grid and filter.
        
        # Create a dense grid of candidate points
        # Range -1.2 to 1.2
        # Step size slightly smaller than expected diameter to ensure enough candidates
        step = 0.08 
        x_range = np.arange(-1.2, 1.2, step)
        y_range = np.arange(-1.2, 1.2, step)
        
        # Hexagonal offset
        points = []
        for i, x in enumerate(x_range):
            for y in y_range:
                if i % 2 == 1:
                    y_offset = step * np.sqrt(3) / 2
                else:
                    y_offset = 0
                points.append([x, y + y_offset])
        
        points = np.array(points)
        
        # Filter points inside unit circle (with margin)
        radii = np.hypot(points[:, 0], points[:, 1])
        mask = radii < 0.95 # Keep well inside initially
        
        candidates = points[mask]
        
        # If we have too many, we need to select N points that maximize min distance.
        # Greedy selection: pick point, remove neighbors, repeat.
        # Or just take the first N? No, that's biased.
        # Let's use a simple greedy max-min distance selection.
        
        if len(candidates) <= N:
             # If not enough, just pad with random? Or expand grid?
             # For N=300, step 0.08 should give plenty.
             # Let's just return all if less, but we expect more.
             return candidates[:N]
        
        selected = []
        remaining = candidates.copy()
        
        # Start with center-most point or random
        # Let's pick the one closest to origin
        dists_origin = np.hypot(remaining[:, 0], remaining[:, 1])
        idx = np.argmin(dists_origin)
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)
        
        while len(selected) < N and len(remaining) > 0:
            selected_arr = np.array(selected)
            # Compute min distance from each remaining candidate to selected set
            # This is O(N_selected * N_remaining). 
            # To speed up, we can just pick the one with max min-dist.
            
            # Vectorized min dist to selected
            # diff: (N_rem, N_sel, 2) -> too big? N_rem ~ 1000, N_sel ~ 300. 1000*300*2 = 600k floats. OK.
            diff = remaining[:, np.newaxis, :] - selected_arr[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            min_dists = np.min(dists, axis=1)
            
            idx = np.argmax(min_dists)
            selected.append(remaining[idx])
            remaining = np.delete(remaining, idx, axis=0)
            
            # Check time
            if time.time() > deadline - 100:
                break
                
        return np.array(selected[:N])

    def optimize_vectorized(pts, max_iters=20000, deadline_check=None):
        """
        Vectorized force-directed optimization.
        Uses NumPy to compute all forces simultaneously.
        """
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        # Adaptive step size
        step_size = 0.01
        # Target radius for forces: slightly above current score to encourage growth
        # But we must ensure we don't explode.
        
        for it in range(max_iters):
            if deadline_check is not None and time.time() > deadline_check:
                break
            
            # Compute pairwise differences and distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Current score
            radii = np.hypot(pts[:, 0], pts[:, 1])
            min_boundary = np.min(1.0 - radii)
            min_pair = np.min(dists) / 2.0
            current_score = min(min_pair, min_boundary)
            
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
                
            # Target radius for force calculation
            # We want to push pairs apart if they are closer than 2*r_target
            # And push points in if they are further than 1-r_target
            r_target = best_score * 1.005 + 0.001
            
            # Forces
            forces = np.zeros_like(pts)
            
            # 1. Pairwise repulsion
            # Force magnitude: proportional to overlap
            # If d < 2*r_target, push apart.
            # We can use a soft repulsion: F = k * (2*r_target - d) if d < 2*r_target else 0
            # Vectorized:
            overlap = 2.0 * r_target - dists
            overlap[overlap < 0] = 0.0
            
            # Normalize direction: diff / dists
            # Handle division by zero
            dists_safe = np.where(dists > 1e-10, dists, 1.0)
            dir_vecs = diff / dists_safe[:, :, np.newaxis]
            
            # Force contribution from each pair i->j is dir_vecs[i,j] * overlap[i,j]
            # Sum over j for each i
            # forces[i] += sum_j dir_vecs[i,j] * overlap[i,j]
            # Note: diff[i,j] = pts[i] - pts[j]. So dir_vecs[i,j] points from j to i.
            # If overlap > 0, we want to push i away from j, so force on i is in direction of diff[i,j]. Correct.
            
            # Multiply direction by overlap magnitude
            weighted_dirs = dir_vecs * overlap[:, :, np.newaxis]
            
            # Sum over axis 1 (j) to get force on i
            forces += np.sum(weighted_dirs, axis=1)
            
            # 2. Boundary attraction/repulsion
            # If radii[i] > 1 - r_target, push in.
            # Force direction: -pts[i] / radii[i]
            # Magnitude: (radii[i] - (1 - r_target))
            
            boundary_violation = radii - (1.0 - r_target)
            boundary_violation[boundary_violation < 0] = 0.0
            
            radii_safe = np.where(radii > 1e-10, radii, 1.0)
            boundary_dirs = -pts / radii_safe[:, np.newaxis]
            
            forces += boundary_dirs * boundary_violation[:, np.newaxis]
            
            # Update
            pts += forces * step_size
            
            # Project inside unit circle (hard constraint)
            new_radii = np.hypot(pts[:, 0], pts[:, 1])
            outside = new_radii > 1.0
            if np.any(outside):
                pts[outside] = pts[outside] / new_radii[outside, np.newaxis]
            
            # Decay step size
            step_size *= 0.999
            
            # If step size gets too small, stop
            if step_size < 1e-6:
                break
                
        return best_pts, best_score

    def local_search(pts, max_iters=5000, deadline_check=None):
        """
        Local search: perturb one point at a time, keep if better.
        """
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        # Precompute distances? No, too expensive to update efficiently.
        # Just recompute score.
        
        temp = 0.01
        for it in range(max_iters):
            if deadline_check is not None and time.time() > deadline_check:
                break
                
            idx = np.random.randint(N)
            orig_pt = pts[idx].copy()
            
            # Perturb
            pts[idx] += np.random.randn(2) * temp
            
            # Project inside
            r = np.hypot(pts[idx, 0], pts[idx, 1])
            if r > 1.0:
                pts[idx] = pts[idx] / r
            
            new_score = compute_score(pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
            else:
                # Revert
                pts[idx] = orig_pt
                
            # Decay temperature
            temp *= 0.9999
            if temp < 1e-5:
                temp = 0.005 # Reset if too small to allow exploration
                
        return best_pts, best_score

    # Main execution
    # 1. Initialize
    pts = generate_hex_lattice()
    if len(pts) < N:
        # Fallback to rings if lattice failed
        points = []
        points.append([0.0, 0.0])
        for k in range(1, 10):
            n_points = 6 * k
            radius = k * 0.055
            angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
            for ang in angles:
                points.append([radius * np.cos(ang), radius * np.sin(ang)])
        remaining = N - len(points)
        if remaining > 0:
            k = 10
            radius = k * 0.055
            angles = np.linspace(0, 2*np.pi, remaining, endpoint=False)
            for ang in angles:
                points.append([radius * np.cos(ang), radius * np.sin(ang)])
        pts = np.array(points[:N])
        
    best_pts = pts.copy()
    best_score = compute_score(pts)
    
    # 2. Vectorized Optimization
    # Run multiple passes to refine
    for _ in range(3):
        if time.time() > deadline - 20:
            break
        pts, score = optimize_vectorized(pts, max_iters=10000, deadline_check=deadline)
        if score > best_score:
            best_score = score
            best_pts = pts
            
    # 3. Local Search for final tweaks
    if time.time() < deadline - 5:
        pts, score = local_search(best_pts, max_iters=2000, deadline_check=deadline)
        if score > best_score:
            best_score = score
            best_pts = pts
            
    # Final check
    final_score = compute_score(best_pts)
    if final_score > best_score:
        best_score = final_score
        
    # Ensure validity
    radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
    if np.any(radii >= 1.0) or not np.all(np.isfinite(best_pts)):
        scale = 0.99 / np.max(radii) if np.max(radii) > 0 else 1.0
        best_pts = best_pts * scale
        
    return best_pts
# EVOLVE-BLOCK-END
