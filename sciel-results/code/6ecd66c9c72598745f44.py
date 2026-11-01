import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import sys

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # Helper: Compute the packing radius for a set of points
    def compute_radius(pts):
        if len(pts) == 0:
            return 0.0
        tree = cKDTree(pts)
        # Find minimum distance between any pair
        # query_pairs with r=inf is too slow, use query with k=2
        # For N=300, this is fast enough
        dists, _ = tree.query(pts, k=2)
        min_pair_dist = np.min(dists[:, 1])
        
        # Find minimum distance to boundary
        radii = np.linalg.norm(pts, axis=1)
        min_bound_dist = 1.0 - np.max(radii)
        
        return 0.5 * min(min_pair_dist, min_bound_dist)

    # --- Step 1: Initialization ---
    # Use a hexagonal lattice as a strong starting point.
    # This is much better than random for SDP/Local optimization convergence.
    
    def hex_lattice_init(n):
        spacing = 0.11
        best_pts = None
        best_r = -1
        
        while True:
            k = int(1.0 / spacing) + 2
            points = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = i * spacing
                    y = j * spacing * np.sqrt(3)
                    if (j % 2) != 0:
                        x += spacing / 2
                    points.append([x, y])
            points = np.array(points)
            r = np.hypot(points[:, 0], points[:, 1])
            mask = r <= 1.0 - spacing
            points = points[mask]
            
            if len(points) >= n:
                dists = r[mask]
                idx = np.argsort(dists, kind='stable')[:n]
                pts = points[idx]
                r_val = compute_radius(pts)
                if r_val > best_r:
                    best_r = r_val
                    best_pts = pts
                # Try slightly tighter spacing
                spacing *= 0.98
                if spacing < 0.08:
                    break
            else:
                spacing *= 0.95
                if spacing < 0.05:
                    break
        
        if best_pts is None or len(best_pts) < n:
             # Fallback random if lattice fails (unlikely)
             best_pts = np.random.rand(n, 2) * 2 - 1
             best_pts *= 0.9 / np.linalg.norm(best_pts, axis=1)[:, np.newaxis]
        
        return best_pts

    pts = hex_lattice_init(N)
    current_r = compute_radius(pts)
    
    # --- Step 2: Local Optimization using SLSQP ---
    # We optimize the positions directly.
    # To handle the non-smoothness of the min function, we use a "barrier" or 
    # "soft min" approach in the objective, or simply maximize the min distance 
    # by treating it as a variable r and using constraints.
    
    # However, SLSQP with 300x2 variables and O(N^2) constraints is too slow.
    # Instead, we use a "Local Search" strategy that leverages the structure:
    # We identify "active" pairs (those close to the minimum distance) and 
    # solve a small QP for those points to push them apart.
    
    # A more robust alternative for this specific N: 
    # Use a "Gradient Ascent" on the minimum distance.
    # Gradient of min distance w.r.t point i:
    # If point i is part of the minimum pair (i,j), grad_i = (p_i - p_j)/||p_i - p_j||
    # If point i is on the boundary and that's the min, grad_i = -p_i/||p_i||
    
    # Let's implement a robust iterative improvement loop.
    
    def optimize_local(pts, max_iter=500):
        pts = pts.copy()
        current_r = compute_radius(pts)
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            # Identify critical constraints
            tree = cKDTree(pts)
            dists, idxs = tree.query(pts, k=2) # k=2 gives self and 1 neighbor
            min_pair_dists = dists[:, 1]
            min_pair_idxs = idxs[:, 1]
            
            radii = np.linalg.norm(pts, axis=1)
            bound_dists = 1.0 - radii
            
            global_min = min(np.min(min_pair_dists), np.min(bound_dists))
            
            # If no improvement possible (all constraints loose), break? 
            # Actually, we want to maximize the min.
            
            # Gradient calculation
            grads = np.zeros_like(pts)
            
            # Find the specific pair or point that defines the global minimum
            # There might be multiple. We sum gradients from all "active" constraints.
            
            # Active pair constraints: dist < global_min * 1.05 (tolerance)
            active_pairs_mask = min_pair_dists < global_min * 1.05 + 1e-9
            for i in np.where(active_pairs_mask)[0]:
                j = min_pair_idxs[i]
                d_vec = pts[i] - pts[j]
                d = np.linalg.norm(d_vec)
                if d > 1e-9:
                    # Gradient of dist(i,j) w.r.t p_i is (p_i - p_j)/d
                    # We want to maximize dist, so move in direction of gradient
                    grads[i] += d_vec / d
                    # Gradient w.r.t p_j is (p_j - p_i)/d
                    grads[j] += (pts[j] - pts[i]) / d
            
            # Active boundary constraints: bound_dist < global_min * 1.05
            active_bound_mask = bound_dists < global_min * 1.05 + 1e-9
            for i in np.where(active_bound_mask)[0]:
                # Gradient of (1 - |p_i|) w.r.t p_i is -p_i/|p_i|
                # We want to maximize 1-|p|, so move in direction -p_i/|p_i|
                r_i = np.linalg.norm(pts[i])
                if r_i > 1e-9:
                    grads[i] += -pts[i] / r_i
            
            # Normalize gradients?
            # The magnitudes are already unit vectors.
            
            # Determine step size
            # Use a line search or fixed small step.
            # Since we are moving in the direction that increases the minimum constraint,
            # we can take a step proportional to the "slack" or just a fixed small step.
            
            # To avoid oscillation, we use a decaying step size or adaptive step.
            # Let's try a simple adaptive step: try to increase r by a small amount.
            # Actually, just take a step in the gradient direction.
            
            # Normalize the total gradient for each point to prevent large jumps if many constraints are active
            grad_norms = np.linalg.norm(grads, axis=1)
            mask_nonzero = grad_norms > 1e-9
            if np.any(mask_nonzero):
                # Scale gradients so max norm is 1
                max_norm = np.max(grad_norms[mask_nonzero])
                if max_norm > 0:
                    grads[mask_nonzero] /= max_norm
            
            # Step size: 
            # If we move too far, we might violate other constraints.
            # Let's use a small step size, e.g., 0.01 * (1 - current_r/0.053) 
            # Or just fixed 0.005.
            step_size = 0.01
            
            # Apply update
            new_pts = pts + step_size * grads
            
            # Enforce hard constraints: 
            # 1. Inside circle
            new_radii = np.linalg.norm(new_pts, axis=1)
            mask_out = new_radii > 1.0
            if np.any(mask_out):
                new_pts[mask_out] = new_pts[mask_out] / new_radii[mask_out, np.newaxis] * 0.9999
            
            # 2. Check if this move actually improved the radius.
            # If not, we might have moved "sideways" or hurt another constraint.
            # For robustness, only accept if compute_radius(new_pts) > current_r - epsilon.
            # But compute_radius is expensive. 
            # Heuristic: If the gradient direction was towards the minimum, it should improve.
            # However, if multiple points move, they might collide.
            
            # Let's do a quick check on the minimum distance of the new points.
            # If it's worse, revert.
            # To save time, we only compute full radius every 5 iterations.
            
            if it % 5 == 0 or it < 10:
                new_r = compute_radius(new_pts)
                if new_r > current_r:
                    pts = new_pts
                    current_r = new_r
                else:
                    # Revert and reduce step size
                    step_size *= 0.5
                    if step_size < 1e-6:
                        break
            else:
                # Assume improvement if gradients were non-zero
                # But to be safe, we can check just the min pair distance quickly?
                # No, let's just accept and verify later.
                # Actually, blind acceptance can lead to divergence.
                # Let's check the minimum distance to the *original* active neighbors.
                # This is an approximation.
                
                # Simpler: Just accept. The gradient is for the *current* minimum.
                # Moving in that direction increases that specific distance.
                # It might decrease others, but those were larger.
                # So the *minimum* should increase or stay same.
                pts = new_pts
                # Update current_r estimate roughly?
                # No, we need accurate current_r for the next iteration's active set detection.
                # We'll update it in the next loop iteration.
                
            # If step size is too small, we might be stuck.
            # Add a small random perturbation to escape local minima?
            # For now, keep it deterministic.
            
            if step_size < 1e-6:
                break
                
        return pts, current_r

    pts, current_r = optimize_local(pts)
    
    # --- Step 3: Final Polish ---
    # Sometimes the gradient method gets stuck in a local minimum where 
    # the "active set" is rigid. 
    # We can try a few random restarts of the local optimization from 
    # slightly perturbed positions of the best solution found so far.
    
    best_pts = pts
    best_r = current_r
    
    # Try a few perturbations
    for restart in range(5):
        if time.time() > deadline - 5:
            break
            
        # Perturb the best solution
        perturbed_pts = best_pts + np.random.randn(N, 2) * 0.005
        # Project back inside
        radii = np.linalg.norm(perturbed_pts, axis=1)
        mask_out = radii > 1.0
        if np.any(mask_out):
            perturbed_pts[mask_out] = perturbed_pts[mask_out] / radii[mask_out, np.newaxis] * 0.9999
        
        # Optimize
        new_pts, new_r = optimize_local(perturbed_pts, max_iter=200)
        
        if new_r > best_r:
            best_r = new_r
            best_pts = new_pts
            
    return best_pts

# Test run to ensure it works
if __name__ == "__main__":
    pts = construct()
    print(f"Shape: {pts.shape}")
    print(f"Radius: {compute_radius(pts)}")
