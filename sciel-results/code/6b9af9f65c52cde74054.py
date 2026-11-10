import numpy as np
import time
from scipy.spatial.distance import cdist

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    def compute_score(points):
        if points.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        # Boundary check
        radii = np.linalg.norm(points, axis=1)
        if np.any(radii >= 1.0):
            return 0.0
        
        dist = cdist(points, points)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        min_boundary_dist = np.min(1.0 - radii)
        
        return min(min_pair_dist / 2.0, min_boundary_dist)

    # --- Initialization: Hexagonal Lattice ---
    # A good initial guess helps the relaxation converge faster and avoid deep local minima.
    # Estimate radius from area: r ~ sqrt(0.9 / (N * pi)) is too rough.
    # Better: use known optimal r ~ 0.053. Spacing = 2*r ~ 0.106.
    # Let's generate a hex lattice with spacing slightly larger than optimal to ensure fit, 
    # or just generate a large hex grid and pick the best N.
    
    # We will use a spacing that allows for some compression.
    # Optimal spacing s_opt = 2 * 0.05315 = 0.1063.
    # Let's start with s_init = 0.11 to be safe, then let forces compress.
    
    def generate_hex_lattice(n_target, s):
        pts = []
        # Generate enough points
        max_range = int(1.0 / s) + 2
        for j in range(-max_range, max_range + 1):
            for i in range(-max_range, max_range + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0:
                    pts.append((x, y))
        
        pts = np.array(pts)
        if len(pts) < n_target:
            # Fallback: random
            return np.random.uniform(-0.9, 0.9, (n_target, 2))
        
        # Sort by distance from center to pick the most central ones
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists, kind='stable')[:n_target]
        return pts[idx]

    # Multi-start strategy: Try a few different initializations or random perturbations
    # But given time constraints, a single robust start with strong relaxation is often better.
    # Let's use the hex lattice start.
    s_init = 0.108 # Slightly larger than 2*0.053
    points = generate_hex_lattice(N, s_init)
    
    # Add small random noise to break symmetry
    points += np.random.normal(0, 0.001, points.shape)
    
    # --- Physics-based Relaxation ---
    # We model the system with:
    # 1. Repulsive force between circles if they are closer than target distance D.
    #    F_rep = k_rep * (D - d) * direction  (if d < D)
    # 2. Confinement force pushing points towards center if they are too close to boundary.
    #    F_conf = k_conf * (1 - r) * direction (if r > 1 - r_margin)
    #    Actually, simpler: Force towards origin proportional to (r - r_max_allowed)
    
    # We want to maximize the minimum distance.
    # Let's iterate:
    # 1. Calculate all pair distances.
    # 2. For pairs with d < D, apply repulsive force.
    # 3. For points with r > R_max, apply inward force.
    # 4. Update positions with velocity damping (Langevin dynamics or simple gradient descent).
    
    # To be efficient, we can use a simple iterative scheme:
    # x_new = x + alpha * force
    
    # Parameters
    # D is the current target diameter. We want to find the max D.
    # Instead of fixing D, we can let the system find its natural equilibrium.
    # A common approach: Minimize the potential energy U = sum_{i<j} max(0, D - d_ij)^2 + sum_i max(0, r_i - R)^2
    # But we don't know D.
    
    # Alternative: "Jiggling" or "Annealing"
    # Start with a D slightly smaller than optimal.
    # Increase D slowly. If the system can relax to fit D, increase D more.
    
    # Let's try a direct optimization of the configuration to maximize min_dist.
    # We will use a simple gradient-free local search or a physics simulation.
    
    # Physics Simulation Setup:
    # Force = sum of repulsions + confinement
    # Repulsion: If d_ij < d_target, force = k * (d_target - d_ij)
    # Confinement: If r_i > 1 - r_circle, force = k * (r_i - (1 - r_circle))
    
    # We don't know r_circle.
    # Let's assume the limiting factor is either pair distance or boundary.
    # We can iterate on a parameter 'r_c' (radius of small circles).
    # For a given r_c, the target distance is 2*r_c.
    # The boundary distance must be >= r_c.
    
    # Strategy:
    # 1. Start with r_c = 0.05.
    # 2. Relax configuration to fit r_c.
    # 3. If successful (all constraints met), increase r_c slightly.
    # 4. Repeat until time limit or convergence.
    
    def relax_configuration(points, r_c, max_iters=500, alpha=0.1):
        """
        Relax points to fit circles of radius r_c.
        Returns relaxed points.
        """
        d_target = 2.0 * r_c
        r_max = 1.0 - r_c
        
        # Pre-calculate indices for pairs? No, N=300 is small enough for cdist each iter if needed, 
        # but force calculation needs directions.
        
        # For efficiency, we can use a simple loop or vectorized ops.
        # Vectorized force calculation:
        # dist matrix D
        # force matrix F
        
        pts = points.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Compute distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            
            # Repulsive forces
            # Only pairs with dist < d_target contribute
            # Force magnitude: k * (d_target - dist)
            # Direction: diff / dist
            
            mask = dist < d_target
            # Avoid division by zero
            safe_dist = np.where(mask, dist, 1.0)
            
            # Force vector for each pair (i,j)
            # F_ij = k * (d_target - d_ij) * (p_i - p_j) / d_ij
            # Note: diff[i,j] = p_i - p_j
            # Force on i is sum over j of F_ij
            # Force on j is sum over i of F_ji = -F_ij
            
            k_rep = 1.0
            force_mag = k_rep * (d_target - dist)
            force_mag[~mask] = 0.0
            # Normalize direction
            # diff / dist
            dir_vec = diff / safe_dist[:, :, np.newaxis]
            
            # Force on i from j: force_mag[i,j] * dir_vec[i,j]
            # Sum over j
            force_rep = np.sum(force_mag[:, :, np.newaxis] * dir_vec, axis=1)
            
            # Confinement force
            # If r_i > r_max, push towards center
            r_i = np.linalg.norm(pts, axis=1)
            mask_conf = r_i > r_max
            k_conf = 1.0
            force_conf_mag = k_conf * (r_i - r_max)
            force_conf_mag[~mask_conf] = 0.0
            # Direction towards center: -p_i / r_i
            safe_r = np.where(r_i > 1e-6, r_i, 1.0)
            force_conf = -force_conf_mag[:, np.newaxis] * pts / safe_r[:, np.newaxis]
            
            total_force = force_rep + force_conf
            
            # Update positions
            # Simple gradient descent / Euler integration
            # Damping
            pts = pts + alpha * total_force
            
            # Optional: Project back inside if they fly out?
            # The confinement force should handle it, but let's clip to be safe
            new_r = np.linalg.norm(pts, axis=1)
            if np.any(new_r > 1.0):
                # Scale back
                pts[new_r > 1.0] /= new_r[new_r > 1.0, np.newaxis]
                
        return pts

    # Main Loop: Increase r_c
    # Start with a conservative r_c
    r_c = 0.050
    best_points = points
    best_score = compute_score(points)
    
    # We will try to increase r_c in steps
    # Step size depends on time
    # Let's say we have ~100 seconds for search.
    # Each relaxation takes ~0.1s (500 iters * cdist(300) ~ 500 * 0.001s = 0.5s? cdist is fast).
    # Let's benchmark roughly.
    
    # To be safe, limit total iterations.
    while time.time() < deadline - 20:
        # Relax for current r_c
        # Use fewer iterations for larger r_c if we are close to limit?
        # Or just fixed iterations.
        relaxed_pts = relax_configuration(best_points, r_c, max_iters=200, alpha=0.05)
        
        # Check if valid
        # Compute score
        score = compute_score(relaxed_pts)
        
        if score >= r_c - 1e-9: # If we successfully fit r_c (min_dist/2 >= r_c and boundary >= r_c)
            # Actually, score is min(min_d/2, min_b). We want score >= r_c.
            # If score < r_c, it means we didn't fit.
            if score < r_c:
                # Failed to fit r_c. Try smaller step or break?
                # If we can't fit r_c, we can't fit larger.
                # But maybe relaxation was bad.
                # Let's try a smaller r_c? No, we are increasing.
                # If it fails, we should stop increasing.
                # However, numerical errors might cause slight failure.
                # Let's allow a small tolerance.
                if score < r_c * 0.99:
                    break
            best_points = relaxed_pts
            best_score = max(best_score, score)
            # Increase r_c
            r_c += 0.0005
        else:
            # Failed to fit. 
            # Maybe the configuration got stuck.
            # Try a different initial configuration?
            # Or just break.
            # Let's try a smaller increase?
            # No, if it can't fit 0.055, it likely can't fit 0.056.
            # But maybe the relaxation was poor.
            # Let's try one more time with more iterations?
            relaxed_pts2 = relax_configuration(best_points, r_c, max_iters=500, alpha=0.02)
            score2 = compute_score(relaxed_pts2)
            if score2 >= r_c * 0.99:
                best_points = relaxed_pts2
                best_score = max(best_score, score2)
                r_c += 0.0005
            else:
                break

    # Final Polish: Local optimization
    # Use a simple local search to squeeze out a bit more.
    # We can use the 'maximin' heuristic.
    # Or just use the result from the physics relaxation.
    
    # Let's do a few steps of "coordinate descent" on the most constrained pairs.
    # This is complex to implement robustly in short time.
    # The physics relaxation is usually quite good.
    
    # Ensure valid output
    if best_points.shape != (N, 2) or not np.all(np.isfinite(best_points)):
        # Fallback
        pts = generate_hex_lattice(N, 0.11)
        return pts
        
    # Final check
    final_score = compute_score(best_points)
    if final_score < 0.01:
        # Something went wrong, fallback
        pts = generate_hex_lattice(N, 0.11)
        return pts

    return best_points

if __name__ == "__main__":
    start = time.time()
    pts = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    score = compute_score(pts)
    print(f"Score: {score}")
