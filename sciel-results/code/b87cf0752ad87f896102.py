import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Initialization: Hexagonal Lattice
    # This is a much better starting point than random rings for dense packing.
    # We generate a hex lattice with spacing s and keep points inside radius 1.
    # We adjust s to get close to N points.
    
    # Area of unit circle is pi.
    # Area per point in hex packing is 2*sqrt(3)*r^2.
    # Approx r ~ sqrt(pi / (2*sqrt(3)*N)).
    # Spacing s = 2*r.
    
    # Let's estimate s
    r_est = np.sqrt(np.pi / (2 * np.sqrt(3) * N))
    s_est = 2 * r_est
    
    # Generate hex lattice points
    # Basis vectors for hex lattice
    a = s_est
    b = a * np.sqrt(3)
    
    # Grid range
    # Max x ~ 1, so i from -ceil(1/a) to ceil(1/a)
    # Max y ~ 1, so j from -ceil(1/b) to ceil(1/b)
    i_max = int(np.ceil(1.0 / a)) + 1
    j_max = int(np.ceil(1.0 / b)) + 1
    
    points = []
    for i in range(-i_max, i_max + 1):
        for j in range(-j_max, j_max + 1):
            x = i * a + (0.5 if j % 2 != 0 else 0) * a
            y = j * b
            if x*x + y*y <= 1.0:
                points.append([x, y])
    
    pts = np.array(points)
    
    # Adjust count to N
    if len(pts) > N:
        # Keep the innermost points (closest to origin)
        dists = np.linalg.norm(pts, axis=1)
        indices = np.argsort(dists)
        pts = pts[indices[:N]]
    elif len(pts) < N:
        # If we have too few, we need to add points.
        # This is unlikely with the estimate, but handle it.
        # Add random points near the boundary? Or just pad with random?
        # Better: shrink s slightly and regenerate?
        # For robustness, let's just pad with random points inside the disk.
        # This is a fallback.
        diff = N - len(pts)
        rand_pts = np.random.randn(diff, 2)
        norms = np.linalg.norm(rand_pts, axis=1)
        # Scale to be inside unit disk
        # Use rejection sampling or simple scaling
        # Simple scaling: divide by max norm * 1.1
        max_norm = np.max(norms)
        if max_norm > 0:
            rand_pts = rand_pts / max_norm * 0.9
        pts = np.vstack([pts, rand_pts])
        
    # 2. Physics-Based Relaxation (Repulsion + Confinement)
    # We want to maximize the minimum distance.
    # We simulate forces:
    # - Repulsive force between all pairs: F ~ 1/d^2 (or softer)
    # - Confinement force towards center if outside, or just clamp.
    # Actually, a better approach for "maximize min distance" is to treat it as an energy minimization
    # where energy is high when points are close.
    # Let's use a simple gradient ascent on the score?
    # Or just move points away from their closest neighbors.
    
    # Let's use a "Repulsion + Centering" step.
    # For each point i:
    #   Find closest neighbor j.
    #   Move i away from j by small step.
    #   If i is too close to boundary, move it inwards.
    
    # To be faster, we can use a global repulsion force field.
    # Force on i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^3  (Coulomb-like)
    # This is O(N^2) per step. N=300, N^2=90k.
    # 1000 steps * 90k = 90M ops. Feasible in 115s?
    # NumPy vectorization helps.
    
    x = pts.copy()
    
    # Precompute indices for efficiency?
    # Let's just use vectorized operations.
    
    # Step size schedule
    step_size = 0.01
    decay = 0.999
    
    # We will run this until time runs out or step size is tiny.
    # We also need to ensure we don't overlap.
    
    # Optimization Loop
    while time.time() < deadline - 10.0:  # Reserve 10s for polish
        # Calculate distances
        # diff[i,j] = x[i] - x[j]
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Avoid division by zero
        np.fill_diagonal(dists_sq, 1.0)
        dists = np.sqrt(dists_sq)
        
        # Calculate forces
        # F_ij = diff_ij / dists_ij^3
        # This pushes points apart.
        # We want to move in the direction of the force.
        # Force on i is sum_j F_ij.
        
        # To avoid large forces from very close points causing instability,
        # we can cap the force or use a softer potential.
        # Let's use F = diff / (dists_sq * dists)
        # This is standard Coulomb.
        
        # Vectorized force calculation
        # We need to sum over j for each i.
        # forces[i] = sum_j (x[i]-x[j]) / |x[i]-x[j]|^3
        
        # To prevent numerical issues with very small distances, add epsilon
        dists_sq_safe = dists_sq + 1e-6
        dists_safe = np.sqrt(dists_sq_safe)
        
        # Factor = 1 / dist^3
        factor = 1.0 / (dists_sq_safe * dists_safe)
        np.fill_diagonal(factor, 0.0) # No self-force
        
        # forces = sum_j diff_ij * factor_ij
        # diff shape (N, N, 2), factor shape (N, N)
        # We need to broadcast factor to (N, N, 1)
        forces = np.sum(diff * factor[:, :, np.newaxis], axis=1)
        
        # Confinement force
        # If point is outside radius 1, pull it in.
        # Actually, we want to keep them inside.
        # Let's just clamp after the move, or add a force towards center if r > 0.95.
        norms = np.linalg.norm(x, axis=1)
        # If norm > 0.9, apply inward force proportional to (norm - 0.9)
        confine_mask = norms > 0.9
        if np.any(confine_mask):
            # Inward direction is -x / norm
            inward_dir = -x[confine_mask] / norms[confine_mask, np.newaxis]
            # Strength proportional to violation
            strength = (norms[confine_mask] - 0.9) * 0.5
            forces[confine_mask] += inward_dir * strength[:, np.newaxis]
            
        # Update positions
        x += forces * step_size
        
        # Clamp to unit disk
        new_norms = np.linalg.norm(x, axis=1)
        outside = new_norms > 1.0
        if np.any(outside):
            x[outside] /= new_norms[outside, np.newaxis]
            
        # Decay step size
        step_size *= decay
        
        # If step size is too small, break early to save time for polish
        if step_size < 1e-5:
            break
            
    # 3. Local Polish: Coordinate Descent
    # Now we do a deterministic local search to maximize the minimum distance.
    # We iterate over points and try to move each one slightly to increase
    # its minimum distance to neighbors and boundary.
    
    # Calculate current score
    def calc_score(x):
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        min_pair = np.min(dists) / 2.0
        min_bound = np.min(1.0 - np.linalg.norm(x, axis=1))
        return min(min_pair, min_bound)
    
    best_x = x.copy()
    best_r = calc_score(best_x)
    
    # Polish loop
    # We will try small moves in 8 directions for each point.
    # If a move increases the local minimum distance, accept it.
    
    polish_steps = 200
    step_polish = 0.001
    
    for _ in range(polish_steps):
        if time.time() > deadline - 2.0:
            break
            
        # Shuffle points to avoid bias
        order = np.random.permutation(N)
        
        for i in order:
            # Current local min for point i
            # Dist to all others
            dists_to_i = np.linalg.norm(x - x[i], axis=1)
            dists_to_i[i] = np.inf
            min_dist_i = np.min(dists_to_i) / 2.0
            bound_i = 1.0 - np.linalg.norm(x[i])
            local_min_i = min(min_dist_i, bound_i)
            
            # Try moves
            best_move = None
            best_local_min = local_min_i
            
            # Directions: 8 neighbors + maybe radial
            for dx in [-step_polish, 0, step_polish]:
                for dy in [-step_polish, 0, step_polish]:
                    if dx == 0 and dy == 0:
                        continue
                    new_x_i = x[i] + np.array([dx, dy])
                    
                    # Check boundary
                    norm_new = np.linalg.norm(new_x_i)
                    if norm_new > 1.0:
                        # Invalid or needs projection
                        # For simplicity, skip if outside
                        continue
                    
                    # Check distances to others
                    # Only need to check if new min > current local min
                    dists_new = np.linalg.norm(x - new_x_i, axis=1)
                    dists_new[i] = np.inf
                    min_dist_new = np.min(dists_new) / 2.0
                    bound_new = 1.0 - norm_new
                    local_min_new = min(min_dist_new, bound_new)
                    
                    if local_min_new > best_local_min:
                        best_local_min = local_min_new
                        best_move = new_x_i
                        
            if best_move is not None:
                x[i] = best_move
                # Update global best if this improved the global score?
                # We can check periodically.
                # For now, just accept local improvements.
                
        # Periodically check global score
        # If time permits, every 10 iterations
        if _ % 10 == 0:
            current_r = calc_score(x)
            if current_r > best_r:
                best_r = current_r
                best_x = x.copy()
                
    # Final check
    final_r = calc_score(x)
    if final_r > best_r:
        best_r = final_r
        best_x = x.copy()
        
    return best_x

# Test
if __name__ == "__main__":
    pts = construct()
    print(pts.shape)
