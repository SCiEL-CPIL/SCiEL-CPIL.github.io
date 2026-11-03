"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    def compute_radii(centers):
        """
        Vectorized computation of maximum non-overlapping radii for given centers.
        centers: (N, 2) array
        returns: (N,) array of radii
        """
        centers = np.asarray(centers)
        # Boundary distances
        d_left = centers[:, 0]
        d_right = 1.0 - centers[:, 0]
        d_bottom = centers[:, 1]
        d_top = 1.0 - centers[:, 1]
        boundary_r = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        
        # For each i, the max radius is min(boundary_r[i], min_{j != i} dist[i, j] / 2)
        np.fill_diagonal(dist, np.inf)
        
        # min distance to any other center, divided by 2
        min_neighbor_dist = np.min(dist, axis=1) / 2.0
        
        radii = np.minimum(boundary_r, min_neighbor_dist)
        return radii

    def objective(centers):
        """
        Returns negative sum of radii (to be minimized).
        """
        radii = compute_radii(centers)
        return -np.sum(radii)

    def is_valid(x, tol=1e-8):
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        # Check bounds
        if np.any(xs - rs < -tol) or np.any(xs + rs > 1 + tol):
            return False
        if np.any(ys - rs < -tol) or np.any(ys + rs > 1 + tol):
            return False
        
        # Check overlaps
        centers = x[:, :2]
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        
        # sum of radii
        r_sum = rs[:, np.newaxis] + rs[np.newaxis, :]
        
        # Check if dist < r_sum - tol
        mask = ~np.eye(N, dtype=bool)
        if np.any(dist[mask] < r_sum[mask] - tol):
            return False
            
        return True

    def fix_validity(x):
        """
        If x is invalid, shrink radii until valid.
        """
        if is_valid(x, tol=1e-8):
            return x
        
        # Try shrinking radii
        factor = 1.0
        for _ in range(100):
            factor *= 0.99
            new_x = x.copy()
            new_x[:, 2] = x[:, 2] * factor
            if is_valid(new_x, tol=1e-8):
                return new_x
        
        # Fallback: very small radii
        new_x = x.copy()
        new_x[:, 2] = 0.01
        return new_x

    def physics_relaxation(centers, max_iter=5000, dt=0.001, pressure_coeff=1.0):
        """
        Physics-based bubble relaxation. Circles are modeled as pressurized gas bubbles
        with internal pressure proportional to r^2. They expand against repulsive contact
        forces and boundary constraints. The system finds mechanical equilibrium which
        naturally maximizes the sum of radii.
        """
        centers = centers.copy()
        n = len(centers)
        
        # Initialize radii
        radii = compute_radii(centers)
        
        # Velocity for each circle
        velocities = np.zeros((n, 2))
        
        # Friction coefficient
        friction = 0.95
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Compute forces on each circle
            forces = np.zeros((n, 2))
            
            # 1. Internal pressure force: proportional to r^2, pushes outward
            #    This drives circles to expand
            for i in range(n):
                # Pressure force magnitude proportional to r_i^2
                pressure_mag = pressure_coeff * radii[i] ** 2
                
                # Direction: away from nearest constraint (boundary or other circle)
                # For simplicity, use the direction that would increase the minimum constraint
                # Compute all constraints for circle i
                d_left = centers[i, 0]
                d_right = 1.0 - centers[i, 0]
                d_bottom = centers[i, 1]
                d_top = 1.0 - centers[i, 1]
                
                # Find which boundary is closest
                min_boundary = min(d_left, d_right, d_bottom, d_top)
                
                # Find closest other circle
                min_other_dist = np.inf
                closest_dir = np.zeros(2)
                for j in range(n):
                    if i == j:
                        continue
                    dist_ij = np.linalg.norm(centers[i] - centers[j])
                    if dist_ij < min_other_dist:
                        min_other_dist = dist_ij
                        closest_dir = centers[i] - centers[j]
                
                # The limiting constraint determines the direction
                if min_boundary < min_other_dist / 2:
                    # Boundary is limiting
                    if min_boundary == d_left:
                        force_dir = np.array([-1.0, 0.0])
                    elif min_boundary == d_right:
                        force_dir = np.array([1.0, 0.0])
                    elif min_boundary == d_bottom:
                        force_dir = np.array([0.0, -1.0])
                    else:
                        force_dir = np.array([0.0, 1.0])
                else:
                    # Another circle is limiting
                    if min_other_dist > 1e-10:
                        force_dir = closest_dir / min_other_dist
                    else:
                        force_dir = np.array([1.0, 0.0])
                
                forces[i] += pressure_mag * force_dir
            
            # 2. Repulsive contact forces between overlapping or nearly-touching circles
            for i in range(n):
                for j in range(i + 1, n):
                    dist_ij = np.linalg.norm(centers[i] - centers[j])
                    if dist_ij < 1e-10:
                        # Random small push
                        forces[i] += np.array([np.random.randn(), np.random.randn()]) * 0.01
                        forces[j] -= np.array([np.random.randn(), np.random.randn()]) * 0.01
                        continue
                    
                    # Contact force if circles are too close
                    # Desired separation: r_i + r_j
                    desired_sep = radii[i] + radii[j]
                    
                    if dist_ij < desired_sep:
                        # Repulsive force proportional to overlap
                        overlap = desired_sep - dist_ij
                        force_mag = 10.0 * overlap  # Stiff spring constant
                        direction = (centers[i] - centers[j]) / dist_ij
                        
                        forces[i] += force_mag * direction
                        forces[j] -= force_mag * direction
            
            # 3. Boundary repulsion: push circles away from walls if too close
            for i in range(n):
                margin = radii[i] * 0.5  # Start pushing when within half radius of wall
                
                # Left wall
                if centers[i, 0] < margin:
                    forces[i, 0] += 10.0 * (margin - centers[i, 0])
                # Right wall
                if centers[i, 0] > 1.0 - margin:
                    forces[i, 0] -= 10.0 * (centers[i, 0] - (1.0 - margin))
                # Bottom wall
                if centers[i, 1] < margin:
                    forces[i, 1] += 10.0 * (margin - centers[i, 1])
                # Top wall
                if centers[i, 1] > 1.0 - margin:
                    forces[i, 1] -= 10.0 * (centers[i, 1] - (1.0 - margin))
            
            # Update velocities and positions
            velocities = friction * velocities + forces * dt
            centers += velocities * dt
            
            # Clamp to unit square (with small margin for numerical stability)
            centers[:, 0] = np.clip(centers[:, 0], 0.01, 0.99)
            centers[:, 1] = np.clip(centers[:, 1], 0.01, 0.99)
            
            # Recompute radii after position update
            radii = compute_radii(centers)
            
            # Adaptive dt: reduce if system is unstable
            max_speed = np.max(np.linalg.norm(velocities, axis=1))
            if max_speed > 0.1:
                dt *= 0.9
            
        return centers, radii

    def polish_local_search(centers, radii, max_iter=3000):
        """
        High-precision polish pass using local search with fine steps.
        """
        current_centers = centers.copy()
        current_radii = radii.copy()
        current_score = np.sum(current_radii)
        
        step_scale = 0.005
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for i in range(N):
                # Try 8 directions with current step scale
                for dx in [-step_scale, 0, step_scale]:
                    for dy in [-step_scale, 0, step_scale]:
                        if dx == 0 and dy == 0:
                            continue
                        
                        new_centers = current_centers.copy()
                        new_centers[i, 0] = np.clip(current_centers[i, 0] + dx, 0.0, 1.0)
                        new_centers[i, 1] = np.clip(current_centers[i, 1] + dy, 0.0, 1.0)
                        
                        new_radii = compute_radii(new_centers)
                        new_score = np.sum(new_radii)
                        
                        if new_score > current_score + 1e-10:
                            current_centers = new_centers
                            current_radii = new_radii
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                step_scale *= 0.7
                if step_scale < 0.0001:
                    break
        
        return current_centers, current_radii

    # Initializations
    inits = []
    # Grid
    side = int(np.ceil(np.sqrt(N)))
    grid_centers = []
    for i in range(side):
        for j in range(side):
            if len(grid_centers) < N:
                grid_centers.append([(i + 0.5) / side, (j + 0.5) / side])
    inits.append(np.array(grid_centers[:N]))
    
    # Randoms
    for seed in [1, 7, 13, 23, 37, 53, 71, 89, 101, 113]:
        rng = np.random.RandomState(seed)
        inits.append(rng.uniform(0.1, 0.9, size=(N, 2)))

    best_result = None
    best_score = -np.inf

    for init_idx, init_centers in enumerate(inits):
        if time.time() > deadline - 15:
            break
        
        # Phase 1: Physics-based bubble relaxation
        phys_centers, phys_radii = physics_relaxation(init_centers, max_iter=3000)
        
        # Phase 2: Polish with local search
        polished_centers, polished_radii = polish_local_search(phys_centers, phys_radii, max_iter=2000)
        
        # Compute final radii
        final_radii = compute_radii(polished_centers)
        x_full = np.column_stack([polished_centers, final_radii])
        
        # Ensure validity
        x_valid = fix_validity(x_full)
        
        score = np.sum(x_valid[:, 2])
        if score > best_score:
            best_score = score
            best_result = x_valid.copy()

    # Fallback
    if best_result is None:
        centers = inits[0]
        radii = compute_radii(centers)
        best_result = np.column_stack([centers, radii])
        best_result = fix_validity(best_result)

    return best_result

# EVOLVE-BLOCK-END
