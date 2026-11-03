import numpy as np
import time

N = 26

def construct():
    # Deadline guard: 120 seconds to be safe within the 150s limit
    deadline = time.time() + 120.0
    
    # Initial best result to return in case of early termination or failure
    best_result = None
    best_score = -1.0

    def check_validity(circles):
        """Check if circles are within bounds and non-overlapping."""
        if circles.shape[0] != N:
            return False
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Boundary check
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return False
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return False
        
        # Non-overlap check
        # Vectorized pairwise distance calculation
        # diff[i, j] = p_i - p_j
        diff = circles[:, np.newaxis, :2] - circles[np.newaxis, :, :2]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        min_dist_required = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Check all pairs i < j
        # dist[i, j] >= min_dist_required[i, j]
        # We only need to check upper triangle to avoid self-interference (dist 0)
        for i in range(N):
            for j in range(i + 1, N):
                if dist[i, j] < min_dist_required[i, j] - 1e-9:
                    return False
        return True

    def score(circles):
        return np.sum(circles[:, 2])

    def relax(circles, max_iters=2000, tol=1e-6):
        """
        Force-directed relaxation to resolve overlaps.
        circles: (N, 3) array [x, y, r]
        Returns: (N, 3) array of relaxed positions
        """
        pos = circles[:, :2].copy()
        radii = circles[:, 2].copy()
        
        # Learning rate for position updates
        lr = 0.1 
        
        for it in range(max_iters):
            # Calculate pairwise distances
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=-1))
            
            # Prevent division by zero
            dist[dist < 1e-10] = 1e-10
            
            # Required distance is r_i + r_j
            req_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
            
            # Overlap amount: req_dist - dist
            # We want to push apart if dist < req_dist
            overlap = req_dist - dist
            overlap = np.maximum(overlap, 0)
            
            # Force is proportional to overlap
            # Unit vector from j to i
            # force[i, j] is the force on i due to j
            # We want to move i away from j
            # Vector diff is p_i - p_j
            # So we add (diff / dist) * overlap * strength
            # But we need to sum forces on each particle
            
            # To be efficient, we can compute the sum of forces on each particle
            # Force on i from j: direction (p_i - p_j) / ||p_i - p_j|| * overlap_ij
            # Note: overlap is symmetric, direction is anti-symmetric
            
            # Let's compute displacement for each particle
            disp = np.zeros_like(pos)
            
            # Vectorized force calculation
            # weights = overlap / dist  (shape N, N)
            weights = overlap / dist
            # Zero out diagonal
            np.fill_diagonal(weights, 0)
            
            # Force on i is sum_j ( (p_i - p_j) / dist_ij * overlap_ij )
            # This is equivalent to: sum_j ( (p_i - p_j) * (overlap_ij / dist_ij) )
            # (p_i - p_j) * w_ij = p_i * w_ij - p_j * w_ij
            # Sum_j (p_i * w_ij) = p_i * sum_j(w_ij)
            # Sum_j (p_j * w_ij) = sum_j(p_j * w_ij) -> This is a matrix product
            
            # Let W be the weight matrix (N, N) where W[i, j] = overlap_ij / dist_ij
            # Force_i = sum_j (p_i - p_j) * W[i, j]
            # Force_i = p_i * sum_j(W[i, j]) - sum_j(p_j * W[i, j])
            
            sum_weights = np.sum(weights, axis=1) # Shape (N,)
            # sum_j(p_j * W[i, j]) is (W.T @ pos) ? 
            # W[i, j] * p_j. Sum over j. 
            # Let M = W.T. Then M[j, i] = W[i, j].
            # (M @ pos)[i] = sum_j M[j, i] * pos[j] = sum_j W[i, j] * pos[j]. Yes.
            
            weighted_pos_sum = weights.T @ pos # Shape (N, 2)
            
            forces = pos * sum_weights[:, np.newaxis] - weighted_pos_sum
            
            # Update positions
            pos += lr * forces
            
            # Check convergence
            max_force = np.max(np.linalg.norm(forces, axis=1))
            if max_force < tol:
                break
            
        return np.column_stack((pos, radii))

    def grow_and_relax(circles, dt=0.005, max_grow_steps=1000):
        """
        Attempt to grow radii uniformly and relax positions.
        Returns new circles if successful, else None.
        """
        current_circles = circles.copy()
        for _ in range(max_grow_steps):
            # Try to grow all radii by dt
            new_radii = current_circles[:, 2] + dt
            # Boundary check for growth: r + dt <= min(x, 1-x, y, 1-y)
            # If any circle hits the boundary limit, we can't grow uniformly anymore
            # unless we move them. But relax handles moving.
            # However, if r is already at the max possible for that position, 
            # relax will push it to the wall, but it can't grow further without moving.
            
            test_circles = np.column_stack((current_circles[:, :2], new_radii))
            
            # Relax the positions to accommodate the larger radii
            relaxed_circles = relax(test_circles, max_iters=500, tol=1e-5)
            
            # Verify if the relaxation actually resolved the overlaps
            # We check if the final configuration is valid
            if check_validity(relaxed_circles):
                current_circles = relaxed_circles
            else:
                # Failed to relax, stop growing
                break
                
        return current_circles

    # Main Loop: Multiple restarts with different initializations
    # We will use a grid initialization as a strong baseline, then perturb.
    
    # 1. Grid Initialization
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                # Initial radius: half the cell width, slightly smaller
                r = (0.5 / side) * 0.8
                cells.append([x, y, r])
    
    initial_circles = np.array(cells[:N])
    
    # 2. Optimization Loop
    # We will alternate between growing and shaking to escape local minima
    
    current_best = initial_circles.copy()
    best_val = score(initial_circles)
    
    # First, try to grow from the grid
    grown = grow_and_relax(initial_circles, dt=0.01, max_grow_steps=200)
    if check_validity(grown):
        current_best = grown
        best_val = score(current_best)
        
    # Now, iterative improvement loop
    # We will use a random restart / perturbation strategy
    # If we get stuck (no improvement in X seconds), we perturb.
    
    last_improvement_time = time.time()
    stagnation_limit = 10.0 # seconds
    
    while time.time() < deadline:
        if time.time() - last_improvement_time > stagnation_limit:
            # Perturb the current best to escape local minimum
            # Add small random noise to positions
            perturbed = current_best.copy()
            noise_scale = 0.05 * np.max(perturbed[:, 2])
            perturbed[:, :2] += np.random.normal(0, noise_scale, (N, 2))
            
            # Ensure perturbed circles are inside bounds (shrink if necessary)
            for i in range(N):
                x, y, r = perturbed[i]
                # Clip x, y to [r, 1-r]
                x = np.clip(x, r, 1-r)
                y = np.clip(y, r, 1-r)
                perturbed[i, 0] = x
                perturbed[i, 1] = y
            
            # Relax the perturbed configuration
            relaxed = relax(perturbed, max_iters=2000, tol=1e-5)
            
            if check_validity(relaxed):
                current_best = relaxed
                # Try to grow again from this new local optimum
                grown = grow_and_relax(current_best, dt=0.005, max_grow_steps=100)
                if check_validity(grown) and score(grown) > score(current_best):
                    current_best = grown
                if score(current_best) > best_val:
                    best_val = score(current_best)
                    last_improvement_time = time.time()
            else:
                # If perturbation failed validity, revert to previous best
                pass

        # Try small incremental growth
        # We attempt to grow the current best configuration slightly
        # If it fails, we stop growing this specific configuration
        # and rely on perturbation
        
        # To be more aggressive, we can try to grow individual circles
        # but uniform growth is safer for stability.
        
        # Check if we can grow
        test_radii = current_best[:, 2] + 0.001
        test_circles = np.column_stack((current_best[:, :2], test_radii))
        relaxed_test = relax(test_circles, max_iters=200, tol=1e-5)
        
        if check_validity(relaxed_test):
            current_best = relaxed_test
            new_val = score(current_best)
            if new_val > best_val:
                best_val = new_val
                last_improvement_time = time.time()
        else:
            # Can't grow, maybe we need to shake more frequently
            # Reduce stagnation limit to force more exploration
            stagnation_limit = max(5.0, stagnation_limit * 0.9)

    # Final cleanup: ensure the returned result is strictly valid
    # Sometimes floating point errors can cause borderline failures
    if check_validity(current_best):
        return current_best
    else:
        # Fallback to a known valid grid if something went wrong
        # This shouldn't happen if we start with valid and only accept valid relaxations
        return initial_circles

# EVOLVE-BLOCK-END
