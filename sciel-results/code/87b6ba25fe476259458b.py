"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Initialization: Hexagonal Grid
    # We want to initialize 200 points in a hexagonal pattern.
    # Approximate radius for 200 circles in unit square:
    # Area per circle ~ (2r)^2 / (2/sqrt(3)) = 2*sqrt(3)*r^2
    # 200 * 2*sqrt(3)*r^2 = 1 => r^2 = 1/(400*sqrt(3)) => r ~ 0.036
    # Let's start with a slightly smaller radius to ensure they fit.
    
    def generate_hex_grid(target_n, square_size=1.0):
        # Estimate spacing
        # Hex packing density is pi/sqrt(12) ~ 0.9069
        # Area per circle = square_size^2 / target_n
        # d^2 * sqrt(3)/2 = Area_per_circle => d = sqrt(2 * Area / sqrt(3))
        area_per_circle = (square_size * square_size) / target_n
        d = np.sqrt(2 * area_per_circle / np.sqrt(3))
        
        # Generate points
        points = []
        # Range of x and y to cover the square [-0.5, 0.5]
        # Hex grid rows are offset
        row_height = d * np.sqrt(3) / 2
        
        # Start y from -0.5 - margin to 0.5 + margin
        y = -0.5 - row_height
        row_idx = 0
        while y <= 0.5 + row_height:
            x_offset = 0 if row_idx % 2 == 0 else d / 2
            x = -0.5 - d + x_offset
            while x <= 0.5 + d:
                # Check if point is inside or near the square
                # We will filter later, but generating a bit extra helps
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append((x, y))
                elif -0.5 - d <= x <= 0.5 + d and -0.5 - d <= y <= 0.5 + d:
                    # Keep some outside points? No, we want exactly N inside.
                    # Better to generate a dense grid and pick the best N or just clip.
                    # For simplicity, let's generate a grid slightly larger and take the first N that are inside.
                    # Actually, simpler: generate grid, keep those inside, if < N, reduce d.
                    pass
                x += d
            y += row_height
            row_idx += 1
        
        # Convert to array
        if len(points) < target_n:
            # If we don't have enough, reduce d and retry
            return generate_hex_grid(target_n, square_size)
            
        points = np.array(points[:target_n])
        return points

    # The above recursive approach might be fragile. Let's do a direct generation.
    def generate_hex_grid_v2(target_n):
        # Start with an estimated d
        # r ~ 0.036 => d ~ 2*r/sqrt(3) ~ 0.041
        # Let's try d = 0.04
        d = 0.04
        while True:
            points = []
            row_height = d * np.sqrt(3) / 2
            y = -0.5
            row_idx = 0
            while y <= 0.5:
                x_offset = 0 if row_idx % 2 == 0 else d / 2
                x = -0.5 + x_offset
                while x <= 0.5:
                    points.append((x, y))
                    x += d
                y += row_height
                row_idx += 1
            
            if len(points) >= target_n:
                # Select the first target_n points
                # To make it more symmetric, we could select the central ones, 
                # but for local search, any valid start is fine.
                # However, picking random ones or the first ones might bias.
                # Let's pick the ones closest to the center to ensure they are "inside" 
                # and not on the very edge if possible, though edge is fine.
                # Actually, just taking the first ones is deterministic and fine.
                return np.array(points[:target_n])
            else:
                # Need more points, decrease d
                d *= 0.95

    centers = generate_hex_grid_v2(N)
    
    # Ensure they are strictly inside? The score function handles boundaries.
    # "A centre on or outside the square ... makes the candidate invalid."
    # So we must ensure -0.5 < c_i < 0.5.
    # The hex grid generation above uses -0.5 <= x <= 0.5.
    # We should clamp them slightly inward.
    centers = np.clip(centers, -0.499, 0.499)
    
    # Helper to compute the radius
    def compute_radius(centers):
        # Min distance to boundary
        min_boundary = np.min(np.minimum(0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1])))
        
        # Min distance between pairs
        # Use broadcasting for N=200 (200x200 matrix is fine)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        r = min(min_boundary, min_pair_dist / 2.0)
        return r

    current_r = compute_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    # Multi-scale optimization
    # Start with a target radius slightly below current and increase
    # Or start with a target radius based on the theoretical max and decrease?
    # The "push-apart" strategy works by setting a target separation.
    
    # Let's define a schedule of target radii.
    # Start from current_r * 1.1 and go up? 
    # Or start from a small value and go up to current_r?
    # Since current_r is from hex grid, it's already good.
    # We want to improve it.
    # Let's try to push towards a higher radius.
    
    # We will iterate over a range of target radii.
    # The algorithm:
    # For target_r in schedule:
    #   For step in range(max_steps):
    #     Move points to satisfy dist >= 2*target_r and inside box.
    #   If successful (min_r >= target_r), update best.
    
    # Schedule: linearly interpolate from current_r to current_r * 1.2
    # But we don't know if 1.2 is achievable.
    # Let's just run a long local search with an increasing target.
    
    max_iterations = 5000
    # We will adjust the target radius dynamically.
    # Start with target = current_r * 1.05
    # If we can achieve it, increase target.
    # If we fail (get stuck), keep the best found.
    
    target_r = current_r * 1.05
    step_size = 0.005
    
    # We need to be careful with the time.
    # Each iteration involves N^2 distance calculation.
    # N=200 -> 40,000 operations.
    # 5000 iterations -> 200M operations.
    # NumPy is fast, this should be well under 1 second per 1000 iterations?
    # Let's test with a smaller number first to be safe, then scale up if time permits.
    
    # Actually, let's use a more efficient update.
    # Only update points that are violating constraints?
    # For N=200, full matrix is fine.
    
    # Let's run the optimization loop
    # We will use a simple gradient-like step.
    
    # To handle the "push-apart", we calculate the force on each point.
    # Force from pair i,j if dist < 2*target_r:
    #   direction = (c_i - c_j) / dist
    #   magnitude = (2*target_r - dist)
    #   F_i += direction * magnitude
    # Force from boundary:
    #   if x < -0.5: F_x += (-0.5 - x)
    #   if x > 0.5: F_x += (0.5 - x)
    #   (same for y)
    
    # Update: c_i += step_size * F_i
    
    # We will run this for a fixed number of iterations, but check time.
    
    # Let's structure the loop to allow multiple "passes" with increasing target_r
    passes = 20
    for p in range(passes):
        if time.time() > deadline - 5.0:
            break
            
        # Increase target radius slightly in each pass
        # Start from best_r + small increment
        # Actually, let's just use the current best_r as the base for the target
        # and try to push it higher.
        
        # A better strategy: 
        # 1. Start with target_r = best_r * 1.01
        # 2. Run local search to satisfy this target.
        # 3. If satisfied, best_r = target_r, increase target_r.
        # 4. If not satisfied after max_steps, break or reduce target_r?
        
        # Let's try a simpler approach: 
        # Just run a long local search where the target radius is fixed at a value 
        # slightly above the current best, and see if we can reach it.
        
        # Since we don't know the optimal, let's try to optimize the radius directly.
        # Objective: maximize r.
        # Constraint: dist(i,j) >= 2r, boundary.
        
        # Let's use the "push-apart" with a dynamic target.
        # Start target at current_r.
        # In each iteration, we push to satisfy dist >= 2*target_r.
        # If the current min_dist/2 > target_r, we can increase target_r.
        
        # This is essentially what the code below does.
        
        # Reset step size for this pass? Or keep it?
        # Let's keep a constant step size for simplicity, but maybe decay it.
        
        # Let's do 1000 iterations per pass.
        iters_per_pass = 1000
        
        for i in range(iters_per_pass):
            if time.time() > deadline - 5.0:
                break
            
            # Compute forces
            forces = np.zeros_like(centers)
            
            # Pairwise forces
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=-1))
            np.fill_diagonal(dist, 1.0) # Avoid div by zero
            
            # Mask for violations
            # We want to push apart if dist < 2 * target_r
            # Force magnitude is proportional to violation
            # Let's use a soft constraint: F = (2*target_r - dist) if dist < 2*target_r else 0
            # To make it smooth, we can just use the difference.
            
            # Vectorized calculation of forces
            # This is tricky to do fully vectorized for the sum of forces.
            # Let's loop over i for now? N=200, 200 loops is fine.
            
            # Actually, let's just do the N^2 loop in Python? No, too slow.
            # Let's use NumPy to accumulate forces.
            
            # dist is NxN.
            # We need to sum (diff_ij / dist_ij) * max(0, 2*target_r - dist_ij) for all j != i.
            
            # Let's create a matrix of "push magnitudes"
            # push_mag[i,j] = max(0, 2*target_r - dist[i,j])
            # Force on i from j = (c_i - c_j) / dist[i,j] * push_mag[i,j]
            
            # Note: (c_i - c_j) is diff[i,j].
            # So Force_ij = diff[i,j] / dist[i,j] * push_mag[i,j]
            
            # We need to sum over j for each i.
            
            # Set diagonal to 0 to avoid self-interaction
            dist_safe = np.where(dist > 1e-9, dist, 1.0)
            
            # Calculate push magnitudes
            # Only apply if dist < 2*target_r
            push_mag = np.where(dist < 2*target_r, 2*target_r - dist, 0.0)
            
            # Force contribution matrix
            # F_ij = diff[i,j] * (push_mag[i,j] / dist_safe[i,j])
            # Note: if dist is 0, we have a problem. But we initialized with grid, so dist > 0.
            # If they coincide, we need to break symmetry.
            
            # To handle coincident points, add small random noise if dist is very small.
            # But for now, assume they don't coincide.
            
            force_matrix = diff * (push_mag / dist_safe)[:, :, np.newaxis]
            
            # Sum over j (axis 1) to get net force on i
            forces += np.sum(force_matrix, axis=1)
            
            # Boundary forces
            # x < -0.5: push right. x > 0.5: push left.
            # Let's use a strong penalty.
            x = centers[:, 0]
            y = centers[:, 1]
            
            # Soft boundary: if outside, push in.
            # If inside but close to boundary, we might not want to push unless necessary.
            # The score function penalizes going outside.
            # So we should only push if outside?
            # Or push if dist_to_boundary < some margin?
            # Let's push if outside.
            
            mask_x_min = x < -0.5
            mask_x_max = x > 0.5
            mask_y_min = y < -0.5
            mask_y_max = y > 0.5
            
            # Apply boundary forces
            # Force = (boundary - pos) * strength
            # Let's use strength 1.0
            forces[mask_x_min, 0] += (-0.5 - x[mask_x_min])
            forces[mask_x_max, 0] += (0.5 - x[mask_x_max])
            forces[mask_y_min, 1] += (-0.5 - y[mask_y_min])
            forces[mask_y_max, 1] += (0.5 - y[mask_y_max])
            
            # Update positions
            centers += step_size * forces
            
            # Clamp to square to ensure validity
            centers = np.clip(centers, -0.5, 0.5)
            
            # Check current radius
            # To save time, check every 10 iterations
            if i % 10 == 0:
                current_r = compute_radius(centers)
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
                    
                    # If we achieved the target, increase it
                    if current_r >= target_r:
                        target_r *= 1.01
                        # Maybe reduce step size as we get closer?
                        # step_size *= 0.99
                        pass
                    else:
                        # If we are stuck, maybe decrease target slightly to escape?
                        # Or just keep trying.
                        pass
                        
        # End of pass
        
        # If we didn't improve, break?
        # Let's just continue if time permits.
        
    # Final Polish
    # Use the best centers found
    centers = best_centers.copy()
    
    # Run a few more iterations with a very small step size and target = best_r
    target_r = best_r
    step_size = 0.001
    max_polish_iters = 1000
    
    for i in range(max_polish_iters):
        if time.time() > deadline - 2.0:
            break
            
        forces = np.zeros_like(centers)
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dist, 1.0)
        
        dist_safe = np.where(dist > 1e-9, dist, 1.0)
        
        # Push to maintain best_r
        push_mag = np.where(dist < 2*target_r, 2*target_r - dist, 0.0)
        force_matrix = diff * (push_mag / dist_safe)[:, :, np.newaxis]
        forces += np.sum(force_matrix, axis=1)
        
        x = centers[:, 0]
        y = centers[:, 1]
        
        mask_x_min = x < -0.5
        mask_x_max = x > 0.5
        mask_y_min = y < -0.5
        mask_y_max = y > 0.5
        
        forces[mask_x_min, 0] += (-0.5 - x[mask_x_min])
        forces[mask_x_max, 0] += (0.5 - x[mask_x_max])
        forces[mask_y_min, 1] += (-0.5 - y[mask_y_min])
        forces[mask_y_max, 1] += (0.5 - y[mask_y_max])
        
        centers += step_size * forces
        centers = np.clip(centers, -0.5, 0.5)
        
        if i % 10 == 0:
            current_r = compute_radius(centers)
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
            else:
                # Revert to best if it got worse?
                # The push-apart should generally improve or maintain.
                # But clamping might hurt.
                pass

    return best_centers
# EVOLVE-BLOCK-END
