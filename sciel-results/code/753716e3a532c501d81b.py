import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    rng = np.random.default_rng(42)
    
    def vectorized_forces(positions, radii, dt, damping):
        """
        Calculate and apply forces.
        Returns updated positions and radii.
        """
        n = len(positions)
        # 1. Pairwise repulsion (inverse square)
        # positions: (N, 2)
        dx = positions[:, 0:1] - positions[0:1, 0]  # (N, N)
        dy = positions[:, 1:2] - positions[0:1, 1]  # (N, N)
        dist_sq = dx**2 + dy**2
        np.fill_diagonal(dist_sq, 1.0) # Avoid div by 0
        
        dist = np.sqrt(dist_sq)
        
        # Force magnitude: k / d^2
        # We want them to repel strongly when close.
        # Ideal separation is roughly r_i + r_j.
        # Let's use a simple repulsive force: F = k * (1/d^2) * (r_i + r_j)
        # Actually, standard force-directed graph uses 1/d. Let's use 1/d^2 for stronger short-range repulsion.
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        
        # Calculate repulsive forces
        # F_ij = k * (r_i + r_j) / d^2 * (dx/d, dy/d)
        # Vectorized:
        r_sum = radii[:, 0:1] + radii[0:1, :]  # (N, N)
        
        # Avoid division by zero in force calculation
        safe_dist_sq = np.where(dist_sq > 1e-12, dist_sq, 1.0)
        
        # Force vector components
        # F_x[i,j] = k * (r_i + r_j) * dx[i,j] / dist_sq[i,j]
        # Note: dx[i,j] is x_i - x_j. So force on i is in direction (x_i - x_j).
        k_repel = 10.0
        
        fx = k_repel * r_sum * dx / safe_dist_sq
        fy = k_repel * r_sum * dy / safe_dist_sq
        
        # Zero out self forces
        fx[~mask] = 0
        fy[~mask] = 0
        
        # Sum forces on each particle
        total_fx = np.sum(fx, axis=1)
        total_fy = np.sum(fy, axis=1)
        
        # 2. Boundary forces (springs)
        # Push inward if too close to boundary
        # Boundary condition: x >= r, x <= 1-r, etc.
        # Let's add a spring force proportional to violation, or just a constant push if close.
        # Better: Force = -k_bound * (r - min_dist) if r > min_dist, else 0.
        # But we also want them to stay away from walls even if not violating, to allow radius growth?
        # Actually, if we maximize radius later, we just need valid positions.
        # Let's use a soft repulsion from walls:
        # F_x_wall = -k_wall * (r - x) if x < r, else 0. (Pushes right if x < r)
        # F_x_wall = -k_wall * (r - (1-x)) if x > 1-r, else 0. (Pushes left if x > 1-r)
        
        k_wall = 50.0
        
        # Left wall: x < r -> push right (+x)
        viol_left = np.maximum(0, radii - positions[:, 0])
        fx_wall = k_wall * viol_left
        
        # Right wall: x > 1-r -> push left (-x)
        viol_right = np.maximum(0, positions[:, 0] - (1 - radii))
        fx_wall -= k_wall * viol_right
        
        # Bottom wall: y < r -> push up (+y)
        viol_bottom = np.maximum(0, radii - positions[:, 1])
        fy_wall = k_wall * viol_bottom
        
        # Top wall: y > 1-r -> push down (-y)
        viol_top = np.maximum(0, positions[:, 1] - (1 - radii))
        fy_wall -= k_wall * viol_top
        
        total_fx += fx_wall
        total_fy += fy_wall
        
        # 3. Update velocities and positions (simple Euler with damping)
        # We don't store velocity explicitly in this simplified version, we just move by force * dt.
        # This is like a heavy particle with high friction.
        
        new_x = positions[:, 0] + total_fx * dt
        new_y = positions[:, 1] + total_fy * dt
        
        # Keep positions strictly inside [0,1] to avoid numerical issues, even if forces push out.
        # The radius constraint will be handled by the "polish" step or by clamping radius.
        # For the simulation, let's clamp positions to [0.01, 0.99] to keep them in the square.
        new_x = np.clip(new_x, 0.01, 0.99)
        new_y = np.clip(new_y, 0.01, 0.99)
        
        return new_x, new_y

    def polish_radii(positions, max_iter=50):
        """
        Given fixed positions, maximize radii such that circles are disjoint and inside square.
        This is a system of constraints:
        r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
        r_i + r_j <= dist(i, j)
        
        We can solve this by iterative relaxation.
        Initialize r_i to boundary limit.
        Then iterate: for each pair, if r_i + r_j > d_ij, reduce the larger one (or both) to satisfy constraint.
        """
        n = len(positions)
        # Initial radii: limited by boundary
        min_dist_to_boundary = np.minimum(np.minimum(positions[:, 0], 1 - positions[:, 0]), 
                                          np.minimum(positions[:, 1], 1 - positions[:, 1]))
        radii = min_dist_to_boundary.copy()
        
        # Precompute pairwise distances
        dx = positions[:, 0:1] - positions[0:1, 0]
        dy = positions[:, 1:2] - positions[0:1, 1]
        dists = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dists, 1e9) # No self constraint
        
        for _ in range(max_iter):
            changed = False
            # Iterate over pairs
            for i in range(n):
                for j in range(i + 1, n):
                    d = dists[i, j]
                    current_sum = radii[i] + radii[j]
                    if current_sum > d:
                        # Need to reduce sum by (current_sum - d)
                        # Reduce the one with more "room" or just split?
                        # To maximize sum, we should reduce the one that is less constrained by other factors?
                        # Simple heuristic: reduce the larger radius, or split proportionally.
                        # Let's split the violation equally if possible, respecting non-negativity.
                        excess = current_sum - d
                        # We can reduce r_i by up to r_i, r_j by up to r_j.
                        # Let's try to keep them as balanced as possible relative to their current sizes?
                        # Or just: new_r_i = max(0, r_i - excess/2), new_r_j = max(0, r_j - excess/2)
                        # If one becomes 0, the other takes the rest.
                        
                        # Better: Reduce the one with larger radius? No, that hurts sum less if the small one is 0?
                        # Actually, to maximize sum(r), any reduction is bad. We must reduce by exactly `excess`.
                        # The choice of who reduces doesn't change the sum, unless one hits 0.
                        # If one hits 0, we might be able to reduce the other less?
                        # No, r_i + r_j = d is the tight constraint.
                        # So sum is fixed to d for this pair.
                        # The distribution matters for other constraints.
                        # Let's reduce the one with the larger radius to keep the smaller one alive?
                        # Or reduce proportionally to current radii?
                        
                        # Let's reduce the one with the larger radius, as it likely has more "slack" from other constraints?
                        # Actually, let's just split the excess equally, but ensure no negative radii.
                        
                        half_excess = excess / 2.0
                        new_ri = radii[i] - half_excess
                        new_rj = radii[j] - half_excess
                        
                        if new_ri < 0:
                            new_ri = 0.0
                            new_rj = d # Since r_i=0, r_j can be d? No, r_j <= d.
                            # Wait, if r_i=0, r_j can be up to d.
                            # But we started with sum > d.
                            # If we set r_i=0, we need r_j <= d.
                            # Current r_j was such that r_i+r_j > d.
                            # If we set r_i=0, r_j_new = min(r_j, d).
                            # But we also need to satisfy the boundary for j.
                            # This simple relaxation might violate boundary if we increase? No, we only decrease.
                            # So if we decrease, we are safe from boundary (since we started valid).
                            
                            # If new_ri < 0, it means r_i < excess/2.
                            # So we can only reduce r_i by r_i.
                            # Remaining excess = excess - r_i.
                            # Reduce r_j by remaining excess.
                            new_ri = 0.0
                            new_rj = radii[j] - (excess - radii[i])
                            if new_rj < 0:
                                # This should not happen if d >= 0 and we started with valid boundary radii?
                                # If d is very small, and both radii were large...
                                # But radii were limited by boundary.
                                # If d < r_i + r_j, and we force sum to d, we might hit 0.
                                new_rj = 0.0
                                # If both 0, sum=0 <= d. Valid.
                        
                        elif new_rj < 0:
                            new_rj = 0.0
                            new_ri = radii[i] - (excess - radii[j])
                            if new_ri < 0:
                                new_ri = 0.0
                                new_rj = 0.0
                        
                        if new_ri != radii[i] or new_rj != radii[j]:
                            radii[i] = new_ri
                            radii[j] = new_rj
                            changed = True
            if not changed:
                break
                
        # Final check: ensure boundary constraints are met (should be, since we only decreased)
        # And ensure non-negative
        radii = np.maximum(radii, 0)
        
        return radii

    # Initial setup
    # Start with a grid to ensure good initial distribution
    side = int(np.ceil(np.sqrt(N)))
    positions = np.zeros((N, 2))
    for i in range(N):
        row = i // side
        col = i % side
        positions[i, 0] = (col + 0.5) / side
        positions[i, 1] = (row + 0.5) / side
    
    # Initial radii (small)
    radii = np.full(N, 0.05)
    
    best_score = -1
    best_positions = positions.copy()
    best_radii = radii.copy()
    
    # Simulation parameters
    dt = 0.01
    damping = 0.9
    n_steps = 2000
    
    # Run simulation
    for step in range(n_steps):
        if time.time() > deadline:
            break
            
        positions[:, 0], positions[:, 1] = vectorized_forces(positions, radii, dt, damping)
        
        # Periodically polish radii to update the "size" of circles for the force calculation
        # The force calculation uses radii to determine repulsion strength.
        # If radii are too small, circles can get too close.
        # If we update radii to their max possible, the repulsion will be stronger where they are close.
        if step % 100 == 0:
            current_radii = polish_radii(positions, max_iter=20)
            # Use a fraction of the polished radii for force calculation to avoid instability?
            # Or use full? Let's use full.
            radii = current_radii
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_positions = positions.copy()
                best_radii = radii.copy()
                
    # Final polish with more iterations
    if time.time() < deadline:
        final_radii = polish_radii(best_positions, max_iter=100)
        final_score = np.sum(final_radii)
        
        # Verify validity of final result
        # Check boundaries
        valid = True
        for i in range(N):
            x, y, r = best_positions[i, 0], best_positions[i, 1], final_radii[i]
            if x < r - 1e-9 or x > 1 - r + 1e-9 or y < r - 1e-9 or y > 1 - r + 1e-9:
                valid = False
                break
        
        if valid:
            # Check overlaps
            for i in range(N):
                for j in range(i + 1, N):
                    dx = best_positions[i, 0] - best_positions[j, 0]
                    dy = best_positions[i, 1] - best_positions[j, 1]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < final_radii[i] + final_radii[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            best_radii = final_radii
            best_score = final_score
        else:
            # If invalid, fall back to the last known valid state from the loop
            # Or try to fix it by shrinking?
            # Let's try to fix it by shrinking the problematic circles slightly.
            # Simple fix: scale all radii by 0.99
            best_radii = best_radii * 0.99
            # Re-check? Assuming 0.99 scale makes it valid if it was close.
            
    # Construct output
    result = np.zeros((N, 3))
    result[:, 0] = best_positions[:, 0]
    result[:, 1] = best_positions[:, 1]
    result[:, 2] = best_radii
    
    return result
