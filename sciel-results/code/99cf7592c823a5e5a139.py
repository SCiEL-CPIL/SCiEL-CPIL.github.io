import numpy as np
from scipy.spatial import KDTree
import time
import sys

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def time_left():
        return deadline - time.time()

    def compute_radius(centers):
        """Compute the maximum radius for given centers."""
        centers = np.asarray(centers)
        if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
            return 0.0
        # Boundary distance: min over all points of min distance to boundary
        # Boundary is at ±0.5
        bdist = 0.5 - np.abs(centers).max(axis=1)
        if bdist.min() < 0:
            return 0.0
        # Pairwise distances
        # Use KDTree for efficiency
        tree = KDTree(centers)
        dists, _ = tree.query(centers, k=2)
        min_pair_dist = dists[:, 1].min()
        return min(bdist.min(), min_pair_dist / 2.0)

    def softmin_neg_radius(centers, beta=500.0):
        """
        Smooth surrogate: -softmin of the radius constraints.
        We want to maximize r = min(bdist, min_pair_dist/2).
        Equivalent to minimizing -r = -min(bdist, min_pair_dist/2).
        Use log-sum-exp for soft-min.
        """
        centers = np.asarray(centers)
        bdist = 0.5 - np.abs(centers).max(axis=1)
        
        # Pairwise distances via KDTree
        tree = KDTree(centers)
        dists, _ = tree.query(centers, k=2)
        min_pair_dists = dists[:, 1] / 2.0
        
        # Combine all constraints into a single array
        # Constraints: bdist (N values) and min_pair_dists (N values)
        # r = min(all these values)
        # -r = -min(all) = softmin_neg(all, beta)
        
        all_constraints = np.concatenate([bdist, min_pair_dists])
        
        # Soft-min: min(x) ≈ -(1/beta) * log(sum(exp(-beta * x)))
        # To avoid overflow, subtract max
        x_min = all_constraints.min()
        # If x_min is very negative, we might have issues, but constraints should be positive-ish
        # Actually, if points overlap or go outside, constraints become negative.
        # We want to penalize this.
        
        # Stable log-sum-exp
        # Let y = -beta * (all_constraints - x_min)
        # Then softmin = x_min - (1/beta) * log(sum(exp(y)))
        
        y = -beta * (all_constraints - x_min)
        # Clip y to avoid overflow in exp
        y = np.clip(y, -100, 100)
        
        log_sum_exp = np.log(np.sum(np.exp(y)))
        soft_min_val = x_min - (1.0 / beta) * log_sum_exp
        
        return -soft_min_val

    def objective_and_grad(params, beta=500.0):
        """
        Objective: -softmin(r) + penalty for hard violations (if any)
        Actually, let's just use the softmin. It handles violations by making the value large (bad).
        Wait, if bdist < 0, then x_min < 0, soft_min_val < 0, -soft_min_val > 0.
        We are minimizing this. So large positive values are bad.
        This is correct.
        """
        centers = params.reshape(N, 2)
        val = softmin_neg_radius(centers, beta)
        
        # Gradient approximation? No, we need analytic or numerical.
        # Let's use numerical gradient for simplicity first, or analytic if easy.
        # Analytic gradient of softmin is complex.
        # Let's use L-BFGS-B with numerical gradient? Too slow for 400 vars.
        # Let's use a simple gradient descent with finite differences?
        # Or use scipy.optimize.minimize with method='L-BFGS-B' and finite diff?
        # 400 dimensions, finite diff takes 400 evaluations per gradient.
        # Each evaluation is O(N log N) for KDTree.
        # 200 * log(200) ~ 1400 ops.
        # 400 * 1400 = 560k ops per gradient step.
        # 1000 steps = 560M ops. Might be too slow in Python.
        
        # Alternative: Use a force-based method which is cheaper?
        # Or use the "bottleneck elimination" idea from inspiration 3.
        
        # Let's try a hybrid:
        # 1. Use force-based refinement (fast, local).
        # 2. Periodically re-evaluate and adjust.
        
        return val

    def force_refine(centers, steps=500, lr=0.01, beta=500.0):
        """
        Refine centers using force-based method.
        Forces are derived from the gradient of the soft-min objective.
        Or simpler: push apart close pairs, push inwards from walls.
        """
        centers = centers.copy()
        r_current = compute_radius(centers)
        
        for step in range(steps):
            if time_left() < 5.0:
                break
            
            # Compute forces
            forces = np.zeros_like(centers)
            
            # 1. Wall forces
            # Push points away from walls if they are too close
            # Use a smooth potential: V_wall = (r_target - dist_to_wall)^2 if dist < r_target
            # But we don't know r_target exactly.
            # Let's use the current estimated r.
            # Actually, let's just push if dist_to_wall < current_r * 0.9
            margin = r_current * 0.95
            # x walls
            for i in range(N):
                # Left wall
                d_left = centers[i, 0] + 0.5
                if d_left < margin:
                    forces[i, 0] += (margin - d_left)
                # Right wall
                d_right = 0.5 - centers[i, 0]
                if d_right < margin:
                    forces[i, 0] -= (margin - d_right)
                # Bottom wall
                d_bottom = centers[i, 1] + 0.5
                if d_bottom < margin:
                    forces[i, 1] += (margin - d_bottom)
                # Top wall
                d_top = 0.5 - centers[i, 1]
                if d_top < margin:
                    forces[i, 1] -= (margin - d_top)
            
            # 2. Pair forces
            # Use KDTree to find close neighbors
            tree = KDTree(centers)
            # Query for neighbors within a certain distance
            # We want to repel if dist < 2 * r_current
            query_dist = 2.0 * r_current * 1.1
            for i in range(N):
                neighbors = tree.query_ball_point(centers[i], query_dist)
                for j in neighbors:
                    if i == j:
                        continue
                    d = centers[i] - centers[j]
                    dist = np.linalg.norm(d)
                    if dist < 1e-8:
                        # Random nudge
                        d = np.random.randn(2)
                        dist = np.linalg.norm(d)
                    
                    # Repulsion force
                    # Stronger if closer than 2*r
                    if dist < 2.0 * r_current:
                        # Overlap
                        mag = (2.0 * r_current - dist) / dist
                    else:
                        # Close but not overlapping
                        # Weak repulsion to pack tightly
                        # Use a smooth falloff
                        excess = dist - 2.0 * r_current
                        if excess < query_dist - 2.0 * r_current:
                            mag = 0.1 * (query_dist - dist) / dist
                        else:
                            mag = 0
                    
                    forces[i] += d * mag
            
            # Update centers
            centers += lr * forces
            
            # Optional: normalize forces to keep steps small
            # Or adapt lr
            
            # Update r_current
            r_current = compute_radius(centers)
            
            # Occasionally, do a global check or reset?
            
        return centers

    def simulated_annealing(centers, steps=1000, t_start=0.01, t_end=0.001):
        """
        Simulated annealing to escape local minima.
        """
        centers = centers.copy()
        r_current = compute_radius(centers)
        
        for step in range(steps):
            if time_left() < 5.0:
                break
            
            # Temperature
            t = t_start * (t_end / t_start) ** (step / steps)
            
            # Generate a random move
            # Move one or a few points randomly
            n_move = np.random.randint(1, 5)
            idxs = np.random.choice(N, n_move, replace=False)
            
            # Random displacement
            scale = 0.02 * (t / t_start) + 0.001
            new_centers = centers.copy()
            for i in idxs:
                new_centers[i] += np.random.randn(2) * scale
            
            # Ensure new centers are within bounds?
            # Let the radius computation handle it (will be 0 if invalid)
            
            r_new = compute_radius(new_centers)
            
            # Accept if better, or with probability exp((r_new - r_current) / t)
            if r_new > r_current:
                centers = new_centers
                r_current = r_new
            else:
                delta = r_new - r_current
                # Avoid division by zero or negative t
                if t > 1e-9:
                    prob = np.exp(delta / t)
                else:
                    prob = 0
                if np.random.random() < prob:
                    centers = new_centers
                    r_current = r_new
        
        return centers

    def hexagonal_initialization():
        """
        Generate a hexagonal lattice packing.
        """
        # Hex grid spacing
        # For N=200, we need a grid that fits ~200 points.
        # Hex grid: rows of alternating offsets.
        # Let's generate a large hex grid and pick the best 200.
        
        # Approximate radius r ~ 0.0366
        # Diameter d = 0.0732
        # Hex spacing: dx = d, dy = d * sqrt(3) / 2
        
        # Let's generate points in [-0.5, 0.5]^2
        # Number of columns: ~ 1 / dx = 13.6 -> 14
        # Number of rows: ~ 1 / dy = 15.8 -> 16
        
        points = []
        dx = 0.074
        dy = dx * np.sqrt(3) / 2
        
        # Center the grid
        n_cols = 14
        n_rows = 16
        
        # Generate grid
        for row in range(n_rows):
            for col in range(n_cols):
                x = (col - n_cols / 2.0 + 0.5) * dx
                y = (row - n_rows / 2.0 + 0.5) * dy
                if row % 2 == 1:
                    x += dx / 2.0
                
                # Check if within bounds
                if -0.5 < x < 0.5 and -0.5 < y < 0.5:
                    points.append([x, y])
        
        points = np.array(points)
        
        # If we have more than N, pick N randomly or by best fit?
        # If less, add random points?
        if len(points) > N:
            # Pick N points that are well-spread
            # Simple: random selection
            np.random.seed(42)
            idx = np.random.choice(len(points), N, replace=False)
            points = points[idx]
        elif len(points) < N:
            # Add random points
            missing = N - len(points)
            rand_points = np.random.uniform(-0.5, 0.5, size=(missing, 2))
            points = np.vstack([points, rand_points])
        
        return points

    def square_grid_initialization():
        """
        Generate a square grid packing.
        """
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        
        if len(grid) > N:
            np.random.seed(42)
            idx = np.random.choice(len(grid), N, replace=False)
            grid = grid[idx]
        elif len(grid) < N:
            missing = N - len(grid)
            rand_points = np.random.uniform(-0.5, 0.5, size=(missing, 2))
            grid = np.vstack([grid, rand_points])
        
        return grid

    def random_initialization():
        """
        Generate random points.
        """
        return np.random.uniform(-0.5, 0.5, size=(N, 2))

    # Main optimization loop
    best_centers = None
    best_r = 0.0
    
    # Try different initializations
    inits = [
        hexagonal_initialization,
        square_grid_initialization,
        random_initialization
    ]
    
    # We have ~145s.
    # Let's do multiple restarts.
    # Each restart: init -> force_refine -> simulated_annealing -> force_refine
    
    # Estimate time per restart
    # Force refine: 500 steps * (N log N) ~ 500 * 1400 = 700k ops
    # Simulated annealing: 1000 steps * (N log N) ~ 1.4M ops
    # Total ~ 2M ops per restart.
    # In Python, maybe 0.1s per restart?
    # So we can do ~1000 restarts?
    # Let's be conservative: 100 restarts.
    
    restarts = 0
    while time_left() > 10.0:
        restarts += 1
        # Cycle through initializations
        init_func = inits[restarts % len(inits)]
        centers = init_func()
        
        # Refine
        centers = force_refine(centers, steps=300, lr=0.01)
        r1 = compute_radius(centers)
        
        # Simulated annealing
        centers = simulated_annealing(centers, steps=500, t_start=0.01, t_end=0.001)
        r2 = compute_radius(centers)
        
        # Final refine
        centers = force_refine(centers, steps=200, lr=0.005)
        r3 = compute_radius(centers)
        
        if r3 > best_r:
            best_r = r3
            best_centers = centers.copy()
        
        # Print progress
        if restarts % 10 == 0:
            print(f"Restart {restarts}, Best R: {best_r:.10f}, Time Left: {time_left():.1f}s")
    
    # Final polish: use scipy.optimize.minimize with L-BFGS-B on the softmin objective
    # This might help squeeze out the last bit of precision.
    if time_left() > 10.0 and best_centers is not None:
        try:
            from scipy.optimize import minimize
            
            def obj(params):
                c = params.reshape(N, 2)
                return softmin_neg_radius(c, beta=1000.0)
            
            # Bounds
            bounds = [(-0.5, 0.5)] * (N * 2)
            
            res = minimize(obj, best_centers.flatten(), method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 100, 'gtol': 1e-6})
            
            r_final = compute_radius(res.x.reshape(N, 2))
            if r_final > best_r:
                best_r = r_final
                best_centers = res.x.reshape(N, 2).copy()
        except Exception as e:
            pass
    
    # Ensure validity
    if best_centers is None:
        # Fallback to a simple grid
        best_centers = square_grid_initialization()
    
    # Final check and correction
    # If any point is outside, clip it
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

# Run construct if called directly? No, the evaluator calls construct().
# But we need to make sure it's importable.

if __name__ == "__main__":
    # For local testing
    centers = construct()
    r = compute_radius(centers)
    print(f"Radius: {r:.10f}")
