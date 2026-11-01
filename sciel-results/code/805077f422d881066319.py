import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # 1. Initialization: Hexagonal lattice
    # We want a dense packing.
    # Start with a hexagonal grid and select the N closest to the origin.
    # Then refine.
    
    def hex_init():
        points = []
        # Generate a hex grid
        # Spacing roughly 2*r. Let's try a few spacings.
        # For N=50, r ~ 0.125, diameter ~ 0.25.
        # Area of unit circle ~ 3.14. Area of 50 circles ~ 50 * pi * 0.125^2 ~ 2.45.
        # Density ~ 0.78.
        # Hex packing density is 0.906.
        # So spacing s = 2*r.
        # Let's just generate a large hex grid and pick the best N.
        
        best_points = None
        best_min_dist = 0
        
        # Try a few grid densities
        for scale in [1.2, 1.3, 1.4, 1.5, 1.6]:
            r_est = 0.12 * scale # rough estimate
            s = 2 * r_est
            dx = s
            dy = s * np.sqrt(3) / 2
            # Range
            range_x = int(1.0 / dx) + 2
            range_y = int(1.0 / dy) + 2
            
            pts = []
            for i in range(-range_x, range_x + 1):
                for j in range(-range_y, range_y + 1):
                    x = i * dx + (j % 2) * dx / 2
                    y = j * dy
                    if np.hypot(x, y) <= 1.0:
                        pts.append([x, y])
            
            if len(pts) < N:
                continue
                
            pts = np.array(pts)
            # Sort by distance to origin to keep central ones
            dists = np.linalg.norm(pts, axis=1)
            idx = np.argsort(dists)
            pts = pts[idx[:N]]
            
            # Check validity and min distance
            # Compute min pair dist
            min_pair_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < min_pair_dist:
                        min_pair_dist = d
            
            min_boundary_dist = np.min(1.0 - np.linalg.norm(pts, axis=1))
            r_candidate = min(min_pair_dist / 2.0, min_boundary_dist)
            
            if r_candidate > best_min_dist:
                best_min_dist = r_candidate
                best_points = pts.copy()
                
        return best_points

    points = hex_init()
    if points is None:
        # Fallback random
        points = np.random.uniform(-0.8, 0.8, (N, 2))
        # Ensure inside
        norms = np.linalg.norm(points, axis=1)
        for i in range(N):
            if norms[i] > 0.9:
                points[i] *= 0.9 / norms[i]
                
    current_r = 0.1 # Start low, will grow
    
    # Optimization Loop
    # We will use a simple push-based local search.
    # To escape local minima, we will use simulated annealing-like steps or just multiple restarts if time permits.
    
    def compute_score(pts):
        # Returns min radius
        min_pair = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_pair:
                    min_pair = d
        min_bound = np.min(1.0 - np.linalg.norm(pts, axis=1))
        return min(min_pair / 2.0, min_bound)
    
    best_points = points.copy()
    best_r = compute_score(best_points)
    
    # Local Search
    # We try to increase r.
    # Strategy: 
    # 1. Set target_r = current_r * (1 + epsilon)
    # 2. Relax forces to satisfy constraints for target_r
    # 3. If successful, accept, else reject.
    
    # Simpler: Iterative push with adaptive step size.
    # For each iteration:
    #   Identify violations for current_r.
    #   Push points to reduce violations.
    #   Slightly increase current_r.
    
    # Let's use a more robust approach: 
    # Maximize min distance directly using gradient ascent on a smooth surrogate?
    # Or just use the "push" method which is very effective for circle packing.
    
    def push_relax(pts, r, steps=500, step_size=0.01):
        pts = pts.copy()
        for s in range(steps):
            moved = False
            # Boundary constraints
            for i in range(N):
                dist = np.linalg.norm(pts[i])
                if dist > 1.0 - r:
                    # Push in
                    if dist > 1e-9:
                        pts[i] *= (1.0 - r) / dist
                    moved = True
            
            # Pair constraints
            for i in range(N):
                for j in range(i+1, N):
                    d_vec = pts[i] - pts[j]
                    d = np.linalg.norm(d_vec)
                    min_d = 2.0 * r
                    if d < min_d:
                        if d > 1e-9:
                            # Push apart
                            diff = (min_d - d) / 2.0
                            d_unit = d_vec / d
                            pts[i] += diff * d_unit
                            pts[j] -= diff * d_unit
                        else:
                            # Random kick
                            pts[i] += np.random.uniform(-0.01, 0.01, 2)
                            pts[j] -= np.random.uniform(-0.01, 0.01, 2)
                        moved = True
            
            # If no moved, we might be stuck or converged.
            # We can break early if no movement, but we usually want to keep trying to increase r.
            # However, this function is called with a fixed r.
            # If it converges, it means the configuration is valid for r.
            
        return pts

    # We will try to increase r in small increments.
    # Start with a valid r (computed from init).
    # Then try to increase it.
    
    r = best_r
    # Increase r slowly
    # Each step, try to relax to new r.
    # If relaxation fails (points go outside or overlap significantly), revert?
    # Actually, the push method naturally keeps them inside if we enforce boundary strictly.
    # The issue is if they can't all fit.
    
    # Let's use a loop that tries to increase r.
    # If we can't increase r in a certain number of steps, we try a restart or perturbation.
    
    max_iter = 2000
    iter_count = 0
    step_size_r = 0.0005 # How much to increase r per successful relaxation
    
    while time.time() < deadline and iter_count < max_iter:
        # Try to increase r
        new_r = r + step_size_r
        # Relax to new_r
        # We need to check if it's feasible.
        # The push method might not converge to a valid state if infeasible.
        # We can check the final min distance.
        
        trial_pts = push_relax(points, new_r, steps=200, step_size=0.05)
        trial_r = compute_score(trial_pts)
        
        if trial_r >= new_r - 1e-5: # Feasible
            points = trial_pts
            r = new_r
            if r > best_r:
                best_r = r
                best_points = points.copy()
            iter_count = 0 # Reset counter for progress
        else:
            # Not feasible, try smaller step or perturb
            step_size_r *= 0.9
            # Perturb current points to escape local minimum
            points += np.random.uniform(-0.01, 0.01, (N, 2))
            # Ensure validity after perturbation
            # Re-project to boundary if needed
            for i in range(N):
                dist = np.linalg.norm(points[i])
                if dist > 1.0 - r:
                    if dist > 1e-9:
                        points[i] *= (1.0 - r) / dist
            
            iter_count += 1
            if iter_count > 50:
                # Stuck, try a bigger restart or different strategy
                # Let's try a random restart from a slightly perturbed hex grid
                # Or just break if we've spent too much time.
                # Given the time limit, let's just continue with smaller steps.
                step_size_r = 0.0001
    
    return best_points

# Run construct to verify syntax and basic logic
if __name__ == "__main__":
    pts = construct()
    # Calculate score
    min_pair = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_pair:
                min_pair = d
    min_bound = np.min(1.0 - np.linalg.norm(pts, axis=1))
    score = min(min_pair / 2.0, min_bound)
    print(f"Score: {score}")
