import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Uses Simulated Annealing with Hexagonal Lattice Initialization
    to find a near-optimal packing of N circles in a unit circle.
    """
    
    def hexagonal_init(n_points):
        """
        Initialize points in a hexagonal lattice pattern scaled to fit in the unit circle.
        """
        points = []
        # Estimate grid size
        # Area of unit circle is pi. Area of hex cell with side s is (3*sqrt(3)/2)*s^2.
        # Number of cells ~ pi / ((3*sqrt(3)/2)*s^2)
        # We want ~ n_points.
        # Let's just generate a large grid and pick the first n_points closest to center?
        # Or generate a grid that covers the circle.
        
        # Simpler: Generate points in a hex lattice in a square bounding box, then filter/scale.
        # Let's create a grid with spacing based on expected radius.
        # Expected r ~ 0.12. Spacing ~ 2*r ~ 0.24.
        # Let's use spacing 0.2.
        spacing = 0.2
        max_coord = 1.0
        points_list = []
        
        # Hex lattice generation
        # Basis vectors: (1, 0), (0.5, sqrt(3)/2)
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, np.sqrt(3)/2])
        
        # Generate points in a range
        range_val = 2.0 / spacing
        
        for i in range(-int(range_val), int(range_val) + 1):
            for j in range(-int(range_val), int(range_val) + 1):
                p = i * v1 * spacing + j * v2 * spacing
                if np.linalg.norm(p) <= 1.0:
                    points_list.append(p)
        
        # If we don't have enough points, we might need to adjust spacing or just use what we have?
        # For N=50, spacing 0.2 should yield plenty.
        # Area pi ~ 3.14. Cell area ~ 0.086. 3.14/0.086 ~ 36. Might be tight.
        # Let's use smaller spacing to ensure we have at least N points.
        if len(points_list) < N:
             spacing = 0.15
             points_list = []
             range_val = 2.0 / spacing
             for i in range(-int(range_val), int(range_val) + 1):
                 for j in range(-int(range_val), int(range_val) + 1):
                     p = i * v1 * spacing + j * v2 * spacing
                     if np.linalg.norm(p) <= 1.0:
                         points_list.append(p)
        
        # Shuffle and take first N? No, take closest to center?
        # Taking closest to center is good for symmetry.
        points_list.sort(key=lambda p: np.linalg.norm(p))
        points = np.array(points_list[:N])
        
        # If we still don't have enough, fill with random
        if len(points) < N:
            remaining = N - len(points)
            random_points = np.random.rand(remaining, 2) * 2.0 - 1.0
            # Scale to be inside
            norms = np.linalg.norm(random_points, axis=1)
            random_points = random_points / norms[:, np.newaxis] * 0.8
            points = np.vstack([points, random_points])
            
        return points

    def compute_energy(points, r):
        """
        Compute the energy (sum of squared overlaps and boundary violations).
        """
        energy = 0.0
        # Pairwise
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(points[i] - points[j])
                if dist < 2 * r:
                    overlap = 2 * r - dist
                    energy += overlap * overlap
        
        # Boundary
        for i in range(N):
            dist_to_origin = np.linalg.norm(points[i])
            if dist_to_origin > 1.0 - r:
                overlap = dist_to_origin - (1.0 - r)
                energy += overlap * overlap
                
        return energy

    def simulate_annealing(points, r, max_iters, start_temp, end_temp):
        """
        Run Simulated Annealing to minimize energy for fixed r.
        """
        current_points = points.copy()
        current_energy = compute_energy(current_points, r)
        
        best_points = current_points.copy()
        best_energy = current_energy
        
        T = start_temp
        cooling_factor = (end_temp / start_temp) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            # Generate a move
            # Pick a random point
            idx = np.random.randint(N)
            # Move by a random vector
            # Step size should be related to T
            step_size = T * 0.1
            move = np.random.normal(0, step_size, 2)
            
            new_points = current_points.copy()
            new_points[idx] += move
            
            # Check if new point is way out of bounds (optimization)
            if np.linalg.norm(new_points[idx]) > 1.5:
                continue
                
            new_energy = compute_energy(new_points, r)
            
            # Metropolis acceptance
            delta_E = new_energy - current_energy
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
                current_points = new_points
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_points = current_points.copy()
            
            T *= cooling_factor
            
        return best_points, best_energy

    # Initialization
    points = hexagonal_init(N)
    
    # Start with a small radius
    r = 0.05
    min_step = 0.001
    max_r = 0.13
    
    start_time = time.time()
    time_limit = 115.0
    
    current_points = points.copy()
    
    best_overall_points = points.copy()
    best_overall_r = r
    
    # SA Parameters
    # We want to spend most time on the higher radii
    # Start with a few iterations, increase as we get closer to max?
    # Or keep it constant?
    
    while time.time() - start_time < time_limit:
        r_new = r + min_step
        
        # Determine number of iterations based on time remaining
        time_remaining = time_limit - (time.time() - start_time)
        # Estimate cost per iteration. Let's say ~1ms per iter for N=50?
        # N=50 -> 1225 pairs. 50 boundary. ~1300 ops. Python loops are slow.
        # Let's assume ~1000 iters per second is optimistic? Maybe 100?
        # Let's try 500 iters per step.
        max_iters = 500
        if r_new > 0.12:
             max_iters = 1000
             
        start_temp = 0.1
        end_temp = 0.001
        
        new_points, energy = simulate_annealing(current_points, r_new, max_iters, start_temp, end_temp)
        
        if energy < 1e-4:
            # Success
            r = r_new
            current_points = new_points
            best_overall_r = r
            best_overall_points = current_points.copy()
            
            # Adjust step size
            if r > 0.12:
                min_step = 0.0005
            elif r > 0.1:
                min_step = 0.001
            else:
                min_step = 0.002
        else:
            # Failure
            min_step *= 0.5
            if min_step < 1e-5:
                break
            
    # Final polish with a small step size and more iterations
    # Try to squeeze out more radius
    r_current = best_overall_r
    for _ in range(5):
        if time.time() - start_time > time_limit - 5:
            break
        r_try = r_current + 0.0001
        new_points, energy = simulate_annealing(best_overall_points, r_try, 2000, 0.05, 0.0001)
        if energy < 1e-4:
            best_overall_r = r_try
            best_overall_points = new_points
            r_current = r_try
        else:
            break

    # Ensure validity
    def get_score(points):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        radii = np.linalg.norm(points, axis=1)
        min_boundary = 1.0 - np.max(radii)
        return min(min_pairwise / 2.0, min_boundary)

    final_score = get_score(best_overall_points)
    
    # If invalid, try to fix
    radii = np.linalg.norm(best_overall_points, axis=1)
    if np.any(radii > 1.0 - final_score):
        scale = (1.0 - final_score) / np.max(radii)
        best_overall_points *= scale
        final_score = get_score(best_overall_points)

    return best_overall_points

# EVOLVE-BLOCK-END
