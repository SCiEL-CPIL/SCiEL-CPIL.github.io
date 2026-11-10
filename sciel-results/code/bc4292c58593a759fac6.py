# EVOLVE-BLOCK-START
import numpy as np
import time


def moving_sofa_2d() -> np.ndarray:
    """
    The 2D moving-sofa problem: find the largest-area rigid shape that can be moved
    through an L-shaped corridor of width 1 that makes a single 90-degree turn.

    The shape is specified IMPLICITLY by a motion "path". Each path point is a triple
    (x, y, angle_degrees) giving the position of the corridor's inner corner and the
    corridor's rotation at one instant of the move. In the reference frame of the sofa
    the corridor sweeps over the plane; the sofa is then the intersection of every
    corridor placement along the path, so by construction it always clears the corridor.
    The objective is to maximize the area of that intersection.

    The evaluator prepends the start state (0, 0, 0) and appends the end state
    (-2, 0, 90), then densifies the path to <=1-degree rotation steps.

    Returns:
        path: np.ndarray of shape (n, 3); each row is (x, y, angle_degrees) for the
            corridor's inner corner.
    """
    start_time = time.time()
    deadline = 625.0  # evaluator kills at 630s, keep safety margin

    # Sofa grid
    x_grid = np.arange(-3, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid)
    P = np.column_stack((X.ravel(), Y.ravel()))  # (N, 2)
    N = P.shape[0]

    # Start and end states (added by evaluator)
    start_state = np.array([0.0, 0.0, 0.0])
    end_state = np.array([-2.0, 0.0, 90.0])

    def densify_path(path):
        """
        Densify the path by adding linear interpolations between consecutive rows.
        Between consecutive rows (x1,y1,a1) and (x2,y2,a2), add m = max(1, ceil(|a2-a1|))
        equal steps of interpolation.
        """
        if len(path) == 0:
            return np.array([start_state, end_state])
        
        full_path = [start_state] + list(path) + [end_state]
        
        densified = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(1, m + 1):
                t = k / m
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = p1[2] + t * (p2[2] - p1[2])
                densified.append([x, y, a])
        
        return np.array(densified)

    def check_point_in_corridor_vectorized(P, x, y, a_deg):
        """
        Vectorized check: which points in P are inside the corridor placement at (x, y, a_deg).
        Returns a boolean array of shape (N,).
        """
        a_rad = np.radians(a_deg)
        dx = P[:, 0] - x
        dy = P[:, 1] - y
        cos_a = np.cos(-a_rad)
        sin_a = np.sin(-a_rad)
        u = cos_a * dx - sin_a * dy
        v = sin_a * dx + cos_a * dy
        
        eps = 1e-9
        in_r1 = (u >= -eps) & (u <= 1 + eps) & (v >= -3 - eps) & (v <= 1 + eps)
        in_r2 = (u >= -3 - eps) & (u <= 1 + eps) & (v >= -eps) & (v <= 1 + eps)
        
        return in_r1 | in_r2

    def evaluate_path(path):
        """
        Evaluate the area of the sofa for a given path.
        The sofa is the set of grid points that are inside every placement.
        """
        densified = densify_path(path)
        
        # Initialize all points as potentially in the sofa
        in_sofa = np.ones(N, dtype=bool)
        
        for x, y, a in densified:
            # Check all points against this placement
            mask = check_point_in_corridor_vectorized(P, x, y, a)
            in_sofa &= mask
            if not in_sofa.any():
                break  # Early exit if no points remain
        
        return in_sofa.sum() * 0.0001

    def evaluate_path_with_mask(path):
        """
        Evaluate the path and also return the mask of remaining sofa points.
        """
        densified = densify_path(path)
        
        in_sofa = np.ones(N, dtype=bool)
        
        for x, y, a in densified:
            mask = check_point_in_corridor_vectorized(P, x, y, a)
            in_sofa &= mask
            if not in_sofa.any():
                break
        
        return in_sofa.sum() * 0.0001, in_sofa

    # Initial path - hand-tuned starting point with more control points
    path = np.array([
        [-0.05, 0.05, 10.0],
        [-0.15, 0.15, 20.0],
        [-0.25, 0.25, 30.0],
        [-0.35, 0.35, 40.0],
        [-0.45, 0.35, 50.0],
        [-0.55, 0.25, 60.0],
        [-0.65, 0.20, 70.0],
        [-0.75, 0.25, 80.0],
        [-0.85, 0.30, 86.0],
    ], dtype=np.float64)

    best_path = path.copy()
    best_score = evaluate_path(path)

    # Coordinate descent with adaptive step sizes
    np.random.seed(42)
    
    def coordinate_descent_optimize(path, max_iterations=500, base_step=0.1):
        """
        Coordinate descent optimization with adaptive step sizes.
        For each point in the path, try moving in each coordinate direction.
        """
        current_path = path.copy()
        current_score = evaluate_path(current_path)
        n_points = len(current_path)
        step = base_step
        
        for iteration in range(max_iterations):
            if (time.time() - start_time) > deadline - 30:
                break
            
            improved = False
            for i in range(n_points):
                if (time.time() - start_time) > deadline - 30:
                    break
                
                # Try each coordinate: x, y, angle
                for coord in range(3):
                    # Try positive and negative directions
                    for direction in [1, -1]:
                        new_path = current_path.copy()
                        if coord < 2:
                            new_path[i, coord] += direction * step
                        else:
                            new_path[i, coord] += direction * step * 2
                            new_path[i, coord] = np.clip(new_path[i, coord], 5, 85)
                        
                        new_score = evaluate_path(new_path)
                        if new_score > current_score + 1e-10:
                            current_path = new_path
                            current_score = new_score
                            improved = True
            
            if not improved:
                step *= 0.7
                if step < 0.001:
                    break
        
        return current_path, current_score

    def simulated_annealing_optimize(path, max_iterations=300, start_temp=0.05, end_temp=0.001):
        """
        Simulated annealing to escape local optima.
        """
        current_path = path.copy()
        current_score = evaluate_path(current_path)
        best_path_sa = current_path.copy()
        best_score_sa = current_score
        n_points = len(current_path)
        
        for iteration in range(max_iterations):
            if (time.time() - start_time) > deadline - 30:
                break
            
            # Temperature schedule
            t = iteration / max_iterations
            temp = start_temp * (end_temp / start_temp) ** t
            
            # Pick a random point and coordinate to perturb
            i = np.random.randint(n_points)
            coord = np.random.randint(3)
            
            new_path = current_path.copy()
            perturbation = np.random.uniform(-0.1, 0.1)
            if coord < 2:
                new_path[i, coord] += perturbation
            else:
                new_path[i, coord] += perturbation * 2
                new_path[i, coord] = np.clip(new_path[i, coord], 5, 85)
            
            new_score = evaluate_path(new_path)
            
            # Accept if better, or with probability exp(-(new_score - current_score) / temp)
            if new_score > current_score:
                current_path = new_path
                current_score = new_score
                if new_score > best_score_sa:
                    best_path_sa = new_path.copy()
                    best_score_sa = new_score
            else:
                delta = new_score - current_score
                if delta < 0:
                    prob = np.exp(delta / temp)
                    if np.random.random() < prob:
                        current_path = new_path
                        current_score = new_score
        
        return best_path_sa, best_score_sa

    def add_point(path, index, new_point):
        """Insert a new point at the given index."""
        path = np.array(path, dtype=np.float64)
        return np.vstack([path[:index], new_point.reshape(1, 3), path[index:]])

    def remove_point(path, index):
        """Remove the point at the given index."""
        path = np.array(path, dtype=np.float64)
        if len(path) <= 3:
            return path
        return np.vstack([path[:index], path[index+1:]])

    def structural_optimize(path, max_iterations=100):
        """
        Try adding and removing points to find the optimal number of control points.
        """
        current_path = path.copy()
        current_score = evaluate_path(current_path)
        n_points = len(current_path)
        
        for iteration in range(max_iterations):
            if (time.time() - start_time) > deadline - 40:
                break
            
            improved = False
            
            # Try adding a point at a random location
            if n_points < 15:
                idx = np.random.randint(0, n_points + 1)
                if idx == 0:
                    new_point = current_path[0].copy()
                    new_point[2] = max(5, new_point[2] - 5)
                elif idx == n_points:
                    new_point = current_path[-1].copy()
                    new_point[2] = min(85, new_point[2] + 5)
                else:
                    # Interpolate between adjacent points
                    p1 = current_path[idx - 1]
                    p2 = current_path[idx]
                    new_point = (p1 + p2) / 2
                    # Add some randomness
                    new_point[0] += np.random.uniform(-0.05, 0.05)
                    new_point[1] += np.random.uniform(-0.05, 0.05)
                    new_point[2] += np.random.uniform(-2, 2)
                    new_point[2] = np.clip(new_point[2], 5, 85)
                
                candidate = add_point(current_path, idx, new_point)
                candidate_score = evaluate_path(candidate)
                if candidate_score > current_score + 1e-10:
                    current_path = candidate
                    current_score = candidate_score
                    n_points += 1
                    improved = True
            
            # Try removing a point
            if n_points > 5:
                idx = np.random.randint(0, n_points)
                candidate = remove_point(current_path, idx)
                candidate_score = evaluate_path(candidate)
                if candidate_score > current_score + 1e-10:
                    current_path = candidate
                    current_score = candidate_score
                    n_points -= 1
                    improved = True
            
            if not improved:
                break
        
        return current_path, current_score

    # Multi-start coordinate descent
    n_starts = 4
    for start_idx in range(n_starts):
        if (time.time() - start_time) > deadline - 90:
            break
        
        if start_idx == 0:
            # Use the initial path
            current_path = path.copy()
        elif start_idx == 1:
            # Perturb the best path found so far
            current_path = best_path.copy()
            noise_scale = 0.15
            for i in range(len(current_path)):
                current_path[i, 0] += np.random.uniform(-noise_scale, noise_scale)
                current_path[i, 1] += np.random.uniform(-noise_scale, noise_scale)
                current_path[i, 2] += np.random.uniform(-noise_scale * 2, noise_scale * 2)
                current_path[i, 2] = np.clip(current_path[i, 2], 5, 85)
        elif start_idx == 2:
            # Try a different initial path structure
            alt_path = np.array([
                [-0.1, 0.1, 12.0],
                [-0.2, 0.2, 25.0],
                [-0.3, 0.3, 38.0],
                [-0.4, 0.35, 50.0],
                [-0.5, 0.3, 62.0],
                [-0.6, 0.25, 72.0],
                [-0.7, 0.25, 82.0],
            ], dtype=np.float64)
            current_path = alt_path
        else:
            # More aggressive perturbation
            current_path = best_path.copy()
            noise_scale = 0.3
            for i in range(len(current_path)):
                current_path[i, 0] += np.random.uniform(-noise_scale, noise_scale)
                current_path[i, 1] += np.random.uniform(-noise_scale, noise_scale)
                current_path[i, 2] += np.random.uniform(-noise_scale * 3, noise_scale * 3)
                current_path[i, 2] = np.clip(current_path[i, 2], 5, 85)
        
        # First do coordinate descent
        optimized_path, optimized_score = coordinate_descent_optimize(
            current_path, 
            max_iterations=200,
            base_step=0.1
        )
        
        if optimized_score > best_score:
            best_score = optimized_score
            best_path = optimized_path.copy()
        
        # Then try simulated annealing from the best
        if (time.time() - start_time) < deadline - 60:
            sa_path, sa_score = simulated_annealing_optimize(
                optimized_path,
                max_iterations=150,
                start_temp=0.03,
                end_temp=0.001
            )
            if sa_score > best_score:
                best_score = sa_score
                best_path = sa_path.copy()
        
        # Try structural optimization (add/remove points)
        if (time.time() - start_time) < deadline - 45:
            struct_path, struct_score = structural_optimize(
                best_path,
                max_iterations=50
            )
            if struct_score > best_score:
                best_score = struct_score
                best_path = struct_path.copy()
                
                # Re-optimize after structural change
                reopt_path, reopt_score = coordinate_descent_optimize(
                    best_path,
                    max_iterations=100,
                    base_step=0.05
                )
                if reopt_score > best_score:
                    best_score = reopt_score
                    best_path = reopt_path.copy()

    # Fine-grained coordinate descent polish
    if (time.time() - start_time) < deadline - 40:
        polish_path, polish_score = coordinate_descent_optimize(
            best_path,
            max_iterations=300,
            base_step=0.01
        )
        if polish_score > best_score:
            best_score = polish_score
            best_path = polish_path.copy()

    # Very fine polish
    if (time.time() - start_time) < deadline - 20:
        ultra_fine_path, ultra_fine_score = coordinate_descent_optimize(
            best_path,
            max_iterations=150,
            base_step=0.002
        )
        if ultra_fine_score > best_score:
            best_score = ultra_fine_score
            best_path = ultra_fine_path.copy()

    # Final ultra-precise polish
    if (time.time() - start_time) < deadline - 10:
        final_path, final_score = coordinate_descent_optimize(
            best_path,
            max_iterations=100,
            base_step=0.0005
        )
        if final_score > best_score:
            best_score = final_score
            best_path = final_path.copy()

    return best_path

# EVOLVE-BLOCK-END
