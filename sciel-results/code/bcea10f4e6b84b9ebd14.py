# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def moving_sofa_2d() -> np.ndarray:
    """
    Moves a sofa through an L-shaped corridor.
    Strategy: Deterministic path construction based on the "pivot" solution.
    Instead of stochastic search, we construct a path that keeps the sofa
    tight against the inner corner of the L.
    
    The path is parameterized by the rotation angle a (0 to 90).
    We define x(a) and y(a) as sums of sinusoidal functions to allow
    for a smooth, flexible curve that can be analytically tuned.
    
    We perform a coarse grid search over the coefficients of these sinusoids
    to find a good path, then return the best one.
    """
    
    # 1. Setup Grid for Evaluation
    # Sofa points: x in [-3, 1.5), y in [-1.5, 1.5), step 0.01
    # Note: The problem statement says np.arange(-3, 1.5, 0.01) for x and y.
    # Let's verify the ranges.
    x_grid = np.arange(-3, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid)
    P = np.column_stack((X.ravel(), Y.ravel()))
    N_points = P.shape[0]
    wx = P[:, 0]
    wy = P[:, 1]
    
    start_state = np.array([0.0, 0.0, 0.0])
    end_state = np.array([-2.0, 0.0, 90.0])
    
    def evaluate_path(path_array):
        """
        Evaluates the area of the sofa for a given path.
        path_array: np.ndarray of shape (n, 3)
        Returns: float area
        """
        if path_array.size == 0:
            full_path = [start_state, end_state]
        else:
            full_path = [start_state] + path_array.tolist() + [end_state]
        
        mask = np.ones(N_points, dtype=bool)
        
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i + 1]
            
            a1 = p1[2]
            a2 = p2[2]
            
            da = a2 - a1
            steps = max(1, int(np.ceil(abs(da))))
            
            if steps == 1:
                eval_points = [p2]
            else:
                eval_points = []
                for j in range(1, steps + 1):
                    t = j / steps
                    nx = p1[0] + t * (p2[0] - p1[0])
                    ny = p1[1] + t * (p2[1] - p1[1])
                    na = a1 + t * da
                    eval_points.append([nx, ny, na])
            
            for p in eval_points:
                x, y, a_deg = p
                a_rad = np.deg2rad(a_deg)
                
                cos_a = np.cos(a_rad)
                sin_a = np.sin(a_rad)
                
                dx = wx - x
                dy = wy - y
                
                u = dx * cos_a + dy * sin_a
                v = -dx * sin_a + dy * cos_a
                
                in_r1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
                in_r2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
                
                in_corridor = in_r1 | in_r2
                
                mask &= in_corridor
                
                if not np.any(mask):
                    return 0.0
                    
        return float(np.sum(mask)) * 0.0001

    # 2. Parametric Path Definition
    # We want to define x(a) and y(a) for a in [0, 90].
    # A good base is a linear interpolation from (0,0) to (-2,0) with some bulge.
    # Let's use:
    # x(a) = -2 * (a/90) + A * sin(pi * a/90) + B * sin(2 * pi * a/90)
    # y(a) = C * sin(pi * a/90) + D * sin(2 * pi * a/90)
    
    # Coefficients to search: A, B, C, D
    # A: horizontal bulge
    # B: higher harmonic horizontal
    # C: vertical bulge (should be positive to go into the corner)
    # D: higher harmonic vertical
    
    # Ranges for coefficients based on intuition:
    # The sofa needs to swing out.
    # A in [-1, 1]
    # B in [-0.5, 0.5]
    # C in [0, 1.5]
    # D in [-0.5, 0.5]
    
    # Grid search over coefficients
    # To keep it fast, we use a coarse grid.
    # 5x5x5x5 = 625 paths. Each path has ~90 steps. 625 * 90 * (N_points operations)
    # N_points is ~135k. 625 * 90 * 135k is too slow for pure Python loops.
    # We need to vectorize the evaluation or reduce the grid.
    
    # Let's reduce the grid to 3x3x3x3 = 81 paths.
    # Or even better, evaluate a few key points first.
    
    # Let's try a smarter approach:
    # 1. Evaluate a set of candidate paths.
    # 2. Pick the best.
    # 3. Do a local search around the best.
    
    # Candidate generation
    # We will sample 100 random sets of coefficients within reasonable bounds.
    
    np.random.seed(42)
    
    best_score = 0.0
    best_coeffs = None
    
    # Coeff bounds
    A_range = (-1.0, 1.0)
    B_range = (-0.5, 0.5)
    C_range = (0.0, 1.5)
    D_range = (-0.5, 0.5)
    
    # Pre-compute angle grid for path generation
    # We will generate paths with 9 points (every 10 degrees)
    angles = np.linspace(10, 80, 7) # 7 points + start + end = 9 segments
    
    # Actually, let's just generate the path points directly from coefficients.
    def generate_path_from_coeffs(coeffs):
        A, B, C, D = coeffs
        path = []
        # Sample angles from 10 to 80
        for a in angles:
            t = a / 90.0
            x = -2.0 * t + A * np.sin(np.pi * t) + B * np.sin(2 * np.pi * t)
            y = C * np.sin(np.pi * t) + D * np.sin(2 * np.pi * t)
            path.append([x, y, a])
        return np.array(path)
    
    # Search
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Phase 1: Random Sampling
    num_samples = 200
    for _ in range(num_samples):
        if time.time() > deadline:
            break
        A = np.random.uniform(*A_range)
        B = np.random.uniform(*B_range)
        C = np.random.uniform(*C_range)
        D = np.random.uniform(*D_range)
        
        coeffs = np.array([A, B, C, D])
        path = generate_path_from_coeffs(coeffs)
        
        score = evaluate_path(path)
        
        if score > best_score:
            best_score = score
            best_coeffs = coeffs
    
    # Phase 2: Local Search around best
    if best_coeffs is not None:
        current_coeffs = best_coeffs
        current_score = best_score
        step_sizes = [0.2, 0.1, 0.05, 0.02, 0.01]
        
        for step in step_sizes:
            improved = False
            for _ in range(50): # 50 iterations per step size
                if time.time() > deadline:
                    break
                # Perturb one coefficient
                i = np.random.randint(4)
                perturbation = np.random.uniform(-step, step)
                new_coeffs = current_coeffs.copy()
                new_coeffs[i] += perturbation
                
                # Clamp to bounds
                bounds = [A_range, B_range, C_range, D_range]
                for j in range(4):
                    new_coeffs[j] = np.clip(new_coeffs[j], bounds[j][0], bounds[j][1])
                
                path = generate_path_from_coeffs(new_coeffs)
                score = evaluate_path(path)
                
                if score > current_score:
                    current_score = score
                    current_coeffs = new_coeffs
                    improved = True
            
            if not improved:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_coeffs = current_coeffs

    # Generate final path from best coefficients
    if best_coeffs is not None:
        # Use a finer grid for the final path to ensure smoothness
        final_angles = np.arange(10, 81, 10) # 10, 20, ..., 80
        final_path = []
        A, B, C, D = best_coeffs
        for a in final_angles:
            t = a / 90.0
            x = -2.0 * t + A * np.sin(np.pi * t) + B * np.sin(2 * np.pi * t)
            y = C * np.sin(np.pi * t) + D * np.sin(2 * np.pi * t)
            final_path.append([x, y, a])
        
        return np.array(final_path)
    else:
        # Fallback to a simple linear path if search fails
        fallback_path = []
        for a in range(10, 81, 10):
            t = a / 90.0
            x = -2.0 * t
            y = 0.0
            fallback_path.append([x, y, a])
        return np.array(fallback_path)

# EVOLVE-BLOCK-END
