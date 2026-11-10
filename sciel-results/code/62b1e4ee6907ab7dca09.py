import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # 50s margin before 630s kill

    # Grid definition
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = X_flat.size

    P_start = np.array([0.0, 0.0, 0.0])
    P_end = np.array([-2.0, 0.0, 90.0])

    # 1D angular trajectory: the path is determined by angle a(t)
    # For L-shaped corridor, translation is derived from angle
    # The inner corner (x, y) must maintain contact with walls
    # 
    # Key insight: For the L-shaped corridor to fit in the room,
    # the corner (x,y) traces a specific curve as angle changes.
    # We parameterize by angle and derive x, y analytically.
    #
    # The base corridor has inner corner at (0,0) with arms extending
    # in +u (length 1) and +v (length 1), and the L extends to -3 in both.
    # When rotated by angle a, the constraint that the corridor stays
    # within bounds determines (x, y) as a function of a.
    
    # For a given angle a (radians), the inner corner position that
    # keeps the L-shaped corridor in contact with the "walls" of the
    # feasible region:
    # 
    # The corridor occupies (in local coords):
    #   Rect1: 0<=u<=1, -3<=v<=1
    #   Rect2: -3<=u<=1, 0<=v<=1
    # 
    # The "tip" of the L is at (1,1) in local coords.
    # When rotated by a, the global position of the tip is:
    #   (x + cos(a)*1 - sin(a)*1, y + sin(a)*1 + cos(a)*1)
    # = (x + cos(a) - sin(a), y + sin(a) + cos(a))
    #
    # The "far end" of arm 1 is at (1, -3): global = (x + cos(a) + 3*sin(a), y + sin(a) - 3*cos(a))
    # The "far end" of arm 2 is at (-3, 1): global = (x - 3*cos(a) - sin(a), y - 3*sin(a) + cos(a))
    #
    # We need to find (x,y) as a function of a that maximizes the
    # intersection. Let's use a parametric family:
    # x(a) = c1 * f1(a), y(a) = c2 * f2(a)
    # where f1, f2 are smooth functions and c1, c2 are optimized.
    
    # Alternative: use a smooth function x(a), y(a) parameterized by a few coefficients
    # Let's use: x(a) = ax0 + ax1*sin(a) + ax2*cos(a) + ax3*sin(2a) + ax4*cos(2a)
    #             y(a) = ay0 + ay1*sin(a) + ay2*cos(a) + ay3*sin(2a) + ay4*cos(2a)
    # with constraints x(0)=0, y(0)=0, x(pi/2)=-2, y(pi/2)=0
    
    n_samples = 100  # number of angle samples
    
    def path_from_angles(angle_coeffs, n_samples=n_samples):
        """
        angle_coeffs: array of 10 values [ax0, ax1, ax2, ax3, ax4, ay0, ay1, ay2, ay3, ay4]
        Returns path of shape (n_samples, 3)
        """
        a = np.linspace(0, np.pi/2, n_samples)
        
        ax0, ax1, ax2, ax3, ax4 = angle_coeffs[0:5]
        ay0, ay1, ay2, ay3, ay4 = angle_coeffs[5:10]
        
        x = ax0 + ax1*np.sin(a) + ax2*np.cos(a) + ax3*np.sin(2*a) + ax4*np.cos(2*a)
        y = ay0 + ay1*np.sin(a) + ay2*np.cos(a) + ay3*np.sin(2*a) + ay4*np.cos(2*a)
        
        # Enforce boundary conditions
        # At a=0: x=0, y=0
        # At a=pi/2: x=-2, y=0
        # We'll project to satisfy these
        x[0] = 0.0
        y[0] = 0.0
        x[-1] = -2.0
        y[-1] = 0.0
        
        a_deg = np.degrees(a)
        path = np.column_stack([x, y, a_deg])
        return path

    def densify_path(path):
        """Add linear interpolation between consecutive placements."""
        full_path = np.vstack([P_start, path, P_end])
        dense_points = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            delta_a = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(delta_a)))
            for k in range(1, m + 1):
                t = k / m
                p_int = p1 + t * (p2 - p1)
                dense_points.append(p_int)
        return np.array(dense_points)

    def count_inside(placements):
        """Count grid points inside all placements using progressive pruning."""
        valid_indices = np.arange(n_points)
        
        for x, y, a_deg in placements:
            a_rad = np.deg2rad(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            Xv = X_flat[valid_indices]
            Yv = Y_flat[valid_indices]
            
            dx = Xv - x
            dy = Yv - y
            
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a
            
            in_rect1 = (u >= -1e-9) & (u <= 1 + 1e-9) & (v >= -3 - 1e-9) & (v <= 1 + 1e-9)
            in_rect2 = (u >= -3 - 1e-9) & (u <= 1 + 1e-9) & (v >= -1e-9) & (v <= 1 + 1e-9)
            inside = in_rect1 | in_rect2
            
            if not np.any(inside):
                return 0
            
            valid_indices = valid_indices[inside]
            if len(valid_indices) == 0:
                return 0
        
        return len(valid_indices)

    def score_path(path):
        dense = densify_path(path)
        count = count_inside(dense)
        return count * 0.0001

    # Initial guess: simple linear interpolation in angle with derived positions
    # For a straight-line path in angle, x and y follow specific curves
    init_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Let's use a better initial guess based on the known solution structure
    # The optimal path for L-shaped corridor typically has x decreasing and y going up then down
    init_coeffs = np.array([
        0.0, 0.5, -0.5, 0.0, 0.0,   # x coefficients
        0.0, 0.8, -0.8, 0.0, 0.0    # y coefficients
    ])
    
    best_path = path_from_angles(init_coeffs)
    best_score = score_path(best_path)
    print(f"Initial 1D trajectory score: {best_score:.6f}")

    def smooth_objective(coeffs):
        """Smooth surrogate using soft-min for gradient-based optimization."""
        if time.time() > deadline:
            return 0.0
        
        path = path_from_angles(coeffs)
        dense = densify_path(path)
        
        # Use subsample for speed
        n_sub = min(3000, n_points)
        indices = np.random.choice(n_points, n_sub, replace=False)
        Xs = X_flat[indices]
        Ys = Y_flat[indices]
        
        min_margins = np.full(n_sub, np.inf)
        
        for x, y, a_deg in dense:
            a_rad = np.deg2rad(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = Xs - x
            dy = Ys - y
            
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a
            
            margin1 = np.minimum.reduce([u, 1 - u, v + 3, 1 - v])
            margin2 = np.minimum.reduce([u + 3, 1 - u, v, 1 - v])
            margin = np.maximum(margin1, margin2)
            
            min_margins = np.minimum(min_margins, margin)
        
        k = 30.0
        soft_count = np.sum(1.0 / (1.0 + np.exp(-k * min_margins)))
        estimated = soft_count * (n_points / n_sub)
        return -estimated * 0.0001

    # Bounds for coefficients
    bounds = [(-1.0, 1.0)] * 10

    # Phase 1: SLSQP with smooth objective
    if time.time() < deadline - 120:
        try:
            t0 = time.time()
            smooth_objective(init_coeffs)
            t1 = time.time()
            avg_eval = t1 - t0
            max_iter = max(30, int((deadline - time.time() - 120) * 0.8 / avg_eval))
            
            result = minimize(
                smooth_objective,
                init_coeffs,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-6, 'disp': False}
            )
            
            if result.fun < 0:
                candidate_path = path_from_angles(result.x)
                exact_score = score_path(candidate_path)
                print(f"SLSQP: smooth={-result.fun:.6f}, exact={exact_score:.6f}")
                if exact_score > best_score:
                    best_score = exact_score
                    best_path = candidate_path
        except Exception as e:
            print(f"SLSQP failed: {e}")

    # Phase 2: Genetic algorithm on the 1D parameter space
    if time.time() < deadline - 60:
        pop_size = 50
        n_generations = 30
        mutation_rate = 0.3
        mutation_scale = 0.1
        
        # Initialize population
        population = []
        population.append(init_coeffs.copy())
        for _ in range(pop_size - 1):
            individual = init_coeffs + np.random.randn(10) * 0.2
            individual = np.clip(individual, [b[0] for b in bounds], [b[1] for b in bounds])
            population.append(individual)
        
        for gen in range(n_generations):
            if time.time() > deadline - 30:
                break
            
            # Evaluate fitness
            fitness = []
            for individual in population:
                path = path_from_angles(individual)
                score = score_path(path)
                fitness.append(score)
            
            fitness = np.array(fitness)
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > best_score:
                best_score = fitness[best_idx]
                best_path = path_from_angles(population[best_idx])
            
            # Selection: tournament
            next_gen = []
            # Keep best individual
            next_gen.append(population[best_idx].copy())
            
            while len(next_gen) < pop_size:
                # Tournament selection
                candidates = np.random.choice(pop_size, 3, replace=False)
                winner = candidates[np.argmax(fitness[candidates])]
                parent = population[winner].copy()
                
                # Mutation
                if np.random.random() < mutation_rate:
                    parent += np.random.randn(10) * mutation_scale
                    parent = np.clip(parent, [b[0] for b in bounds], [b[1] for b in bounds])
                
                next_gen.append(parent)
            
            population = next_gen
    
    # Phase 3: Local refinement with Nelder-Mead on exact objective
    if time.time() < deadline - 20:
        try:
            def exact_objective(coeffs):
                if time.time() > deadline:
                    return 0.0
                path = path_from_angles(coeffs)
                return -score_path(path)
            
            nm_result = minimize(
                exact_objective,
                init_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-8}
            )
            nm_score = -nm_result.fun
            if nm_score > best_score:
                best_score = nm_score
                best_path = path_from_angles(nm_result.x)
            print(f"Nelder-Mead result: {best_score:.6f}")
        except Exception as e:
            print(f"Nelder-Mead failed: {e}")

    # Final coordinate ascent polish
    if time.time() < deadline - 10:
        current = init_coeffs.copy()
        current_score = best_score
        step = 0.05
        improved = True
        while improved and step > 1e-5 and time.time() < deadline - 5:
            improved = False
            for i in range(10):
                for direction in [-1, 1]:
                    new_val = current[i] + direction * step
                    if new_val < bounds[i][0] or new_val > bounds[i][1]:
                        continue
                    old_val = current[i]
                    current[i] = new_val
                    new_score = score_path(path_from_angles(current))
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                    else:
                        current[i] = old_val
            step *= 0.5
            if improved and time.time() < deadline - 5:
                if current_score > best_score:
                    best_score = current_score
                    best_path = path_from_angles(current)

    # Final validation
    final_score = score_path(best_path)
    if final_score > best_score:
        best_score = final_score
    
    print(f"Final score: {best_score:.6f}")
    
    return best_path
