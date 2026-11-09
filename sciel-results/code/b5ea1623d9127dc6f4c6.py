"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for Linear Slope function.
    Strategy:
    1. Estimate gradient using central differences (more robust).
    2. Predict the optimal vertex based on gradient signs.
    3. Evaluate predicted vertex.
    4. Use remaining budget to refine gradient estimate and verify.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s margin
        
        rng = np.random.default_rng(42)
        d = self.dim
        calls_made = 0
        
        def safe_call(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return None
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            return fx
        
        # Phase 1: Estimate gradient using central differences
        # For linear function f(x) = a^T x + b
        # Central difference: a_i ≈ (f(center + h*e_i) - f(center - h*e_i)) / (2*h)
        
        # First, get a good starting point - use center
        center = np.zeros(d)
        f_center = safe_call(center)
        if f_center is None:
            return center, np.inf
        
        # Use central differences with h=0.5 (stays in bounds for all dimensions)
        h = 0.5
        grad_est = np.zeros(d)
        
        for i in range(d):
            if calls_made >= self.budget or time.time() > deadline:
                break
            
            # Positive direction
            x_plus = center.copy()
            x_plus[i] += h
            f_plus = safe_call(x_plus)
            if f_plus is None:
                break
            
            # Negative direction
            x_minus = center.copy()
            x_minus[i] -= h
            f_minus = safe_call(x_minus)
            if f_minus is None:
                break
            
            grad_est[i] = (f_plus - f_minus) / (2 * h)
        
        # Determine optimal vertex based on gradient signs
        optimal_vertex = np.zeros(d)
        for i in range(d):
            if grad_est[i] > 0:
                optimal_vertex[i] = -5.0  # Move in negative direction
            elif grad_est[i] < 0:
                optimal_vertex[i] = 5.0   # Move in positive direction
            else:
                optimal_vertex[i] = 0.0   # Neutral
        
        # Evaluate the predicted optimal vertex
        f_opt = safe_call(optimal_vertex)
        if f_opt is None:
            f_opt = f_center
            best_x = center.copy()
        else:
            best_x = optimal_vertex.copy()
            best_f = f_opt
        
        # Phase 2: Refine gradient estimate using more evaluation points
        # For a linear function, we can use any two points to estimate the gradient
        # Use multiple pairs to get a more robust estimate
        
        # Collect several random points and use them to estimate gradient
        # f(x) = a^T x + b, so f(x1) - f(x2) = a^T (x1 - x2)
        # We can set up a system of equations to solve for a
        
        # Use the center and several other points
        points = [center]
        values = [f_center]
        
        # Add the optimal vertex if not already included
        if not np.allclose(optimal_vertex, center):
            if calls_made < self.budget and time.time() < deadline:
                f_v = safe_call(optimal_vertex)
                if f_v is not None:
                    points.append(optimal_vertex.copy())
                    values.append(f_v)
        
        # Add random points to help estimate the gradient
        n_refine = min(20, max(0, self.budget - calls_made))
        for _ in range(n_refine):
            if calls_made >= self.budget or time.time() > deadline:
                break
            x_rand = -5 + 10 * rng.random(d)
            f_rand = safe_call(x_rand)
            if f_rand is not None:
                points.append(x_rand)
                values.append(f_rand)
        
        # If we have at least d+1 points, we can solve for the gradient
        if len(points) >= d + 1:
            X = np.array(points)
            y = np.array(values)
            
            # Solve for a in y = X @ a + b
            # Use least squares
            A = np.column_stack([X, np.ones(len(X))])
            try:
                result = np.linalg.lstsq(A, y, rcond=None)
                grad_est_refined = result[0][:d]
                
                # Update optimal vertex with refined gradient
                optimal_vertex_refined = np.zeros(d)
                for i in range(d):
                    if grad_est_refined[i] > 0:
                        optimal_vertex_refined[i] = -5.0
                    elif grad_est_refined[i] < 0:
                        optimal_vertex_refined[i] = 5.0
                    else:
                        optimal_vertex_refined[i] = 0.0
                
                # Evaluate the refined optimal vertex
                f_refined = safe_call(optimal_vertex_refined)
                if f_refined is not None and f_refined < best_f:
                    best_f = f_refined
                    best_x = optimal_vertex_refined.copy()
            except:
                pass
        
        # Phase 3: Final verification - try a few vertices near the best
        # For a linear function, the minimum must be at a vertex
        # Try some random vertices to verify
        
        n_verify = min(10, max(0, self.budget - calls_made))
        for _ in range(n_verify):
            if calls_made >= self.budget or time.time() > deadline:
                break
            # Random vertex: each coordinate is either -5 or 5
            x_vertex = np.where(rng.random(d) < 0.5, -5.0, 5.0)
            f_vertex = safe_call(x_vertex)
            if f_vertex is not None and f_vertex < best_f:
                best_f = f_vertex
                best_x = x_vertex.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
