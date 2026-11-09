import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls = 0
        
        def safe_call(x):
            nonlocal calls
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = func(x)
            calls += 1
            return val
        
        # Phase 1: Estimate gradient at origin
        x0 = np.zeros(dim)
        f0 = safe_call(x0)
        if f0 is None:
            return x0, float('inf')
        
        best_x = x0.copy()
        best_f = f0
        
        # Use central differences with small step to estimate gradient
        delta = 0.1
        grad = np.zeros(dim)
        for i in range(dim):
            if time.time() > deadline or calls >= budget:
                break
            xp = x0.copy()
            xm = x0.copy()
            xp[i] += delta
            xm[i] -= delta
            fp = safe_call(xp)
            fm = safe_call(xm)
            if fp is None or fm is None:
                break
            grad[i] = (fp - fm) / (2 * delta)
        
        # Phase 2: Determine optimal vertex based on gradient signs
        # For linear f(x) = a^T x + b, min is at x_i = -5 if a_i > 0, +5 if a_i < 0
        optimal_vertex = np.zeros(dim)
        for i in range(dim):
            if grad[i] > 0:
                optimal_vertex[i] = -5.0
            elif grad[i] < 0:
                optimal_vertex[i] = 5.0
            else:
                optimal_vertex[i] = 0.0
        
        # Evaluate the optimal vertex
        f_opt_vertex = safe_call(optimal_vertex)
        if f_opt_vertex is not None and f_opt_vertex < best_f:
            best_f = f_opt_vertex
            best_x = optimal_vertex.copy()
        
        # Phase 3: If we have budget left, do local refinement
        # For a linear function, the minimum is at a vertex, so refinement won't help
        # But let's check nearby vertices to be safe
        # Generate all vertices and evaluate the top few by predicted value
        
        # Predict value at any point: f0 + grad^T x
        # For vertices, this is f0 + grad^T v
        
        # Generate all 2^dim vertices
        import itertools
        verts = np.array(list(itertools.product([-5.0, 5.0], repeat=dim)))
        
        # Predict values
        pred_values = f0 + verts @ grad
        
        # Sort by predicted value
        order = np.argsort(pred_values)
        
        # We already evaluated the predicted best vertex (optimal_vertex)
        # Now evaluate other vertices in order of predicted value
        # Skip the one we already evaluated
        for idx in order:
            if calls >= budget or time.time() > deadline:
                break
            v = verts[idx]
            # Check if we already evaluated this vertex
            if np.allclose(v, best_x, atol=1e-10):
                continue
            fv = safe_call(v)
            if fv is None:
                break
            if fv < best_f:
                best_f = fv
                best_x = v.copy()
        
        return best_x, best_f
