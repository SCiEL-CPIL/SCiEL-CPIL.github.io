"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Direct Affine Model Fitting Optimizer for BBOB f5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # We need at least d+1 points to fit the affine model f(x) = w^T x + b
        # If budget is too small, fallback to random
        if budget < dim + 1:
            x = np.zeros(dim)
            try:
                func(x)
            except Exception:
                pass
            return x, 0.0

        # 1. Construct an orthogonal simplex centered at the origin (0,0,...,0)
    # Points:
    # p0 = [0, 0, ..., 0]
    # p1 = [d, 0, ..., 0]
    # p2 = [0, d, ..., 0]
    # ...
    # pd = [0, 0, ..., d]
    
    # We choose d=1.0 for the step size. 
    # Since the minimizer is in [-4, 4]^5 and the box is [-5, 5]^5, 
    # a step of 1.0 from the origin stays well within the box and provides 
    # good numerical conditioning for the linear system.
    
    step = 1.0
    points = []
    points.append(np.zeros(dim)) # p0
    
    for i in range(dim):
        pi = np.zeros(dim)
        pi[i] = step
        points.append(pi)
        
    # 2. Evaluate the function at these d+1 points
    fvals = np.zeros(dim + 1)
    calls_made = 0
    
    for i in range(dim + 1):
        if time.time() > deadline:
             break
        if calls_made >= budget:
             break
        fvals[i] = func(points[i])
        calls_made += 1
        
    # If we couldn't evaluate all points (should not happen with budget=500, dim=5)
    if calls_made < dim + 1:
        # Fallback to best point found so far
        if calls_made > 0:
             best_idx = np.argmin(fvals[:calls_made])
             return points[best_idx], fvals[best_idx]
        else:
             x = np.zeros(dim)
             return x, 0.0

    # 3. Solve for the affine parameters w and b
    # f(p_i) = w^T p_i + b
    
    # Let A be the matrix of points p_1 ... p_d (excluding p_0)
    # Let y be the vector of f(p_1) ... f(p_d)
    # Let f0 = f(p_0) = b
    
    b = fvals[0]
    
    # Construct matrix A where rows are p_1 ... p_d
    A = np.array(points[1:]) # Shape (dim, dim)
    # Construct vector y where elements are f(p_1) ... f(p_d)
    y = fvals[1:]
    
    # Since A is diagonal with 'step' on the diagonal, we can solve easily:
    # w_i * step = y_i - b  =>  w_i = (y_i - b) / step
    
    w = (y - b) / step
    
    # 4. Determine the optimal vertex
    # Minimize f(x) = w^T x + b subject to -5 <= x_i <= 5
    # Since the function is linear, the minimum is at a vertex of the box.
    # For each dimension i:
    # If w_i > 0, we want x_i as small as possible -> x_i = -5
    # If w_i < 0, we want x_i as large as possible -> x_i = 5
    # If w_i == 0, any x_i is optimal; choose 0 for stability
    
    x_opt = np.zeros(dim)
    for i in range(dim):
        if w[i] > 0:
            x_opt[i] = -5.0
        elif w[i] < 0:
            x_opt[i] = 5.0
        else:
            x_opt[i] = 0.0
            
    # 5. Verify the solution and use remaining budget to refine if necessary
    # The model is exact for a linear function, so x_opt should be the global minimum.
    # However, due to floating point or potential non-linearity (though f5 is linear),
    # we can evaluate x_opt.
    
    f_opt = func(x_opt)
    calls_made += 1
    
    best_x = x_opt
    best_f = f_opt
    
    # If we have budget left, we can perform a local refinement using coordinate descent
    # or simply check neighbors, but for a strictly linear function, the vertex is exact.
    # Given the "Linear slope" description, the vertex is the true optimum.
    
    # To be safe and utilize budget if the function has any curvature (it shouldn't for f5),
    # we can do a few steps of coordinate descent starting from x_opt.
    # But since f5 is linear, extra calls are wasted. 
    # We will just return the computed vertex.
    
    return best_x, best_f

# EVOLVE-BLOCK-END
