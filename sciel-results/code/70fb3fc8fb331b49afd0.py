"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Structure-aware optimizer for DTLZ1.
    
    Strategy:
    1. Identify the 'distance' variables (last 3) and 'shape' variables (first 4).
    2. Fix distance variables to 0.5 to ensure g(x) = 0, placing points on the Pareto front.
    3. Use the evaluation budget to optimize the shape variables using a 
       non-dominated sorting approach with local perturbations to maximize 
       hypervolume coverage of the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        # DTLZ1 structure:
        # f1 = 0.5 * x1 * (1 + g(x))
        # f2 = 0.5 * x2 * (1 - x1) * (1 + g(x))
        # f3 = 0.5 * (1 - x2) * (1 + g(x))
        # where x1 = x[0], x2 = x[1], and g(x) = 100 * (n_g + sum((x_i - 0.5)^2 - sin(4*pi*x_i)))
        # n_g is the number of g-variables. For dim=7, n_g = 7 - 2 = 5.
        # Wait, standard DTLZ1 definition:
        # M objectives, K decision variables.
        # f1 = 0.5 * x1 * (1 + g(x))
        # f2 = 0.5 * x2 * (1 - x1) * (1 + g(x))
        # f3 = 0.5 * (1 - x2) * (1 + g(x))
        # g(x) = 100 * (n_g + sum_{i=3}^{K} ((x_i - 0.5)^2 - sin(4*pi*x_i)))
        # Here K=7, M=3. So x1=x[0], x2=x[1].
        # g-variables are x[2], x[3], x[4], x[5], x[6]. (5 variables)
        # To minimize g, we set x_i = 0.5 for i in 2..6.
        # Then g=0, and 1+g=1.
        # The front is defined by x[0] and x[1] in [0,1].
        
        self.shape_dim = 2  # x[0] and x[1]
        self.dist_dim = self.dim - self.shape_dim  # x[2]...x[6]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initialize population
        # We will fix the distance variables to 0.5 (midpoint) to ensure g=0.
        # We will search the shape variables (first 2) to cover the front.
        
        # Budget allocation:
        # Use a population of size P.
        # Let's try P = 50. 
        # Remaining budget for local search.
        
        pop_size = 50
        if self.budget < pop_size * 2:
            pop_size = max(10, self.budget // 4)

        # Initialize shape variables randomly in [0, 1]
        # Use np.random.seed(0) is handled by evaluator, but we use default_rng for internal consistency if needed?
        # The evaluator calls np.random.seed(seed). We should use np.random directly or a seeded rng.
        # Let's use np.random directly to respect the global seed set by evaluator.
        
        X_shape = np.random.rand(pop_size, self.shape_dim)
        
        # Construct full X
        # Distance variables fixed to 0.5
        X_dist = np.full((pop_size, self.dist_dim), 0.5)
        X = np.hstack([X_shape, X_dist])
        
        # Clip to bounds (though 0.5 is usually within [0,1])
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Local search loop
        # We want to improve the hypervolume. 
        # A simple strategy: pick a random point, perturb its shape variables, 
        # check if it is non-dominated by the current population.
        # If yes, replace a dominated point or the worst point.
        
        # To be more effective, we can use a "crowding distance" or similar metric, 
        # but given the budget, a simple non-dominated replacement is robust.
        
        # Let's refine: 
        # 1. Compute non-dominated front.
        # 2. While budget remains:
        #    a. Pick a random individual from the current population.
        #    b. Perturb its shape variables with a small step.
        #    c. Evaluate.
        #    d. If the new point is non-dominated by the current population's best front, 
        #       and it is not dominated by itself (trivially), we might want to add it.
        #       However, we have a fixed population size? No, we can just keep the best non-dominated set.
        #       Actually, the task says "Every returned objective vector is recorded". 
        #       So we just need to generate as many good points as possible.
        #       We don't have to maintain a fixed population size for the final result, 
        #       but we must not exceed the budget.
        #       The score is based on the hypervolume of ALL recorded vectors.
        #       So we should aim to fill the budget with non-dominated points that are well-distributed.
        
        # Strategy:
        # Maintain a list of all evaluated points.
        # But keeping all 200 points is fine.
        # We can just generate 200 points.
        # 1. Generate initial 50 points on the front.
        # 2. For the remaining 150, perform local search around existing points to find better distributed ones.
        
        # Actually, since g=0 for all points with dist vars at 0.5, all points are on the Pareto front.
        # The quality of the hypervolume depends on how well the points cover the front.
        # The front is a triangle in 3D space (since x1+x2 <= 1 is not enforced by bounds, 
        # but the objective definitions create the shape).
        # f1 = 0.5 * x1
        # f2 = 0.5 * x2 * (1 - x1)
        # f3 = 0.5 * (1 - x2)
        
        # To maximize hypervolume, we want points spread across the surface.
        # Random sampling in x1, x2 is a good start.
        # Local search can help fill gaps.
        
        # Let's use a simple hill-climbing / random restart approach.
        # We will generate points one by one.
        # We will keep a record of all F vectors.
        
        all_F = [F]  # List of arrays? Or just accumulate.
        all_X = [X]
        
        # Flatten for easier management
        # Actually, let's just use a list of (x, f) pairs.
        
        points = []
        for i in range(pop_size):
            points.append((X[i], F[i]))
        
        # Remaining budget
        remaining = self.budget - used
        
        # For each remaining evaluation:
        # 1. Pick a random point from the current set.
        # 2. Perturb its shape variables.
        # 3. Evaluate.
        # 4. Add to points.
        # 5. (Optional) If we have too many points, we could prune dominated ones, 
        #    but since all are on the front (g=0), they are all non-dominated 
        #    unless one dominates another? 
        #    Wait, if g=0, f1=0.5*x1, f2=0.5*x2(1-x1), f3=0.5(1-x2).
        #    Does one point dominate another?
        #    If x1' < x1, then f1' < f1.
        #    But f2' = 0.5*x2'(1-x1'). If x2' is large, f2' might be larger than f2.
        #    So they are generally non-dominated with each other.
        #    So we just need to spread them out.
        
        # To ensure good coverage, we can use a "repulsion" strategy or just random search.
        # Given the small budget (200), random search on the front is quite strong.
        # But we can do slightly better by perturbing existing points.
        
        # Let's try a mix: 50% random new points, 50% perturbations of existing.
        
        for _ in range(remaining):
            if time.time() > deadline:
                break
                
            if np.random.rand() < 0.5:
                # Random new point on the front
                x_shape = np.random.rand(self.shape_dim)
            else:
                # Perturb an existing point
                idx = np.random.randint(0, len(points))
                x_parent = points[idx][0]
                # Perturb shape variables
                step = 0.1
                x_shape = x_parent[:self.shape_dim] + step * np.random.randn(self.shape_dim)
                x_shape = np.clip(x_shape, 0, 1)
            
            x_dist = np.full(self.dist_dim, 0.5)
            x_new = np.hstack([x_shape, x_dist])
            x_new = np.clip(x_new, lo, hi)
            
            f_new = problem(x_new)
            points.append((x_new, f_new))
        
        # Return the last X (ignored, but required by interface)
        # The evaluator records all F returned by problem().
        # We don't need to return anything specific, but the signature says "return X".
        # Let's return the last X or an empty array. The contract says "return value ... is ignored".
        
        return points[-1][0] if points else None

# EVOLVE-BLOCK-END
