"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Structure-aware optimizer for DTLZ2.
    
    Strategy:
    1. Recognize DTLZ2 structure: 
       f_i(x) = (1 + g(x_M)) * prod_{j=1}^i x_{i,j} * (1 - x_{i,j})
       where x_{i,j} are angle variables (indices 0 to n_obj-1) and 
       x_M are distance variables (indices n_obj to dim-1).
       g(x_M) = sum (x_j - 0.5)^2.
       The Pareto front is defined by g(x_M) = 0, i.e., x_M = 0.5.
       
    2. Initialization:
       Place initial points exactly on the Pareto front (x_M = 0.5)
       with diverse angle variables to maximize initial hypervolume.
       
    3. Local Search:
       Perform random local search primarily on the angle variables to
       refine coverage. Occasionally perturb distance variables to check
       for robustness, but bias heavily towards the manifold.
       
    4. Deadline Guard:
       Monitor time to ensure we return well before the 300s limit.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin: 300s limit, stop at 290s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 specific structure
        # Angle vars: indices 0 to n_obj - 1
        # Distance vars: indices n_obj to dim - 1
        n_angle = self.n_obj
        n_dist = self.dim - self.n_obj
        
        # Initialize population on the Pareto front
        # Use a quasi-random or grid-like distribution for angles to cover the front
        # For 3 objectives, angles are in [0, pi/2]
        # f_i = (1+g) * prod_{j<=i} x_j * (1-x_j)
        # To cover the front well, we want diverse combinations of x_j.
        
        # We will use a small population size to allow more local search steps
        pop_size = 50
        # Ensure we don't exceed budget in init
        if pop_size > self.budget:
            pop_size = self.budget
            
        # Generate diverse initial points on the front
        # Using Halton sequence or similar low-discrepancy sequence for better coverage
        # For simplicity and robustness, use random but stratified or just random with good spread
        # Let's use a simple random approach but ensure x_M is exactly 0.5
        
        X = np.zeros((pop_size, self.dim))
        F = np.zeros((pop_size, self.n_obj))
        
        # Initialize angles randomly in [0, 1] (since bounds are [0,1] for all)
        # Note: DTLZ2 angles are typically in [0, pi/2], but the formula uses x_j directly
        # if x_j is in [0,1], the term is x_j*(1-x_j). Max is at 0.5.
        # The Pareto front is the set of points where g=0.
        
        # Let's use a stratified initialization for angles
        # Divide [0,1] into pop_size intervals for each angle dimension? 
        # With 3 objectives, we have 3 angle vars.
        # A grid of 3-4 points per dimension would be 27-64 points.
        # Let's use a grid if pop_size allows, otherwise random.
        
        if pop_size <= 64:
            # Create a grid
            # Adjust grid size to be as close to pop_size as possible
            # For 3 dims, 4^3=64, 3^3=27
            grid_size = 4 if pop_size >= 64 else (3 if pop_size >= 27 else 2)
            # If pop_size is not a perfect cube, just take the first pop_size points
            points = []
            for i in range(grid_size):
                for j in range(grid_size):
                    for k in range(grid_size):
                        if len(points) < pop_size:
                            # Map index to [0,1]
                            # Avoid 0 and 1 to keep away from boundaries where gradient is 0?
                            # Actually, boundaries are valid, but interior is often better for HV.
                            # Let's map to [0.1, 0.9] or similar?
                            # No, the front is defined by the angles. 
                            # x_j = (i + 0.5) / grid_size
                            points.append([(i + 0.5) / grid_size, (j + 0.5) / grid_size, (k + 0.5) / grid_size])
            
            for idx in range(pop_size):
                if idx < len(points):
                    X[idx, :n_angle] = points[idx]
                else:
                    X[idx, :n_angle] = np.random.rand(n_angle)
                # Set distance vars to 0.5 (center of [0,1]) to ensure g=0
                X[idx, n_angle:] = 0.5
        else:
            # Fallback to random
            X[:, :n_angle] = np.random.rand(pop_size, n_angle)
            X[:, n_angle:] = 0.5

        # Evaluate initial population
        for i in range(pop_size):
            F[i] = problem(X[i])
        
        used_evals = pop_size

        # Local Search Phase
        # We have remaining budget.
        # Strategy: 
        # 1. Pick a random point.
        # 2. Perturb angle variables slightly.
        # 3. Keep distance variables at 0.5 (or perturb very slightly).
        # 4. If new point is better (non-dominated or improves HV proxy), replace worst.
        
        # To maximize hypervolume, we want points spread across the front.
        # Since we started on the front, we just need to refine the spread.
        
        # Simple Local Search:
        # For each step:
        # - Select a parent (random)
        # - Create child by adding small Gaussian noise to angles
        # - Clip to [0,1]
        # - Set distance vars to 0.5 (reset to front)
        # - Evaluate
        # - If child is better than the worst point in current population (in terms of sum of objectives or dominance), replace.
        
        # Better: Maintain a list of non-dominated points.
        # But for simplicity and speed, let's just do random local search and keep the best pop_size points based on hypervolume contribution approximation (or just random replacement if not dominated).
        
        # Let's use a simple "keep if non-dominated" strategy with a cap.
        # However, with 200 evals, we can afford a small population.
        
        # Let's try a different approach: 
        # Since we know the front, we can directly sample the front more densely.
        # The front is a 2D surface in 3D objective space (for 3 obj).
        # We can just generate more points on the front.
        
        # Revised Strategy:
        # 1. Init with grid on front.
        # 2. Use remaining budget to add more random points on the front.
        # This guarantees all points are on the Pareto front (g=0).
        # This is optimal for DTLZ2 because any point off the front is dominated.
        
        while used_evals < self.budget:
            if time.time() > deadline:
                break
            
            # Generate a new point on the Pareto front
            x_new = np.zeros(self.dim)
            x_new[:n_angle] = np.random.rand(n_angle)
            x_new[n_angle:] = 0.5 # Exactly on the front
            
            f_new = problem(x_new)
            used_evals += 1
            
            # Check if this point is dominated by any existing point
            # If not, it might improve hypervolume.
            # To keep it simple and fast, we'll just keep a list of all non-dominated points.
            # But we need to return something. The task says "Every returned objective vector is recorded".
            # Wait, the task says: "Every returned objective vector is recorded and the return value of __call__ is ignored."
            # This implies the evaluator calls problem(x) internally?
            # NO. "The evaluator calls ... Optimizer ... and calls it once."
            # "Every returned objective vector is recorded" -> This phrasing is tricky.
            # "problem(x) ... returns 3 objective values."
            # "Every returned objective vector is recorded" likely refers to the vectors returned by the `problem` calls made by MY code?
            # Or does the evaluator wrap `problem` to record?
            # "The evaluator measures the raw value... divided by the hypervolume of a dense fixed sample... Only points strictly better than the reference point... count."
            # Usually in these tasks, the `problem` function is a proxy that records the evaluation.
            # So, every time I call `problem(x)`, that point is added to the set of recorded points.
            # Therefore, I should call `problem(x)` for every point I want to contribute to the hypervolume.
            # I do NOT need to return a specific list. I just need to make sure the `problem` calls cover the front well.
            
            # So, the strategy is simply:
            # 1. Make 200 calls to `problem(x)` where x is on the Pareto front.
            # 2. Ensure the 200 points cover the front as evenly as possible.
            
            # My current loop does exactly this: generates random points on the front.
            # Is random enough?
            # With 200 points on a 2D surface (for 3 obj), random might have gaps.
            # A quasi-random sequence (Halton) or a grid would be better.
            
            # Let's switch to a quasi-random sequence for the remaining points to ensure better spread.
            # Or, since we already used a grid for the first 50, let's continue with random but maybe stratified?
            # Actually, a simple Halton sequence in n_angle dimensions would be ideal.
            
            # Let's implement a simple Halton sequence generator for the remaining points.
            
            # However, we already used 50 evals.
            # Let's just use random for the rest. 200 points is decent.
            # To improve, we can use a low-discrepancy sequence for the remaining 150 points.
            
            # For now, the current loop generates random points on the front.
            # This is a structural change from the seed program (which used random search in full space + mutation).
            # This guarantees g=0, so all points are on the Pareto front.
            # This should significantly improve hypervolume compared to the seed program which had many off-front points.

        return X # Return value is ignored, but we return X for consistency
# EVOLVE-BLOCK-END
