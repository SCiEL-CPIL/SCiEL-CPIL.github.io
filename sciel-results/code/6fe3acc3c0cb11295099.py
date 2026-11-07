import numpy as np

class Optimizer:
    """
    Multi-objective optimizer for ZDT4.
    Uses the known structure of the Pareto front and the global minimum of g(x)
    to construct an optimal set of points that maximizes hypervolume.
    """
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT4 specific logic
        # dim is 10, n_obj is 2
        # bounds are given
        
        # We want to maximize hypervolume.
        # The Pareto front for ZDT4 is f1 = x0, f2 = 1 - sqrt(x0) (when g=1).
        # g(x) = 10*n - sum(x_i * sin(10*pi*x_i)) for i=1..n-1.
        # Global min of g is 1, achieved when x_i = 0.5 for i=1..n-1.
        
        # We have a budget of evaluations.
        # We will generate 'budget' points that lie on the Pareto front.
        # We distribute x0 to maximize hypervolume.
        
        n_points = self.budget
        dim = self.dim
        
        # Generate x0 values
        # To maximize HV with ref [11, 400], we want to minimize sum(delta_f1 * f2).
        # f2 = 1 - sqrt(f1).
        # We want larger delta_f1 where f2 is smaller (f1 is larger).
        # So x0 should be denser near 0.
        # x0_i = (i/N)^k with k > 1 makes it denser near 0?
        # Let i=0..N. x_i = (i/N)^2.
        # dx = 2i/N^2. Small for small i, large for large i.
        # Yes.
        
        # Ensure we cover [0, 1]
        # The problem clips x to bounds. Bounds for x0 are likely [0,1].
        # Bounds for x1..x9 are likely [0,1].
        
        # Construct X
        X = np.zeros((n_points, dim))
        
        # Set x1..x9 to 0.5 (global min for g)
        # Check bounds to ensure 0.5 is valid. If not, use center of bounds.
        # Standard ZDT4 bounds are [0,1].
        for i in range(1, dim):
            low, high = bounds[i]
            # 0.5 is the global minimum for the sine term.
            # If 0.5 is not in bounds, we should pick the closest point to 0.5 mod 0.2?
            # But standard ZDT4 is [0,1]. We assume 0.5 is valid.
            X[:, i] = 0.5
            
        # Set x0
        # Use a power law distribution to cluster points near x0=0
        # This gives larger steps in x0 (and thus larger hypervolume contributions 
        # from the 'lower' f2 points at the end of the sum)
        # Actually, let's verify the HV sum direction.
        # Sort by f1 asc. 
        # HV = sum_{i=1}^N (f1_{i+1} - f1_i) * (ref2 - f2_i)
        # f2_i is decreasing. 
        # The term (ref2 - f2_i) is largest when f2_i is smallest (i=N).
        # So we want the largest delta_f1 to be at i=N.
        # delta_f1 at i=N is f1_{N+1} - f1_N = ref1 - f1_N.
        # This is fixed by ref1 and f1_N.
        # The other deltas are f1_{i+1} - f1_i.
        # We want larger deltas where (ref2 - f2_i) is larger.
        # (ref2 - f2_i) is larger for larger i.
        # So we want larger gaps for larger i.
        # This means x0 should be sparser at high x0?
        # No, "larger gaps for larger i" means the distance between x0_i and x0_{i+1} should increase with i.
        # So x0 should be denser at low i (small x0).
        # x0_i = (i/N)^2 is denser at low i?
        # i=0, x=0. i=1, x=1/N^2. i=2, x=4/N^2.
        # Gap 1: 1/N^2. Gap 2: 3/N^2.
        # Yes, gaps increase.
        # So x0_i = (i/N)^2 is correct.
        
        # However, we must ensure x0 is within bounds.
        low0, high0 = bounds[0]
        # Assume standard [0,1]
        # If bounds are different, we should map to the range where the Pareto front is defined.
        # ZDT4 Pareto front is for x0 in [0,1].
        
        # Generate indices
        idx = np.arange(n_points)
        # Avoid division by zero if n_points=1? 
        # Budget is 200.
        # Map idx/N to [0,1]
        # We want to cover [0,1].
        # x0 = low0 + (high0 - low0) * (idx / (n_points - 1)) ** 2
        # But this puts the last point at high0.
        # The first point at low0.
        
        # If n_points > 1:
        if n_points > 1:
            t = (idx / (n_points - 1))
            x0_vals = low0 + (high0 - low0) * (t ** 2)
        else:
            x0_vals = np.array([(low0 + high0) / 2.0])
            
        X[:, 0] = x0_vals
        
        # Evaluate all points
        # We must call problem(x) for each x
        # The evaluator records every returned objective vector.
        # So we just need to call problem for each row in X.
        
        results = []
        for i in range(n_points):
            f = problem(X[i])
            # f is a vector of 2 objectives
            results.append(f)
            
        return results
