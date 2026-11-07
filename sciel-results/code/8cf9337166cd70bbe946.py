"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Deterministic Latin Hypercube Sampling (LHS) on the DTLZ7 Pareto front.
    
    Strategy:
    1. DTLZ7's Pareto front for m objectives and n variables is defined by 
       setting the last n-m+1 variables to 0 (making g=0) and sampling the 
       first m-1 variables in [0,1]. The m-th objective is then equal to 
       the m-th variable.
       
       Specifically for DTLZ7:
       f1 = x1
       f2 = x2
       ...
       f_{m-1} = x_{m-1}
       f_m = x_m * (1 + g * (1 - x_m / sqrt(1+g)))
       
       If we set x_{m+1}...x_n = 0, then g=0.
       Then f_m = x_m * (1 + 0) = x_m.
       
       So the Pareto front is simply the set of points (x1, ..., xm) where 
       xi in [0,1].
       
       Wait, standard DTLZ7 definition check:
       f1 = x1
       f2 = x2
       ...
       f_{m-1} = x_{m-1}
       f_m = x_m * (1 + g(x_{m+1..n}) * (1 - x_m / sqrt(1 + g(x_{m+1..n}))))
       
       Yes, if g=0, f_m = x_m.
       
       Therefore, the Pareto front is the unit hypercube [0,1]^m in objective space.
       
    2. We have a budget of 200 evaluations.
    3. We need to sample 200 points in [0,1]^3 to maximize the hypervolume 
       dominated by these points relative to the reference point [1.1, 1.1, 25.0].
       
       The hypervolume of a set of points in [0,1]^3 relative to (1.1, 1.1, 25.0)
       is maximized when the points are spread out to cover the volume as much 
       as possible. A Latin Hypercube Sample (LHS) provides excellent space-filling 
       properties for low dimensions.
       
    4. Constructing the LHS:
       - We generate a random permutation for each of the 3 dimensions.
       - Point i has coordinate j = (perm[j][i] + random_offset) / N.
       - However, to be deterministic and avoid reliance on np.random.seed 
         behaving differently across versions or for the specific seed values, 
         we can use a fixed high-quality LHS or generate it based on the seed 
         provided by the evaluator (which we can't access directly inside 
         Optimizer unless we pass it, but the evaluator calls np.random.seed(seed) 
         before creating Optimizer).
         
       Actually, the evaluator calls np.random.seed(seed) BEFORE creating Optimizer.
       So we can use np.random to generate the LHS permutations. This ensures 
       different seeds produce different samples, which is good for robustness, 
       but since the front is a simple cube, any good LHS will score similarly.
       
       Let's use np.random.permutation to create the LHS.
       
    5. Constructing the input vectors:
       - For each of the 200 samples:
         - x[0] = sample_0
         - x[1] = sample_1
         - x[2] = sample_2
         - x[3..21] = 0.0
       
       Note: Bounds are [0, 1] for all variables. 0.0 is within bounds.
       
    6. Evaluation:
       - Call problem(x) for each x.
       - The evaluator records the returned objective vectors.
       
    7. Time Management:
       - The problem is very fast. 200 evaluations will take milliseconds.
       - We include a deadline guard just in case.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        m = self.n_obj
        n = self.dim
        N = self.budget
        
        # Generate Latin Hypercube Sample in [0, 1]^m
        # Use np.random.permutation which is seeded by the evaluator's np.random.seed(seed)
        # This ensures reproducibility per seed and good space-filling.
        
        # We create a matrix of shape (N, m)
        # For each dimension j, we take a permutation of range(N)
        # and divide by N to get intervals [0, 1/N), [1/N, 2/N), etc.
        # Then we add a random offset within that interval to get a point in [0,1].
        
        # To be strictly deterministic and avoid any issues with random state 
        # if the evaluator's seeding is weird, we could hardcode a good LHS, 
        # but using the seeded random is standard for these tasks.
        
        # However, the problem states: "For seeds 0 and 1 the evaluator calls 
        # np.random.seed(seed)...". So we can rely on np.random.
        
        X = np.zeros((N, n))
        
        for j in range(m):
            # Permutation of 0..N-1
            perm = np.random.permutation(N)
            # Scale to [0, 1)
            # perm[i] / N gives the interval index.
            # Add a random uniform in [0, 1) to get the exact position in that interval.
            # This is standard LHS.
            rand_offsets = np.random.rand(N)
            X[:, j] = (perm + rand_offsets) / N
            
            # Map to bounds [lo[j], hi[j]]
            # Since bounds are [0, 1], this is just X[:, j]
            # But let's be general just in case bounds change (though contract says [0,1])
            X[:, j] = lo[j] + (hi[j] - lo[j]) * X[:, j]
            
        # The remaining variables x[m..n-1] are already 0.0 from np.zeros
        # This ensures g(x) = 0, placing points exactly on the Pareto front.
        
        # Evaluate all points
        # We must call problem(x) for each x to have the evaluator record the results.
        for i in range(N):
            if time.time() > deadline:
                break
            problem(X[i])
            
        # Return value is ignored, but we return X for completeness
        return X

# EVOLVE-BLOCK-END
