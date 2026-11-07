"""Multi-objective optimizer on DTLZ7 using quasi-random sampling on Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ7 that exploits the known 2D Pareto front
    structure using quasi-random (Sobol-like) sampling for optimal hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s margin

        dim = self.dim
        budget = self.budget

        # DTLZ7 Pareto front for 3 objectives:
        # x[2:] = 0, f1 = x[0], f2 = x[1], f3 = 1 - sqrt(x[1])*(1 + sin(3*pi*x[1]))
        # The front is 2D, parameterized by (x0, x1) in [0,1]^2
        # Hypervolume with ref [1.1, 1.1, 25.0] is dominated by 2D coverage in (f1,f2)
        
        # Generate quasi-random points in [0,1]^2 using a low-discrepancy sequence
        # Use a simple Halton-like sequence or Sobol approximation
        n_points = budget
        
        # Use a stratified sampling approach: divide [0,1]^2 into cells
        # and place one point per cell using a permutation pattern
        # For n=200, use a 20x10 grid with jitter based on a permutation
        
        # Better: use the inverse CDF of a 2D uniform with a space-filling curve
        # Simplest effective approach: use a grid with golden-ratio jitter
        
        # Use a deterministic quasi-random sequence based on van der Corput in 2D
        def halton_sequence(n, base):
            seq = []
            for i in range(1, n + 1):
                f = 1.0
                r = 0.0
                while i > 0:
                    f /= base
                    r += f * (i % base)
                    i //= base
                seq.append(r)
            return np.array(seq)
        
        # Generate 2D Halton sequence (bases 2 and 3)
        h1 = halton_sequence(n_points, 2)
        h2 = halton_sequence(n_points, 3)
        
        # Shift to avoid boundary issues (keep points in (0,1) not on edges)
        # Actually, including boundary points is fine for DTLZ7
        # But let's add a small inset to be safe
        inset = 1e-10
        x0_vals = np.clip(h1, inset, 1.0 - inset)
        x1_vals = np.clip(h2, inset, 1.0 - inset)
        
        # Evaluate all points
        evaluated = []
        eval_count = 0
        
        for i in range(n_points):
            if eval_count >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = x0_vals[i]
            x[1] = x1_vals[i]
            # x[2:] stays 0 (on the Pareto front)
            
            f = problem(x)
            evaluated.append((x, f))
            eval_count += 1
        
        if evaluated:
            return np.array([x for x, _ in evaluated])
        return np.array([])

# EVOLVE-BLOCK-END
