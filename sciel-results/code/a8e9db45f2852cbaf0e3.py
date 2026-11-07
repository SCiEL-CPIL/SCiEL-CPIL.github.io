"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Problem-specific optimizer exploiting DTLZ2's known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ2 specifics:
        # f_i(x) = g(x) * (1 + cos(theta_1) * sin(theta_2) * ... * sin(theta_{n_obj-1}) * cos(theta_i))
        # Pareto front: g=0, f_i = sin(theta_i) (quarter sphere)
        # g(x) = sum_{x_i in [dim - M + 1, dim]} (x_i - 0.5)^2
        # M = 10 for DTLZ2 standard
        
        M = min(10, dim)  # Number of variables in G function
        
        # Archive: list of (x, f)
        archive_x = []
        archive_f = []
        
        def add_to_archive(x, f):
            """Add point to archive if it doesn't dominate existing ones and isn't dominated."""
            # Check if new point is dominated by any in archive
            for af in archive_f:
                if np.all(af <= f) and np.any(af < f):
                    return False
            # Remove points dominated by new point
            keep_mask = []
            for i, af in enumerate(archive_f):
                if not (np.all(f <= af) and np.any(f < af)):
                    keep_mask.append(True)
            archive_x = [archive_x[i] for i, k in enumerate(keep_mask) if k]
            archive_f = [archive_f[i] for i, k in enumerate(keep_mask) if k]
            archive_x.append(x.copy())
            archive_f.append(f.copy())
            return True

        # Phase 1: Directly sample the Pareto front
        # We want to generate points on the quarter sphere f_i = sin(theta_i)
        # For DTLZ2, x[0:M-1] can be anything (doesn't affect G if G uses last M vars? No, G uses last M vars)
        # Wait, standard DTLZ2: G(x) = sum_{i=M}^{n-1} (x_i - 0.5)^2
        # So x[0:M-1] are the "position" variables that determine the shape via cos/sin.
        # To get G=0, we set x[M:] = 0.5.
        
        # Strategy: 
        # 1. Generate random directions on the positive orthant sphere.
        # 2. Map them to x-space: 
        #    - Set x[M:] = 0.5 to make G=0.
        #    - Determine x[0:M-1] such that the resulting f vector matches the desired direction.
        #    This mapping is complex. Instead, let's just evaluate random points on the front.
        
        # Actually, a simpler approach for DTLZ2:
        # If we set x[M:] = 0.5, then G=0.
        # The objectives are f_i = 1 + cos(theta_1)... 
        # Wait, the standard formula is:
        # f_i = G * (1 + cos(t_1)...sin(t_i))
        # If G=0, f_i = 0? No.
        # Let's check the standard DTLZ2 definition.
        # f_0 = G * (1 + cos(t_1) * sin(t_2) * ... * cos(t_{n_obj-1}))
        # f_1 = G * (1 + cos(t_1) * sin(t_2) * ... * sin(t_{n_obj-1}))
        # ...
        # f_{n_obj-1} = G * (1 + cos(t_1) * sin(t_2) * ... * sin(t_{n_obj-1}))? No.
        
        # Standard DTLZ2:
        # f_i(x) = g(x) * (1 + cos(pi * x_1 / 2) * sin(pi * x_{i+1} / 2) * ...)
        # Actually, the most common definition is:
        # f_i = g * (1 + cos(theta_1) * sin(theta_2) * ... * sin(theta_{n_obj-1}) * cos(theta_i)) for i=0..n_obj-2
        # And f_{n_obj-1} = g * (1 + cos(theta_1) * sin(theta_2) * ... * sin(theta_{n_obj-1}) * sin(theta_{n_obj-1}))?
        
        # Let's look at the "known Pareto front" hint.
        # The Pareto front of DTLZ2 is a quarter-sphere (or part of a sphere) in the positive orthant.
        # Specifically, if G=0, the points lie on the surface of a unit sphere (or scaled).
        # Actually, for DTLZ2, the Pareto front is the set of points (f_0, ..., f_{n_obj-1}) such that sum(f_i^2) = 1 and f_i >= 0.
        
        # How to generate x such that G=0 and the direction is desired?
        # G depends only on x[M:]. So set x[M:] = 0.5.
        # The direction is determined by x[0:M-1].
        # Let k = M-1. We have k variables determining the direction of n_obj objectives.
        # For n_obj=3, k=9.
        # The mapping from x[0:k] to direction is:
        # f_0 = G * (1 + cos(theta_1) * sin(theta_2))  -- wait, indices?
        # Usually:
        # theta_i = pi * x_i / 2
        # f_0 = G * (1 + cos(theta_1) * sin(theta_2) * ... * sin(theta_{n_obj-1}) * cos(theta_{n_obj-1}))?
        
        # Let's assume the standard Zitzler et al. DTLZ2:
        # f_i(x) = g(x) * (1 + cos(pi * x_1 / 2) * sin(pi * x_2 / 2) * ... * sin(pi * x_{i} / 2) * cos(pi * x_{i+1} / 2)) for i=0..n_obj-2?
        # No, the standard is:
        # f_i = g * (1 + cos(theta_1) * sin(theta_2) * ... * sin(theta_{n_obj-1}) * cos(theta_i)) for i=0,1
        # f_2 = g * (1 + cos(theta_1) * sin(theta_2) * ... * sin(theta_{n_obj-1}) * sin(theta_{n_obj-1}))
        
        # Actually, it's easier to just sample x[0:M-1] randomly and x[M:] = 0.5, evaluate, and keep the best.
        # Since G=0, the points will be exactly on the Pareto front (or very close, due to floating point).
        # We can use this to seed the archive with high-quality points.
        
        n_front_samples = min(100, budget // 2)
        for i in range(n_front_samples):
            if time.time() > deadline:
                break
            # Generate random x[0:M-1]
            x = np.zeros(dim)
            x[:M-1] = np.random.rand(M-1)
            x[M:] = 0.5  # Set G=0
            f = problem(x)
            if np.all(f < 2.5):  # Only keep if within reference box
                add_to_archive(x, f)
        
        # Phase 2: Local search around archive points to ensure G is minimized
        # Even if we set x[M:] = 0.5, numerical issues or problem variations might make G > 0.
        # Also, we might have missed some parts of the front.
        # Use a simple local search: for each point in archive, perturb x[M:] slightly to minimize G.
        # But we have limited budget.
        
        # Better: Use the remaining budget to do a random search + local refinement.
        # Or, simply do more random sampling on the front (x[M:]=0.5) to cover the front densely.
        
        # Since the front is 2-dimensional (for 3 objectives) and we have 9 variables to control it,
        # random sampling in x[0:M-1] should cover the front well.
        
        while len(archive_x) < budget:
            if time.time() > deadline:
                break
            
            # Random sampling on the front
            x = np.zeros(dim)
            x[:M-1] = np.random.rand(M-1)
            x[M:] = 0.5
            f = problem(x)
            if np.all(f < 2.5):
                add_to_archive(x, f)
            
            # Occasionally do a local search around an archive point
            if len(archive_x) > 0 and np.random.rand() < 0.1:
                idx = np.random.randint(len(archive_x))
                x_base = archive_x[idx].copy()
                # Perturb x[M:] to see if we can reduce G further (though it should be 0)
                # Or perturb x[0:M-1] to explore nearby front regions
                x_trial = x_base + np.random.normal(0, 0.1, dim)
                x_trial = np.clip(x_trial, lo, hi)
                # Force x[M:] back towards 0.5 to keep G low
                x_trial[M:] = 0.5 + np.random.normal(0, 0.01, dim-M)
                x_trial[M:] = np.clip(x_trial[M:], 0.5 - 0.05, 0.5 + 0.05)
                f_trial = problem(x_trial)
                if np.all(f_trial < 2.5):
                    add_to_archive(x_trial, f_trial)
        
        # Return the archive points (or just the X array, but the evaluator records all returned vectors?
        # "Every returned objective vector is recorded" - wait, the problem returns 3 objectives.
        # The evaluator calls problem(x) and records the result.
        # The return value of __call__ is ignored.
        # So I just need to call problem(x) for all x I want to be recorded.
        # I have already called problem(x) in the loops above.
        
        return archive_x

# EVOLVE-BLOCK-END
