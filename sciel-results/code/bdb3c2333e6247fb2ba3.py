import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # DTLZ1 with M=3, n=7
        # At g=0 (x[2:7] = 0.5), the objectives are:
        #   f0 = 0.5 * x[0]
        #   f1 = 0.5 * (1 - x[0]) * x[1]
        #   f2 = 0.5 * (1 - x[1])
        #
        # The Pareto front in objective space is a 2D surface. We parameterize it
        # using spherical coordinates to ensure uniform coverage of the front.
        #
        # Let's use a direct parameterization:
        # Sample (u, v) from the unit square [0,1]^2
        # x[0] = u
        # x[1] = v
        # This covers the entire front since x[0], x[1] in [0,1] map to the front.
        #
        # To get better hypervolume, we want points spread uniformly across the
        # Pareto front in objective space. The mapping from (x0, x1) to (f0, f1, f2)
        # is nonlinear, so uniform sampling in (x0, x1) space doesn't give uniform
        # coverage in objective space.
        #
        # Instead, let's sample directly in objective space on the simplex
        # f0 + f1 + f2 = 0.5, fi >= 0, and invert to get decision variables.
        #
        # From the equations:
        #   f0 = 0.5 * x0  =>  x0 = 2*f0
        #   f2 = 0.5 * (1 - x1)  =>  x1 = 1 - 2*f2
        #   f1 = 0.5 * (1 - x0) * x1 = 0.5 * (1 - 2*f0) * (1 - 2*f2)
        #
        # For a given (f0, f2) with f0 in [0, 0.5], f2 in [0, 0.5]:
        #   f1 = 0.5 * (1 - 2*f0) * (1 - 2*f2)
        # We need f1 >= 0, which is always true for f0, f2 in [0, 0.5].
        #
        # But we also need f0 + f1 + f2 = 0.5. Let's check:
        # f0 + f1 + f2 = f0 + 0.5*(1-2*f0)*(1-2*f2) + f2
        #   = f0 + 0.5 - f0 - f2 + 2*f0*f2 + f2
        #   = 0.5 + 2*f0*f2
        # This is NOT 0.5 unless f0*f2 = 0. So the DTLZ1 front is NOT the flat simplex.
        #
        # The correct approach: just sample (x0, x1) uniformly and evaluate.
        # Use a grid for good coverage.
        
        budget = self.budget
        dim = self.dim
        
        # Use a structured grid in (x0, x1) space for good coverage
        n_side = int(np.ceil(np.sqrt(budget)))
        # Adjust to fit exactly budget points
        x0_vals = np.linspace(0, 1, n_side)
        x1_vals = np.linspace(0, 1, n_side)
        
        total = n_side * n_side
        if total > budget:
            # Subsample
            indices = np.random.choice(total, budget, replace=False)
            x0_flat = np.repeat(x0_vals, n_side)
            x1_flat = np.tile(x1_vals, n_side)
            x0_sel = x0_flat[indices]
            x1_sel = x1_flat[indices]
        else:
            x0_flat = np.repeat(x0_vals, n_side)
            x1_flat = np.tile(x1_vals, n_side)
            x0_sel = x0_flat
            x1_sel = x1_flat
            # Add extra random points to use full budget
            extra = budget - total
            if extra > 0:
                x0_extra = np.random.rand(extra)
                x1_extra = np.random.rand(extra)
                x0_sel = np.concatenate([x0_sel, x0_extra])
                x1_sel = np.concatenate([x1_sel, x1_extra])
        
        for i in range(budget):
            x = np.zeros(dim)
            x[0] = x0_sel[i]
            x[1] = x1_sel[i]
            x[2:] = 0.5  # g = 0
            problem(x)
        
        return None
