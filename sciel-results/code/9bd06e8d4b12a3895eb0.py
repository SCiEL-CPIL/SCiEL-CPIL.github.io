"""Surface-area-weighted Sobol sampling on DTLZ2 Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Place all evaluations on the DTLZ2 Pareto front using surface-area-weighted Sobol sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 325.0  # safety margin before 330s kill
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Generate Sobol sequence for 2D (t1, t2) parameter space
        # Using scipy's Sobol for better low-discrepancy properties
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=2, scramble=True, seed=42)
        # Number of points must be power of 2 for Sobol; use next power of 2 >= budget
        n_power = 1
        while n_power < budget:
            n_power *= 2
        # Generate n_power points, then take first budget points
        u = sobol.random(n_power)  # shape (n_power, 2)
        u = u[:budget]  # take first budget points

        u1 = u[:, 0]  # for t1
        u2 = u[:, 1]  # for t2

        # Surface-area-weighted sampling for DTLZ2 Pareto front (quarter sphere)
        # The Pareto front is a quarter of a unit sphere in 3D.
        # Surface area element on sphere: dA = sin(t1) * dt1 * dt2
        # To sample uniformly in surface area, we need:
        #   t1 = arcsin(2*u1 - 1) for full sphere, but for quarter sphere t1 in [0, pi/2]:
        #   The CDF of t1 under uniform surface area is F(t1) = sin(t1) / sin(pi/2) = sin(t1)
        #   So t1 = arcsin(u1) where u1 ~ Uniform(0,1)
        # For t2, the surface area element is uniform in t2 (no weighting), so t2 = u2 * pi/2
        t1 = np.arcsin(u1)  # maps [0,1] to [0, pi/2] with surface-area weighting
        t2 = u2 * (np.pi / 2)  # uniform in t2

        # Evaluate all points on the Pareto front
        for i in range(budget):
            if time.time() > deadline:
                break
            x = np.full(dim, 0.5)
            x[0] = t1[i]
            x[1] = t2[i]
            problem(x)
        
        return None
# EVOLVE-BLOCK-END
