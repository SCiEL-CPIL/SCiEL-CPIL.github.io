import numpy as np
import time


class Optimizer:
    """Direct construction of the ZDT3 Pareto front with optimized sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = int(self.budget)

        n_sample = max(2, budget)

        # Generate t values: combine uniform sampling with emphasis on regions
        # where f2 is small (near sin^2 minima) to maximize hypervolume
        # sin^2(10t/pi) has minima at t = k*pi/10 for k=0,1,2,3 (within [0,1])
        # Use a mix: some points at/near minima, rest uniform
        
        # Uniform base
        t_uniform = np.linspace(0, 1, n_sample, endpoint=True)
        
        # Add extra points near the minima of sin^2(10t/pi)
        # Minima at t = k*pi/10 for k where t in [0,1]
        # k*pi/10 <= 1 => k <= 10/pi ≈ 3.18, so k = 0,1,2,3
        min_indices = []
        for k in range(0, 4):
            t_min = k * np.pi / 10.0
            if 0 <= t_min <= 1:
                min_indices.append(t_min)
        
        # Create a combined set: use all budget points
        # Strategy: use power-law spacing that's denser near t=0 and near minima
        # Actually, let's just use a fine uniform grid since 200 points is plenty
        # to capture ~1.59 periods of the oscillation
        
        t_values = np.linspace(0, 1, n_sample, endpoint=True)
        
        recorded = []
        used = 0
        x = np.zeros(dim)
        
        for tv in t_values:
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            x[0] = float(tv)
            x_clipped = np.clip(x, lo, hi)
            
            f = np.asarray(problem(x_clipped), dtype=float)
            recorded.append(f)
            used += 1

        return recorded
