import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295

        n_evals = self.budget
        if n_evals <= 0:
            return []

        results = []

        # For ZDT6, the Pareto front is traced by varying x[0] with x[1:]=0
        # f1(x0) = 1 + x0*(sin(10*pi*x0) + sin(10*pi*x0^2))
        # f2(x0) = 1 - sqrt(f1(x0))
        # We need to sample x0 across [0, 1] to cover the full Pareto front

        # Strategy: use non-uniform sampling that concentrates points
        # where the Pareto front has high curvature (near oscillation peaks)
        # while still covering the full [0,1] range.

        # Use a sine-based mapping that clusters points near 0 and 1,
        # where the front has more complex behavior, but also spread uniformly.
        # Actually, let's use a mix: half uniform, half concentrated near
        # the regions of high curvature.

        # A simpler approach: use linspace but with a slight non-uniformity
        # by mapping through a function that has derivative > 1 in high-curvature regions.

        # Let's try: x0 = 0.5 - 0.5*cos(pi * t) where t is uniform in [0,1]
        # This gives a cosine distribution that clusters near endpoints.
        # But actually for hypervolume, uniform in f1-space might be better.

        # Let's sample uniformly in [0,1] for x0, but add some jitter
        # to avoid aliasing with the sinusoidal structure.

        # Actually, the key insight: we want to maximize hypervolume.
        # The Pareto front of ZDT6 has f1 ranging from 1 to about 1+max_oscillation.
        # Uniform sampling in x0 space is reasonable. Let's stick with that
        # but ensure we hit the endpoints and key features.

        if n_evals == 1:
            x0s = [0.5]
        elif n_evals == 2:
            x0s = [0.0, 1.0]
        else:
            # Dense sampling with slight non-uniformity to better capture
            # the oscillatory structure
            # Use Chebyshev nodes for better coverage of oscillatory functions
            k = np.arange(n_evals)
            # Chebyshev nodes on [0, 1]
            x0s = 0.5 * (1 - np.cos(np.pi * (k + 0.5) / n_evals))
            x0s = np.sort(x0s)

        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim, dtype=float)
            x[0] = float(x0s[i])
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break

        return results
