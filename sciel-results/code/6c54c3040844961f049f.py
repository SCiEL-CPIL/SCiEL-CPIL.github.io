"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Multi-objective Optimizer using Sobol Sampling and Pareto Front Extraction."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin against 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        eval_count = 0
        all_points = []
        all_objs = []

        # 1. Generate a set of deterministic, low-discrepancy sample points
        # Sobol sequences provide excellent coverage of the unit hypercube.
        # We use the full budget (200) for sampling.
        # qmc.Sobol requires the number of points to be a power of 2 for best properties,
        # but we can just take the first 200 points or use a power of 2 close to 200 (e.g., 128 or 256).
        # Given budget is exactly 200, let's use 128 points to be safe and fast, 
        # or perhaps use all 200 if we generate enough.
        # Actually, Sobol generator can produce any number, but statistical properties are best for powers of 2.
        # Let's use 128 points. This leaves 72 evaluations unused? 
        # No, the task says "More than 200 evaluations... makes the whole result invalid".
        # It does not say we MUST use exactly 200. It says budget=200.
        # Using fewer evaluations is valid as long as we don't exceed.
        # However, hypervolume is calculated on *recorded* vectors.
        # More points usually means better hypervolume coverage.
        # Let's use 128 Sobol points. This is a very robust global sampling strategy.
        
        # Alternatively, we can mix: 100 Sobol + 100 Random?
        # Sobol is deterministic and non-random, which is good for reproducibility across seeds 
        # (though the evaluator seeds np.random, Sobol doesn't use it unless we randomize the offset).
        # The problem instance is fixed (RE37), so deterministic sampling is ideal.
        
        n_samples = 128
        sobol = qmc.Sobol(d=dim, scramble=False)
        # Generate points in [0, 1]^d
        # scramble=False ensures determinism. The evaluator seeds np.random, but Sobol doesn't use it.
        # This makes the result independent of the seed? 
        # Wait, the scoring is "mean over the two seeds". If the result is identical for both seeds,
        # that's fine.
        
        # However, if we want to leverage the seed, we could scramble the Sobol sequence.
        # qmc.Sobol(d=dim, scramble=True, seed=seed)
        # But we don't have access to the seed in __init__ or __call__ directly, 
        # although np.random.seed(seed) is called before.
        # We can get the current state or just use a fixed seed for Sobol to be safe?
        # If we use scramble=True, we should use a seed derived from np.random.
        # Let's stick to scramble=False for pure deterministic coverage first.
        # If the problem is non-convex, Sobol might miss narrow basins, but with 128 points in 4D,
        # it's a solid baseline.
        
        # Let's try to use all 200 budget.
        # Sobol points are best for powers of 2. 128 is 2^7. 256 is 2^8.
        # If we use 256, we exceed budget (200).
        # So we must use <= 200.
        # Let's use 128 Sobol points.
        
        samples = sobol.random(n_samples)
        
        # Scale from [0, 1] to [lo, hi]
        samples = lo + samples * (hi - lo)
        
        # 2. Evaluate all samples
        # We can vectorize if problem supports it, but problem(x) takes a single vector.
        # We must loop.
        for x in samples:
            if eval_count >= self.budget:
                break
            if time.time() > deadline:
                break
            try:
                f = np.array(problem(x))
                eval_count += 1
                all_points.append(x)
                all_objs.append(f)
            except Exception:
                eval_count += 1
                continue
        
        # If we have budget left, we can add random points to fill the gap?
        # 200 - 128 = 72 points.
        # Let's add random points to utilize the full budget.
        # This combines global structured coverage with stochastic exploration.
        while eval_count < self.budget:
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.rand(dim)
            try:
                f = np.array(problem(x))
                eval_count += 1
                all_points.append(x)
                all_objs.append(f)
            except Exception:
                eval_count += 1
                continue

        return None
# EVOLVE-BLOCK-END
