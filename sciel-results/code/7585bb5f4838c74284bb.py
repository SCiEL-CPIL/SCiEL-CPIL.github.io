# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    Strategy:
    1. Use a deterministic Sobol sequence to generate initial candidates in [0, 1]^30.
       This provides better space-filling properties than random or simple grid sampling,
       increasing the chance of hitting the narrow "valleys" where g(x) is minimal.
    2. Evaluate all candidates.
    3. Identify the best points (based on a simple scalarization or distance to origin)
       and perform a local refinement around them to ensure we densely cover the 
       non-dominated segments.
    4. Since the budget is small (200), we allocate most to the initial Sobol spread 
       and use the remainder for local perturbations of the best found points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_sequence(self, n, dim):
        """
        Generate n points in [0, 1]^dim using a simple Sobol-like sequence.
        Since scipy.stats.qmc is available, we use it for robustness.
        """
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True)
            # We need n points. qmc requires power of 2 for base, but can generate any n.
            # However, for small n, simple Halton or just using the first n points of a larger set is fine.
            # Let's generate a power of 2 >= n to be safe with some implementations, or just n.
            points = sampler.random(n)
            return points
        except Exception:
            # Fallback to Halton if Sobol fails
            def halton(n, dim):
                def halton_index(i, base):
                    f = 1.0
                    r = 0.0
                    while i > 0:
                        f /= base
                        r += f * (i % base)
                        i //= base
                    return r
                points = np.zeros((n, dim))
                for j in range(dim):
                    base = 2 + j  # primes: 2, 3, 5, ...
                    for i in range(n):
                        points[i, j] = halton_index(i + 1, base)
                return points
            return halton(n, dim)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # We will use a mix of global exploration and local exploitation.
        # ZDT3 has 7 disconnected Pareto front segments.
        # A pure random or grid search might miss the narrow valleys where g is low.
        # Sobol sequence helps, but let's also explicitly try to find low g points.
        
        # Split budget: 150 for global Sobol, 50 for local refinement
        n_global = 150
        n_local = budget - n_global
        
        # Generate global points
        points_global = self._sobol_sequence(n_global, dim)
        # Map to bounds
        X_global = points_global * (hi - lo) + lo
        
        # Store evaluated points and their objectives
        X_all = []
        F_all = []
        
        # Evaluate global points
        for i in range(n_global):
            if time.time() > deadline:
                break
            x = X_global[i]
            f = problem(x)
            X_all.append(x)
            F_all.append(f)
        
        if not X_all:
            return np.array(X_all)
            
        X_arr = np.array(X_all)
        F_arr = np.array(F_all)
        
        # Find the best point(s) to use as seeds for local refinement.
        # We want points that are close to the Pareto front.
        # A simple proxy is the sum of objectives or distance to (0,0).
        # However, ZDT3's front is not at (0,0) in objective space for all x1.
        # But g(x) >= 1, so f1 >= 0 and f2 >= 0.
        # Points with low f1 + f2 are likely on or near the front.
        
        # Sort by sum of objectives
        sums = F_arr[:, 0] + F_arr[:, 1]
        # Take the top 5 best points for local refinement
        top_k = min(5, len(X_arr))
        top_indices = np.argsort(sums)[:top_k]
        
        # Generate local perturbations around these best points
        # Use a small Gaussian perturbation
        # The width of the valleys in ZDT3 is roughly 0.05-0.1 in decision space for some variables.
        # Let's use a sigma of 0.05.
        sigma = 0.05
        
        X_local = []
        for idx in top_indices:
            center = X_arr[idx]
            # Generate n_local // top_k points around this center
            n_per_center = n_local // top_k
            if n_per_center == 0:
                continue
                
            # Generate random directions
            random_vecs = np.random.randn(n_per_center, dim)
            # Perturb
            X_pert = center + random_vecs * sigma
            # Clip to bounds
            X_pert = np.clip(X_pert, lo, hi)
            X_local.append(X_pert)
            
        if X_local:
            X_local = np.vstack(X_local)
            # Ensure we don't exceed budget
            if len(X_local) > n_local:
                X_local = X_local[:n_local]
            
            # Evaluate local points
            for i in range(len(X_local)):
                if time.time() > deadline:
                    break
                x = X_local[i]
                f = problem(x)
                X_all.append(x)
                F_all.append(f)
        
        # Return the array of all evaluated points (though return value is ignored, 
        # it's good practice to return something meaningful if the interface expects it)
        return np.array(X_all)

# EVOLVE-BLOCK-END
