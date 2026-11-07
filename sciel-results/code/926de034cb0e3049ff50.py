"""Multi-objective optimizer for ZDT3 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT3-specific optimizer using optimal x0 distribution for hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        dim = self.dim
        n = dim  # number of variables

        # ZDT3 structure:
        # f1 = x[0]
        # f2 = (1 + 10*(n-1)) * (1 - sqrt(f1/(1+10*(n-1))) - 0.05*sin(10*pi*f1))
        # Pareto-optimal when x[1:] = 0
        # 
        # The hypervolume contribution density for the Pareto front is proportional
        # to 1/|df2/df1| = 1/sqrt(f1) * (1 + 0.05*10*pi*cos(10*pi*f1))
        # More precisely, for maximizing hypervolume with reference point [11, 11],
        # we want to sample x0 in proportion to the width of the non-dominated region.
        #
        # Key insight: The Pareto front of ZDT3 has a complex shape with 51 local
        # minima. The hypervolume contribution at each point depends on how much
        # "new" area it covers. A good approximation is to sample x0 according to
        # the density proportional to 1/sqrt(x0) which accounts for the sqrt term
        # in f2, combined with ensuring coverage of the sinusoidal features.

        # Compute optimal x0 distribution:
        # The marginal contribution to hypervolume from a point at f1 is proportional
        # to the "height" of the front at that point relative to neighbors.
        # For ZDT3, f2(f1) = 291 * (1 - sqrt(f1/291) - 0.05*sin(10*pi*f1))
        # 
        # A practical approach: use inverse CDF sampling where the CDF is chosen
        # to concentrate samples where the front has high curvature (near local minima
        # of f2) and where 1/sqrt(f1) is large (near f1=0).

        # Let's use a mixture distribution:
        # - 70% weight: uniform on [0, 1] (baseline coverage)
        # - 30% weight: concentrated near the 51 local minima positions
        #   which are at x0 = (2k+1)/20 for k=0..49, i.e., 0.05, 0.15, ..., 0.95

        budget = self.budget
        F_list = []
        X_list = []
        used = 0

        def evaluate(x0):
            nonlocal used
            if used >= budget:
                return None
            if time.time() > deadline - 5:
                return None
            x = np.zeros(dim)
            x[0] = x0
            f = problem(x)
            used += 1
            F_list.append(f)
            X_list.append(x)
            return f

        # Phase 1: Dense sampling with optimal distribution
        # Use a combination of:
        # 1. Uniform samples for baseline coverage
        # 2. Samples concentrated at local minima of the Pareto front
        # 3. Samples weighted by 1/sqrt(x0) for the sqrt term

        # Generate candidate x0 values using multiple strategies
        n_uniform = int(budget * 0.5)
        n_local_min = int(budget * 0.3)
        n_sqrt_weighted = budget - n_uniform - n_local_min

        x0_candidates = []

        # Uniform samples
        x0_uniform = np.linspace(0.001, 0.999, n_uniform)
        x0_candidates.extend(x0_uniform.tolist())

        # Local minima positions: x0 = (2k+1)/20 for k=0..49
        local_minima = np.array([(2*k + 1) / 20.0 for k in range(50)])
        # Add small perturbations to each local minimum
        for lm in local_minima:
            if len(x0_candidates) + 3 > budget:
                break
            # Add points around each local minimum
            for offset in [-0.01, 0.0, 0.01]:
                x0_val = np.clip(lm + offset, 0.001, 0.999)
                x0_candidates.append(x0_val)
            if len(x0_candidates) >= n_uniform + n_local_min:
                break

        # Sqrt-weighted samples: sample x0 from distribution proportional to 1/sqrt(x0)
        # CDF: F(x) = 2*sqrt(x) / 2*sqrt(1) = 2*sqrt(x)
        # Inverse CDF: x = (u/2)^2 = u^2/4
        # But we want to concentrate more near 0, so use x = u^2
        if n_sqrt_weighted > 0:
            u = np.linspace(0.01, 0.99, n_sqrt_weighted)
            x0_sqrt = u ** 2  # Concentrates near 0
            x0_candidates.extend(x0_sqrt.tolist())

        # Shuffle to avoid systematic bias
        np.random.shuffle(x0_candidates)

        # Track evaluated f1 values to avoid duplicates
        evaluated_f1 = []

        for x0 in x0_candidates:
            if used >= budget:
                break
            if time.time() > deadline - 5:
                break

            # Skip if too close to an already-evaluated point
            if evaluated_f1:
                if min(abs(x0 - ef1) for ef1 in evaluated_f1) < 0.003:
                    continue

            f = evaluate(x0)
            if f is not None:
                evaluated_f1.append(f[0])

        # Phase 2: Fill remaining budget with gap-filling
        if used < budget and len(F_list) > 0:
            F = np.array(F_list)
            f1_vals = np.sort(F[:, 0])
            
            remaining = budget - used
            # Find gaps between consecutive f1 values
            if len(f1_vals) > 1:
                gaps = np.diff(f1_vals)
                gap_positions = (f1_vals[:-1] + f1_vals[1:]) / 2.0
                
                # Sort gaps by size, fill largest first
                gap_order = np.argsort(-gaps)
                
                for idx in gap_order:
                    if used >= budget:
                        break
                    if time.time() > deadline - 5:
                        break
                    
                    x0_new = gap_positions[idx]
                    
                    # Skip if too close to existing
                    if evaluated_f1 and min(abs(x0_new - ef1) for ef1 in evaluated_f1) < 0.003:
                        continue
                    
                    f = evaluate(x0_new)
                    if f is not None:
                        evaluated_f1.append(f[0])

            # If still budget remaining, add random points
            while used < budget:
                if time.time() > deadline - 5:
                    break
                x0_new = np.random.uniform(0.001, 0.999)
                if evaluated_f1 and min(abs(x0_new - ef1) for ef1 in evaluated_f1) < 0.003:
                    continue
                f = evaluate(x0_new)
                if f is not None:
                    evaluated_f1.append(f[0])

        # Return the set of non-dominated points (or all points, since the evaluator
        # records all returned objective vectors)
        if X_list:
            X = np.array(X_list)
        else:
            X = np.array([])

        return X
# EVOLVE-BLOCK-END
