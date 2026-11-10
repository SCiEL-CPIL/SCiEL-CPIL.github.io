"""Multi-objective optimizer for ZDT4 with adaptive Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer exploiting the structure of ZDT4.
    
    Key insight: For ZDT4, f1 depends only on x[0], and the Pareto front
    is achieved when x[1:] minimizes g(x[1:]). We sample x[0] values
    to cover the front, and use a local search to find the optimal tail
    for each x[0] value.
    
    Strategy:
    1. Use a small number of evaluations to identify a good tail (x[1:])
    2. Use remaining budget to sample x[0] values with that tail
    3. Use a non-uniform distribution that clusters near the ends of the
       x[0] range where the front has higher curvature
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 315.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        dim = self.dim

        if dim < 2:
            # Degenerate case
            x = np.clip(lo, lo, hi)
            problem(x)
            return None

        # Phase 1: Find a good tail (x[1:]) that minimizes g
        # For standard ZDT4, g is minimized at x[1:] = 0
        # Try a few candidates and pick the one that gives lowest f2
        n_tail_trials = 5
        tail_candidates = []
        
        # Candidate 1: all zeros (clipped to bounds)
        tail0 = np.full(dim - 1, 0.0)
        tail0 = np.clip(tail0, lo[1:], hi[1:])
        tail_candidates.append(tail0)
        
        # Candidate 2: midpoints of bounds
        tail_mid = (lo[1:] + hi[1:]) / 2.0
        tail_candidates.append(tail_mid)
        
        # Candidates 3-5: random points in the tail space
        for _ in range(3):
            tail_rand = lo[1:] + np.random.rand(dim - 1) * (hi[1:] - lo[1:])
            tail_candidates.append(tail_rand)

        # Evaluate each tail candidate at a fixed x0 (e.g., midpoint)
        x0_test = (lo[0] + hi[0]) / 2.0
        best_tail = tail_candidates[0]
        best_f2 = np.inf

        for tail in tail_candidates:
            if time.time() > deadline:
                break
            x = np.concatenate(([x0_test], tail))
            x = np.clip(x, lo, hi)
            try:
                f = problem(x)
                f2 = f[1] if len(f) > 1 else np.inf
                if f2 < best_f2:
                    best_f2 = f2
                    best_tail = tail.copy()
            except Exception:
                continue

        # Phase 2: Local refinement of the tail
        # Use a few evaluations to try to improve the tail
        n_refine = 10
        current_tail = best_tail.copy()
        current_f2 = best_f2
        step_size = 0.1 * (hi[1:] - lo[1:])
        
        for _ in range(n_refine):
            if time.time() > deadline:
                break
            # Try perturbing each dimension
            improved = False
            for d in range(dim - 1):
                for direction in [-1, 1]:
                    new_tail = current_tail.copy()
                    new_tail[d] += direction * step_size[d]
                    new_tail = np.clip(new_tail, lo[1:], hi[1:])
                    x = np.concatenate(([x0_test], new_tail))
                    x = np.clip(x, lo, hi)
                    try:
                        f = problem(x)
                        f2 = f[1] if len(f) > 1 else np.inf
                        if f2 < current_f2:
                            current_f2 = f2
                            current_tail = new_tail.copy()
                            improved = True
                    except Exception:
                        continue
            if not improved:
                step_size *= 0.5
                if np.max(step_size) < 1e-10:
                    break

        best_tail = current_tail

        # Phase 3: Sample x[0] values with the best tail
        # Use remaining budget for x[0] sampling
        evals_used = n_tail_trials + n_refine * (dim - 1) * 2  # approximate
        remaining = self.budget - evals_used
        if remaining < 10:
            remaining = self.budget - n_tail_trials  # fallback

        n_sweep = max(remaining, 50)

        # Use a non-uniform distribution for x[0]
        # The Pareto front for ZDT4 has higher curvature near the ends
        # Use a combination of clustered and uniform samples
        n_clustered = int(n_sweep * 0.6)
        n_uniform = n_sweep - n_clustered

        # Clustered near both ends: use |2u - 1|^0.5 transformation
        u_c = np.linspace(0.0, 1.0, n_clustered, endpoint=True)
        # Transform to cluster near 0 and 1
        u_c_transformed = np.where(u_c < 0.5, 
                                    (2 * u_c) ** 0.5 / 2,
                                    1 - (2 * (1 - u_c)) ** 0.5 / 2)
        x0_clustered = lo[0] + (hi[0] - lo[0]) * u_c_transformed

        # Uniform samples
        u_u = np.linspace(0.0, 1.0, n_uniform, endpoint=True)
        x0_uniform = lo[0] + (hi[0] - lo[0]) * u_u

        # Combine and sort
        x0_samples = np.concatenate([x0_clustered, x0_uniform])
        x0_samples = np.sort(x0_samples)

        # Remove near-duplicates
        if len(x0_samples) > 1:
            range_x0 = hi[0] - lo[0]
            if range_x0 > 0:
                tol = range_x0 * 1e-9
                mask = np.ones(len(x0_samples), dtype=bool)
                for i in range(1, len(x0_samples)):
                    if x0_samples[i] - x0_samples[i - 1] < tol:
                        mask[i] = False
                x0_samples = x0_samples[mask]

        # Evaluate all x[0] samples with the best tail
        for i in range(len(x0_samples)):
            if time.time() > deadline:
                break

            x0 = x0_samples[i]
            x = np.concatenate(([x0], best_tail))
            x = np.clip(x, lo, hi)

            try:
                problem(x)
            except Exception:
                break

        return None
# EVOLVE-BLOCK-END
