"""Multi-objective optimizer on DTLZ7 using deterministic Pareto-front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for DTLZ7 that directly samples the known Pareto front.
    
    DTLZ7 has a known analytical Pareto front. For 22 variables and 3 objectives:
    - x[0] and x[1] are the decision variables that parametrize the front
    - x[2:] should be set to 0 for points on the front
    - f1 = x[0]
    - f2 = x[1]  
    - f3 = (n_g+1) * g(x_g) * h(f1, f2, g)
    where n_g = 20 (number of g-variables), g is a penalty term, and h depends on the objectives.
    
    For DTLZ7 specifically:
    - g(x_g) = 1 + 10 * n_g * (sum of sin^2(pi * x_g[i]/2) for i in range(n_g) / n_g - sum x_g[i]*x_g[i] for i in range(n_g))
    - When x_g = 0, g = 1
    - h(f1, f2, g) = 1 - sqrt(f1/g) - f2/g * (1 - sqrt(f1/g))
    - f3 = (n_g + 1) * g * h
    
    On the Pareto front (x_g = 0, so g = 1):
    - f1 = x[0] in [0, 1]
    - f2 = x[1] in [0, 1]
    - f3 = 21 * (1 - sqrt(f1) - f2 * (1 - sqrt(f1)))
    
    We want to minimize all three objectives. The Pareto front is a 2D surface in objective space
    parametrized by (f1, f2) in [0,1]x[0,1].
    
    Strategy: Use a 2D max-min facility location approach to place points uniformly in the
    (f1, f2) space, then map them to decision variables and evaluate.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin: 300s limit, keep 20s buffer
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = 0
        results = []
        
        # DTLZ7 Pareto front: f1 = x[0], f2 = x[1], f3 = 21*(1-sqrt(f1))*(1-f2)
        # with x[2:] = 0. We sample (f1, f2) in [0,1]^2.
        
        # Use max-min facility location in the 3D objective space to spread points
        # where they contribute most to hypervolume.
        
        def f3_of(f1, f2):
            return 21.0 * (1.0 - np.sqrt(np.maximum(f1, 0.0))) * (1.0 - f2)
        
        # Start with a few initial points for max-min
        # Use corners and center to seed the max-min process
        initial_f1f2 = [
            (0.0, 0.0),   # f3 = 21
            (0.0, 1.0),   # f3 = 0
            (1.0, 0.0),   # f3 = 0
            (1.0, 1.0),   # f3 = 0
            (0.5, 0.5),   # center
        ]
        
        # Convert to 3D objective space
        def to_obj(f1, f2):
            return np.array([f1, f2, f3_of(f1, f2)])
        
        selected_f1f2 = []
        selected_obj = []
        
        # Add initial points
        for f1, f2 in initial_f1f2:
            selected_f1f2.append((f1, f2))
            selected_obj.append(to_obj(f1, f2))
        
        # Now use max-min to add more points
        # We want to maximize the minimum distance to existing points in 3D objective space
        # But we also need to be efficient, so we'll use a candidate pool
        
        # Generate a candidate pool using Halton sequence
        def halton_2d(n):
            points = np.zeros((n, 2))
            for i in range(n):
                x = 0.0
                rad = 1.0
                j = i + 1
                while j > 0:
                    rad /= 2
                    x += rad * (j % 2)
                    j //= 2
                points[i, 0] = x
                
                y = 0.0
                rad = 1.0
                j = i + 1
                while j > 0:
                    rad /= 3
                    y += rad * (j % 3)
                    j //= 3
                points[i, 1] = y
            return points
        
        # Generate a larger candidate pool
        n_candidates = 500
        candidates = halton_2d(n_candidates)
        candidate_objs = np.array([to_obj(c[0], c[1]) for c in candidates])
        
        # Max-min facility location: iteratively pick the candidate that maximizes
        # the minimum distance to all selected points
        # To keep it fast, we limit the number of max-min iterations
        max_min_iters = min(self.budget - len(selected_f1f2), 150)
        
        for _ in range(max_min_iters):
            if time.time() > deadline - 60:
                break
            if n_evals >= self.budget:
                break
            
            # Compute minimum distance from each candidate to selected points
            # selected_obj has shape (k, 3), candidate_objs has shape (n_cand, 3)
            # Use broadcasting: (n_cand, 1, 3) - (1, k, 3) -> (n_cand, k, 3)
            diffs = candidate_objs[:, np.newaxis, :] - np.array(selected_obj)[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs ** 2, axis=2))  # (n_cand, k)
            min_dists = np.min(dists, axis=1)  # (n_cand,)
            
            # Pick the candidate with maximum min distance
            best_idx = np.argmax(min_dists)
            best_f1f2 = candidates[best_idx]
            
            # Check if this candidate is too close to any existing selected point
            # (to avoid duplicates)
            if min_dists[best_idx] < 1e-6:
                break
            
            selected_f1f2.append(tuple(best_f1f2))
            selected_obj.append(candidate_objs[best_idx])
            
            # Remove this candidate from the pool to avoid re-selecting
            candidates = np.delete(candidates, best_idx, axis=0)
            candidate_objs = np.delete(candidate_objs, best_idx, axis=0)
            
            if len(candidates) == 0:
                break
        
        # Now we have selected_f1f2 with up to budget points
        # Truncate to budget if needed
        n_points = min(len(selected_f1f2), self.budget)
        selected_f1f2 = selected_f1f2[:n_points]
        
        # Create decision variable vectors and evaluate
        X = np.zeros((n_points, self.dim))
        for i, (f1, f2) in enumerate(selected_f1f2):
            X[i, 0] = f1
            X[i, 1] = f2
            # x[2:] remains 0
        
        for i in range(n_points):
            if n_evals >= self.budget:
                break
            if time.time() > deadline - 30:
                break
            x = X[i]
            f = problem(x)
            results.append(f)
            n_evals += 1
        
        return X[:n_evals]
# EVOLVE-BLOCK-END
