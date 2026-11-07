"""Adaptive lattice search with L-BFGS-B polishing for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Adaptive lattice search with L-BFGS-B polishing for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        calls = [0]
        best_state = [float('inf'), np.zeros(dim)]
        rng = np.random.default_rng(12345)
        
        # All evaluated points for lattice-based exploration
        all_points = []
        
        def eval_and_track(x):
            if calls[0] >= budget or time.time() > deadline:
                return best_state[0]
            calls[0] += 1
            try:
                x_arr = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                val = float(func(x_arr))
                if val < best_state[0]:
                    best_state[0] = val
                    best_state[1] = x_arr.copy()
                all_points.append((val, x_arr.copy()))
                return val
            except Exception:
                return best_state[0]
        
        def lbfgsb_polish(x0, max_evals):
            """Run L-BFGS-B from x0 with limited evaluations."""
            if calls[0] >= budget or time.time() > deadline:
                return
            
            x_start = np.clip(np.asarray(x0, dtype=float), -5.0, 5.0)
            used = [0]
            
            def wrapper(x):
                if used[0] >= max_evals or calls[0] >= budget or time.time() > deadline:
                    return best_state[0]
                used[0] += 1
                return eval_and_track(x)
            
            try:
                result = minimize(wrapper, x_start, method='L-BFGS-B',
                                  bounds=[(-5.0, 5.0)] * dim,
                                  options={'maxiter': max_evals, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                pass
        
        # Phase 1: Structured initial exploration - stratified lattice
        # Use a coarse grid to quickly find promising regions
        n_coarse = min(40, budget // 10)
        if n_coarse > 0:
            # Create stratified points in the box
            per_dim = max(2, int(round(n_coarse ** (1.0 / dim))))
            grid_coords = np.linspace(-5.0, 5.0, per_dim)
            grid_points = np.array(np.meshgrid(*([grid_coords] * dim))).reshape(dim, -1).T
            # Shuffle and take n_coarse points
            idx = rng.permutation(len(grid_points))[:n_coarse]
            for i in idx:
                if calls[0] >= budget or time.time() > deadline:
                    break
                eval_and_track(grid_points[i])
        
        # Phase 2: Adaptive lattice refinement
        # Maintain a dynamic set of active points, refine the most promising region
        remaining = budget - calls[0]
        
        # Run multiple rounds of lattice refinement + polishing
        n_rounds = min(5, max(2, remaining // 80))
        per_round = max(20, remaining // n_rounds)
        
        for round_idx in range(n_rounds):
            if calls[0] >= budget or time.time() > deadline:
                break
            
            # Determine refinement region based on current best
            x_best = best_state[1]
            # Region size shrinks over rounds
            region_size = max(1.0, 5.0 * (0.6 ** round_idx))
            
            # Create a local lattice around the best point
            n_local = min(per_round, budget - calls[0])
            if n_local < 5:
                break
            
            per_dim_local = max(2, int(round(n_local ** (1.0 / dim))))
            # Center the grid on the best point
            center = np.clip(x_best, -5.0 + region_size/2, 5.0 - region_size/2)
            grid_center = np.linspace(center[0] - region_size/2, center[0] + region_size/2, per_dim_local)
            grid_points_local = np.array(np.meshgrid(*([grid_center] * dim))).reshape(dim, -1).T
            grid_points_local = np.clip(grid_points_local, -5.0, 5.0)
            
            # Shuffle and evaluate
            idx_local = rng.permutation(len(grid_points_local))[:n_local]
            for i in idx_local:
                if calls[0] >= budget or time.time() > deadline:
                    break
                eval_and_track(grid_points_local[i])
            
            # Polish the best point found so far
            if calls[0] < budget - 10 and time.time() < deadline:
                lbfgsb_polish(best_state[1], max(15, (budget - calls[0]) // 4))
            
            # Also try polishing a few other good points for diversity
            if len(all_points) > 5 and calls[0] < budget - 20 and time.time() < deadline:
                # Get top few distinct points
                sorted_points = sorted(all_points, key=lambda t: t[0])
                candidates = []
                for val, pt in sorted_points[:10]:
                    if np.linalg.norm(pt - best_state[1]) > 0.5:
                        candidates.append(pt)
                    if len(candidates) >= 2:
                        break
                for cand in candidates:
                    if calls[0] < budget - 10 and time.time() < deadline:
                        lbfgsb_polish(cand, max(10, (budget - calls[0]) // 5))
        
        # Phase 3: Final intensive polishing if budget remains
        if calls[0] < budget - 5 and time.time() < deadline:
            # Multiple L-BFGS-B restarts from best and nearby points
            n_final_polish = min(3, (budget - calls[0]) // 15)
            for i in range(n_final_polish):
                if calls[0] >= budget or time.time() > deadline:
                    break
                if i == 0:
                    start_pt = best_state[1].copy()
                else:
                    # Perturb the best point
                    start_pt = best_state[1] + 0.1 * rng.standard_normal(dim)
                    start_pt = np.clip(start_pt, -5.0, 5.0)
                lbfgsb_polish(start_pt, max(10, (budget - calls[0]) // max(1, n_final_polish - i)))
        
        # Phase 4: If we have budget left, do coordinate-wise local search
        if calls[0] < budget - 3 and time.time() < deadline:
            x_best = best_state[1].copy()
            step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
            for ss in step_sizes:
                if calls[0] >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    for direction in [-1, 1]:
                        if calls[0] >= budget or time.time() > deadline:
                            break
                        x_try = x_best.copy()
                        x_try[d] += direction * ss
                        x_try = np.clip(x_try, -5.0, 5.0)
                        val = eval_and_track(x_try)
                        if val < best_state[0]:
                            x_best = x_try.copy()
                            improved = True
                if not improved:
                    break
        
        return best_state[1], best_state[0]
# EVOLVE-BLOCK-END
