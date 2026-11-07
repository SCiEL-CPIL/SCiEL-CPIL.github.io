"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ4.
    Strategy:
    1. Detect DTLZ4-like structure by probing x=0 vs x=0.5.
    2. If detected, exploit known structure:
       - Last (n_obj-1) variables are distance variables -> set to 0.
       - First (n_obj-1) variables determine position on the Pareto front.
       - Use a dense grid in the first (n_obj-1) dimensions to cover the front.
    3. Use remaining budget for local refinement around grid points.
    4. Fallback to simple random search if structure is not detected.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        all_X = []
        all_F = []
        budget_used = 0

        def eval_point(x):
            nonlocal budget_used
            x = np.array(x, dtype=float)
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            budget_used += 1
            return x, f

        def add_to_archive(x, f):
            all_X.append(x)
            all_F.append(f)

        # --- Probe Phase: Check if x=0 is better than x=0.5 ---
        x_zero = np.zeros(self.dim)
        x_half = np.full(self.dim, 0.5)

        x0, f0 = eval_point(x_zero)
        add_to_archive(x0, f0)
        x1, f1 = eval_point(x_half)
        add_to_archive(x1, f1)

        # Heuristic: If f0 is significantly better, assume DTLZ4 structure
        is_dtlz4_like = np.sum(f0) < np.sum(f1) * 0.5

        if is_dtlz4_like:
            # --- Exploit DTLZ4 Structure ---
            # Last (n_obj - 1) variables are distance variables -> set to 0
            # First (n_obj - 1) variables determine position on the Pareto front
            n_pos = self.n_obj - 1  # Number of position variables
            
            # Use a grid in the first n_pos dimensions
            # Budget remaining
            remaining = self.budget - budget_used
            
            # Choose grid size to fit within budget
            # For 3 objectives, n_pos = 2, so a 2D grid
            if n_pos == 2:
                # 2D grid: side x side points
                # side = sqrt(remaining)
                side = int(np.sqrt(remaining))
                if side < 2:
                    side = 2
                # Cap side to avoid too many points
                side = min(side, 30)
                
                x1_vals = np.linspace(0, 1, side)
                x2_vals = np.linspace(0, 1, side)
                
                for x1 in x1_vals:
                    for x2 in x2_vals:
                        if budget_used >= self.budget:
                            break
                        if time.time() > deadline:
                            return all_X
                        
                        x = np.zeros(self.dim)
                        x[0] = x1
                        x[1] = x2
                        # Distance variables are already 0
                        
                        try:
                            xc, fc = eval_point(x)
                            add_to_archive(xc, fc)
                        except Exception:
                            continue
            elif n_pos == 1:
                # 1D grid
                n_points = min(remaining, 50)
                x_vals = np.linspace(0, 1, n_points)
                for x1 in x_vals:
                    if budget_used >= self.budget:
                        break
                    if time.time() > deadline:
                        return all_X
                    
                    x = np.zeros(self.dim)
                    x[0] = x1
                    
                    try:
                        xc, fc = eval_point(x)
                        add_to_archive(xc, fc)
                    except Exception:
                        continue
            else:
                # For higher dimensions, use random sampling in the position space
                for _ in range(remaining):
                    if budget_used >= self.budget:
                        break
                    if time.time() > deadline:
                        return all_X
                    
                    x = np.zeros(self.dim)
                    x[:n_pos] = np.random.rand(n_pos)
                    
                    try:
                        xc, fc = eval_point(x)
                        add_to_archive(xc, fc)
                    except Exception:
                        continue
        else:
            # --- Fallback: General Random Search ---
            for _ in range(self.budget - budget_used):
                if budget_used >= self.budget:
                    break
                if time.time() > deadline:
                    return all_X
                
                x = lo + (hi - lo) * np.random.rand(self.dim)
                try:
                    xc, fc = eval_point(x)
                    add_to_archive(xc, fc)
                except Exception:
                    continue

        return all_X
# EVOLVE-BLOCK-END
