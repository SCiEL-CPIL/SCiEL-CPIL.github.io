"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f15 (Rastrigin rotated).
    Strategy:
    1. Sobol quasi-random scan to identify promising basins.
    2. (1+1)-ES with adaptive step size for local refinement.
    3. Event-driven restarts to escape local minima.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        calls_used = 0

        def f(x):
            nonlocal calls_used
            if calls_used >= budget:
                return 1e10
            calls_used += 1
            return func(x)

        # --- Phase 1: Sobol Quasi-Random Scan ---
        # Systematic coverage to identify multiple promising basins.
        def sobol_sequence(n, dim, seed=42):
            try:
                from scipy.stats import qmc
                sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
                points = sampler.random(n)
                return points
            except ImportError:
                # Fallback: Halton sequence
                def halton(index, base):
                    f = 1.0
                    r = 0.0
                    while index > 0:
                        f /= base
                        r += f * (index % base)
                        index //= base
                    return r
                points = np.zeros((n, dim))
                for d in range(dim):
                    base = d + 2
                    for i in range(n):
                        points[i, d] = halton(i + 1, base)
                return points

        n_sobol = min(50, budget // 10)
        if n_sobol < 5:
            n_sobol = 5

        sobol_points = sobol_sequence(n_sobol, dim)
        sobol_points = sobol_points * 10.0 - 5.0  # Map [0,1] to [-5,5]

        candidates = []
        for pt in sobol_points:
            if calls_used >= budget or time.time() > deadline:
                break
            pt_clipped = np.clip(pt, -5, 5)
            fp = f(pt_clipped)
            candidates.append((fp, pt_clipped.copy()))

        if not candidates:
            best_x = np.zeros(dim)
            best_f = f(best_x)
            candidates = [(best_f, best_x)]

        # Sort candidates by fitness (best first)
        candidates.sort(key=lambda x: x[0])
        
        global_best_x = candidates[0][1].copy()
        global_best_f = candidates[0][0]

        # --- Phase 2: (1+1)-ES with 1/5 Success Rule and Restarts ---
        # We maintain a list of starting points (from Sobol scan).
        # We run ES from the best candidate. If it stagnates, we switch to the next best candidate.
        
        # To avoid wasting budget on a very poor candidate, we only consider the top K candidates for restarts.
        K = 5
        restart_candidates = [c[1] for c in candidates[:K]]
        restart_fits = [c[0] for c in candidates[:K]]
        
        current_idx = 0
        x = restart_candidates[current_idx].copy()
        fx = restart_fits[current_idx]
        
        # Initialize sigma based on distance between best two points or default
        if len(restart_candidates) > 1:
            sigma = np.linalg.norm(restart_candidates[0] - restart_candidates[1]) / 2.0
            sigma = np.clip(sigma, 0.5, 5.0)
        else:
            sigma = 2.0

        rng = np.random.default_rng(0)
        
        # 1/5 success rule parameters
        # We track a window of the last 5 steps. If success rate > 0.5, increase sigma, else decrease.
        # For simplicity and robustness, we use a simpler adaptive rule often effective in (1+1)-ES:
        # Increase sigma by 1.5 on success, decrease by 0.87 on failure.
        # This is a standard heuristic for (1+1)-ES.
        
        # Stagnation threshold: number of steps without improvement before triggering a restart
        # This is a structural parameter. 30-50 is typical for dim=5.
        stagnation_limit = 40 
        current_stagnation = 0
        
        # We will loop until budget or deadline.
        # We track the best result globally.
        
        while calls_used < budget and time.time() < deadline:
            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = f(y)
            
            if fy < fx:
                # Improvement
                x, fx = y, fy
                sigma *= 1.5
                current_stagnation = 0
                
                # Update global best
                if fx < global_best_f:
                    global_best_f = fx
                    global_best_x = x.copy()
            else:
                # No improvement
                sigma *= 0.87
                current_stagnation += 1
                
                # Structural Change: Event-Driven Restart
                # If we stagnate, and there are other promising candidates, switch to them.
                if current_stagnation >= stagnation_limit:
                    # Try the next best candidate
                    current_idx += 1
                    if current_idx < len(restart_candidates):
                        x = restart_candidates[current_idx].copy()
                        fx = restart_fits[current_idx]
                        sigma = 2.0 # Reset sigma for new basin
                        current_stagnation = 0
                    else:
                        # No more distinct Sobol candidates. 
                        # Option: Random restart from a point far from current best, or continue with larger sigma.
                        # Let's try a random restart within the box, but biased towards the center or just uniform.
                        # Actually, if we've exhausted top Sobol points, we might be in the global basin but stuck in a local one.
                        # A "jump" restart: move to a random point.
                        x = np.clip(rng.uniform(-5, 5, dim), -5, 5)
                        fx = f(x)
                        sigma = 3.0 # Larger sigma for exploration
                        current_stagnation = 0
                        
                        # If this random restart is better than global best, update
                        if fx < global_best_f:
                            global_best_f = fx
                            global_best_x = x.copy()

        return global_best_x, global_best_f

# EVOLVE-BLOCK-END
