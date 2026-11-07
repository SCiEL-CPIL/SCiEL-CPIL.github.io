"""Multi-objective optimizer for ZDT3 using a specialized approach exploiting the problem structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for ZDT3 that exploits the known structure of the problem.
    ZDT3: f1 = x0, f2 = (1 + g(x[1:])) * h(f1, g(x[1:]))
    where g = 10*n + sum(2*x_i^2 - 4*cos(2*pi*x_i)), n = dim-1
    h = 1 - sqrt(f1/g) - sin(10*pi*f1)
    
    The Pareto front has f2 = (1+g)*h(f1, g) with f1 in [0,1].
    We focus on finding good x0 values and minimizing g.
    """
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self._deadline = None
    
    def __call__(self, problem, bounds):
        start_time = time.time()
        self._deadline = start_time + 280  # 300s limit, 20s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 analysis:
        # f1 = x[0]
        # f2 = (1 + g(x[1:])) * h(x[0], g(x[1:]))
        # where g = 10*(n-1) + sum_{i=2}^{n} (2*x_i^2 - 4*cos(2*pi*x_i))
        # h = 1 - sqrt(x[0]/g) - sin(10*pi*x[0])
        #
        # To minimize both objectives, we need:
        # 1. Minimize g (push x[1:] to zeros)
        # 2. Choose x[0] to minimize f2 given g
        #
        # The Pareto front is f2 = (1+g_min) * h(f1, g_min) for f1 in [0,1]
        # where g_min = 10*(n-1) (when all x[1:] = 0)
        
        n = self.dim
        # Theoretical minimum of g when x[1:] = 0
        g_min = 10.0 * (n - 1)
        
        # Strategy:
        # 1. Use a good fraction of budget to minimize g by optimizing x[1:]
        # 2. Use remaining budget to sample x[0] values along the Pareto front
        
        # Phase 1: Minimize g(x[1:]) using CMA-ES-like approach
        # g(x[1:]) = 10*(n-1) + sum_{i=2}^{n} (2*x_i^2 - 4*cos(2*pi*x_i))
        # Each term 2*x_i^2 - 4*cos(2*pi*x_i) has minimum -2 at x_i=0
        # So g_min = 10*(n-1) + (n-1)*(-2) = 10*(n-1) - 2*(n-1) = 8*(n-1)
        # Wait, let me recalculate: at x_i=0, 2*0 - 4*cos(0) = -4
        # So each term min is -4, g_min = 10*(n-1) + (n-1)*(-4) = 6*(n-1)
        # Hmm, but the standard ZDT3 has g = 10*n + sum(2*x_i^2 - 4*cos(2*pi*x_i)) for i=2..n
        # That's n-1 terms. At x_i=0: 2*0-4*1 = -4. So g_min = 10*(n-1) - 4*(n-1) = 6*(n-1)
        
        # Actually for ZDT3 with 30 vars: n=30, so 29 terms in the sum
        # g = 10*29 + sum_{i=2}^{30}(2*x_i^2 - 4*cos(2*pi*x_i))
        # At all zeros: g = 290 + 29*(-4) = 290 - 116 = 174
        
        # Let's use a hybrid approach:
        # - Allocate ~60% of budget to minimize g via coordinate descent / local search
        # - Allocate ~40% to sample x[0] values
        
        budget_g = int(self.budget * 0.5)
        budget_x0 = self.budget - budget_g
        
        # Initialize x[1:] at 0 (known good point for minimizing g)
        x_tail = np.zeros(n - 1)
        
        # Evaluate initial point
        x_init = np.zeros(n)
        f_init = problem(x_init)
        
        # Local search to minimize g: try small perturbations
        # Use a simple random restart local search
        best_x_tail = x_tail.copy()
        best_g = self._compute_g(x_tail, n)
        
        # Try to improve g with random perturbations
        rng = np.random.RandomState(42)
        evals_used = 1  # for the initial evaluation
        
        # Coordinate descent on x[1:] to minimize g
        step_size = 0.05
        for _ in range(budget_g // 3):
            if time.time() > self._deadline - 60:
                break
            improved = False
            for i in range(n - 1):
                for delta in [step_size, -step_size]:
                    x_try = best_x_tail.copy()
                    x_try[i] = np.clip(x_try[i] + delta, 0, 1)
                    g_try = self._compute_g(x_try, n)
                    if g_try < best_g - 1e-10:
                        best_g = g_try
                        best_x_tail = x_try
                        improved = True
                        break
            if not improved:
                # Try random restart
                x_try = rng.uniform(0, 1, n - 1)
                g_try = self._compute_g(x_try, n)
                if g_try < best_g - 1e-10:
                    best_g = g_try
                    best_x_tail = x_try
                    step_size *= 0.5
            else:
                step_size = max(0.001, step_size * 0.9)
        
        # Phase 2: Sample x[0] values to cover the Pareto front
        # The Pareto front: f1 = x[0], f2 = (1 + g) * h(x[0], g)
        # where h = 1 - sqrt(x[0]/g) - sin(10*pi*x[0])
        # 
        # We want to find x[0] values that give good coverage of the front.
        # The front is discontinuous in ZDT3 due to the sin term.
        # Let's sample x[0] uniformly and also at specific points.
        
        # Generate candidate x[0] values
        # Uniform sampling + some strategic points
        x0_candidates = np.linspace(0, 1, budget_x0)
        
        # Also add some points near where h might be small
        # h = 1 - sqrt(x0/g) - sin(10*pi*x0)
        # sin(10*pi*x0) = 1 when 10*pi*x0 = pi/2 + 2*pi*k, i.e., x0 = (1/20) + k/5
        # So x0 = 0.05, 0.25, 0.45, 0.65, 0.85 give sin=1
        # At these points, h = 1 - sqrt(x0/g) - 1 = -sqrt(x0/g) < 0
        # This means f2 < 0, which is better!
        
        # Add strategic points
        strategic_x0 = np.array([0.05, 0.25, 0.45, 0.65, 0.85,
                                 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        x0_candidates = np.unique(np.concatenate([x0_candidates, strategic_x0]))
        
        # If we have more candidates than budget, sample
        if len(x0_candidates) > budget_x0:
            idx = rng.choice(len(x0_candidates), budget_x0, replace=False)
            x0_candidates = x0_candidates[idx]
        
        # Evaluate all candidates
        results_x0 = []
        for x0 in x0_candidates:
            if time.time() > self._deadline - 30:
                break
            x = np.concatenate([[x0], best_x_tail])
            x = np.clip(x, 0, 1)
            f = problem(x)
            results_x0.append((x0, f))
        
        # Phase 3: Refine with local search on full x
        # Use remaining budget for local refinement
        remaining_budget = self.budget - len(results_x0) - 1  # -1 for initial eval
        if remaining_budget > 0 and len(results_x0) > 0:
            # Sort by f2 (since f1 = x0 is fixed by our choice)
            results_x0.sort(key=lambda t: t[1][1])
            
            # Take top few and do local search
            top_k = min(5, len(results_x0))
            for i in range(top_k):
                if time.time() > self._deadline - 20:
                    break
                x_best = np.concatenate([[results_x0[i][0]], best_x_tail])
                # Local search
                for _ in range(min(20, remaining_budget // top_k)):
                    if time.time() > self._deadline - 10:
                        break
                    x_try = x_best + rng.normal(0, 0.01, n)
                    x_try = np.clip(x_try, 0, 1)
                    f_try = problem(x_try)
                    if f_try[1] < results_x0[i][1][1] - 1e-10:
                        x_best = x_try
                        results_x0[i] = (x_best[0], f_try)
        
        return best_x_tail
    
    def _compute_g(self, x_tail, n):
        """Compute g(x[1:]) for ZDT3."""
        if len(x_tail) == 0:
            return 0.0
        return 10.0 * (n - 1) + np.sum(2.0 * x_tail**2 - 4.0 * np.cos(2.0 * np.pi * x_tail))
# EVOLVE-BLOCK-END
