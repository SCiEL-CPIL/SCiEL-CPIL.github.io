"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart random hill-climbing (stochastic local search)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin within 300s limit

        best_x = None
        best_f = np.inf

        # We'll do multiple restarts. Each restart: random init + local hill climb.
        # Total budget is 500 function evaluations.
        # Split into restarts: each restart gets a chunk of evaluations.
        # Use adaptive: start with more restarts, fewer evals each, then refine.
        
        # Strategy: 
        # Phase 1: 20 restarts, each with ~10 evals (random init + 9 hill-climb steps)
        # Phase 2: take best 5 from phase 1, do 20 evals each (4 * 5 = 20 restarts * 10 = 200)
        # Actually, let's just do a simple multi-restart with fixed budget per restart.
        
        # Total 500 evals. Let's do 50 restarts, each with 10 evals (1 init + 9 climbs).
        # Or 25 restarts with 20 evals each.
        
        # Better: use a schedule. Early restarts are cheap, later ones focus on best.
        
        # Simple approach: 50 restarts, 10 evals each.
        num_restarts = 50
        evals_per_restart = budget // num_restarts  # 10
        if evals_per_restart < 2:
            evals_per_restart = 2
            num_restarts = budget // evals_per_restart

        # Actually, let's be smarter. Use a "best-of-restarts" with refinement.
        # Phase 1: 30 cheap restarts (5 evals each) = 150 evals
        # Phase 2: 10 medium restarts (10 evals each) = 100 evals  
        # Phase 3: 5 detailed refinements (14 evals each) = 70 evals
        # Phase 4: 2 final deep refinements (37.5 evals each) = 75 evals
        # Total: 150+100+70+75 = 395, remaining 105 for extra refinement
        
        # Let's just do a clean multi-restart with hill climbing.
        
        def hill_climb(x0, fx0, max_evals, rng):
            """Simple stochastic hill climbing from x0."""
            x = x0.copy()
            fx = fx0
            sigma = 1.0
            for _ in range(max_evals):
                if time.time() > deadline:
                    break
                # Try a random perturbation
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                    if sigma < 0.01:
                        sigma = 0.01
            return x, fx

        # Phase 1: Many cheap restarts to find good basins
        phase1_restarts = 30
        phase1_evals = 5
        used_evals = 0

        for i in range(phase1_restarts):
            if time.time() > deadline:
                break
            if used_evals >= budget:
                break
            x0 = rng.uniform(-5, 5, dim)
            fx0 = func(x0)
            used_evals += 1
            if fx0 < best_f:
                best_f, best_x = fx0, x0.copy()
            
            remaining = min(phase1_evals - 1, budget - used_evals)
            if remaining > 0:
                x, fx = hill_climb(x0, fx0, remaining, rng)
                used_evals += remaining
                if fx < best_f:
                    best_f, best_x = fx, x.copy()

        # Phase 2: Medium restarts around promising areas
        phase2_restarts = 15
        phase2_evals = 10
        for i in range(phase2_restarts):
            if time.time() > deadline:
                break
            if used_evals >= budget:
                break
            # Mix: some random, some near best
            if i < phase2_restarts // 2 and best_x is not None:
                x0 = np.clip(best_x + rng.standard_normal(dim) * 1.5, -5, 5)
            else:
                x0 = rng.uniform(-5, 5, dim)
            fx0 = func(x0)
            used_evals += 1
            if fx0 < best_f:
                best_f, best_x = fx0, x0.copy()
            
            remaining = min(phase2_evals - 1, budget - used_evals)
            if remaining > 0:
                x, fx = hill_climb(x0, fx0, remaining, rng)
                used_evals += remaining
                if fx < best_f:
                    best_f, best_x = fx, x.copy()

        # Phase 3: Detailed refinement of the best point
        phase3_evals = min(budget - used_evals, 100)
        if phase3_evals > 0 and best_x is not None and time.time() < deadline:
            x, fx = hill_climb(best_x, best_f, phase3_evals, rng)
            used_evals += phase3_evals
            if fx < best_f:
                best_f, best_x = fx, x.copy()

        # Phase 4: Final deep refinement
        phase4_evals = budget - used_evals
        if phase4_evals > 0 and best_x is not None and time.time() < deadline:
            x, fx = hill_climb(best_x, best_f, phase4_evals, rng)
            if fx < best_f:
                best_f, best_x = fx, x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
