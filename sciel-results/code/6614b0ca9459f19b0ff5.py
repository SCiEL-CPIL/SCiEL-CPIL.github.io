"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global grid exploration followed by targeted local search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s

        # Phase 1: Global grid-based exploration
        # For dim=5, use a coarse grid to identify the basin of attraction
        # We'll use a stratified sampling approach for better coverage
        
        # First, do a few random starts to get a sense of the landscape
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Initial random samples to get a rough idea
        n_initial = 20
        for _ in range(n_initial):
            if calls_made >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Grid-based exploration around promising regions
        # Use a coarse grid over [-5, 5]^dim
        # For dim=5, a full grid of 3^5 = 243 points is feasible
        # But we'll be smarter: use stratified sampling
        
        # Let's use a structured approach:
        # Sample on a grid with 3 levels per dimension (low, mid, high)
        # That's 3^5 = 243 points, but we already used 20, so we have room
        
        # Actually, let's do a more adaptive approach:
        # 1. Start with a coarse grid
        # 2. Refine around the best points
        
        # Coarse grid: 3 points per dimension
        grid_points = []
        for d in range(self.dim):
            for _ in range(3):
                pass
        
        # Generate all combinations of 3 levels per dimension
        levels = np.linspace(-5, 5, 3)
        from itertools import product
        grid = list(product(levels, repeat=self.dim))
        
        # Shuffle to avoid bias
        rng.shuffle(grid)
        
        for x in grid:
            if calls_made >= self.budget or time.time() > deadline:
                break
            x = np.array(x)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 3: Local search from the best point
        # Use a (1+1)-ES with adaptive step size
        if calls_made < self.budget and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5  # Start with a moderate step size
            
            # Adaptive local search
            no_improve = 0
            while calls_made < self.budget and time.time() < deadline:
                # Try multiple candidates per iteration for better exploration
                n_candidates = 5
                best_y = None
                best_fy = np.inf
                
                for _ in range(n_candidates):
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < best_fy:
                        best_fy = fy
                        best_y = y.copy()
                
                if best_y is not None and best_fy < fx:
                    x = best_y
                    fx = best_fy
                    no_improve = 0
                    # Increase step size slightly on success
                    sigma *= 1.3
                    if sigma > 2.0:
                        sigma = 2.0
                else:
                    no_improve += 1
                    # Decrease step size on failure
                    sigma *= 0.7
                    if sigma < 0.01:
                        sigma = 0.01
                    
                    # If stuck for too long, try a larger jump
                    if no_improve > 10:
                        x = np.clip(x + 1.0 * rng.standard_normal(self.dim), -5, 5)
                        fx = func(x)
                        calls_made += 1
                        no_improve = 0
                        sigma = 0.5
        
        return best_x, best_f
# EVOLVE-BLOCK-END
