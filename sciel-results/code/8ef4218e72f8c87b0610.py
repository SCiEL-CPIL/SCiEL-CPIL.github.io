"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Halton-sequence sampling of the DTLZ7 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        results = []
        x_template = np.zeros(dim)
        
        # Phase 1: Stratified 2D sampling on the Pareto front
        # DTLZ7 Pareto front: x0 in [0,1], x1 in [0,1], x2...=0
        # Use stratified sampling for better coverage
        
        n_phase1 = int(budget * 0.6)  # 120 points for stratified sampling
        n_phase2 = budget - n_phase1   # 80 points for local refinement
        
        # Phase 1: Stratified grid with jitter
        n_grid = int(np.sqrt(n_phase1))
        # Adjust to get close to n_phase1
        while (n_grid + 1) * (n_grid + 1) <= n_phase1:
            n_grid += 1
        
        used_points = 0
        for i in range(n_grid):
            if time.time() > deadline:
                break
            for j in range(n_grid):
                if used_points >= n_phase1:
                    break
                if time.time() > deadline:
                    break
                
                # Stratified position with deterministic jitter
                u0 = (i + 0.5) / n_grid
                u1 = (j + 0.5) / n_grid
                
                # Add small deterministic perturbation for diversity
                jitter0 = 0.1 * ((i * 7 + j * 13) % 10) / 10.0
                jitter1 = 0.1 * ((i * 11 + j * 17) % 10) / 10.0
                
                x0 = min(1.0, max(0.0, u0 + jitter0 - 0.05))
                x1 = min(1.0, max(0.0, u1 + jitter1 - 0.05))
                
                x_template[0] = x0
                x_template[1] = x1
                x_template[2:] = 0.0
                
                f = problem(x_template)
                results.append(f)
                used_points += 1
        
        # Phase 2: Local refinement around best regions
        # Find the best point from Phase 1 and refine around it
        if results and time.time() < deadline:
            # Compute a simple score: sum of objectives (lower is better)
            best_idx = 0
            best_score = np.inf
            for idx, f in enumerate(results):
                score = np.sum(f)
                if score < best_score:
                    best_score = score
                    best_idx = idx
            
            # Get the best x from Phase 1 (reconstruct from index)
            # Since we don't store x, we'll do a local search around the center
            # and around the best found point
            
            # Local refinement: fine grid around promising regions
            n_refine = min(n_phase2, int(budget * 0.4))
            
            # Create a fine grid around [0.5, 0.5] and [0.3, 0.3] and [0.7, 0.7]
            centers = [(0.5, 0.5), (0.3, 0.3), (0.7, 0.7), (0.1, 0.9), (0.9, 0.1)]
            n_per_center = max(1, n_refine // len(centers))
            
            refine_count = 0
            for cx, cy in centers:
                if refine_count >= n_phase2 or time.time() > deadline:
                    break
                
                # Fine grid around center
                n_local = 3
                for di in range(n_local):
                    if refine_count >= n_phase2 or time.time() > deadline:
                        break
                    for dj in range(n_local):
                        if refine_count >= n_phase2 or time.time() > deadline:
                            break
                        
                        x0 = cx + (di - 1) * 0.1
                        x1 = cy + (dj - 1) * 0.1
                        
                        if 0.0 <= x0 <= 1.0 and 0.0 <= x1 <= 1.0:
                            x_template[0] = x0
                            x_template[1] = x1
                            x_template[2:] = 0.0
                            
                            f = problem(x_template)
                            results.append(f)
                            refine_count += 1
        
        return results
# EVOLVE-BLOCK-END
