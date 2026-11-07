"""Multi-objective optimizer on WFG4 using MOEA/D with Sobol initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """MOEA/D based multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # 1. Generate Weight Vectors for MOEA/D
        # We use a simplex lattice. For 3 objectives, we can generate points on the simplex.
        # Let's aim for about 20-30 sub-problems.
        # Number of points in simplex lattice of order k for n_obj is C(k+n_obj-1, n_obj-1).
        # For n_obj=3:
        # k=4 -> C(6,2) = 15
        # k=5 -> C(7,2) = 21
        # k=6 -> C(8,2) = 28
        # Let's use k=5 for 21 sub-problems.
        k = 5
        weights = []
        for i in range(k + 1):
            for j in range(k + 1 - i):
                l = k - i - j
                weights.append([i / k, j / k, l / k])
        weights = np.array(weights)
        n_sub = len(weights)

        # 2. Initialize with Sobol Sequence
        # We need at least n_sub points, but let's take a bit more for initial archive diversity.
        n_init = max(n_sub, 30)
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        # Sobol requires power of 2 for perfect generation, but it works with rounding.
        # To get exactly n_init, we might need to generate a power of 2 and slice, or use randomize.
        # scipy qmc.Sobol randomize method allows non-power-of-2 if we just call random.
        # However, for low discrepancy, generating power of 2 is better.
        # Next power of 2 >= n_init
        n_power2 = 1
        while n_power2 < n_init:
            n_power2 *= 2
        points = sobol.random(n_power2)
        X_init = lo + (hi - lo) * points[:n_init]
        
        F_init = np.array([problem(x) for x in X_init])
        evaluations = n_init

        # 3. Assign initial points to sub-problems
        # For each sub-problem i, find the point that minimizes the weighted sum w_i . f
        # To save time, we can just assign the first n_sub points arbitrarily or by nearest weight.
        # Better: For each weight, pick the best point from the initial set.
        # Since n_init >= n_sub, we can do this.
        
        sub_champions_X = [None] * n_sub
        sub_champions_F = [None] * n_sub
        sub_champion_scores = [np.inf] * n_sub

        # Precompute weighted sums for all initial points for all weights?
        # F_init shape (n_init, 3), weights shape (n_sub, 3)
        # scores[i, j] = weights[j] . F_init[i]
        # Use broadcasting: F_init[:, np.newaxis, :] * weights[np.newaxis, :, :] -> sum axis 2
        # Shape (n_init, n_sub)
        weighted_scores = np.sum(F_init[:, np.newaxis, :] * weights[np.newaxis, :, :], axis=2)
        
        for j in range(n_sub):
            # Find index i in 0..n_init-1 that minimizes weighted_scores[i, j]
            best_i = np.argmin(weighted_scores[:, j])
            sub_champions_X[j] = X_init[best_i]
            sub_champions_F[j] = F_init[best_i]
            sub_champion_scores[j] = weighted_scores[best_i, j]

        # 4. Build Neighborhood for each sub-problem
        # Calculate distance between all weight pairs
        # dist[i, j] = ||weights[i] - weights[j]||
        diff = weights[:, np.newaxis, :] - weights[np.newaxis, :, :]
        dist = np.sum(diff ** 2, axis=2)
        # Sort indices for each i by distance
        neighborhood = []
        for i in range(n_sub):
            sorted_idx = np.argsort(dist[i])
            # Take the 5 nearest neighbors (including self)
            neighborhood.append(sorted_idx[:5])

        # 5. Global Archive for Non-Dominated Points
        archive_X = []
        archive_F = []

        def is_dominated(f_new, archive_F):
            if not archive_F:
                return False
            leq = np.all(archive_F <= f_new, axis=1)
            lt = np.any(archive_F < f_new, axis=1)
            return np.any(leq & lt)

        def update_archive(x_new, f_new):
            nonlocal archive_X, archive_F
            if not is_dominated(f_new, archive_F):
                # Remove dominated points
                if archive_F:
                    leq = np.all(f_new <= archive_F, axis=1)
                    lt = np.any(f_new < archive_F, axis=1)
                    dominated_mask = leq & lt
                    if np.any(dominated_mask):
                        archive_X = [archive_X[i] for i in range(len(archive_X)) if not dominated_mask[i]]
                        archive_F = [archive_F[i] for i in range(len(archive_F)) if not dominated_mask[i]]
                archive_X.append(x_new)
                archive_F.append(f_new)
                # Prune archive if too large to keep computation fast
                # Max archive size 50
                if len(archive_X) > 50:
                    # Simple pruning: keep points with lowest hypervolume contribution? 
                    # Or just random? Or crowding?
                    # For speed, let's just keep the first 50 or use a simple filter.
                    # Actually, for WFG4 with 200 evals, archive won't get huge.
                    # But let's implement a simple crowding distance prune if it gets big.
                    pass # Skip complex prune for now, 200 evals is small.

        # Initialize global archive
        for i in range(n_init):
            update_archive(X_init[i], F_init[i])

        # 6. Main Loop
        rng = np.random.default_rng(42)
        
        # Parameters
        F_factor = 0.8
        # We will use the remaining budget for variation
        # Reserve last 20% for polishing?
        # Let's do 160 MOEA/D steps and 40 polishing steps.
        polish_budget = 40
        main_budget = budget - polish_budget

        # Track best solution per sub-problem
        # We already have sub_champions
        
        for _ in range(main_budget):
            if time.time() > deadline:
                break
            if evaluations >= budget:
                break

            # Select a sub-problem
            i = rng.integers(n_sub)
            
            # Select neighbors
            neighbors = neighborhood[i]
            # Pick 2 distinct neighbors (excluding self if possible, or just random from list)
            # We need r1, r2, r3 for DE/rand/1
            # Let's pick 3 random indices from the neighborhood list
            # Ensure they are distinct if possible
            if len(neighbors) >= 3:
                r_indices = rng.choice(neighbors, size=3, replace=False)
            else:
                # If neighborhood is small, just pick from all
                r_indices = rng.choice(n_sub, size=3, replace=False)
            
            r1, r2, r3 = r_indices[0], r_indices[1], r_indices[2]
            
            x_r1 = sub_champions_X[r1]
            x_r2 = sub_champions_X[r2]
            x_r3 = sub_champions_X[r3]
            
            # DE variation
            x_off = x_r1 + F_factor * (x_r2 - x_r3)
            
            # Add small noise
            sigma = 0.1
            x_off = x_off + sigma * (hi - lo) * rng.standard_normal(dim)
            
            x_off = np.clip(x_off, lo, hi)
            
            f_off = problem(x_off)
            evaluations += 1
            
            # Update sub-problem champion if better in weighted sum
            # We only update the selected sub-problem i? Or all?
            # Standard MOEA/D updates the sub-problem if the offspring is better than the current one.
            # Let's check if it's better for sub-problem i.
            score_off = np.dot(weights[i], f_off)
            if score_off < sub_champion_scores[i]:
                sub_champions_X[i] = x_off
                sub_champions_F[i] = f_off
                sub_champion_scores[i] = score_off
                
            # Also, check if it should replace any other sub-problem? 
            # No, typically each sub-problem is optimized for its specific weight.
            # But we can check if it's better for neighbors too? 
            # For efficiency, just update i.
            
            # Update global archive
            update_archive(x_off, f_off)

        # 7. Polishing Phase
        # Take the best non-dominated points and perturb them.
        if archive_F:
            # Select top K points from archive based on crowding or just random
            # Let's just take all points in archive if small, else random sample
            n_polish_points = min(len(archive_X), 10)
            if n_polish_points > 0:
                # Indices of points to polish
                polish_indices = rng.choice(len(archive_X), size=n_polish_points, replace=False)
                
                for _ in range(polish_budget):
                    if time.time() > deadline:
                        break
                    if evaluations >= budget:
                        break
                    
                    # Pick a random point from the polish candidates
                    idx = polish_indices[rng.integers(n_polish_points)]
                    x_base = archive_X[idx]
                    
                    # Small perturbation
                    sigma = 0.02
                    x_off = x_base + sigma * (hi - lo) * rng.standard_normal(dim)
                    x_off = np.clip(x_off, lo, hi)
                    
                    f_off = problem(x_off)
                    evaluations += 1
                    
                    update_archive(x_off, f_off)
                    
                    # Also update sub-champions if this new point is better for some weight?
                    # This might be expensive. Let's skip for polishing to save time.
                    # The global archive is what matters for hypervolume.

        return archive_X
# EVOLVE-BLOCK-END
