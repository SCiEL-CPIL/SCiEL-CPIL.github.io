"""Multi-objective optimizer on WFG4 using Hypervolume-guided search with local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided multi-objective optimizer with local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use the global numpy random state (seeded by evaluator)
        rng = np.random

        # Reference point for hypervolume calculation (based on task description)
        ref_point = np.array([3.0, 5.0, 7.0], dtype=np.float64)
        
        # If n_obj != 3, we need a safe reference point. 
        # The task specifies WFG4 (3 objs) and ref [3,5,7].
        # If n_obj is different, we assume the problem is standard or we just use a large bound.
        if n_obj != 3:
             # Fallback: use max observed or a large value
             ref_point = np.full(n_obj, 10.0) 

        # Population size for the main search
        pop_size = 40
        if budget < pop_size:
            pop_size = max(2, budget)

        # Archive stores (x, f) pairs that are non-dominated
        archive_X = []
        archive_F = []
        
        # Helper to check dominance
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        # Helper to add to archive
        def add_to_archive(x, f):
            # Check if f is dominated by any in archive
            for af in archive_F:
                if dominates(af, f):
                    return False
            
            # Check if f dominates any in archive, remove them
            new_F = []
            new_X = []
            for i, af in enumerate(archive_F):
                if dominates(f, af):
                    continue
                # Check for duplicates
                if np.allclose(f, af, atol=1e-10):
                    continue
                new_F.append(af)
                new_X.append(archive_X[i])
            
            new_F.append(f)
            new_X.append(x)
            
            archive_F[:] = new_F
            archive_X[:] = new_X
            return True

        # Hypervolume calculation for 3D (simplified for speed, exact for small sets)
        # Since archive size is small (<100), we can use a simple decomposition or grid-based approx.
        # However, for selection, we just need relative contribution.
        # A simpler metric: sum of volumes of boxes defined by each point and its neighbors in the non-dominated set.
        # For efficiency in selection, we can approximate HV contribution.
        
        def calculate_hv_contributions(archive_F_list):
            """
            Approximate HV contribution of each point in the archive.
            Uses a simple box decomposition where each point 'owns' the volume
            between itself and the 'next' points in each dimension.
            This is an approximation but sufficient for ranking.
            """
            if not archive_F_list:
                return []
            
            n = len(archive_F_list)
            contributions = [0.0] * n
            
            # Sort by first objective to help with box definition
            # This is a heuristic approach. For exactness, we'd need a proper HV calculator.
            # Given the small size, we can compute exact HV for the whole set and use it for global score,
            # but for selection, we can use a simpler 'dominated volume' metric.
            
            # Let's use a standard simple HV approximation:
            # For each point i, its contribution is the volume of the region dominated by i 
            # but not dominated by any other point.
            # This is hard to compute exactly per point. 
            # Instead, we will just maintain the archive and select parents based on 
            # random sampling from the archive, or use a simple diversity metric.
            
            # Given the inspirations, "Hypervolume-guided" often implies using the HV to select 
            # which points to expand. 
            # Let's stick to a simpler, robust method: 
            # 1. Maintain non-dominated archive.
            # 2. Select parents from archive using tournament selection on a 'potential' metric 
            #    (e.g., distance to reference point or just random).
            # 3. Use local search to improve.
            
            return contributions

        # Initialize population
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))
        used = 0
        
        for i in range(pop_size):
            # Stratified initialization
            x = lo + span * rng.random(dim)
            # Or better: Latin Hypercube style
            # x = lo + span * (rng.random(dim) + i/pop_size) % 1.0
            
            X[i] = x
            F[i] = problem(x)
            used += 1
            if used >= budget:
                break
            
            # Add to archive
            # We need to copy x and f to avoid reference issues
            add_to_archive(X[i].copy(), F[i].copy())

        # Main Loop
        while used < budget:
            if time.time() > deadline - 10.0:
                break
            
            # 1. Select a parent from the archive
            # If archive is empty (shouldn't happen), pick random
            if not archive_X:
                parent_idx = 0
            else:
                # Random selection from archive, or weighted by some metric
                parent_idx = rng.randint(0, len(archive_X))
            
            parent_x = archive_X[parent_idx].copy()
            
            # 2. Generate offspring via mutation
            # Adaptive mutation: start large, shrink over time
            progress = used / budget
            sigma = 0.5 * (1.0 - progress) + 0.01
            
            child_x = parent_x + sigma * span * rng.randn(dim)
            child_x = np.clip(child_x, lo, hi)
            
            child_f = problem(child_x)
            used += 1
            if used >= budget:
                break
            
            # 3. Local Refinement
            # Try to improve the child locally
            # Simple coordinate descent or small random steps
            local_steps = 5
            current_x = child_x.copy()
            current_f = child_f.copy()
            
            for _ in range(local_steps):
                if used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                
                # Try a small perturbation
                step_size = 0.05 * (1.0 - progress) + 0.001
                perturb_x = current_x + step_size * span * rng.randn(dim)
                perturb_x = np.clip(perturb_x, lo, hi)
                
                perturb_f = problem(perturb_x)
                used += 1
                
                # If perturbation is better (non-dominated and improves HV or just non-dominated), accept
                # For simplicity, accept if it is non-dominated by current_f
                if not dominates(current_f, perturb_f):
                    # Check if perturb_f is dominated by current_f? 
                    # If current_f dominates perturb_f, reject.
                    # If they are non-dominated, we might accept to maintain diversity, 
                    # but for local search, we usually want to move to a better point.
                    # Let's accept if perturb_f is not dominated by current_f.
                    current_x = perturb_x
                    current_f = perturb_f
            
            # 4. Update Archive
            # Add the best local solution to the archive
            add_to_archive(current_x.copy(), current_f.copy())
            
            # Also, we might want to keep the population diverse. 
            # The current approach is a pure archive-based search (like PESA-II or simple HV-based).
            # To ensure we explore the whole space, we should occasionally add random points.
            if rng.random() < 0.1:
                if used >= budget:
                    break
                random_x = lo + span * rng.random(dim)
                random_f = problem(random_x)
                used += 1
                add_to_archive(random_x.copy(), random_f.copy())

        return X
# EVOLVE-BLOCK-END
