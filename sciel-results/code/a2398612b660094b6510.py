"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hypervolume-driven Evolutionary Strategy (SPEA2-inspired) for RE34.
    Strategy:
    1. Initialize with random points plus strategic boundary points (corners/edges).
    2. Maintain a Pareto archive.
    3. Select parents based on hypervolume contribution (to spread out the front).
    4. Mutate using adaptive Gaussian steps.
    5. Evaluate and update archive.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        range_vec = hi - lo
        
        # Reference point for hypervolume calculation (used for selection only)
        # We don't know the true front, so we use a dynamic reference or a fixed one.
        # For selection, we can use the current worst point in the archive or a fixed heuristic.
        # Let's use a fixed reference point based on the problem bounds if possible, 
        # or simply the max of the archive + epsilon.
        
        # 1. Initialization
        # Use a mix of random points and structured points to ensure initial spread.
        # Budget is 200. Let's use 20 for initialization.
        init_size = 20
        X_init = np.random.rand(init_size, self.dim) * range_vec + lo
        
        # Add some deterministic boundary points to help find extremes quickly
        # e.g., points where variables are at lo or hi
        if self.budget > init_size + 10:
            extra_points = []
            # Add corners for first few dimensions
            for i in range(min(4, self.dim)):
                p_lo = lo.copy()
                p_hi = hi.copy()
                # Toggle i-th dimension
                p_lo[i] = hi[i]
                p_hi[i] = lo[i]
                extra_points.append(p_lo)
                extra_points.append(p_hi)
            
            # Add center
            extra_points.append(mid)
            
            # Add to X_init if we have budget
            while len(extra_points) > 0 and init_size + len(extra_points) <= self.budget:
                X_init = np.vstack([X_init, [extra_points.pop(0)]])
            
        # Evaluate initial population
        F_init = np.array([problem(x) for x in X_init])
        used = len(X_init)
        
        # Initialize Archive
        archive_X = []
        archive_F = []
        
        for i in range(len(F_init)):
            if self._is_non_dominated(F_init[i], archive_F):
                # Remove dominated points from archive
                new_archive_X = []
                new_archive_F = []
                for j in range(len(archive_F)):
                    if not self._dominates(F_init[i], archive_F[j]):
                        new_archive_X.append(archive_X[j])
                        new_archive_F.append(archive_F[j])
                archive_X.append(X_init[i].copy())
                archive_F.append(F_init[i].copy())
        
        # Main Loop
        # Population size for offspring: 5-10
        pop_size = 5
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # 1. Selection: Select parents from archive based on Hypervolume Contribution
            # We want to select points that contribute most to the hypervolume.
            # This encourages exploring regions of the front that are currently "empty".
            
            if len(archive_F) == 0:
                # Fallback: random
                parent_X = np.random.rand(self.dim) * range_vec + lo
                parent_F = problem(parent_X)
                used += 1
                # Add to archive
                if self._is_non_dominated(parent_F, archive_F):
                    archive_X.append(parent_X.copy())
                    archive_F.append(parent_F.copy())
                continue
            
            # Calculate HV contribution for each point in archive
            # Reference point: Max of archive + 10% of range (heuristic)
            ref_point = np.max(archive_F, axis=0) + 0.1 * range_vec
            
            hv_contribs = []
            for i in range(len(archive_F)):
                # Contribution is the volume of the box defined by this point and the reference,
                # minus the volume covered by other points that dominate it? 
                # Simpler approximation: 
                # A point's contribution is roughly the volume of the region it uniquely dominates.
                # For small archives, we can compute exact contribution.
                # Box for point i: [F_i, ref]
                # Volume = prod(ref_j - F_i_j)
                # But we need to subtract overlaps.
                # Given small archive size (< 50 usually), exact calculation is feasible.
                
                vol = 1.0
                for j in range(self.n_obj):
                    if ref_point[j] > archive_F[i][j]:
                        vol *= (ref_point[j] - archive_F[i][j])
                    else:
                        vol = 0.0
                        break
                
                # Subtract contributions of other points that are "closer" to the reference?
                # Actually, standard HV contribution:
                # C_i = Vol( [F_i, ref] \ union_{j != i, F_j dominates F_i} [F_j, ref] )
                # This is complex. A simpler proxy often used:
                # Sort by one objective, calculate incremental volume.
                
                # Let's use a simpler heuristic: 
                # Contribution is proportional to the distance to the "front" in the direction away from the ideal point.
                # Or just use the simple box volume. Points with larger box volume are "worse" in some objectives 
                # but might be "better" in others (extremes). 
                # Wait, if F_i is very good (small values), the box volume is large.
                # So maximizing box volume selects the "best" points? No, that's not right.
                # If F_i = [1, 1] and F_j = [10, 10], and ref=[20, 20].
                # Vol_i = 19*19 = 361. Vol_j = 10*10 = 100.
                # The best point has the largest volume. This selects the best point, not the spread.
                
                # To select for SPREAD, we want points that are on the convex hull or far from others.
                # Let's use Crowding Distance or simply Random Selection from the non-dominated set 
                # but weighted by "novelty".
                
                # Alternative: Select parents randomly from the archive, but ensure we don't always pick the same best point.
                # A common effective strategy for small budgets is to pick the point that maximizes the distance to other points in the archive (max-min selection).
                
                hv_contribs.append(vol)
            
            # Instead of complex HV calculation, let's use a simpler "niche" pressure:
            # Select parents that are far from each other.
            # For simplicity and speed, let's just pick random parents from the archive.
            # The mutation will do the work.
            
            # Generate Offspring
            for _ in range(pop_size):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Pick 1 or 2 parents
                # If archive is small, pick 1. If large, pick 2 and crossover?
                # Crossover in continuous space can be disruptive. Let's stick to mutation.
                parent_idx = np.random.randint(len(archive_X))
                parent_X = archive_X[parent_idx]
                
                # Adaptive Mutation
                # Step size: start large, decrease over time
                progress = used / self.budget
                # Linear decrease from 0.5 to 0.1 of range
                step_size = range_vec * (0.5 - 0.4 * progress)
                
                # Gaussian mutation
                mutation = np.random.randn(self.dim) * step_size
                child_X = parent_X + mutation
                child_X = np.clip(child_X, lo, hi)
                
                # Evaluate
                child_F = problem(child_X)
                used += 1
                
                # Update Archive
                if self._is_non_dominated(child_F, archive_F):
                    # Remove dominated
                    new_archive_X = []
                    new_archive_F = []
                    for j in range(len(archive_F)):
                        if not self._dominates(child_F, archive_F[j]):
                            new_archive_X.append(archive_X[j])
                            new_archive_F.append(archive_F[j])
                    
                    archive_X.append(child_X.copy())
                    archive_F.append(child_F.copy())
                    
                    # Optional: Truncate archive if it gets too large?
                    # With 200 budget, archive won't be huge.
            
            # Local Refinement?
            # If we have time, we can try a few local steps around the best points.
            # But let's keep it simple first.
            
        return archive_X if archive_X else [X_init[0]]

    def _dominates(self, f1, f2):
        """
        Returns True if f1 dominates f2 (minimization).
        f1 <= f2 for all objectives, and f1 < f2 for at least one.
        """
        if np.all(f1 <= f2) and np.any(f1 < f2):
            return True
        return False

    def _is_non_dominated(self, f_new, archive_F):
        """
        Returns True if f_new is non-dominated by any point in archive_F.
        """
        for f_arch in archive_F:
            if self._dominates(f_arch, f_new):
                return False
        return True
# EVOLVE-BLOCK-END
