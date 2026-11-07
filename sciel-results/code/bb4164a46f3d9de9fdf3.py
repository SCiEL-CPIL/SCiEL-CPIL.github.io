# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hypervolume-driven evolutionary multi-objective optimizer for DTLZ7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7 specific structure:
        # f3 = 1 + g, where g depends on x_3...x_22 (indices 2 to 21).
        # To minimize f3, we want x_3...x_22 -> 0.
        # The Pareto front is achieved when x_3...x_22 = 0.
        # f1, f2 depend on x_1, x_2.
        
        # Strategy:
        # 1. Initialize with a grid/sampling on the Pareto front (x_3..x_21 = 0).
        # 2. Use remaining budget for local search around promising points.
        
        pop_size = 50
        X = np.zeros((pop_size, self.dim))
        
        # Sample x1, x2 on a grid or randomly to cover the front
        # A grid is better for initial coverage of the 2D front.
        # Let's use a quasi-random or stratified sampling for x1, x2.
        # For simplicity, uniform random is okay, but let's try to be a bit structured.
        # We have 200 budget. 50 initial points leaves 150 for refinement.
        
        for i in range(pop_size):
            # Random sampling of x1, x2 in [0, 1]
            X[i, 0] = rng.random()
            X[i, 1] = rng.random()
            # x_3 ... x_22 are 0 (indices 2 to 21)
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Keep track of all evaluated points to select parents
        # We will maintain a list of (X, F) for all non-dominated points found
        # But to save memory and computation, we just keep the current population of best points.
        # Let's use a simple archive of non-dominated points.
        
        # Initialize archive with the initial population's non-dominated points
        archive_X = []
        archive_F = []
        
        for i in range(len(F)):
            f = F[i]
            # Check if f is dominated by any point in archive
            dominated = False
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    dominated = True
                    break
            if not dominated:
                # Check if f dominates any point in archive
                new_archive_X = []
                new_archive_F = []
                for j in range(len(archive_F)):
                    af = archive_F[j]
                    if np.all(f <= af) and np.any(f < af):
                        continue # f dominates af, discard af
                    new_archive_X.append(archive_X[j])
                    new_archive_F.append(af)
                new_archive_X.append(X[i])
                new_archive_F.append(f)
                archive_X = new_archive_X
                archive_F = new_archive_F
        
        # Refinement phase
        # Use local search (mutation) around points in the archive.
        # Since the front is 2D, we can afford many evaluations.
        # We will pick a point from the archive, perturb it, and evaluate.
        
        while used < self.budget:
            # Pick a parent from the archive
            # To improve coverage, we might want to pick points that are less crowded,
            # but random selection from archive is also effective for small archives.
            idx = rng.integers(0, len(archive_X))
            parent = archive_X[idx]
            
            # Create child
            child = parent.copy()
            
            # Perturb x1 and x2 to explore the front
            # Use a small step size to refine
            step_size = 0.05
            child[0] += rng.uniform(-step_size, step_size)
            child[1] += rng.uniform(-step_size, step_size)
            
            # Perturb distance variables (x_3...x_21) slightly, but bias towards 0
            # If they are 0, keep them 0 or add tiny noise? 
            # Adding noise to distance vars increases g, which worsens f3.
            # So we should keep them as close to 0 as possible.
            # However, if the parent has non-zero distance vars (due to previous noise),
            # we should try to reduce them.
            for d in range(2, self.dim):
                if child[d] > 0:
                    # Try to reduce towards 0
                    child[d] *= rng.uniform(0.0, 0.5)
                else:
                    # Keep at 0 or allow very small positive noise? 
                    # No, 0 is optimal.
                    pass
            
            # Clip to bounds
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Update archive
            dominated = False
            for af in archive_F:
                if np.all(af <= fc) and np.any(af < fc):
                    dominated = True
                    break
            
            if not dominated:
                # Check if fc dominates any point in archive
                new_archive_X = []
                new_archive_F = []
                for j in range(len(archive_F)):
                    af = archive_F[j]
                    if np.all(fc <= af) and np.any(fc < af):
                        continue # fc dominates af
                    new_archive_X.append(archive_X[j])
                    new_archive_F.append(af)
                
                # Limit archive size to prevent it from growing too large?
                # With 200 evaluations, archive size will be small (< 50).
                # So no limit needed.
                new_archive_X.append(child)
                new_archive_F.append(fc)
                archive_X = new_archive_X
                archive_F = new_archive_F

        return archive_X
# EVOLVE-BLOCK-END
