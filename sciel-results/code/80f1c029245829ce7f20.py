"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution for Multi-Objective Optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        used = 0
        archive = []  # List of (x, f)
        
        def is_dominated(f_new, archive_F):
            """Check if f_new is dominated by any point in archive_F."""
            for af in archive_F:
                if np.all(af <= f_new) and np.any(af < f_new):
                    return True
            return False

        def update_archive(x, f):
            """Add to archive if non-dominated, remove dominated points."""
            nonlocal archive
            f = np.array(f)
            
            # Check if dominated by existing
            if is_dominated(f, [af for _, af in archive]):
                return

            # Remove points dominated by new
            new_archive = []
            for ax, af in archive:
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append((ax, af))
            
            new_archive.append((x, f))
            archive = new_archive

        # 1. Initialize Population
        # Use a reasonably sized population for DE. 
        # With 200 budget, we can afford a pop size of ~20-40.
        # Let's start with 30 to ensure good initial coverage.
        pop_size = 30
        if self.budget < 50:
            pop_size = 10
            
        population = []
        
        # Latin Hypercube Sampling for initial population for better spread
        lhs = np.zeros((pop_size, self.dim))
        for d in range(self.dim):
            perm = np.arange(pop_size)
            np.random.shuffle(perm)
            lhs[:, d] = (perm + 0.5) / pop_size
        
        X_pop = lo + lhs * (hi - lo)
        
        for x in X_pop:
            if used >= self.budget or time.time() > deadline:
                break
            f = problem(x)
            used += 1
            population.append((x.copy(), np.array(f)))
            update_archive(x, f)

        # 2. Evolutionary Loop
        # DE Parameters
        # CR: Crossover rate. 0.5 is standard, 0.9 can be better for continuous.
        # F: Scaling factor. 0.5-0.8 is typical.
        CR = 0.9
        F = 0.8
        
        # We will run generations until budget or time runs out.
        # Each generation consumes pop_size evaluations (approx, due to rejection).
        
        while used < self.budget and time.time() < deadline:
            current_pop = population
            new_pop = []
            
            # If population is too small to perform DE (needs at least 3 distinct individuals for rand/1),
            # just do random search or restart.
            if len(current_pop) < 3:
                # Fill up with random points
                while len(current_pop) < 3 and used < self.budget:
                    x_rand = np.random.uniform(lo, hi)
                    f_rand = problem(x_rand)
                    used += 1
                    current_pop.append((x_rand, np.array(f_rand)))
                    update_archive(x_rand, f_rand)
                if len(current_pop) < 3:
                    break

            for i in range(len(current_pop)):
                if used >= self.budget or time.time() > deadline:
                    break
                
                # Select three distinct random individuals
                # Ensure they are distinct from i and each other
                while True:
                    r1, r2, r3 = np.random.choice(len(current_pop), 3, replace=False)
                    if r1 != i and r2 != i and r3 != i:
                        break
                
                x1 = current_pop[r1][0]
                x2 = current_pop[r2][0]
                x3 = current_pop[r3][0]
                xi = current_pop[i][0]
                fi = current_pop[i][1]
                
                # Mutation: v = x1 + F * (x2 - x3)
                v = x1 + F * (x2 - x3)
                
                # Clip to bounds
                v = np.clip(v, lo, hi)
                
                # Crossover: uniform
                # jrand: random index for uniform crossover
                jrand = np.random.randint(self.dim)
                u = np.random.rand(self.dim)
                child = np.where(u < CR, v, xi)
                child[jrand] = v[jrand] # Ensure at least one gene is from mutant
                
                # Evaluate Child
                if used >= self.budget:
                    break
                f_child = problem(child)
                used += 1
                f_child = np.array(f_child)
                
                # Selection: Keep the better one (non-dominated)
                # If child dominates parent -> replace
                # If parent dominates child -> keep parent
                # If non-dominated -> keep both (population grows? No, standard DE keeps size fixed)
                # For MOEA, we often expand population or use external archive.
                # Here, we will maintain a fixed-size population for DE dynamics, 
                # but rely on the global archive for the final result.
                
                # Simple selection: If child is non-dominated w.r.t parent, replace.
                # This is a simplified approach. A more robust one would be to keep both
                # in a larger pool, but that complicates the DE loop.
                # Let's use: If child is not dominated by parent, replace. 
                # (This might lose diversity if parent was non-dominated w.r.t child, but child is worse in some sense).
                # Actually, for DE, it's better to just accept if it's not strictly dominated by the current parent.
                
                dominated_by_parent = np.all(fi <= f_child) and np.any(fi < f_child)
                
                if not dominated_by_parent:
                    # Child is better or non-dominated
                    new_pop.append((child, f_child))
                    update_archive(child, f_child)
                else:
                    new_pop.append((xi, fi))
                    # Parent remains in archive if it was there

            population = new_pop
            
            # Optional: If population stagnates or shrinks, inject diversity
            # For now, we just loop.

        return archive

# EVOLVE-BLOCK-END
