import numpy as np
import scipy.optimize as so
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = 0
        self.time_limit = 275  # Safety margin for 300s limit

    def __call__(self, func):
        self.start_time = time.time()
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        def check_deadline():
            return time.time() - self.start_time > self.time_limit
        
        def evaluate(x):
            if self.calls >= self.budget or check_deadline():
                return None
            self.calls += 1
            return func(x)
        
        best_x = None
        best_f = np.inf
        all_points = []
        all_values = []
        
        # Phase 1: Latin Hypercube Sampling for initial exploration
        n_init = min(40, self.budget // 4)
        lhs = np.zeros((n_init, dim))
        for i in range(dim):
            perm = rng.permutation(n_init)
            lhs[:, i] = (perm + 0.5) / n_init
        
        x_init = lhs * 10 - 5
        
        for x in x_init:
            fx = evaluate(x)
            if fx is None:
                break
            all_points.append(x)
            all_values.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            return None, np.inf

        # Phase 2: Derandomized Evolutionary Strategy (DES)
        # This replaces the random restarts with a structured population-based search.
        # We maintain a population of the best candidates and apply deterministic mutations.
        
        # Build initial population from LHS points, sorted by fitness
        pop_indices = np.argsort(all_values)
        pop_size = min(10, len(pop_indices))
        population = [all_points[i].copy() for i in pop_indices[:pop_size]]
        
        # Define deterministic mutation vectors (Hadamard-like or simple orthogonal basis)
        # For 5D, we can use a set of fixed directions.
        # Let's use a simple set of orthogonal directions plus diagonal ones.
        mutation_vectors = []
        # Coordinate axes
        for i in range(dim):
            v = np.zeros(dim)
            v[i] = 1.0
            mutation_vectors.append(v)
        # Diagonal directions (all same sign, alternating)
        v_diag1 = np.ones(dim)
        mutation_vectors.append(v_diag1 / np.linalg.norm(v_diag1))
        v_diag2 = np.ones(dim)
        v_diag2[0] = -1
        mutation_vectors.append(v_diag2 / np.linalg.norm(v_diag2))
        
        # Normalize mutation vectors
        for i in range(len(mutation_vectors)):
            mutation_vectors[i] = mutation_vectors[i] / np.linalg.norm(mutation_vectors[i])
        
        # DES Loop
        # We will iterate, trying to improve the population.
        # Each iteration, we take the best individual, mutate it, and if it improves, replace the worst.
        # We also keep a global best.
        
        # Allocation: Use 60% of remaining budget for DES, 40% for final polishing
        des_budget = int((budget - self.calls) * 0.6)
        
        des_calls = 0
        max_des_iters = des_budget // 2 # Approximate, as each iter might use 1-2 calls
        
        for it in range(max_des_iters):
            if self.calls >= budget or check_deadline():
                break
            
            # Select the best individual in the current population
            # Since we don't track fitness of pop members directly in a list, let's re-evaluate or track.
            # Actually, we should track fitness.
            # Let's maintain a list of (x, f) for the population.
            
            # Re-initialize pop structure to track fitness
            # We already have all_points and all_values.
            # Let's just use the top `pop_size` from all_points/all_values as the active population.
            # But as we generate new points, we should insert them into this set.
            
            # Simpler approach:
            # 1. Get current top `pop_size` from all_points/all_values.
            # 2. Take the best one.
            # 3. Mutate it using a deterministic vector.
            # 4. Evaluate the mutant.
            # 5. If it's better than the worst in the population, replace it.
            # 6. Update global best.
            
            # Get current top candidates
            if len(all_points) == 0:
                break
            sorted_idx = np.argsort(all_values)
            top_idx = sorted_idx[:pop_size]
            # Sort top_idx by fitness to ensure population[0] is best
            top_idx = top_idx[np.argsort([all_values[i] for i in top_idx])]
            
            best_pop_idx = top_idx[0]
            x_parent = all_points[best_pop_idx].copy()
            f_parent = all_values[best_pop_idx]
            
            # Choose mutation vector deterministically
            # Cycle through vectors based on iteration
            vec_idx = it % len(mutation_vectors)
            v = mutation_vectors[vec_idx]
            
            # Adapt step size
            # Start with a larger step, decrease over time
            base_step = 2.0
            step = base_step * (0.9 ** (it / 10.0)) # Exponential decay
            
            # Generate mutant
            x_mut = x_parent + step * v
            x_mut = np.clip(x_mut, -5, 5)
            
            f_mut = evaluate(x_mut)
            if f_mut is None:
                break
            
            all_points.append(x_mut)
            all_values.append(f_mut)
            
            if f_mut < best_f:
                best_f = f_mut
                best_x = x_mut.copy()
            
            # Update population: if mutant is better than worst in top `pop_size`, it's implicitly there
            # No explicit replacement needed if we just keep all points and re-select top K each time.
            # This is computationally cheap for K=10 and N=500.
            
            # Optional: If we stagnate (no improvement in last 10 iters), increase step size or try different vector
            # For now, the deterministic cycling and decay should handle it.
        
        # Phase 3: Local Refinement using L-BFGS-B from top candidates
        # Use remaining budget
        if self.calls < budget and not check_deadline():
            # Select top 3 candidates for local search
            if len(all_points) > 0:
                sorted_idx = np.argsort(all_values)
                n_local = min(3, len(sorted_idx), (budget - self.calls) // 20)
                
                for i in range(n_local):
                    if self.calls >= budget or check_deadline():
                        break
                    
                    start_x = all_points[sorted_idx[i]].copy()
                    
                    # Wrap func for L-BFGS-B
                    def local_func(x):
                        if self.calls >= self.budget or check_deadline():
                            return best_f
                        self.calls += 1
                        return func(x)
                    
                    try:
                        res = so.minimize(
                            local_func, start_x,
                            method='L-BFGS-B',
                            bounds=[(-5, 5)] * dim,
                            options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
                        )
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x.copy()
                    except Exception:
                        pass
        
        # Phase 4: Final high-precision polish if budget remains
        if self.calls < budget - 10 and not check_deadline() and best_x is not None:
            def final_func(x):
                if self.calls >= self.budget or check_deadline():
                    return best_f
                self.calls += 1
                return func(x)
            
            try:
                res = so.minimize(
                    final_func, best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        return best_x, best_f
