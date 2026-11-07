"""Global LHS search + Adaptive Differential Evolution for BBOB f4 (Büche-Rastrigin)."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        call_count = 0
        best_x = np.zeros(dim, dtype=float)
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def do_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                raise ValueError("Budget exceeded")
            call_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_ok():
            return time.time() < deadline

        # ---------- Phase 1: Latin Hypercube Global Search ----------
        # LHS provides better space-filling properties than random sampling
        n_lhs = 80
        # Generate LHS points in [0, 1]^dim
        lhs_pts = np.zeros((n_lhs, dim))
        for i in range(dim):
            # Shuffle the sequence [0, 1/n, 2/n, ...] to create stratified sampling
            order = rng.permutation(n_lhs)
            lhs_pts[:, i] = (order + rng.uniform(0, 1, n_lhs)) / n_lhs
        
        # Map to [-5, 5]^dim
        lhs_pts = lhs_pts * 10.0 - 5.0

        # Evaluate LHS points
        for p in lhs_pts:
            if call_count >= budget or not time_ok():
                break
            f = do_call(p)

        # ---------- Phase 2: Differential Evolution (DE) ----------
        # Initialize population around the best point found so far
        # But keep some diversity to escape local minima
        pop_size = 20
        # 1. Best point
        # 2. Points from the LHS that were good (top 5)
        # 3. Random points near best
        # We don't have stored LHS f-values easily without overhead, so we just sample
        # around the current best and keep some random global points
        
        # Let's initialize population with a mix
        pop = []
        pop.append(best_x.copy())
        
        # Add some random points in a wider box around best, but clipped
        for _ in range(pop_size - 1):
            # Mix of local and global
            if rng.random() < 0.3:
                # Global random
                p = rng.uniform(-5.0, 5.0, dim)
            else:
                # Local random around best
                p = best_x + rng.normal(0, 1.5, dim)
            p = np.clip(p, -5.0, 5.0)
            pop.append(p)
        
        # Evaluate initial population (skip the one we already have if it's exact match? 
        # Actually, do_call handles budget, so we just call. To save calls, we can 
        # check if p is close to best_x, but best_x might not be in pop exactly if 
        # we added it. Let's just evaluate all. 
        # Wait, we already evaluated best_x in LHS? Not necessarily, best_x is the 
        # argmin of LHS. So we have its value. But do_call doesn't know that.
        # To be safe and simple, we evaluate the new population.
        
        pop_f = []
        for i, p in enumerate(pop):
            if call_count >= budget or not time_ok():
                break
            if i == 0:
                # We know best_f, but do_call will increment count.
                # Is it worth saving 1 call? Yes.
                # But do_call updates best_x/best_f. Since p[0] IS best_x, 
                # calling it again is redundant.
                # However, the interface requires __call__ to return best_x, best_f.
                # And the evaluator records calls.
                # If we skip the call, we save 1 budget unit.
                # But we need to be careful. Let's just call it to be safe with state.
                # Actually, best_x is a copy. p[0] is a copy.
                # Let's just call it. It's 1 out of 500.
                f = do_call(p)
            else:
                f = do_call(p)
            pop_f.append(f)
        
        # Sort population by fitness
        pop_idx = np.argsort(pop_f)
        pop = [pop[i] for i in pop_idx]
        pop_f = [pop_f[i] for i in pop_idx]
        
        # DE Parameters
        # F: Mutation factor. Start high for exploration, decrease for exploitation.
        # CR: Crossover rate.
        
        max_de_iter = 50
        iter_count = 0
        
        while call_count < budget - 10 and time_ok() and iter_count < max_de_iter:
            iter_count += 1
            
            # Adaptive F: decrease as we get closer to convergence or budget
            # Simple linear decay
            progress = iter_count / max_de_iter
            F = 0.8 - 0.5 * progress  # 0.8 -> 0.3
            CR = 0.5
            
            new_pop = []
            new_pop_f = []
            
            for i in range(pop_size):
                if call_count >= budget - 10 or not time_ok():
                    break
                    
                # Select parents
                # Avoid i
                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(candidates, size=3, replace=False)
                
                # Mutation: v = x_a + F * (x_b - x_c)
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Crossover
                trial = np.zeros(dim)
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip
                trial = np.clip(trial, -5.0, 5.0)
                
                # Evaluate
                f_trial = do_call(trial)
                
                # Selection
                if f_trial <= pop_f[i]:
                    new_pop.append(trial)
                    new_pop_f.append(f_trial)
                else:
                    new_pop.append(pop[i])
                    new_pop_f.append(pop_f[i])
            
            if not new_pop:
                break
                
            # Update population
            pop = new_pop
            pop_f = new_pop_f
            
            # Re-sort
            pop_idx = np.argsort(pop_f)
            pop = [pop[i] for i in pop_idx]
            pop_f = [pop_f[i] for i in pop_idx]

        return best_x, best_f
