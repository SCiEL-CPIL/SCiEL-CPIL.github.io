import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.archive = []
        self.eval_count = 0
        self.start_time = 0
        self.deadline = 325  # Safety margin for 330s limit

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Initialization: Latin Hypercube Sampling
        # Use a moderate number of initial points to get a rough front
        n_init = 40
        perm = np.random.permutation(n_init)
        X_init = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            X_init[:, d] = np.linspace(lo[d], hi[d], n_init, endpoint=False)[perm] + (hi[d]-lo[d]) * np.random.random() * (1/n_init)
            
        for i in range(n_init):
            f = self._evaluate(problem, X_init[i])
            self._update_archive(X_init[i], f)
            
        # 2. ZDT6-Specific Seeding
        # The Pareto front of ZDT6 is achieved when g(x) = 0.
        # g(x) = 1 + 10(n-1) + sum_{i=2}^n (x_i^2 - 10cos(4*pi*x_i*t))
        # To minimize g, we want x_i^2 - 10cos(4*pi*x_i*t) to be close to -1 (so the term is 0).
        # This happens when x_i is near 0 or 1 (cos(0)=1, x^2=0 -> -10+1=-9? No.
        # Let's check: x_i=0 -> 0 - 10cos(0) = -10. Term = 1 + 10(n-1) - 10(n-1) = 1. Wait.
        # g(x) = 1 + 10(n-1) + sum(x_i^2 - 10cos(4*pi*x_i*t))
        # If x_i=0 for all i>=2: sum = n-1 * (0 - 10) = -10(n-1).
        # g = 1 + 10(n-1) - 10(n-1) = 1.
        # So g(x) = 1 is the minimum? No, g(x) is usually defined to have min 0 or 1.
        # For ZDT6, the Pareto front is f1 + sqrt(f2) = 1? No.
        # ZDT6: f1 = 1 + sin(10*pi*x1)
        # f2 = 1 - sqrt(f1/g(x))
        # If g(x) = 1, then f2 = 1 - sqrt(f1).
        # The known front is f1 + sqrt(f2) = 1? No.
        # The front is f2 = 1 - sqrt(f1).
        # So we want g(x) = 1.
        
        # Let's seed points with x1 varied, and x2..xn = 0.
        n_seed = 20
        x1_vals = np.linspace(0.0, 1.0, n_seed)
        for x1 in x1_vals:
            if self.eval_count >= self.budget: break
            if time.time() > self.deadline: break
            
            x = np.zeros(self.dim)
            x[0] = x1
            # x[1:] = 0.0
            # Add small noise to escape local minima if 0,0,...,0 is not exactly g=1 due to floating point or specific definition
            x[1:] = 0.0 + 0.01 * np.random.randn(self.dim - 1)
            x = np.clip(x, lo, hi)
            
            f = self._evaluate(problem, x)
            self._update_archive(x, f)

        # 3. Local Search / Refinement
        # We will perform a simple local search to refine the points on the front.
        # For each point in the archive, we will try to minimize g(x) (which corresponds to maximizing f2 for a given f1? No, minimizing g increases f2).
        # Actually, for a fixed x1, minimizing g(x) makes f2 larger (closer to 1).
        # Since f2 = 1 - sqrt(f1/g), smaller g means larger sqrt(f1/g) means smaller f2?
        # Wait. f2 = 1 - sqrt(f1/g). If g decreases, f1/g increases, sqrt increases, f2 decreases.
        # We want to minimize objectives. So we want small f1 and small f2.
        # Small f1 comes from small x1.
        # Small f2 comes from large sqrt(f1/g), i.e., small g.
        # So yes, minimizing g is good.
        
        # Let's refine the seeded points.
        # We'll take the best points from the archive (those with small f2 for their f1) and do local search.
        
        # Sort archive by f1 to ensure coverage
        self.archive.sort(key=lambda x: x[1][0])
        
        # Select points to refine
        # We have 200 evals total. Used ~60. ~140 left.
        # Let's pick 10 points evenly spaced in f1.
        if len(self.archive) > 10:
            indices = np.linspace(0, len(self.archive)-1, 10, dtype=int)
            selected = [self.archive[i] for i in indices]
        else:
            selected = self.archive[:]
            
        for x, f in selected:
            if self.eval_count >= self.budget: break
            if time.time() > self.deadline: break
            
            # Local search: try to reduce g by adjusting x2..xn
            # We don't know g explicitly, but we can use the objective values.
            # If we change x2..xn, f1 stays constant (depends only on x1).
            # f2 changes. We want to minimize f2.
            # So we can do a 1D search or random perturbation on x2..xn to minimize f2.
            
            best_x = x.copy()
            best_f = f.copy()
            step_size = 0.1
            
            for _ in range(10): # 10 steps of local search
                if self.eval_count >= self.budget: break
                if time.time() > self.deadline: break
                
                improved = False
                # Try random perturbations
                for _ in range(5):
                    x_new = best_x.copy()
                    # Perturb only x2..xn
                    x_new[1:] += step_size * np.random.randn(self.dim - 1)
                    x_new = np.clip(x_new, lo, hi)
                    
                    f_new = self._evaluate(problem, x_new)
                    
                    # Check if f_new is better (smaller f2, same f1 approx)
                    # Since f1 depends only on x1, f1 should be same.
                    if f_new[1] < best_f[1]:
                        best_x = x_new
                        best_f = f_new
                        improved = True
                        break
                    else:
                        # If not improved, maybe try smaller step?
                        pass
                
                if improved:
                    step_size *= 0.8
                else:
                    step_size *= 0.5
                    if step_size < 0.01:
                        break
                
                self._update_archive(best_x, best_f)
        
        # 4. Final Polish
        while self.eval_count < self.budget:
            if time.time() > self.deadline: break
            
            # Pick a random point from the archive
            if not self.archive: break
            x, f = np.random.choice(self.archive)
            
            # Perturb
            x_new = x + 0.02 * np.random.randn(self.dim)
            x_new = np.clip(x_new, lo, hi)
            
            f_new = self._evaluate(problem, x_new)
            self._update_archive(x_new, f_new)
            
        return self.archive

    def _evaluate(self, problem, x):
        self.eval_count += 1
        return np.array(problem(x))

    def _update_archive(self, x, f):
        # Check if x is dominated by any in archive
        for ax, af in self.archive:
            if np.all(f >= af) and np.any(f > af):
                return # Dominated, skip
        
        # Remove any in archive dominated by x
        self.archive = [(ax, af) for ax, af in self.archive if not (np.all(f <= af) and np.any(f < af))]
        self.archive.append((x, f))
