"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer for ZDT6 using structured local search on the Pareto front manifold."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy:
        # 1. The Pareto front of ZDT6 is primarily determined by x_1.
        # 2. For a given x_1, the optimal x_2..x_n minimize the penalty term in f1.
        # 3. We select a set of x_1 values.
        # 4. For each x_1, we perform a local search to find the best x_2..x_n.
        # 5. We collect all evaluated points and return the non-dominated set (or all if < budget).
        
        # We have 200 evaluations.
        # Let's divide them into K groups, each dedicated to a specific x_1.
        # Within each group, we do a local search.
        # Let K = 40 distinct x_1 values.
        # Each gets 200/40 = 5 evaluations.
        # 5 evaluations is enough for a very simple local refinement or a few random perturbations.
        
        K = 40
        evals_per_group = self.budget // K
        remaining_evals = self.budget - (K * evals_per_group)
        
        # Distribute remaining evaluations to the first few groups to give them more depth
        # Or just keep it simple. Let's use K=20, 10 evals each.
        K = 20
        evals_per_group = self.budget // K
        remaining_evals = self.budget - (K * evals_per_group)
        
        # x_1 values: Uniformly spaced, but slightly inset to avoid boundary issues if any
        x1_vals = np.linspace(0.01, 0.99, K)
        
        # Initial guess for x_2..x_n based on analytic minimizer of penalty term
        # x_i = c * x_1^(1/3) where c = 0.25^(2/3)
        c = 0.25 ** (2/3)
        
        all_points = []
        all_objs = []
        
        used_evals = 0
        
        for i in range(K):
            if time.time() > deadline:
                break
                
            x1 = x1_vals[i]
            
            # Initial point for x_2..x_n
            x_rest_init = c * (x1 ** (1/3))
            x_rest_init = np.clip(x_rest_init, lo[1], hi[1])
            
            current_x = np.zeros(self.dim)
            current_x[0] = x1
            current_x[1:] = x_rest_init
            
            # Evaluate initial point
            f = problem(current_x)
            all_points.append(current_x.copy())
            all_objs.append(f)
            used_evals += 1
            
            # Local search: Random perturbations with acceptance if f1 decreases (or f improves in some way)
            # Since we are on the front manifold, minimizing f1 for fixed x1 is the primary goal to stay on the "ideal" curve.
            # However, f2 also depends on x_rest via g(x).
            # A simple random hill climbing on x_rest to minimize f1 is a good proxy.
            
            best_f1 = f[0]
            best_x = current_x.copy()
            
            steps_in_group = evals_per_group - 1
            if i < remaining_evals:
                steps_in_group += 1
                remaining_evals -= 1
                
            # Use a simple random restart / hill climb
            # We perturb x_rest slightly
            for step in range(steps_in_group):
                if time.time() > deadline:
                    break
                if used_evals >= self.budget:
                    break
                    
                # Generate a perturbation
                # Scale of perturbation: decrease as we iterate?
                # Let's use a fixed small scale relative to the range
                scale = 0.1
                perturbation = np.random.uniform(-scale, scale, size=self.dim - 1)
                new_x_rest = best_x[1:] + perturbation
                new_x_rest = np.clip(new_x_rest, lo[1], hi[1])
                
                new_x = np.zeros(self.dim)
                new_x[0] = x1
                new_x[1:] = new_x_rest
                
                new_f = problem(new_x)
                used_evals += 1
                
                # Accept if f1 is lower (closer to minimum for this x1)
                # Or if it's non-dominated by the current best in this group?
                # Since x1 is fixed, lower f1 is generally better for the front position.
                if new_f[0] < best_f1:
                    best_f1 = new_f[0]
                    best_x = new_x.copy()
                    all_points.append(best_x.copy())
                    all_objs.append(new_f)
                else:
                    # Even if not accepted as best, we might want to record it if it's non-dominated globally?
                    # The task records ALL returned objective vectors.
                    # So we should append every evaluation to the global list.
                    all_points.append(new_x.copy())
                    all_objs.append(new_f)
                    
            # Ensure the best point found in this group is definitely in the list
            # (It is, because we appended every eval, but let's make sure the BEST is there)
            # The loop above appends every new_x. If new_x wasn't better, it's still appended.
            # So all evaluations are recorded.
            
        # Return the points. The evaluator records all objective vectors.
        # We don't need to return the points themselves, but the signature requires it.
        # The task says: "The return value of __call__ is ignored."
        # So we can just return None or the points.
        
        return all_points

# EVOLVE-BLOCK-END
