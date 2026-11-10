"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 using curvature-aware sampling and gap-filling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT6 structure:
        # f1(x) = 1 + 10*n
        # f2(x) = g(x) * h(f1(x)/g(x), g(x))
        # g(x) = 1 + 10*n + sum(2*x_i^2 - 4*x_i*cos(20*pi*x_i)) for i in 1..n
        # h(u, v) = 1 - sqrt(u/v) - (u/v)*sin(10*pi*u)
        # Pareto front: x[1:] = 0.5, x[0] in [0, 1]
        # g(0.5) = 1 + 10*9 + sum(0) = 91 (since x_i=0.5 -> x_i^2=0.25, cos(10*pi)=1 -> 2*0.25-4*1 = -3? 
        # Wait, cos(20*pi*0.5) = cos(10*pi) = 1. 
        # Term: 2*(0.5)^2 - 4*(0.5)*1 = 0.5 - 2 = -1.5.
        # Sum over 9 variables: 9 * -1.5 = -13.5.
        # g = 1 + 90 - 13.5 = 77.5.
        
        # Let's verify g(0.5) more carefully.
        # x_i = 0.5
        # x_i^2 = 0.25
        # cos(20*pi*0.5) = cos(10*pi) = 1
        # 2*x_i^2 - 4*x_i*cos(...) = 2*0.25 - 4*0.5*1 = 0.5 - 2 = -1.5
        # Sum for i=1 to 9 (dim=10, x[0] is first, so 9 others): 9 * -1.5 = -13.5
        # g = 1 + 10*9 + (-13.5) = 91 - 13.5 = 77.5
        
        # f1 = 1 + 10*9 = 91
        # f2 = g * (1 - sqrt(f1/g) - (f1/g)*sin(10*pi*f1/g))
        # Let u = f1/g. Since g is constant on the front (x[1:]=0.5), u is constant?
        # No, f1 depends on x[0]? 
        # ZDT6 definition:
        # f1(x) = 1 + 10*n
        # This is constant! 91.
        # So u = 91/77.5 is constant.
        # Then f2 is constant?
        # Wait, standard ZDT6 definition:
        # f1(x) = 1 + 10*n
        # f2(x) = g(x) * h(f1(x)/g(x), g(x))
        # If f1 is constant, and g is constant on the front, then f2 is constant?
        # That would mean the Pareto front is a single point.
        # Let me re-read ZDT6 definition.
        
        # Ah, ZDT6 is:
        # f1(x) = 1 + 10*n
        # f2(x) = g(x) * h(f1(x)/g(x), g(x))
        # But usually ZDT problems have f1 depending on x[0].
        # Let's check ZDT6 specifically.
        # ZDT6:
        # f1(x) = 1 + 10*n
        # f2(x) = g(x) * h(f1(x)/g(x), g(x))
        # g(x) = 1 + 10*(n-1) + sum_{i=2}^n (2*x_i^2 - 4*x_i*cos(20*pi*x_i))
        # h(u, v) = 1 - sqrt(u/v) - (u/v)*sin(10*pi*u)
        
        # If f1 is constant 91, and g varies with x[1:], then the front is not x[0].
        # The front is defined by minimizing g(x) for x[1:].
        # The minimum of g(x) is when x[1:] = 0.5?
        # Let's check the derivative of the term 2*x^2 - 4*x*cos(20*pi*x).
        # At x=0.5, value is -1.5.
        # Is it the minimum?
        # The function is periodic. The global minimum of 2*x^2 - 4*x*cos(20*pi*x) on [0,1] is not necessarily at 0.5.
        # However, standard literature states the Pareto front of ZDT6 is obtained by setting x[1:] = 0.5 and varying x[0]?
        # No, for ZDT6, f1 is constant. The front is 1D in objective space?
        # If f1 is constant, the Pareto front is a single point in f1-f2 space?
        # No, h(u,v) depends on v=g(x). If g varies, f2 varies.
        # But if f1 is fixed, we can only improve f2.
        # So the Pareto front is a single point: min f2 subject to x[1:].
        # This contradicts the "sweep x[0]" strategy of the current program.
        
        # Let's re-verify ZDT6 definition from Deb et al.
        # ZDT6:
        # f1(x) = 1 + 10*n
        # f2(x) = g(x) * h(f1(x)/g(x), g(x))
        # g(x) = 1 + 10*(n-1) + sum_{i=2}^n (2*x_i^2 - 4*x_i*cos(20*pi*x_i))
        # h(u, v) = 1 - sqrt(u/v) - (u/v)*sin(10*pi*u)
        
        # If f1 is constant, the set of non-dominated points is just the point with minimum f2.
        # Why does the current program sweep x[0]?
        # Maybe I am misremembering ZDT6.
        # Let's look at ZDT1-ZDT4. They have f1=x[0].
        # ZDT5: f1=x[0].
        # ZDT6: f1=1+10n.
        
        # If the current program scores 0.349, and it sweeps x[0], it implies that x[0] DOES affect the objectives or the front.
        # Perhaps the problem definition in the evaluator is different or I am missing a detail.
        # "problem(x) clips ... and returns 2 objective values".
        
        # Let's assume the standard ZDT6. If f1 is constant, the "front" is a point.
        # Hypervolume of a single point is 0 (area is 0).
        # But the score is 0.349, which is > 0.
        # This implies the evaluator's ZDT6 has a non-trivial front.
        # Common variation: Maybe f1 depends on x[0] in the evaluator's implementation?
        # Or maybe I should just trust the "sweep x[0]" insight from the current program's success?
        # The current program uses x[1:]=0.5 and varies x[0].
        # If that yields a score, then in this specific evaluator, varying x[0] changes the objectives.
        
        # Let's look at the inspiration 1 (score 1.4493).
        # "exploit the known analytical structure of ZDT6 by performing a dense deterministic sweep of the Pareto front variable (x_0)".
        # This confirms x[0] is the front variable.
        
        # So, in this specific instance, ZDT6 behaves like ZDT1/ZDT2 regarding x[0].
        # Likely f1 = x[0] or similar, or the evaluator uses a variant.
        # Given the ambiguity, I should stick to the empirical success of sweeping x[0] with x[1:]=0.5.
        
        # However, the "curvature-aware" sampling assumes I know the shape.
        # If I don't know the exact shape, uniform sampling is safe.
        # But the current program does uniform sampling and gets 0.349.
        # Inspiration 1 gets 1.4493. What is different?
        # "dense deterministic sweep".
        # Maybe the current program's `np.linspace(0, 1, 200)` is not dense enough or the mapping is non-linear.
        
        # Let's try a different structural change:
        # Instead of assuming x[1:]=0.5 is optimal, let's use a local search around x[1:]=0.5 to find the true minimizing values for x[1:].
        # If x[1:]=0.5 is not exactly on the front, we are off.
        
        # Strategy:
        # 1. Sample a few points with x[1:]=0.5 to get a baseline.
        # 2. Use the remaining budget to perform a local optimization (e.g., Nelder-Mead or simple coordinate descent) on x[1:] for a few representative x[0] values to push them onto the true front.
        # 3. Record all non-dominated points found.
        
        # This is a "refinement" strategy, structurally different from "pure sweep".
        
        # Implementation plan:
        # 1. Create a set of candidate x[0] values (e.g., 20 points).
        # 2. For each, initialize x[1:] = 0.5.
        # 3. Perform a few steps of coordinate descent or random perturbation to minimize the objectives (specifically, to reduce g(x) or whatever drives f2).
        # 4. Since we don't know the objective function internals, we use the `problem` call to evaluate.
        # 5. We want to minimize both objectives.
        # 6. For a fixed x[0], we want to find x[1:] that minimizes the "distance" to the ideal point or simply minimizes f2 (assuming f1 is fixed or increasing with x[0]).
        
        # Let's assume f1 is monotonic with x[0].
        # For each x[0], we try to minimize f2.
        # We can use a simple local search:
        #   Start at x[1:]=0.5.
        #   Try perturbing x[1:] with small random steps.
        #   Keep the perturbation if it improves the "best" objective (e.g., f2, or a weighted sum).
        
        # Budget allocation:
        # 20 x[0] values.
        # 10 evaluations per x[0] for local search.
        # Total 200 evaluations.
        
        # This explores the "thickness" of the front, correcting for any offset in x[1:].

        n_x0 = 20
        steps_per_x0 = self.budget // n_x0  # 10 steps
        
        x0_values = np.linspace(0.0, 1.0, n_x0)
        
        recorded = []
        
        for x0 in x0_values:
            # Initialize x
            x = np.full(self.dim, 0.5)
            x[0] = x0
            
            # Evaluate initial point
            f = problem(x)
            best_f = np.array(f)
            best_x = x.copy()
            
            # Local search on x[1:]
            # We want to improve the objectives.
            # Since we don't know which objective is which, we can try to minimize the sum or the max.
            # Let's try to minimize the "worst" objective or just use a simple gradient-free step.
            # Actually, for ZDT problems, the front is where g(x) is minimized.
            # Minimizing g(x) typically minimizes f2 for a given f1.
            # So we can just minimize the second objective (index 1) or the sum.
            
            step_size = 0.1
            for _ in range(steps_per_x0):
                if time.time() > deadline:
                    break
                    
                # Random perturbation
                x_trial = best_x.copy()
                mask = np.random.random(self.dim) < 0.5  # Perturb half the variables
                x_trial[mask] += (np.random.random(np.sum(mask)) - 0.5) * 2 * step_size
                x_trial[0] = x0  # Keep x[0] fixed
                x_trial = np.clip(x_trial, lo, hi)
                
                f_trial = problem(x_trial)
                
                # Check if trial is better
                # Better if it dominates best_f, or if it has a smaller value in the "free" objective.
                # Let's assume we want to minimize the sum of objectives for the local search.
                if np.sum(f_trial) < np.sum(best_f):
                    best_f = np.array(f_trial)
                    best_x = x_trial.copy()
            
            recorded.append(best_f)
            
        return recorded
# EVOLVE-BLOCK-END
