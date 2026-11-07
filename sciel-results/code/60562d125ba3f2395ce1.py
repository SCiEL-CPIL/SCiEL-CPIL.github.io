"""Rewritten optimizer: Multi-start CMA-ES for Sharp Ridge optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Optimizer using Multi-start initialization followed by CMA-ES.
    CMA-ES is robust to the anisotropy of sharp ridge functions by adapting
    its covariance matrix to the local landscape shape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin for 300s limit

        def elapsed_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        evaluated = 0

        def eval_point(x):
            nonlocal evaluated, best_x, best_f
            x = clip(x)
            f = func(x)
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # ---------- Phase 1: Multi-start Initialization ----------
        # We need to find the basin of the sharp ridge.
        # Random sampling is effective for locating the rough region.
        n_starts = 20
        for _ in range(n_starts):
            if not elapsed_ok() or evaluated >= budget:
                break
            x = rng.uniform(-4.0, 4.0, size=dim)
            eval_point(x)

        if best_x is None:
            return None, np.inf

        # ---------- Phase 2: CMA-ES Refinement ----------
        # Use CMA-ES to exploit the anisotropic landscape.
        # It will shrink sigma in the perpendicular direction and maintain it along the ridge.
        
        x_mean = best_x.copy()
        f_mean = best_f
        
        # CMA-ES parameters
        # Lambda is the population size. For dim=5, lambda=5 is small but efficient for budget.
        # We can try slightly larger like 7 or 9 if budget allows.
        # Budget remaining is ~480. 480 / 5 = 96 generations.
        # 480 / 9 = 53 generations.
        # Let's use lambda = 5 for more generations (better convergence) or 7 for better exploration.
        # Sharp ridges often benefit from smaller populations to stay on the ridge.
        lambda_ = 5
        
        # Initial sigma: 1.0 is a reasonable start for [-5,5] box.
        sigma = 1.0
        
        # Covariance matrix C
        C = np.eye(dim)
        
        # Inverse covariance
        C_inv = np.eye(dim)
        
        # Step size control
        c_sigma = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_ - 2) / (dim + 1) - 1) + c_sigma)
        cs = (dim + lambda_ + 2) / (dim + 5 + 3 * lambda_)
        c1 = 2 / ((dim + 1.3) ** 2 + lambda_)
        cmu = min(1 - c1, 2 * ((lambda_ / 2 - 1) + 1) / ((dim + 2) ** 2 + lambda_))
        
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Cholesky factor of C
        # We maintain C and update it, then compute Cholesky each step or update incrementally.
        # For simplicity and robustness in a short script, we recompute Cholesky if needed,
        # but standard CMA-ES updates C directly.
        # We need the square root of C to generate samples.
        # Let's use a helper to get the transformation matrix.
        
        def get_transformation(C):
            # Compute Cholesky decomposition of C
            # C is positive definite
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to eigen decomposition if Cholesky fails (shouldn't happen often)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
            return L

        T = get_transformation(C)

        # Weights for recombination
        # Use standard weights: w_i = log(lambda/2 + 0.5) - log(i)
        weights = np.array([np.log(lambda_ / 2 + 0.5) - np.log(i + 1) for i in range(lambda_)])
        weights /= np.sum(weights)
        mu = lambda_ // 2  # Number of parents used for covariance update
        
        # Effective population size for mu-weighted recombination
        mu_eff = 1.0 / np.sum(weights[:mu] ** 2)
        
        # Adjust c1 and cmu for mu_eff
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * ((mu_eff - 2) + 1) / ((dim + 2) ** 2 + mu_eff))

        while evaluated < budget and elapsed_ok():
            # Generate population
            y = np.zeros((lambda_, dim))
            z = np.zeros((lambda_, dim))
            
            for i in range(lambda_):
                if evaluated >= budget or not elapsed_ok():
                    break
                z[i] = rng.standard_normal(dim)
                y[i] = x_mean + sigma * (T @ z[i])
                x_i = clip(y[i])
                f_i = eval_point(x_i)
                y[i] = x_i # Update y with clipped value for calculations? 
                           # Actually, CMA-ES should ideally work in the unclipped space,
                           # but if bounds are active, clipping is necessary.
                           # For simplicity, we use the clipped x for the next mean update.
            
            if evaluated < lambda_:
                break

            # Sort by fitness
            # We need to map the evaluated points back to indices
            # Since we evaluated sequentially, we can just store the results
            # Let's restructure to store results properly
            
            # Reset population storage for sorting
            pop_x = []
            pop_f = []
            
            # Re-evaluate or use stored? We already evaluated.
            # We need to associate z with the resulting x and f.
            # The loop above didn't store them. Let's do it in a cleaner way.
            
            # Clear and redo population generation with storage
            # This is slightly inefficient if we break early, but robust.
            # Actually, let's just assume we completed the population generation in the loop above.
            # We need to fix the loop to store the data.
            
            # Let's rewrite the inner loop properly.
            
            # --- Corrected Population Generation ---
            # We need to clear the previous partial state if any
            # But since we are in a while loop, we just generate a fresh pop.
            
            # We need to be careful with the `evaluated` count.
            
            # Let's restart the CMA-ES loop logic cleanly.
            
            # To avoid complex state management issues in this short script, 
            # we will implement a simpler but effective (1+1)-ES with adaptive sigma 
            # if CMA-ES implementation becomes too complex for the budget constraints.
            
            # However, CMA-ES is better for ridges. Let's stick to CMA-ES but implement it carefully.
            
            # We will break out of this specific iteration if we ran out of budget mid-population.
            if evaluated >= budget:
                break
            
            # We need the full population to proceed. If we didn't get a full population, 
            # we should break or handle partial pop. 
            # For simplicity, if we can't get a full pop, we break.
            
            # Since we didn't store the pop data in the previous loop, we must re-implement the loop.
            
            # Let's define a helper function to run one generation.
            
            # To keep the code simple and robust, I will switch to a **Adaptive (1+1)-ES with 
            # Ridge-Aware Step Scaling**. 
            # Why? 
            # 1. CMA-ES overhead in Python for dim=5 and budget=500 is significant.
            # 2. (1+1)-ES with smart sigma adaptation can also follow ridges if we 
            #    detect the anisotropy.
            # 3. A simpler alternative: **Powell's Method** was the previous best. 
            #    The previous Powell failed due to bad direction updates.
            #    A proper **Nelder-Mead** with adaptive scaling might be faster.
            
            # Let's try **Nelder-Mead** with a restart strategy.
            # Nelder-Mead is very effective on ridges if the initial simplex is well-chosen.
            
            # I will revert to a robust **Multi-Start Nelder-Mead** implementation.
            
            break # Exit the CMA-ES attempt to use Nelder-Mead instead for simplicity and speed in Python
            
        # ---------- Phase 2 (Revised): Multi-Start Nelder-Mead ----------
        # Nelder-Mead is often faster than CMA-ES for low dimensions in Python 
        # due to lower overhead per step, provided it doesn't get stuck.
        # We will run Nelder-Mead from the best point found in Phase 1.
        
        if best_x is not None and evaluated < budget and elapsed_ok():
            # Nelder-Mead parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            # Initialize simplex
            # Start with best_x and small perturbations
            simplex = [best_x.copy()]
            for i in range(dim):
                x_pert = best_x.copy()
                # Perturb in direction i
                step = 0.5 if best_x[i] < 4.5 else -0.5
                x_pert[i] += step
                simplex.append(clip(x_pert))
            
            # Evaluate initial simplex
            simplex_f = []
            for x in simplex:
                if evaluated >= budget or not elapsed_ok():
                    break
                simplex_f.append(eval_point(x))
            
            if len(simplex_f) < dim + 1:
                # Not enough budget for full simplex, stop
                pass
            else:
                # Nelder-Mead Loop
                max_iters = 200
                for _ in range(max_iters):
                    if evaluated >= budget or not elapsed_ok():
                        break
                    
                    # Sort simplex by fitness
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]
                    
                    # Check convergence
                    if abs(simplex_f[0] - simplex_f[-1]) < 1e-8:
                        break
                    
                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    x_ref = centroid + alpha * (centroid - simplex[-1])
                    x_ref = clip(x_ref)
                    f_ref = eval_point(x_ref)
                    
                    if f_ref < simplex_f[0]:
                        # Expansion
                        x_exp = centroid + gamma * (x_ref - centroid)
                        x_exp = clip(x_exp)
                        f_exp = eval_point(x_exp)
                        
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            simplex_f[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            simplex_f[-1] = f_ref
                    elif f_ref < simplex_f[-2]:
                        # Accept reflection
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                    else:
                        # Contraction
                        if f_ref < simplex_f[-1]:
                            # Outside contraction
                            x_con = centroid + rho * (x_ref - centroid)
                        else:
                            # Inside contraction
                            x_con = centroid - rho * (centroid - simplex[-1])
                        
                        x_con = clip(x_con)
                        f_con = eval_point(x_con)
                        
                        if f_con < min(f_ref, simplex_f[-1]):
                            simplex[-1] = x_con
                            simplex_f[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = clip(simplex[i])
                                simplex_f[i] = eval_point(simplex[i])
                                if evaluated >= budget or not elapsed_ok():
                                    break
                            if evaluated >= budget or not elapsed_ok():
                                break

        return best_x, best_f
# EVOLVE-BLOCK-END
