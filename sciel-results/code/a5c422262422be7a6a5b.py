"""Hybrid Differential Evolution and CMA-ES for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """
    Hybrid optimizer using Differential Evolution for global exploration 
    and CMA-ES for local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s kill

        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls
            if n_calls >= budget or check_time():
                return None, None
            n_calls += 1
            try:
                x_clipped = np.clip(x, -5, 5)
                f = float(func(x_clipped))
                return f, x_clipped
            except Exception:
                return None, None

        rng = np.random.default_rng(42)

        # --- Phase 1: Differential Evolution (Global Exploration) ---
        # DE is robust for multimodal landscapes like Weierstrass.
        # We allocate ~60% of the budget to DE.
        de_budget = int(budget * 0.6)
        
        # DE Parameters
        pop_size = 10  # Small population for 5D
        F = 0.8        # Differential weight
        CR = 0.9       # Crossover rate
        max_de_gens = de_budget // pop_size
        
        # Initialize Population
        # Use a mix of random and structured points to ensure coverage
        pop = np.zeros((pop_size, dim))
        # First point: origin (often good starting point)
        pop[0] = np.zeros(dim)
        # Rest: random uniform in [-5, 5]
        for i in range(1, pop_size):
            pop[i] = rng.uniform(-5, 5, dim)
        
        fvals = np.zeros(pop_size)
        
        # Evaluate initial population
        for i in range(pop_size):
            f, x_eval = call_func(pop[i])
            if f is None:
                # Out of budget or time, break
                return best_x, best_f
            fvals[i] = f
            if f < best_f:
                best_f = f
                best_x = x_eval.copy()
        
        # DE Main Loop
        gen = 0
        stagnation_counter = 0
        no_improve_threshold = 5 # Stop if no improvement for 5 generations
        
        while gen < max_de_gens and n_calls < de_budget and not check_time():
            # 1. Mutation
            new_pop = np.zeros((pop_size, dim))
            for i in range(pop_size):
                # Pick 3 distinct random indices
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                # Ensure r1 != i
                while r1 == i:
                    r1 = np.random.choice(pop_size)
                
                # Mutation: x_r1 + F * (x_r2 - x_r3)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                
                # 2. Crossover (Binomial)
                trial = np.zeros(dim)
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip to box
                trial = np.clip(trial, -5, 5)
                new_pop[i] = trial
            
            # 3. Evaluation and Selection
            improved_in_gen = False
            for i in range(pop_size):
                f_trial, x_trial = call_func(new_pop[i])
                if f_trial is None:
                    return best_x, best_f
                
                # Greedy Selection
                if f_trial <= fvals[i]:
                    pop[i] = x_trial
                    fvals[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = x_trial.copy()
                        improved_in_gen = True
            
            gen += 1
            if improved_in_gen:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                if stagnation_counter > no_improve_threshold:
                    break

        # --- Phase 2: CMA-ES (Local Refinement) ---
        # Use remaining budget for CMA-ES
        remaining_budget = budget - n_calls
        if remaining_budget > 10 and not check_time():
            # CMA-ES Parameters
            # Start from best point found by DE
            x0 = best_x.copy()
            # Estimate initial sigma based on population spread or use a heuristic
            # For Weierstrass, a moderate sigma is good
            sigma0 = 1.0
            
            # CMA-ES implementation
            n = dim
            # Adjust lambda based on remaining budget
            # Standard lambda is 4 + 3*log(n). For n=5, lambda=10.
            lam = max(4, int(4 + 3 * np.log(n)))
            # Ensure we don't exceed remaining budget
            max_cma_calls = remaining_budget
            # If lam is too large for remaining budget, reduce it
            if lam > max_cma_calls // 2:
                 lam = max(4, max_cma_calls // 2)
                 
            mu = lam // 2

            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)

            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)

            local_best_x = x.copy()
            local_best_f = best_f
            cma_calls = 0
            stagnation_counter_cma = 0
            no_improve_threshold_cma = 3 * lam

            # Initial population for CMA-ES
            pop_cma = np.zeros((lam, n))
            fvals_cma = np.zeros(lam)
            
            for i in range(lam):
                if cma_calls >= max_cma_calls or check_time():
                    break
                z = rng.standard_normal(n)
                x_trial = x + sigma * z
                f, x_eval = call_func(x_trial)
                if f is None:
                    break
                cma_calls += 1
                pop_cma[i] = x_eval
                fvals_cma[i] = f
                if f < local_best_f:
                    local_best_f = f
                    local_best_x = x_eval.copy()
                    stagnation_counter_cma = 0
                else:
                    stagnation_counter_cma += 1

            # Main CMA-ES Loop
            while cma_calls < max_cma_calls and not check_time():
                order = np.argsort(fvals_cma)
                pop_cma = pop_cma[order]
                fvals_cma = fvals_cma[order]

                x_new = np.sum(weights[:, None] * pop_cma[:mu], axis=0)

                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-20, None)
                    Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    Cinv_sqrt = np.eye(n)

                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * Cinv_sqrt @ (x_new - x)
                x = x_new

                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (cma_calls // lam)))) < (1.4 + 2.0 / (n + 1)) * chi_n
                pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * (x_new - x)

                C = (1.0 + c1 * hsig * (1.0 - hsig) - c1 - cmu) * C + c1 * np.outer(pc, pc)
                for i in range(mu):
                    y_i = (pop_cma[i] - x) / sigma
                    C += cmu * weights[i] * np.outer(y_i, y_i)
                C = (C + C.T) / 2.0

                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = np.clip(sigma, 1e-15, 1e6)

                improved_in_gen = False
                for i in range(lam):
                    if cma_calls >= max_cma_calls or check_time():
                        break
                    z = rng.standard_normal(n)
                    x_trial = x + sigma * (Cinv_sqrt @ z)
                    f, x_eval = call_func(x_trial)
                    if f is None:
                        break
                    cma_calls += 1
                    pop_cma[i] = x_eval
                    fvals_cma[i] = f
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = x_eval.copy()
                        stagnation_counter_cma = 0
                        improved_in_gen = True
                    else:
                        stagnation_counter_cma += 1
                
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()

                if stagnation_counter_cma > no_improve_threshold_cma:
                    break
                if sigma < 1e-10:
                    break

            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

        # --- Phase 3: Powell (Final Polish) ---
        remaining = budget - n_calls
        if remaining >= 10 and not check_time():
            def wrapped_func(x):
                nonlocal n_calls, best_f, best_x
                if n_calls >= budget or check_time():
                    return best_f
                x_clipped = np.clip(x, -5, 5)
                n_calls += 1
                try:
                    f = float(func(x_clipped))
                    if f < best_f:
                        best_f = f
                        best_x = x_clipped.copy()
                    return f
                except Exception:
                    return best_f

            try:
                # Use Powell for final refinement
                # maxiter is not direct eval count, but limit it
                max_iter = min(remaining, 50)
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    best_x, 
                    method='Powell',
                    options={'maxiter': max_iter, 'disp': False}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
