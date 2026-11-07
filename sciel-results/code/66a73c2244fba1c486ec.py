"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """
    Multi-Scale Differential Evolution Optimizer.
    Maintains three populations of different sizes to balance global exploration 
    and local exploitation. Uses adaptive crossover rates and a shared best archive.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s, but task says 300s limit.
        # We use 290s to be safe against overhead.
        deadline = start_time + 290

        x_best = np.zeros(dim)
        f_best = float('inf')
        calls = 0

        def safe_call(x):
            nonlocal calls, x_best, f_best
            if calls >= budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            calls += 1
            if val < f_best:
                f_best = val
                x_best = x_clipped.copy()
            return val

        rng = np.random.default_rng(42)

        # --- Population Definitions ---
        # Large population for global exploration
        N_large = 50
        # Medium population for basin identification
        N_med = 20
        # Small population for fine-tuning
        N_small = 5

        # Initialize populations
        # Large: uniform random
        pop_large = rng.uniform(-5, 5, (N_large, dim))
        # Medium: uniform random
        pop_med = rng.uniform(-5, 5, (N_med, dim))
        # Small: will be initialized around the best found after first round
        
        # Evaluate initial populations
        for i in range(N_large):
            val = safe_call(pop_large[i])
            if val is None:
                return x_best, f_best
            pop_large[i, -1] = val if dim < 5 else val # This is a hack, let's use separate arrays
            
        # Let's use separate arrays for clarity and safety
        pop_large_f = np.full(N_large, np.inf)
        pop_med_f = np.full(N_med, np.inf)
        pop_small = np.zeros((N_small, dim))
        pop_small_f = np.full(N_small, np.inf)

        # Re-initialize and evaluate properly
        for i in range(N_large):
            val = safe_call(pop_large[i])
            if val is None: return x_best, f_best
            pop_large_f[i] = val
            
        for i in range(N_med):
            val = safe_call(pop_med[i])
            if val is None: return x_best, f_best
            pop_med_f[i] = val

        # Initialize small population around current best
        best_idx_large = np.argmin(pop_large_f)
        best_idx_med = np.argmin(pop_med_f)
        global_best_idx = 0
        if pop_med_f[best_idx_med] < pop_large_f[best_idx_large]:
            global_best_idx = 1 # Just a flag, we already updated x_best in safe_call
            
        # x_best is already updated by safe_call
        for i in range(N_small):
            # Gaussian perturbation around x_best
            sigma = 0.5
            pop_small[i] = x_best + rng.normal(0, sigma, dim)
            val = safe_call(pop_small[i])
            if val is None: return x_best, f_best
            pop_small_f[i] = val

        # --- Adaptive Crossover Rates ---
        # Start with high exploration (high CR), decrease as we converge
        # We will update CR based on improvement
        cr_large = 0.9
        cr_med = 0.7
        cr_small = 0.5
        
        # Decay factor for CR
        cr_decay = 0.995

        # --- Main Loop ---
        # We run in rounds. In each round, we evaluate one new individual from each population.
        # Total calls per round = 3.
        # We have ~300 calls, so ~100 rounds.
        
        # To maximize efficiency, we can batch evaluations if we want, 
        # but the interface is sequential. 
        # However, we can generate multiple candidates and pick the best one to evaluate?
        # No, DE requires evaluating the trial to compare with parent.
        # So we do 1 trial per population per iteration.
        
        # Optimization: If time is running low, switch to pure local search (Nelder-Mead) 
        # or just keep doing DE with shrinking sigma.
        
        # Let's add a phase transition: 
        # Phase 1 (first 50% budget): Full DE
        # Phase 2 (last 50% budget): Focus on Small Pop + Local Polish
        
        phase1_end = budget // 2
        phase2_start = phase1_end

        while calls < budget and time.time() < deadline:
            # Update adaptive parameters
            # Decay CR slowly to shift from exploration to exploitation
            cr_large = max(0.5, cr_large * cr_decay)
            cr_med = max(0.4, cr_med * cr_decay)
            cr_small = max(0.3, cr_small * cr_decay)

            # --- Large Population Update (Global) ---
            if calls < budget:
                i = rng.integers(0, N_large)
                # Select distinct r1, r2, r3
                candidates = list(range(N_large))
                candidates.remove(i)
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                
                # DE/best/1 mutation: x_best + F * (x_r1 - x_r2)
                # Use x_best as the best vector
                F = 0.8
                mutant = x_best + F * (pop_large[r1] - pop_large[r2])
                
                # Crossover
                trial = np.empty(dim)
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < cr_large or j == j_rand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = pop_large[i, j]
                        
                trial = np.clip(trial, -5, 5)
                val = safe_call(trial)
                if val is None:
                    break
                if val < pop_large_f[i]:
                    pop_large[i] = trial
                    pop_large_f[i] = val
                # Update global best is handled in safe_call

            # --- Medium Population Update (Basin ID) ---
            if calls < budget:
                i = rng.integers(0, N_med)
                candidates = list(range(N_med))
                candidates.remove(i)
                r1, r2 = rng.choice(candidates, size=2, replace=False)
                
                # DE/rand/1 mutation: x_r1 + F * (x_r2 - x_i)
                F = 0.7
                mutant = pop_med[r1] + F * (pop_med[r2] - pop_med[i])
                
                trial = np.empty(dim)
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < cr_med or j == j_rand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = pop_med[i, j]
                        
                trial = np.clip(trial, -5, 5)
                val = safe_call(trial)
                if val is None:
                    break
                if val < pop_med_f[i]:
                    pop_med[i] = trial
                    pop_med_f[i] = val

            # --- Small Population Update (Fine-tuning) ---
            if calls < budget:
                i = rng.integers(0, N_small)
                
                # For small pop, use a simpler local mutation: Gaussian around x_best
                # Or DE/best/1 with smaller F
                F = 0.3
                # Pick two random from small pop
                if N_small > 2:
                    others = [j for j in range(N_small) if j != i]
                    r1, r2 = rng.choice(others, size=2, replace=False)
                    mutant = x_best + F * (pop_small[r1] - pop_small[r2])
                else:
                    mutant = x_best + rng.normal(0, 0.1, dim)

                trial = np.empty(dim)
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < cr_small or j == j_rand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = pop_small[i, j]
                        
                trial = np.clip(trial, -5, 5)
                val = safe_call(trial)
                if val is None:
                    break
                if val < pop_small_f[i]:
                    pop_small[i] = trial
                    pop_small_f[i] = val

            # --- Phase Transition: Local Polish ---
            # If we are in the second half, and haven't improved in the last few iterations,
            # try a local Nelder-Mead step from the current best to squeeze out precision.
            # This is optional but can help with smooth functions like Sphere/Rosenbrock.
            # We need to be careful with budget.
            # Let's check if we have enough budget left for a short LM run.
            if calls > phase1_end and calls < budget - 20:
                # Check if we've made progress recently? 
                # Simple heuristic: Every 10 iterations in phase 2, try a local polish
                # if the step count is a multiple of 10.
                if (calls - phase1_end) % 10 == 0:
                    # Run a limited Nelder-Mead
                    # We need to wrap func to count calls and stop early
                    lm_calls = 0
                    def lm_func(x):
                        nonlocal lm_calls
                        if lm_calls >= 15 or time.time() > deadline:
                            return f_best # Return current best to stop improvement
                        val = safe_call(x)
                        lm_calls += 1
                        return val if val is not None else f_best
                    
                    try:
                        res = scipy.optimize.minimize(
                            lm_func, x_best, method='Nelder-Mead',
                            options={'maxiter': 15, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                        )
                        if res.fun < f_best:
                            x_best = res.x
                            f_best = res.fun
                    except Exception:
                        pass

        return x_best, f_best

# EVOLVE-BLOCK-END
