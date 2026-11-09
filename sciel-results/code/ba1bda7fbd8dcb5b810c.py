"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with time-adaptive budget and safety margin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        
        # CMA-ES parameters
        dim = self.dim
        # Initial x0: Uniform in [-5, 5] is standard for BBOB, but often 0 is better if shift is small.
        # However, shift is in [-4, 4], so uniform is safer against unknown shift direction.
        x0 = np.zeros(dim) 
        # Actually, for BBOB f6 (Attractive sector), the optimum is shifted. 
        # A random start or 0 is fine. Let's use 0 as it's often central.
        # Wait, the contract says "minimizer lies in [-4, 4]^5". 
        # Uniform random is safer if we don't know the shift. But 0 is a good heuristic.
        # Let's stick to 0 for simplicity, or maybe a small random perturbation.
        # Let's use 0.
        
        sigma0 = 2.0
        # Population size lambda
        # Standard CMA-ES: lambda = 4 + floor(3 * log(dim))
        # For dim=5, lambda = 4 + floor(3*1.6) = 4 + 4 = 8.
        # We can scale this up if time allows.
        
        # Adaptive strategy:
        # 1. Estimate time per function call.
        # 2. Determine max lambda and max generations.
        
        # Initial probe to estimate time per call
        # We must be careful not to waste too many calls on probing.
        # Let's do a small probe.
        x_probe = np.random.default_rng(0).uniform(-5, 5, dim)
        t0 = time.time()
        f_probe = func(x_probe)
        t1 = time.time()
        t_call = t1 - t0
        if t_call < 1e-6: t_call = 1e-6
        
        # Reserve 10% of time for overhead and safety
        time_available = deadline - time.time()
        time_per_call_est = t_call * 1.2 # 20% buffer
        
        # We want to maximize the number of evaluations.
        # Total calls = lambda * generations + 1 (initial)
        # Let's aim for at least 100 calls, maybe more.
        # If time allows, we can increase lambda.
        
        # Let's try to fit as many evaluations as possible.
        # Max evaluations allowed by budget is 500.
        # Max evaluations allowed by time: time_available / time_per_call_est
        max_evals_by_time = int(time_available / time_per_call_est)
        max_evals = min(self.budget, max_evals_by_time)
        
        if max_evals < 10:
             # Fallback to simple random search if time is extremely tight
             best_x = x_probe
             best_f = f_probe
             rng = np.random.default_rng(1)
             while time.time() < deadline:
                 x_new = rng.uniform(-5, 5, dim)
                 f_new = func(x_new)
                 if f_new < best_f:
                     best_x = x_new
                     best_f = f_new
             return best_x, best_f

        # Choose lambda
        # Larger lambda gives better parallelism but more overhead per generation.
        # For dim=5, lambda=8 is standard. lambda=16 might be okay if fast.
        # Let's stick to a robust lambda.
        # If max_evals is large, we can use more generations.
        
        # Let's use a fixed lambda for simplicity and robustness, 
        # but ensure we don't exceed budget.
        lambda_ = 8
        # If we have a lot of time, maybe increase lambda?
        # CMA-ES performance is sensitive to lambda. 
        # Let's keep lambda=8 for dim=5.
        
        generations = max(1, (max_evals - 1) // lambda_)
        
        # CMA-ES implementation
        x = x0
        sigma = sigma0
        C = np.eye(dim)
        pc = np.zeros(dim)
        cs = np.zeros(dim)
        
        # Learning rates
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Initial evaluation
        f_best = f_probe # Use probe as first eval? No, probe was random.
        # Let's re-initialize properly to avoid confusion.
        # The probe was just for timing. Let's start fresh for CMA-ES.
        # But we already used 1 call. 
        # We can use the probe point as one of the offspring or just ignore it for the mean.
        # It's cleaner to start CMA-ES with x0 and evaluate it? No, that wastes a call.
        # Let's just run the loop.
        
        # Actually, the probe was random. Let's just start the CMA-ES loop.
        # We have max_evals - 1 calls left.
        
        rng = np.random.default_rng(42)
        
        # Pre-compute Cholesky of C (initially I)
        # We will update C and sigma.
        
        # To speed up, we can use a simplified CMA-ES or just Nelder-Mead?
        # CMA-ES is robust.
        
        # Let's implement a basic CMA-ES loop.
        # Note: We must strictly track calls.
        
        calls_made = 1 # The probe
        best_x = x_probe
        best_f = f_probe
        
        # Main loop
        for gen in range(generations):
            if time.time() > deadline - 1.0: # Stop if close to deadline
                break
            
            # Generate lambda_ offspring
            # We need to be careful with the number of calls.
            # Each offspring is 1 call.
            
            # Update C and sigma based on previous generation? 
            # We haven't done any generation yet.
            
            # For the first generation, we just generate offspring from x, sigma, C=I.
            
            # Optimization: Instead of full CMA-ES which is complex to implement correctly in 50 lines,
            # maybe use a simpler derivative-free method like Nelder-Mead with restarts?
            # Or a simple Multi-Start Nelder-Mead?
            
            # Given the "Attractive Sector" landscape, which is multi-modal, 
            # a global method like CMA-ES or Differential Evolution is better.
            
            # Let's try Differential Evolution (DE) which is simpler and often robust.
            # DE: x_new = x_r1 + F * (x_r2 - x_r3)
            
            # Let's switch strategy to DE if CMA-ES implementation is too long/risky.
            # DE is very simple to code correctly.
            
            # Reset to DE strategy
            # Population size NP
            # For dim=5, NP=10-20 is good.
            # Let's use NP = 10.
            NP = 10
            F = 0.8
            CR = 0.9
            
            # Initialize population
            # We have 1 call used.
            # We can use the probe as pop[0].
            pop = [x_probe]
            f_pop = [f_probe]
            
            # Fill rest of population
            for i in range(1, NP):
                if time.time() > deadline - 1.0:
                    break
                x_new = rng.uniform(-5, 5, dim)
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                calls_made += 1
                pop.append(x_new)
                f_pop.append(f_new)
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new
            
            if len(pop) < 3:
                break # Not enough population for DE
            
            # Main DE Loop
            while time.time() < deadline - 1.0 and calls_made < self.budget:
                # Update each individual
                for i in range(NP):
                    if time.time() > deadline - 1.0 or calls_made >= self.budget:
                        break
                    
                    # Select r1, r2, r3 distinct from i
                    indices = [j for j in range(NP) if j != i]
                    r1, r2, r3 = rng.choice(indices, 3, replace=False)
                    
                    # Crossover
                    jrand = rng.integers(0, dim)
                    v = np.zeros(dim)
                    for j in range(dim):
                        if rng.random() < CR or j == jrand:
                            v[j] = pop[r1][j] + F * (pop[r2][j] - pop[r3][j])
                        else:
                            v[j] = pop[i][j]
                    
                    v = np.clip(v, -5, 5)
                    
                    # Evaluate
                    f_v = func(v)
                    calls_made += 1
                    
                    if f_v < f_pop[i]:
                        pop[i] = v
                        f_pop[i] = f_v
                        if f_v < best_f:
                            best_f = f_v
                            best_x = v
        
        return best_x, best_f

# EVOLVE-BLOCK-END
