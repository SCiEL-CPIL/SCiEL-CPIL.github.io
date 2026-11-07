"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES implementation for BBOB-style optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # lambda is population size. For dim=5, lambda=10-20 is reasonable.
        # We have 500 budget. If lambda=10, we can do 49 generations (490 calls) + 1 initial.
        # This leaves room for safety.
        # Let's use lambda = 12 to allow ~41 generations.
        # Or adapt lambda? Fixed is safer for budget control.
        # Let's try lambda = 10. 500 / 10 = 50 generations.
        # We must be careful: initial point + 50*10 = 501 calls? 
        # No, CMA-ES usually starts with a center point, then samples lambda.
        # Total calls = 1 (center) + generations * lambda.
        # If budget=500, max generations = floor((500-1)/10) = 49.
        
        lambda_pop = 10
        max_generations = (budget - 1) // lambda_pop
        
        # Initialize
        # Start at 0,0,0,0,0 is often good for BBOB if shift is small, 
        # but shift is in [-4,4]. Random start might be safer against bad luck,
        # but CMA-ES is robust. Let's start at 0.
        m = np.zeros(dim)
        f_m = func(m)
        
        # Initial sigma: 
        # Box is [-5, 5]. Diameter is 10.
        # A large sigma helps global search. 
        # Standard CMA-ES starts with sigma0 = 0.25 * (max - min) or similar.
        # Here range is 10. 0.25 * 10 = 2.5.
        # Let's use 2.0 to stay safely inside initially but explore widely.
        sigma = 2.0
        
        # Covariance matrix C
        C = np.eye(dim)
        
        # Step size control parameters (standard CMA-ES)
        mu = lambda_pop // 2  # Number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Track best for return value (though evaluator ignores it, we must not crash)
        best_x = m.copy()
        best_f = f_m
        
        # Pre-compute constants for efficiency
        norm_sigma = 1.0
        # We need to update C every generation.
        # For dim=5, np.linalg.eigh is fast.
        
        for gen in range(max_generations):
            if time.time() > deadline - 5.0:
                break
                
            # Sample points
            # We need to ensure we don't exceed budget. 
            # We are in generation gen, so we have used 1 + gen*lambda calls.
            # Next batch is lambda calls.
            
            Y = np.zeros((lambda_pop, dim))
            F = np.zeros(lambda_pop)
            
            for i in range(lambda_pop):
                # Standard normal
                z = np.random.randn(dim)
                # Apply covariance sqrt
                # C = B * D * B.T
                # We update C, so we need its square root.
                # For dim=5, this is cheap.
                # However, doing eigh every generation for dim=5 is fine.
                # Optimization: Cache eigen decomposition? 
                # C changes every step, so we must recompute.
                
                # To avoid expensive eigh in the inner loop, we can update C first?
                # No, C update depends on selected points.
                # So:
                # 1. Sample using current C (need sqrt)
                # 2. Evaluate
                # 3. Update C
                
                # Recomputing sqrt(C) every generation is O(dim^3). Dim=5, so 125 ops. Negligible.
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip negative eigenvalues to 0 for numerical stability
                eigvals = np.clip(eigvals, 1e-10, None)
                sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                
                y = m + sigma * (sqrt_C @ z)
                y = np.clip(y, -5, 5)
                Y[i] = y
                F[i] = func(y)
                
            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = Y[idx_min].copy()
            
            # Sort
            sorted_idx = np.argsort(F)
            Y_sorted = Y[sorted_idx]
            F_sorted = F[sorted_idx]
            
            # Select parents
            Y_parents = Y_sorted[:mu]
            
            # New mean
            m_new = np.sum(weights[:, None] * Y_parents, axis=0)
            
            # Check if mean moved
            h_sigma = 1.0
            # Simple check: if distance is small, h_sigma might be 0?
            # Standard CMA-ES uses a threshold.
            dist = np.linalg.norm(m_new - m)
            if dist < 1e-10:
                 h_sigma = 0.0
            else:
                 # Approximation for h_sigma. Usually involves C and sigma.
                 # For simplicity and speed, we can often set h_sigma=1 unless sigma is tiny.
                 # Let's use a simple heuristic: if dist > 0.1 * sigma, h_sigma = 1, else 0?
                 # Actually, standard implementation is complex. 
                 # Given dim=5, let's use a simplified CMA-ES or just (1+1)-ES logic if CMA is too heavy?
                 # No, CMA-ES is better. Let's implement the update carefully.
                 
                 # ps update
                 ps = (1 - c_sigma) * ps + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C @ (m_new - m))
                 
                 # pc update
                 h_sigma = 1.0 # Simplified: assume significant move
                 # More accurate h_sigma:
                 # h_sigma = 1 if ||ps||^2 / (1 - (1-c_sigma)^{2(gen+1)}) < (1+2/sqrt(mu_eff))^2 * chi_norm^2
                 # This is getting complicated. 
                 
                 # Alternative: Use a simpler covariance matrix adaptation or just (1+1) with larger population?
                 # Let's stick to a robust simplified CMA-ES.
                 
                 # Actually, for dim=5, a full CMA-ES is overkill if we can't get the details right quickly.
                 # Let's use a "CMA-like" approach: 
                 # 1. Sample from N(m, sigma^2 * C)
                 # 2. Update m
                 # 3. Update C based on selected points
                 # 4. Adapt sigma based on success rate (like 1/5 rule but for population)
                 
                 # Let's revert to a simpler, more robust strategy: 
                 # Multi-start (1+1)-ES with a twist?
                 # The inspiration 3 suggested multi-start.
                 # But CMA-ES is a different family.
                 
                 # Let's try a simpler "Population-based" (1+lambda)-ES with covariance adaptation.
                 
                 # Update C
                 # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - (1 - c_sigma)^2) * C) + c_mu * sum(weights * (y-m_new)(y-m_new)^T)
                 
                 # We need pc.
                 pc = (1 - c_c) * pc + h_sigma * c_c / sigma * (m_new - m)
            # --- Clean CMA-ES Update Step ---
            
            # 1. Calculate h_sigma
            # We need C_old for this.
            # Let's save C_old before updating.
            
            # This implementation is getting too complex for the time limit/risk.
            # Let's switch to a simpler, effective strategy:
            # **Adaptive Multi-Start (1+1)-ES**
            # 
            # Why?
            # 1. Current program is single-start (1+1)-ES.
            # 2. BBOB f6 (Attractive Sector) has a specific structure but hidden shift/rotation.
            # 3. Single start can get stuck.
            # 4. Multi-start allows exploring different basins.
            # 5. AOCC rewards early progress. A good start helps.
            
            # New Strategy:
            # Divide budget into K restarts.
            # For each restart, run (1+1)-ES with a large initial sigma.
            # If progress stalls (no improvement for X steps), restart.
            
            # Let's implement this. It is materially different from the current single-trajectory.
            
            # However, I already started writing CMA-ES. 
            # Let's abandon CMA-ES due to complexity/risk of bugs in 50 lines.
            # Let's implement **Multi-Start (1+1)-ES with Adaptive Step Size**.
            
            break # Break out of the complex CMA-ES loop to use the simpler strategy below.

        # --- ACTUAL IMPLEMENTATION: Multi-Start (1+1)-ES ---
        
        # We will restart the loop logic with the multi-start strategy.
        # The code above is discarded.
        
        # Reset state
        best_x = None
        best_f = float('inf')
        
        rng = np.random.default_rng(42)
        
        # Allocate budget
        # Total budget: 500
        # We will do up to 5 restarts.
        # Each restart gets 100 calls.
        # If a restart finds a very good point, we might continue it?
        # Simpler: Fixed allocation. 5 restarts * 100 calls.
        
        num_restarts = 5
        calls_per_restart = budget // num_restarts
        
        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
                
            # New random start
            # Start far from previous best? Or random?
            # Random is safer.
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            # (1+1)-ES parameters
            sigma = 3.0 # Large initial step
            successes = 0
            attempts = 0
            no_improve_count = 0
            max_no_improve = 20 # If no improvement for 20 steps, break early to save budget for next restart
            
            for step in range(calls_per_restart):
                if time.time() > deadline - 5.0:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    no_improve_count = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    no_improve_count += 1
                    
                attempts += 1
                
                # 1/5 success rule
                if attempts >= 5:
                    success_rate = successes / attempts
                    if success_rate > 0.2:
                        sigma *= 1.5
                    elif success_rate < 0.2:
                        sigma *= 0.7
                    successes = 0
                    attempts = 0
                    
                sigma = max(0.001, min(sigma, 10.0))
                
                # Early termination if stuck
                if no_improve_count > max_no_improve:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
