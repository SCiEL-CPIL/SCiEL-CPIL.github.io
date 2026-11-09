"""Simulated Annealing with adaptive cooling for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Simulated Annealing optimizer for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit. 280s is a safe margin to ensure we return before kill.
        deadline = start_time + 280.0
        
        d = self.dim
        budget = self.budget
        
        # Global best tracking
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        # Random number generator
        rng = np.random.default_rng(42)
        
        # Phase 1: Broad uniform scout to initialize the SA process
        # We want a diverse starting point. Let's just pick a random point.
        # Or maybe a few random points to pick the best one?
        # The directive says "initialized from a broad uniform scout".
        # Let's do a small number of random evaluations to find a decent start.
        n_scout = max(5, budget // 50)
        
        for _ in range(n_scout):
            if time.time() > deadline or calls_used >= budget:
                break
            x = rng.uniform(-5.0, 5.0, d)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if best_x is None:
            x = np.zeros(d)
            best_f = func(x)
            calls_used += 1
            best_x = x.copy()
            
        # Phase 2: Simulated Annealing
        # Current state
        current_x = best_x.copy()
        current_f = best_f
        
        # SA Parameters
        # Temperature schedule: T(t) = T0 * alpha^t
        # We need to choose T0 and alpha such that we have enough steps to cool down.
        # Total steps available: budget - calls_used
        # Let's aim for roughly 1000-5000 steps if possible, but constrained by budget.
        # Actually, budget is 500. So we have ~500 - n_scout steps.
        
        # Let's estimate remaining budget
        remaining_budget = budget - calls_used
        
        # Initial temperature T0
        # f7 values can be large. Let's estimate a scale.
        # If we are at a random point, f might be around 10-100.
        # Let's set T0 to a value that allows significant uphill moves initially.
        # T0 = 10.0 seems reasonable for a scale of ~10.
        T0 = 10.0
        
        # Cooling factor alpha
        # We want T to drop to near 0 after remaining_budget steps.
        # T(remaining_budget) = T0 * alpha^remaining_budget ~ 0
        # Let's aim for T_end = 0.01
        # alpha = (T_end / T0) ** (1 / remaining_budget)
        # But if remaining_budget is small, alpha might be weird.
        # Let's just fix alpha to something like 0.95 or 0.99.
        # With 490 steps, 0.95^490 is very small.
        # 0.99^490 is ~0.007.
        # Let's use alpha = 0.99.
        alpha = 0.99
        
        T = T0
        
        # Step size
        # We need a step size that is large enough to explore but not too large.
        # A common choice is related to the domain size.
        # Let's start with a step size of 1.0 and adapt it.
        # Or use a fixed step size scaled by temperature?
        # Let's try a fixed step size first, maybe 0.5 or 1.0.
        # Given the box is [-5, 5], a step of 1.0 is reasonable.
        step_size = 1.0
        
        # Track stagnation to adapt step size or restart?
        # SA should handle plateaus via uphill moves.
        # But if we are stuck on a flat plateau, uphill moves (which are accepted with prob 1 if T is high) 
        # might just keep us on the plateau if the function is flat.
        # Actually, if f(x_new) == f(x_current), it's an uphill move (delta = 0), so exp(0/T) = 1.
        # So we will always accept moves to the same value. This is good for crossing plateaus.
        # However, we might drift randomly on the plateau.
        # To escape a large plateau, we might need larger steps.
        # Let's monitor if we are making progress in terms of best_f.
        
        stagnation_counter = 0
        stagnation_limit = 50
        
        while calls_used < budget:
            if time.time() > deadline:
                break
                
            # Generate neighbor
            # Gaussian perturbation
            z = rng.standard_normal(d)
            x_new = current_x + step_size * z
            x_new = np.clip(x_new, -5.0, 5.0)
            
            f_new = func(x_new)
            calls_used += 1
            
            # Update global best
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
            # Acceptance criterion
            delta_f = f_new - current_f
            if delta_f < 0:
                # Improvement
                current_x = x_new
                current_f = f_new
            else:
                # Uphill or neutral move
                # Probability of acceptance: exp(-delta_f / T)
                # If T is very small, this is 0.
                # If delta_f is 0, prob is 1.
                if T > 1e-10:
                    prob = np.exp(-delta_f / T)
                    if rng.random() < prob:
                        current_x = x_new
                        current_f = f_new
                # else: reject (T is effectively 0)
                
            # Cool down
            T *= alpha
            
            # Adaptive step size?
            # If we are stagnating (best_f not improving), maybe increase step size?
            # Or if we are improving, decrease it?
            # Standard SA doesn't usually adapt step size, but for plateaus, it might help.
            # If stagnation_counter > stagnation_limit:
            #     step_size *= 1.1
            #     stagnation_counter = 0 # Reset to give it time
            #     # Optionally restart T? No, let's just increase step size.
            
            # If we are very close to the end, we might want to reduce step size to fine-tune?
            # The cooling schedule already reduces the probability of large jumps (effectively).
            # But the step size itself is fixed.
            # Let's try to reduce step size as T drops?
            # step_size = max(0.01, T / T0) * initial_step_size?
            # This might be too aggressive.
            # Let's stick to fixed step size for now, as the cooling schedule handles the exploration/exploitation balance.
            
            # If we are completely stuck (no acceptance for a long time), we might be in a local min.
            # But SA accepts uphill moves, so it shouldn't get stuck in a local min unless T is too low.
            # If T is low and we are in a local min, we will reject all moves and current_x stays put.
            # This is fine, we are converged.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
