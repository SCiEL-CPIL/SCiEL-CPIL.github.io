"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Stochastic Hill Climbing with geometric step decay and random kicks."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        n = self.dim
        total_budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= total_budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            try:
                fx = float(func(x))
            except:
                fx = np.inf
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Initialize at origin
        x = np.zeros(n)
        fx = safe_func(x)
        if fx == np.inf:
            return best_x, best_f
            
        current_x = x.copy()
        current_f = fx
        
        # Initial step size: cover a significant portion of the domain
        # Domain is [-5, 5], so width is 10.
        # Start with sigma ~ 1.5 to allow large jumps but not too large.
        sigma = 1.5
        
        # Decay rate for successful steps
        decay_success = 0.95
        # Decay rate for failed steps
        decay_fail = 0.7
        
        # Kick probability when stuck
        kick_threshold_sigma = 0.05
        kick_magnitude = 2.0
        
        # Stagnation counter
        stagnant_calls = 0
        stagnation_limit = 20 # If no improvement in 20 calls, consider a kick
        
        while calls_made < total_budget and time.time() < deadline:
            # Generate candidate
            # Use Gaussian perturbation
            direction = rng.standard_normal(n)
            candidate = current_x + sigma * direction
            
            # Evaluate candidate
            fc = safe_func(candidate)
            
            if fc == np.inf:
                break # Budget or time exhausted
            
            if fc < current_f:
                # Improvement
                current_x = candidate
                current_f = fc
                sigma *= decay_success
                stagnant_calls = 0
            else:
                # No improvement
                sigma *= decay_fail
                stagnant_calls += 1
                
                # If stagnant for too long and step size is small, kick
                if stagnant_calls > stagnation_limit and sigma < kick_threshold_sigma:
                    # Random kick to escape local minimum
                    kick_dir = rng.standard_normal(n)
                    kick_vec = kick_magnitude * kick_dir
                    candidate_kick = current_x + kick_vec
                    fc_kick = safe_func(candidate_kick)
                    
                    if fc_kick != np.inf:
                        # Update current position to the kick result if it's not worse than current
                        # Actually, just move to the kick location to restart search from there
                        # Even if it's worse, we might be in a different basin.
                        # But to be safe, only move if it's better or if we really want to explore.
                        # Let's move if it's better, otherwise just reset sigma and try again nearby?
                        # Better: Move to the kick point. This is a "restart" in a sense.
                        if fc_kick < current_f:
                            current_x = candidate_kick
                            current_f = fc_kick
                        else:
                            # If kick was worse, we still want to reset the local search dynamics
                            # but keep the best point found so far as the anchor?
                            # No, standard hill climbing keeps current_x as the "current" state.
                            # If we move to a worse point, we might get stuck in a bad area.
                            # Let's only kick if it improves, or if we've been stuck for a VERY long time.
                            pass
                        
                        # Reset sigma and stagnation counter after a kick attempt
                        sigma = kick_magnitude * 0.5
                        stagnant_calls = 0
            
            # Ensure sigma doesn't become too small
            if sigma < 1e-8:
                # If step size is tiny, we are likely at a local minimum or plateau
                # Try a kick immediately
                kick_dir = rng.standard_normal(n)
                candidate_kick = current_x + kick_magnitude * kick_dir
                fc_kick = safe_func(candidate_kick)
                
                if fc_kick != np.inf:
                    if fc_kick < current_f:
                        current_x = candidate_kick
                        current_f = fc_kick
                    # Regardless, reset sigma
                    sigma = kick_magnitude * 0.5
                    stagnant_calls = 0
                else:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
