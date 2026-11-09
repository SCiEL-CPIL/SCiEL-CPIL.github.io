"""Island-based (1+1)-ES with migration for Schaffers F7."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin for 300s limit
        
        n_islands = 5
        calls = 0
        dim = self.dim
        
        # Track global best
        best_x_global = None
        best_f_global = float('inf')
        
        # Initialize islands
        # Each island has: position x, current function value fx, step size sigma
        # We start with diverse initial points to explore different regions
        # Island 0 starts at origin, others at random points
        islands = []
        for i in range(n_islands):
            if i == 0:
                x = np.zeros(dim)
            else:
                # Random initialization in [-4, 4] to stay within box
                x = np.random.uniform(-4, 4, size=dim)
            
            fx = func(x)
            calls += 1
            
            if fx < best_f_global:
                best_f_global = fx
                best_x_global = x.copy()
            
            # Initial step size: larger for diversity
            sigma = 1.5
            
            islands.append({
                'x': x,
                'fx': fx,
                'sigma': sigma,
                'best_x': x.copy(),
                'best_f': fx,
                'successes': 0,
                'window': 10
            })
        
        # Ensure first call is made immediately (already done in init)
        # calls is now n_islands
        
        # Main loop: alternate between islands
        # Each "round" gives each island one evaluation
        # When an island stagnates (no improvement in its window), it migrates
        # from the current global best island
        
        # Migration logic: if an island's best hasn't improved in 'stagnation_threshold' evaluations,
        # it resets to a point near the current global best
        stagnation_threshold = 20  # evaluations without improvement triggers migration
        
        while calls < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Determine which island to update
            # Round-robin: island index = (calls - n_islands) % n_islands
            island_idx = (calls - n_islands) % n_islands
            isl = islands[island_idx]
            
            # Generate offspring
            noise = np.random.standard_normal(dim)
            y = isl['x'] + isl['sigma'] * noise
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls += 1
            
            # Check for improvement
            if fy < best_f_global:
                best_f_global = fy
                best_x_global = y.copy()
            
            # Update island's local best
            if fy < isl['best_f']:
                isl['best_f'] = fy
                isl['best_x'] = y.copy()
                isl['successes'] += 1
            else:
                # No improvement in local best
                pass
            
            # Update island position if offspring is better than current position
            if fy < isl['fx']:
                isl['x'] = y
                isl['fx'] = fy
            
            # Adapt sigma based on success rate over a window
            # Count successes in the last 'window' evaluations for this island
            # Simplified: use a counter that resets
            # For simplicity, let's track consecutive successes/failures
            # Actually, let's use a proper window per island
            
            # To keep it simple and fast, we'll use a heuristic:
            # If the island improved its local best, increase sigma slightly
            # If not, decrease sigma
            # This is a crude but effective adaptation for low-dim problems
            
            # Better: track success rate over last 10 evaluations for this island
            # We need to store history per island
            
            # Let's implement a simple success counter per island
            # that resets every 'window' evaluations for that island
            
            # For efficiency, let's use a different approach:
            # Each island keeps a 'recent_successes' count and 'recent_total'
            # Every 10 evaluations for that island, adapt sigma
            
            # We'll track evaluations per island
            if not hasattr(isl, 'eval_count'):
                isl['eval_count'] = 0
                isl['recent_successes'] = 0
                isl['recent_total'] = 0
            
            isl['eval_count'] += 1
            isl['recent_total'] += 1
            
            # Did this evaluation improve the local best?
            if fy < isl['best_f']:
                isl['recent_successes'] += 1
            
            # Adapt every 10 evaluations
            if isl['recent_total'] >= 10:
                success_rate = isl['recent_successes'] / isl['recent_total']
                if success_rate > 0.5:
                    isl['sigma'] *= 1.3
                elif success_rate < 0.5:
                    isl['sigma'] *= 0.7
                else:
                    pass
                isl['sigma'] = np.clip(isl['sigma'], 1e-6, 10.0)
                isl['recent_successes'] = 0
                isl['recent_total'] = 0
            
            # Migration: if island stagnates, reset it near global best
            # Check if island's local best hasn't improved in 'stagnation_threshold' evaluations
            # We need to track when the island's best was last improved
            
            if not hasattr(isl, 'last_improve_eval'):
                isl['last_improve_eval'] = 0
            
            # If current eval count - last improve eval > stagnation_threshold
            if isl['eval_count'] - isl['last_improve_eval'] > stagnation_threshold:
                # Migrate: reset island to a point near global best
                # Add some noise to explore around the global best
                if best_x_global is not None:
                    noise = np.random.standard_normal(dim) * 0.5
                    new_x = best_x_global + noise
                    new_x = np.clip(new_x, -5, 5)
                    
                    # Evaluate the new point
                    new_f = func(new_x)
                    calls += 1
                    
                    if new_f < best_f_global:
                        best_f_global = new_f
                        best_x_global = new_x.copy()
                    
                    # Update island
                    isl['x'] = new_x
                    isl['fx'] = new_f
                    isl['best_x'] = new_x.copy()
                    isl['best_f'] = new_f
                    isl['sigma'] = 1.0  # Reset step size
                    isl['eval_count'] = 0
                    isl['recent_successes'] = 0
                    isl['recent_total'] = 0
                    isl['last_improve_eval'] = 0
                else:
                    # Should not happen, but handle gracefully
                    pass
            
            # Update last_improve_eval if this eval improved local best
            if fy < isl['best_f']:
                isl['last_improve_eval'] = isl['eval_count']
        
        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
