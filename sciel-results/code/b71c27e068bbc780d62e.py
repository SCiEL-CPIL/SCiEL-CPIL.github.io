"""Multi-start (1+1)-ES with plateau-aware adaptation for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start (1+1)-ES with plateau-aware adaptation for BBOB f7."""

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
        
        # Number of parallel lineages
        # For d=5, 4-6 lineages is a good balance.
        n_lineages = 5
        
        # Initialize lineages
        lineages = []
        
        # Initial points: center and random points
        # Ensure we cover the space reasonably
        init_points = []
        init_points.append(np.zeros(d))
        
        # Add random points
        for _ in range(n_lineages - 1):
            # Sample from a box slightly smaller than the domain to avoid immediate clipping issues
            # [-4, 4] is safe, minimizer is in [-4, 4]
            p = rng.uniform(-4.0, 4.0, size=d)
            init_points.append(p)
            
        # Evaluate initial points and initialize lineages
        for p in init_points:
            if calls_used >= budget:
                break
            f_p = func(p)
            calls_used += 1
            if f_p < best_f:
                best_f = f_p
                best_x = p.copy()
            
            # Each lineage starts with its own state
            # Initial sigma: large enough to explore, small enough to not jump out of box
            sigma = 2.0 
            # We track 'stagnation' by counting steps without improvement
            stagnation_counter = 0
            # Last best value for this lineage
            lineage_best_f = f_p
            lineage_best_x = p.copy()
            
            lineages.append({
                'x': p.copy(),
                'f': f_p,
                'sigma': sigma,
                'stagnation': 0,
                'best_f': f_p,
                'best_x': p.copy()
            })
        
        # If we ran out of budget during initialization
        if calls_used >= budget:
            return best_x, best_f

        # Main loop
        # We cycle through lineages. 
        # To favor early progress (AOCC), we might want to give more budget to better lineages initially?
        # But for plateau escaping, equal budget or round-robin is safer.
        # Let's do round-robin.
        
        while calls_used < budget:
            if time.time() > deadline:
                break
            
            # Select a lineage to update
            # Simple round-robin: pick the one with the lowest 'last_update_time' or just cycle
            # For simplicity and fairness, cycle through them.
            # However, if a lineage is clearly stuck (high stagnation), we might want to restart it?
            # Let's stick to the adaptation mechanism first.
            
            for i, lin in enumerate(lineages):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Generate candidate
                # Step size adaptation:
                # If stagnation is high, increase sigma to jump.
                # If improving, keep sigma or slightly decrease to refine.
                
                # Base sigma adjustment
                # If stagnation > threshold, we are likely on a plateau.
                # f7 has plateaus. A larger step is needed.
                
                # Heuristic:
                # If stagnation > 10, multiply sigma by 1.5
                # If stagnation > 30, multiply sigma by 2.0
                # If improved, reset stagnation and maybe shrink sigma slightly? 
                # No, shrinking too fast traps us. Keep sigma constant or grow slowly.
                
                # Let's try:
                # sigma *= 1.05 if stagnation > 5 else 1.0
                # This slowly increases step size when stuck.
                
                # Actually, for (1+1)-ES on plateaus, a significant jump is often needed.
                # Let's use a more aggressive strategy:
                # If stagnation > 20, sigma *= 1.5
                # If stagnation > 50, sigma *= 2.0
                # Cap sigma to avoid going out of bounds too wildly (though clipping handles it, 
                # very large steps are inefficient as they always clip).
                
                if lin['stagnation'] > 50:
                    lin['sigma'] *= 2.0
                elif lin['stagnation'] > 20:
                    lin['sigma'] *= 1.5
                elif lin['stagnation'] > 5:
                    lin['sigma'] *= 1.2
                
                # Cap sigma
                lin['sigma'] = min(lin['sigma'], 5.0)
                
                # Sample
                z = rng.standard_normal(d)
                x_new = lin['x'] + lin['sigma'] * z
                x_new = np.clip(x_new, -5.0, 5.0)
                
                # Evaluate
                if calls_used >= budget:
                    break
                f_new = func(x_new)
                calls_used += 1
                
                # Update global best
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                
                # Update lineage state
                if f_new < lin['best_f']:
                    # Improvement
                    lin['best_f'] = f_new
                    lin['best_x'] = x_new.copy()
                    lin['x'] = x_new.copy()
                    lin['stagnation'] = 0
                    # Slight shrinkage to refine? Or keep?
                    # Keeping sigma constant allows exploration of the new neighborhood.
                    # Shrinkage might be premature. Let's keep it constant on improvement.
                else:
                    # No improvement
                    lin['stagnation'] += 1
                    # Move x to new point? 
                    # In standard (1+1)-ES, x stays at best. 
                    # But for plateau escaping, moving x to the new point (even if worse) 
                    # might help if the function is non-convex? 
                    # No, f7 is convex-ish (ellipsoidal steps). 
                    # Staying at the best known point in the local neighborhood is safer 
                    # unless we are explicitly restarting.
                    # However, if we stay at x, and sigma grows, we are sampling around x.
                    # This is standard.
                    
                    # Optional: If stagnation is very high, "restart" the lineage from the global best?
                    # Or just let sigma grow.
                    # Let's try: if stagnation > 100, reset x to global best and sigma to 1.0?
                    # This might be too aggressive.
                    
                    # Let's stick to sigma growth.
        
        return best_x, best_f
# EVOLVE-BLOCK-END
