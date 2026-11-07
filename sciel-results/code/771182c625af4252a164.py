"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

class Optimizer:
    """
    Multi-start (1+1)-ES with Island Model and Crossover.
    
    Strategy:
    1. Divide budget among K islands.
    2. Each island runs a (1+1)-ES with adaptive step size (1/5 rule).
    3. Periodically, islands exchange their best individual via crossover (recombination).
    4. This tests if 'cross' (recombination between independent lineages) improves convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin
        
        best_x_global = None
        best_f_global = np.inf
        call_count = 0
        rng = np.random.default_rng(42)

        def make_call(x):
            nonlocal call_count, best_f_global, best_x_global
            if call_count >= self.budget:
                return None
            fx = func(x)
            call_count += 1
            if np.isfinite(fx) and fx < best_f_global:
                best_f_global = fx
                best_x_global = np.array(x)
            return fx

        def run_island_es(start_x, sigma, max_calls, rng_local, exchange_callback=None, exchange_interval=50):
            """
            Runs a (1+1)-ES loop.
            exchange_callback: If provided, called every 'exchange_interval' steps with current best.
            Returns (best_x, best_f).
            """
            nonlocal call_count
            
            x_best = start_x.copy()
            f_best = np.inf
            sigma_current = sigma
            
            # Evaluate start
            if call_count >= self.budget:
                return x_best, f_best
            
            f_best = make_call(x_best)
            if f_best is None:
                return x_best, f_best
            
            steps = 0
            success_count = 0
            window_size = 5
            window = []
            
            while call_count < self.budget:
                # Check deadline
                if time.time() > deadline - 10.0:
                    break
                
                # Generate candidate
                z = rng_local.standard_normal(self.dim)
                x_child = x_best + sigma_current * z
                x_child = np.clip(x_child, -5, 5)
                
                f_child = make_call(x_child)
                if f_child is None:
                    break
                
                # Check for improvement
                improved = False
                if np.isfinite(f_child) and f_child < f_best:
                    x_best = x_child.copy()
                    f_best = f_child
                    improved = True
                    
                # 1/5 Rule adaptation
                window.append(1 if improved else 0)
                if len(window) > window_size:
                    window.pop(0)
                
                if len(window) == window_size:
                    success_rate = sum(window) / window_size
                    if success_rate > 0.2:
                        sigma_current *= 1.05
                    elif success_rate < 0.1:
                        sigma_current *= 0.95
                    # Clamp sigma
                    sigma_current = max(1e-6, min(sigma_current, 10.0))
                
                steps += 1
                
                # Exchange logic: If exchange_callback exists, it might update x_best
                if exchange_callback and steps % exchange_interval == 0:
                    # The callback returns a potentially better individual from another island
                    # We adopt it if it's better, or just use it as a new parent
                    # Here we assume the callback handles the global state and returns a new best for this island
                    new_x, new_f = exchange_callback(x_best, f_best)
                    if new_x is not None:
                        # If the exchanged individual is better, adopt it
                        if new_f < f_best:
                            x_best = new_x
                            f_best = new_f
                        # Even if not better, we might want to jump to it to explore new area?
                        # For robustness, only adopt if strictly better to avoid regression, 
                        # but the "cross" aspect implies recombination.
                        # Let's implement a simple crossover here inside the callback logic or here?
                        # The prompt asks to test if 'cross' changes the chance.
                        # We will implement crossover in the callback.

            return x_best, f_best

        # --- Main Logic: Island Model ---
        
        # Parameters
        n_islands = 5
        # Allocate budget roughly equally, but ensure enough for overhead
        budget_per_island = self.budget // n_islands
        # We will run them sequentially but with shared global best tracking.
        # To simulate parallelism and allow crossover, we will manage states manually.
        
        # Initialize islands
        island_states = []
        for i in range(n_islands):
            # Diverse starting points
            if i == 0:
                x_start = np.zeros(self.dim)
            else:
                x_start = rng.uniform(-4, 4, self.dim)
            
            sigma_start = 1.5 if i == 0 else 1.0
            island_states.append({
                'x': x_start,
                'f': np.inf,
                'sigma': sigma_start,
                'rng': np.random.default_rng(42 + i * 100)
            })
        
        # Initialize global best
        # We need to evaluate initial points to get f values
        for i in range(n_islands):
            if call_count >= self.budget:
                break
            fx = make_call(island_states[i]['x'])
            if fx is not None:
                island_states[i]['f'] = fx
                if fx < best_f_global:
                    best_f_global = fx
                    best_x_global = island_states[i]['x'].copy()
        
        # Crossover function
        def perform_crossover(parent1, parent2):
            """
            Simple uniform crossover or arithmetic blend.
            Since the landscape is continuous, arithmetic crossover (blend) is often better.
            But 'cross' usually implies genetic operators.
            Let's use a mix: 50% chance of uniform crossover, 50% arithmetic blend.
            """
            if rng.random() < 0.5:
                # Uniform crossover
                mask = rng.random(self.dim) < 0.5
                child = np.where(mask, parent1, parent2)
            else:
                # Arithmetic crossover (blend)
                alpha = rng.random()
                child = alpha * parent1 + (1 - alpha) * parent2
            
            return np.clip(child, -5, 5)

        # Main loop: Iterate through islands, allowing them to run for a few steps, 
        # then exchange/cross.
        
        # We will run each island for a fixed number of generations, then cross.
        # Total budget is 500.
        # Let's do 50 generations per island initially, then cross, then continue.
        
        # Simpler approach: 
        # 1. Run all islands for 20 steps.
        # 2. Pick best 2 islands, cross their bests, create new individual, evaluate.
        # 3. Replace the worst island's state with this new individual (if better) or just use it for exploration.
        # 4. Repeat.
        
        steps_per_round = 20
        
        while call_count < self.budget:
            if time.time() > deadline - 5.0:
                break
            
            # 1. Run each island for steps_per_round steps
            for i in range(n_islands):
                if call_count >= self.budget:
                    break
                if time.time() > deadline - 10.0:
                    break
                
                state = island_states[i]
                x_best = state['x']
                f_best = state['f']
                sigma = state['sigma']
                rng_local = state['rng']
                
                for _ in range(steps_per_round):
                    if call_count >= self.budget:
                        break
                    if time.time() > deadline - 10.0:
                        break
                    
                    z = rng_local.standard_normal(self.dim)
                    x_child = x_best + sigma * z
                    x_child = np.clip(x_child, -5, 5)
                    
                    f_child = make_call(x_child)
                    if f_child is None:
                        break
                    
                    improved = False
                    if np.isfinite(f_child) and f_child < f_best:
                        x_best = x_child
                        f_best = f_child
                        improved = True
                    
                    # Adapt sigma (1/5 rule simplified: update every step)
                    # Using a simpler adaptive rule for stability in short windows
                    if improved:
                        sigma *= 1.02
                    else:
                        sigma *= 0.98
                    sigma = max(1e-6, min(sigma, 10.0))
                
                state['x'] = x_best
                state['f'] = f_best
                state['sigma'] = sigma
            
            if call_count >= self.budget:
                break
                
            # 2. Crossover Phase
            # Find the two best islands
            sorted_islands = sorted(range(n_islands), key=lambda i: island_states[i]['f'])
            best_idx_1 = sorted_islands[0]
            best_idx_2 = sorted_islands[1]
            
            p1 = island_states[best_idx_1]['x']
            p2 = island_states[best_idx_2]['x']
            
            # Generate offspring via crossover
            child = perform_crossover(p1, p2)
            
            # Evaluate offspring
            if call_count < self.budget:
                f_child = make_call(child)
                if f_child is not None and np.isfinite(f_child):
                    # If this child is better than the worst island, replace the worst island with it
                    # This injects the "crossed" genetic material into the population
                    worst_idx = sorted_islands[-1]
                    if f_child < island_states[worst_idx]['f']:
                        island_states[worst_idx]['x'] = child
                        island_states[worst_idx]['f'] = f_child
                        # Reset sigma for the new island to explore around this new point
                        island_states[worst_idx]['sigma'] = 1.0
                        
                    # Also, if this child is better than the global best, update global
                    # (Already handled in make_call, but ensure state is consistent)

        # Final refinement: Use remaining budget for a local search on the global best
        if best_x_global is not None and call_count < self.budget:
            # Simple gradient-free local refinement (Nelder-Mead style or just fine ES)
            # Let's use a small (1+1)-ES with very small sigma
            x_cur = best_x_global.copy()
            f_cur = best_f_global
            sigma = 0.1
            rng_local = np.random.default_rng(999)
            
            while call_count < self.budget:
                if time.time() > deadline - 2.0:
                    break
                
                z = rng_local.standard_normal(self.dim)
                x_child = x_cur + sigma * z
                x_child = np.clip(x_child, -5, 5)
                
                f_child = make_call(x_child)
                if f_child is None:
                    break
                
                if np.isfinite(f_child) and f_child < f_cur:
                    x_cur = x_child
                    f_cur = f_child
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                sigma = max(1e-8, min(sigma, 1.0))

        if best_x_global is None:
            # Fallback
            x = np.zeros(self.dim)
            fx = make_call(x)
            if fx is not None and np.isfinite(fx):
                best_x_global = x.copy()
                best_f_global = fx

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
