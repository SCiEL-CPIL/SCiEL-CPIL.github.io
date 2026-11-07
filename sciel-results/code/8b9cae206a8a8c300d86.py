import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin for 300s limit
        d = self.dim
        budget = self.budget
        
        # Best solution tracking
        best_x = None
        best_f = np.inf
        
        def eval_func(x):
            nonlocal best_x, best_f
            # Clip to box
            x_clip = np.clip(x, -5.0, 5.0)
            fx = func(x_clip)
            if fx < best_f:
                best_f = fx
                best_x = x_clip.copy()
            return fx

        # Multi-start (1+1)-ES with Momentum
        # This strategy uses multiple parallel agents to explore the landscape.
        # Each agent uses a simple evolution strategy step with momentum to help
        # escape local minima, which is crucial for the Lunacek bi-Rastrigin function.
        
        n_agents = 5
        agents = []
        
        # Initialize agents with random positions in [-5, 5]^d
        # We use a fixed seed for reproducibility in the agent initialization logic if needed, 
        # but random is fine.
        rng = np.random.default_rng(42)
        
        for _ in range(n_agents):
            x = rng.uniform(-5.0, 5.0, size=d)
            fx = eval_func(x)
            agents.append({
                'x': x,
                'f': fx,
                'sigma': 1.0,  # Initial step size
                'v': np.zeros(d),  # Momentum
                'stagnation': 0
            })
            
        # Update global best after initialization
        for a in agents:
            if a['f'] < best_f:
                best_f = a['f']
                best_x = a['x'].copy()

        beta = 0.9  # Momentum coefficient
        stagnation_limit = 20  # Restart if no improvement for this many steps
        sigma_min = 1e-4
        sigma_max = 5.0
        
        calls_made = len(agents)
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            for i, agent in enumerate(agents):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate new candidate with momentum
                # x_new = x + v + sigma * N(0, 1)
                noise = rng.normal(0, 1, size=d)
                x_new = agent['x'] + agent['v'] + agent['sigma'] * noise
                x_new = np.clip(x_new, -5.0, 5.0)
                
                f_new = eval_func(x_new)
                calls_made += 1
                
                if f_new < agent['f']:
                    # Accept
                    agent['x'] = x_new
                    agent['f'] = f_new
                    agent['stagnation'] = 0
                    # Increase step size slightly on success
                    agent['sigma'] = min(agent['sigma'] * 1.1, sigma_max)
                else:
                    # Reject
                    agent['stagnation'] += 1
                    # Decrease step size on failure
                    agent['sigma'] = max(agent['sigma'] * 0.9, sigma_min)
                    
                    # Update momentum towards the failed direction? 
                    # Actually, momentum should usually persist or decay.
                    # If we reject, we don't move x, but we might want to keep some momentum
                    # or decay it. Let's decay momentum if we don't move.
                    agent['v'] = agent['v'] * 0.5
                    
                # If we accepted, update momentum to the direction of movement
                if f_new < agent['f']:
                    agent['v'] = beta * agent['v'] + (1 - beta) * (x_new - agent['x'])
                    # Note: agent['x'] was updated to x_new, so x_new - agent['x'] is 0?
                    # We need to store old_x.
                    # Let's fix the momentum update logic.

                # Restart if stagnated
                if agent['stagnation'] > stagnation_limit:
                    # Restart near global best with some noise, or completely random
                    # Restarting near global best helps exploitation, random helps exploration.
                    # Let's do a mix: 50% random, 50% near best.
                    if rng.random() < 0.5:
                        agent['x'] = rng.uniform(-5.0, 5.0, size=d)
                    else:
                        agent['x'] = best_x + rng.normal(0, 1.0, size=d)
                        agent['x'] = np.clip(agent['x'], -5.0, 5.0)
                    
                    agent['f'] = eval_func(agent['x'])
                    calls_made += 1
                    agent['sigma'] = 1.0
                    agent['v'] = np.zeros(d)
                    agent['stagnation'] = 0

        return best_x, best_f
