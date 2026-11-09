"""Parallel Multi-Start Adaptive (1+1)-ES Portfolio for Schaffers F7."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Parallel Multi-Start Adaptive (1+1)-Evolution Strategy Portfolio."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin (300s limit)
        
        dim = self.dim
        budget = self.budget
        
        # Number of parallel agents
        # K=10 seems a good balance for 500 budget in 5D
        K = 10
        
        # Initialize K agents
        agents = []
        for k in range(K):
            # Random start in [-4, 4]^5 to ensure we are in the interesting region
            x = np.random.uniform(-4, 4, dim)
            fx = func(x)
            # Initial sigma
            sigma = 1.0
            # State for 1/5 rule
            successes = 0
            total = 0
            window = 10
            
            agents.append({
                'x': x,
                'fx': fx,
                'sigma': sigma,
                'successes': successes,
                'total': total,
                'window': window
            })
            
        calls = K
        
        # Track global best
        global_best_x = agents[0]['x'].copy()
        global_best_fx = agents[0]['fx']
        for a in agents:
            if a['fx'] < global_best_fx:
                global_best_fx = a['fx']
                global_best_x = a['x'].copy()
        
        # Main loop
        # We have budget - K calls left
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Round-robin selection of agent
            # We can just cycle through 0..K-1
            idx = (calls - K) % K
            agent = agents[idx]
            
            # Generate perturbation
            z = np.random.randn(dim)
            y = agent['x'] + agent['sigma'] * z
            
            # Clip to box
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls += 1
            
            if fy < agent['fx']:
                # Success
                agent['x'] = y
                agent['fx'] = fy
                agent['successes'] += 1
                
                # Update global best
                if fy < global_best_fx:
                    global_best_fx = fy
                    global_best_x = agent['x'].copy()
            else:
                # Failure
                pass
                
            agent['total'] += 1
            
            # Adapt step size using 1/5 success rule
            if agent['total'] >= agent['window']:
                success_rate = agent['successes'] / agent['total']
                if success_rate > 0.5:
                    agent['sigma'] *= 1.5
                elif success_rate < 0.5:
                    agent['sigma'] *= (1.0 / 1.5)
                agent['successes'] = 0
                agent['total'] = 0
            
            # Safety check for sigma
            if agent['sigma'] < 1e-6:
                agent['sigma'] = 1e-6
            elif agent['sigma'] > 10.0:
                agent['sigma'] = 10.0

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
