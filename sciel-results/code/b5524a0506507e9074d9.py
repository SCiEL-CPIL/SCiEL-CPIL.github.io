import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Time management with safety margin
        start_time = time.time()
        time_limit = 320.0  # Safety margin before 330s kill
        
        # Number of parallel agents
        num_agents = 5
        
        # Initialize agents with diverse starting points
        agents = []
        for i in range(num_agents):
            if i == 0:
                x = np.zeros(n)
            else:
                x = np.random.uniform(-5, 5, n)
            try:
                fx = func(x)
            except Exception:
                fx = np.inf
            agents.append({
                'x': x.copy(),
                'fx': fx,
                'sigma': 2.0,
                'success_window': [],
                'stagnation': 0,
                'alive': True
            })
        
        # Track global best
        x_best = agents[0]['x'].copy()
        f_best = agents[0]['fx']
        for a in agents[1:]:
            if a['fx'] < f_best:
                f_best = a['fx']
                x_best = a['x'].copy()
        
        eval_count = num_agents
        sigma_min = 1e-8
        sigma_max = 10.0
        window_size = 10
        restart_threshold = 25
        
        # Main loop: cycle through agents
        while eval_count < budget:
            if time.time() - start_time > time_limit:
                break
            
            # Cycle through agents, skipping dead ones
            for idx in range(num_agents):
                if eval_count >= budget or time.time() - start_time > time_limit:
                    break
                
                agent = agents[idx]
                if not agent['alive']:
                    continue
                
                # Adaptive sigma based on success rate
                if len(agent['success_window']) >= window_size:
                    success_rate = np.mean(agent['success_window'])
                    if success_rate > 0.2:
                        agent['sigma'] *= 1.05
                    elif success_rate < 0.2:
                        agent['sigma'] *= 0.95
                    agent['success_window'] = agent['success_window'][1:]
                
                # Generate candidate
                z = np.random.standard_normal(n)
                y = agent['x'] + agent['sigma'] * z
                y = np.clip(y, -5, 5)
                
                try:
                    fy = func(y)
                except Exception:
                    agent['alive'] = False
                    continue
                    
                eval_count += 1
                
                if fy < agent['fx']:
                    # Success
                    agent['x'] = y
                    agent['fx'] = fy
                    agent['success_window'].append(1)
                    agent['stagnation'] = 0
                    
                    if fy < f_best:
                        f_best = fy
                        x_best = y.copy()
                else:
                    # Failure
                    agent['success_window'].append(0)
                    agent['stagnation'] += 1
                
                # Clamp sigma
                agent['sigma'] = np.clip(agent['sigma'], sigma_min, sigma_max)
                
                # Restart if stuck
                if agent['stagnation'] >= restart_threshold:
                    # Reseed this agent with a new random point
                    x_new = np.random.uniform(-5, 5, n)
                    try:
                        f_new = func(x_new)
                        eval_count += 1
                    except Exception:
                        agent['alive'] = False
                        continue
                    agent['x'] = x_new.copy()
                    agent['fx'] = f_new
                    agent['sigma'] = 2.5
                    agent['success_window'] = []
                    agent['stagnation'] = 0
                    
                    # If this is better than global best, update
                    if f_new < f_best:
                        f_best = f_new
                        x_best = x_new.copy()
                
                # Occasionally reseed a random agent for diversity
                if eval_count % 50 == 0 and np.random.random() < 0.3:
                    # Pick a random agent to reseed
                    reseed_idx = np.random.randint(num_agents)
                    ra = agents[reseed_idx]
                    if ra['alive']:
                        x_new = np.random.uniform(-5, 5, n)
                        try:
                            f_new = func(x_new)
                            eval_count += 1
                        except Exception:
                            ra['alive'] = False
                            continue
                        ra['x'] = x_new.copy()
                        ra['fx'] = f_new
                        ra['sigma'] = 2.5
                        ra['success_window'] = []
                        ra['stagnation'] = 0
                        
                        if f_new < f_best:
                            f_best = f_new
                            x_best = x_new.copy()
        
        # Final polish: use remaining budget for fine-tuning the best agent
        if eval_count < budget and time.time() - start_time < time_limit:
            x = x_best.copy()
            fx = f_best
            sigma = 0.1
            
            max_polish_steps = min(budget - eval_count, 30)
            
            for _ in range(max_polish_steps):
                if eval_count >= budget or time.time() - start_time > time_limit:
                    break
                
                z = np.random.standard_normal(n)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                
                try:
                    fy = func(y)
                except Exception:
                    break
                    
                eval_count += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    if fy < f_best:
                        f_best = fy
                        x_best = y.copy()
                    sigma *= 0.95
                else:
                    sigma *= 0.9
                
                sigma = np.clip(sigma, 1e-12, 0.5)
        
        return x_best, f_best
