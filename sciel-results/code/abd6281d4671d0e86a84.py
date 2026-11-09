"""Multi-Start (1+1)-ES with plateau-aware adaptation for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-Start (1+1)-ES strategy for BBOB f7.
    
    Key features:
    1. Maintains multiple independent (1+1)-ES runs (trajectories).
    2. Each trajectory uses simple step-size adaptation based on success rate.
    3. Trajectories are restarted if they stall (no improvement for a while).
    4. Budget is distributed among active trajectories.
    5. Box constraints handled by clipping.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # 5s safety margin
        
        d = self.dim
        budget = self.budget
        
        # Number of parallel trajectories
        # For 500 budget and d=5, we can afford several restarts.
        # Let's start with 5 trajectories, potentially more if they die.
        num_starts = 5
        # Each trajectory gets a portion of the budget initially, but we manage calls globally.
        
        # Global best tracking
        global_best_f = np.inf
        global_best_x = np.zeros(d)
        
        # Trajectory state
        # Each trajectory: x, sigma, calls_used, stall_count, active
        trajectories = []
        
        # Initialize trajectories
        for i in range(num_starts):
            x0 = np.random.uniform(-4.5, 4.5, d)
            f0 = func(x0)
            if f0 < global_best_f:
                global_best_f = f0
                global_best_x = x0.copy()
            
            trajectories.append({
                'x': x0,
                'f': f0,
                'sigma': 1.0,  # Initial step size
                'calls': 1,
                'stall': 0,
                'active': True,
                'best_local_f': f0
            })
        
        calls_used = num_starts
        rng = np.random.default_rng(42)
        
        # Step size adaptation parameters
        # 1/5 rule: if success rate in last 10 steps is 0.2, adjust sigma
        # Simpler: if improved, sigma *= 1.05, else sigma *= 0.95
        # Better: track success rate in a window.
        
        success_window = 10
        # We will implement a simple success rate tracker per trajectory
        
        while calls_used < budget:
            if time.time() > deadline:
                break
            
            # Check if all trajectories are inactive (stalled too much)
            # If so, start new ones
            active_trajs = [t for t in trajectories if t['active']]
            if not active_trajs:
                # Start new trajectories
                for i in range(num_starts):
                    if calls_used >= budget:
                        break
                    x0 = np.random.uniform(-4.5, 4.5, d)
                    f0 = func(x0)
                    calls_used += 1
                    if f0 < global_best_f:
                        global_best_f = f0
                        global_best_x = x0.copy()
                    
                    trajectories.append({
                        'x': x0,
                        'f': f0,
                        'sigma': 1.0,
                        'calls': 1,
                        'stall': 0,
                        'active': True,
                        'best_local_f': f0,
                        'successes': [] # List of 1s and 0s for last 10
                    })
                continue
            
            # Select one active trajectory to step
            # Round-robin or random? Random is fine.
            traj_idx = rng.integers(len(trajectories))
            while not trajectories[traj_idx]['active']:
                traj_idx = rng.integers(len(trajectories))
            
            t = trajectories[traj_idx]
            
            # Generate candidate
            # Sample from N(0, sigma^2)
            step = rng.normal(0, 1, d) * t['sigma']
            x_candidate = t['x'] + step
            
            # Clip to box
            x_candidate = np.clip(x_candidate, -5.0, 5.0)
            
            # Evaluate
            f_candidate = func(x_candidate)
            calls_used += 1
            
            # Update success history
            if 'successes' not in t:
                t['successes'] = []
            
            improved = f_candidate < t['f']
            if improved:
                t['successes'].append(1)
                t['f'] = f_candidate
                t['x'] = x_candidate
                t['stall'] = 0
                if f_candidate < global_best_f:
                    global_best_f = f_candidate
                    global_best_x = x_candidate.copy()
                if f_candidate < t['best_local_f']:
                    t['best_local_f'] = f_candidate
            else:
                t['successes'].append(0)
                t['stall'] += 1
            
            # Keep only last 10
            if len(t['successes']) > success_window:
                t['successes'] = t['successes'][-success_window:]
            
            # Adapt sigma based on success rate
            if len(t['successes']) >= success_window:
                success_rate = np.mean(t['successes'])
                if success_rate > 0.2:
                    t['sigma'] *= 1.05
                else:
                    t['sigma'] *= 0.95
            
            # Clamp sigma
            t['sigma'] = np.clip(t['sigma'], 1e-4, 5.0)
            
            # Stall detection: if no improvement for 50 steps, restart this trajectory
            if t['stall'] > 50:
                t['active'] = False
                # Optionally, we could immediately restart it, but the loop above handles it
                # by starting new ones when all are inactive.
                # To be more aggressive, we can restart it right away if we have budget.
                if calls_used < budget:
                    # Restart around global best or random?
                    # Random is better for diversity.
                    x0 = np.random.uniform(-4.5, 4.5, d)
                    f0 = func(x0)
                    calls_used += 1
                    if f0 < global_best_f:
                        global_best_f = f0
                        global_best_x = x0.copy()
                    
                    t['x'] = x0
                    t['f'] = f0
                    t['sigma'] = 1.0
                    t['calls'] = 1
                    t['stall'] = 0
                    t['active'] = True
                    t['best_local_f'] = f0
                    t['successes'] = []

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
