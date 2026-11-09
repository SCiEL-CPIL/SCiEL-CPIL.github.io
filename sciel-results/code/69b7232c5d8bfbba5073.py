"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start CMA-ES optimizer for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit
        
        dim = self.dim
        total_budget = self.budget
        
        # We will run multiple independent CMA-ES instances.
        # For dim=5, a small number of restarts (e.g., 3-5) is often effective for multimodal functions.
        # We allocate the budget dynamically.
        
        # Parameters for each CMA-ES instance
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # State for each instance
        # We start with 3 instances to cover space, but can merge or stop if they converge.
        # To keep it simple and robust within time limits, we will run a fixed number of 
        # parallel instances that share the global call counter.
        
        num_instances = 3
        rng = np.random.default_rng(42)
        
        # Initialize instances with diverse starts
        # Instead of purely random, we use a grid + noise or just distinct random points
        # to ensure coverage.
        instances = []
        for i in range(num_instances):
            # Diverse starting points
            # Use vertices of a hypercube scaled to [-4, 4] plus noise? 
            # Or just uniform random but seeded differently.
            # For Schaffers F7, local minima are frequent. 
            # Let's use uniform random in [-4, 4] for the mean.
            x0 = rng.uniform(-4, 4, dim)
            
            inst = {
                'x': x0,
                'sigma': 2.0,
                'C': np.eye(dim),
                'C_sqrt': np.eye(dim),
                'pc': np.zeros(dim),
                'ps': np.zeros(dim),
                'best_x': x0,
                'best_f': float('inf'),
                'calls_used': 0,
                'active': True
            }
            # Evaluate initial point
            fx = func(x0)
            inst['best_f'] = fx
            inst['best_x'] = x0
            inst['calls_used'] = 1
            instances.append(inst)
            
        global_best_f = min(inst['best_f'] for inst in instances)
        global_best_x = min(instances, key=lambda x: x['best_f'])['best_x'].copy()
        
        total_calls = num_instances
        
        # Main loop
        while total_calls < total_budget:
            # Check time
            if time.time() > deadline - 5.0:
                break
                
            # Determine which instances are still active
            # Simple heuristic: If an instance hasn't improved in last K generations, 
            # or if sigma is too small, it's done. 
            # For simplicity and speed, we just run all active instances.
            # We allocate calls to instances in a round-robin fashion, but 
            # we can bias towards the one with the best current best_f?
            # Actually, standard multi-start just runs them independently until budget runs out.
            # Since budget is shared, we just pick an active instance to step.
            
            # To be fair and simple, we cycle through active instances.
            # But some might converge faster. 
            # Let's just iterate through them. If one is "stuck" (sigma < 1e-8), mark inactive.
            
            for inst in instances:
                if not inst['active']:
                    continue
                if total_calls >= total_budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                    
                # Generate population for this instance
                x = inst['x']
                sigma = inst['sigma']
                C_sqrt = inst['C_sqrt']
                
                ys = []
                fs = []
                
                for _ in range(lam):
                    if total_calls >= total_budget:
                        break
                    if time.time() > deadline - 5.0:
                        break
                        
                    z = rng.standard_normal(dim)
                    y = x + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    
                    fy = func(y)
                    total_calls += 1
                    inst['calls_used'] += 1
                    
                    if fy < inst['best_f']:
                        inst['best_f'] = fy
                        inst['best_x'] = y.copy()
                        
                    if fy < global_best_f:
                        global_best_f = fy
                        global_best_x = y.copy()
                        
                    ys.append(y)
                    fs.append(fy)
                    
                if len(fs) < lam:
                    # Budget or time ran out mid-generation
                    inst['active'] = False # Stop this instance for now
                    continue
                    
                # Sort
                idx = np.argsort(fs)
                ys = [ys[i] for i in idx]
                fs = [fs[i] for i in idx]
                
                # Update mean
                x_old = x.copy()
                x_new = sum(weights[i] * ys[i] for i in range(mu))
                
                # Update step-size control (CSA)
                ps = (1 - cs) * inst['ps'] + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (C_sqrt.T @ (x_new - x_old))
                norm_ps = np.linalg.norm(ps)
                
                # Update covariance matrix
                hsig = (norm_ps / chi_n) < (1.4 + 2 / (dim + 1)) * (1 - cc)
                
                pc = (1 - cc) * inst['pc'] + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (x_new - x_old) / sigma
                
                # C update
                C = inst['C']
                C = (1 - c1 - cmu) * C
                C += c1 * (np.outer(pc, pc) + (hsig - 1 + cc) * cc * C)
                for i in range(mu):
                    yi = (ys[i] - x_old) / sigma
                    C += cmu * weights[i] * np.outer(yi, yi)
                
                C = (C + C.T) / 2
                
                try:
                    C_sqrt_new = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-9 * np.eye(dim)
                    C_sqrt_new = np.linalg.cholesky(C)
                    
                # Update sigma
                sigma_new = sigma * np.exp(cs / damps * (norm_ps / chi_n - 1))
                
                # Update instance state
                inst['x'] = x_new
                inst['sigma'] = sigma_new
                inst['C'] = C
                inst['C_sqrt'] = C_sqrt_new
                inst['pc'] = pc
                inst['ps'] = ps
                
                # Stopping criteria for this instance
                if sigma_new < 1e-10 or norm_ps > 1e10:
                    inst['active'] = False
                    
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
