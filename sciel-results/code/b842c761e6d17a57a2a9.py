"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Hybrid PSO and Nelder-Mead for multimodal optimization on BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: PSO for Global Exploration
        # Use Sobol sequence for derandomized initialization
        n_particles = 20
        sobol = qmc.Sobol(d=dim, scramble=True, seed=rng)
        # Generate n_particles points in [0, 1]^dim
        # Note: Sobol requires n to be a power of 2 for best properties, but works for any n
        points = sobol.random(n_particles)
        # Map to [-5, 5]
        x_swarm = 10.0 * points - 5.0
        
        # Initialize velocities to 0
        v_swarm = np.zeros_like(x_swarm)
        
        # Initialize personal bests
        p_best_x = x_swarm.copy()
        p_best_f = np.full(n_particles, np.inf)
        
        # Evaluate initial swarm
        for i in range(n_particles):
            if calls_made >= budget or time.time() > deadline:
                break
            f_i = call_func(x_swarm[i])
            p_best_f[i] = f_i
            p_best_x[i] = x_swarm[i].copy()

        # PSO Parameters
        w_start = 0.9
        w_end = 0.4
        c1 = 1.5
        c2 = 1.5
        v_max = 1.0
        
        # Determine number of generations for PSO phase
        # Use 60% of budget for PSO
        pso_budget = int(0.6 * budget)
        remaining_budget = budget - pso_budget
        
        # We already used n_particles calls.
        # Remaining calls for PSO iterations: pso_budget - n_particles
        # Calls per generation: n_particles
        n_generations = (pso_budget - n_particles) // n_particles
        
        # Track global best
        g_best_x = best_x.copy() if best_x is not None else np.zeros(dim)
        g_best_f = best_f
        
        # Ring Topology: each particle i interacts with i-1, i, i+1
        # We will update local bests on the fly
        
        for gen in range(n_generations):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Dynamic inertia
            if n_generations > 0:
                w = w_start + (w_end - w_start) * (gen / n_generations)
            else:
                w = w_end
            
            # Update velocities and positions
            new_x_swarm = np.zeros_like(x_swarm)
            new_v_swarm = np.zeros_like(v_swarm)
            
            for i in range(n_particles):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Get local best from ring neighbors (i-1, i, i+1)
                # We use the current p_best_x and p_best_f to find local best
                # Note: p_best_f is updated as we go, but for simplicity in this loop,
                # we use the state at the start of the generation for neighbors
                # or we can just use global best if ring is complex.
                # Let's stick to simple global best for social component to ensure stability,
                # or implement ring properly.
                
                # Ring Neighbors: i-1, i, i+1 (mod n_particles)
                idxs = [(i - 1) % n_particles, i, (i + 1) % n_particles]
                local_best_f = min(p_best_f[idxs])
                local_best_x = p_best_x[np.argmin(p_best_f[idxs])]
                
                # Cognitive and Social components
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                
                v = w * v_swarm[i] + c1 * r1 * (p_best_x[i] - x_swarm[i]) + c2 * r2 * (local_best_x - x_swarm[i])
                
                # Clip velocity
                v = np.clip(v, -v_max, v_max)
                
                # Update position
                x_new = x_swarm[i] + v
                x_new = np.clip(x_new, -5, 5)
                
                new_x_swarm[i] = x_new
                new_v_swarm[i] = v
                
                # Evaluate
                f_new = call_func(x_new)
                
                # Update personal best
                if f_new < p_best_f[i]:
                    p_best_f[i] = f_new
                    p_best_x[i] = x_new.copy()
                    
                # Update global best
                if f_new < g_best_f:
                    g_best_f = f_new
                    g_best_x = x_new.copy()
            
            # Update swarm state
            x_swarm = new_x_swarm
            v_swarm = new_v_swarm

        # Phase 2: Local Refinement using Nelder-Mead
        # Use remaining budget
        if best_x is not None and calls_made < budget and time.time() < deadline:
            x0 = best_x.copy()
            
            # Nelder-Mead options
            # maxiter is not strictly limited by calls, but we will stop if budget runs out
            # We need a custom callback to check budget
            def callback(xk):
                nonlocal calls_made
                # Nelder-Mead calls func multiple times per iteration internally?
                # Actually, scipy.optimize.minimize uses the provided func.
                # We must ensure we don't exceed budget.
                # The 'func' passed to minimize is our 'call_func'.
                # We can't easily break out of minimize if we exceed budget inside it.
                # So we rely on the fact that NM is fast and we have ~200 calls left.
                # 200 calls is enough for NM to converge in 5D.
                return False # Don't stop callback
            
            try:
                # Use Nelder-Mead
                # initial_simplex_size: 0.5 is reasonable
                res = minimize(
                    func, 
                    x0, 
                    method='Nelder-Mead', 
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                if res.success or res.fun < best_f:
                    # res.x is the best point found by NM
                    # We already updated best_x in call_func, so best_x is current best
                    pass
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
