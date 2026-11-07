"""Rewritten optimizer: PSO for early progress + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """PSO + Nelder-Mead polish for BBOB f19 (Griewank-Rosenbrock) on [-5,5]^5.
    
    PSO is chosen because it evaluates particles immediately and updates based on
    best-so-far, giving rapid early progress which the AOCC metric rewards heavily.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _pso(self, func, x0, budget, rng, deadline, c1=2.0, c2=2.0, w0=0.9, w1=0.4):
        """Run Particle Swarm Optimization. Returns (best_x, best_f, calls_used)."""
        dim = self.dim
        n_particles = max(4, dim + 1)  # 6 particles for dim=5
        
        # Initialize particles around x0
        particles = np.array([x0 + rng.normal(0, 0.5, dim) for _ in range(n_particles)])
        particles = np.clip(particles, -5, 5)
        velocities = rng.normal(0, 0.3, (n_particles, dim))
        
        # Evaluate initial particles
        fits = np.array([func(p) for p in particles])
        calls_used = n_particles
        
        pbest = particles.copy()
        pbest_f = fits.copy()
        gbest_idx = np.argmin(fits)
        gbest = pbest[gbest_idx].copy()
        gbest_f = fits[gbest_idx]
        
        # PSO parameters
        w = w0
        max_iter = (budget - calls_used) // n_particles
        
        for it in range(max_iter):
            if calls_used >= budget or time.time() >= deadline:
                break
            
            # Update inertia weight (linear decrease)
            if max_iter > 0:
                w = w0 + (w1 - w0) * (it / max_iter)
            
            # Update velocities and positions
            for i in range(n_particles):
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                velocities[i] = (w * velocities[i] 
                                + c1 * r1 * (pbest[i] - particles[i])
                                + c2 * r2 * (gbest - particles[i]))
                # Clamp velocity
                velocities[i] = np.clip(velocities[i], -1.0, 1.0)
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], -5, 5)
            
            # Evaluate new positions
            new_fits = np.array([func(p) for p in particles])
            calls_used += n_particles
            
            # Update personal bests
            for i in range(n_particles):
                if new_fits[i] < pbest_f[i]:
                    pbest_f[i] = new_fits[i]
                    pbest[i] = particles[i].copy()
            
            # Update global best
            gbest_idx = np.argmin(pbest_f)
            if pbest_f[gbest_idx] < gbest_f:
                gbest_f = pbest_f[gbest_idx]
                gbest = pbest[gbest_idx].copy()
        
        return gbest, gbest_f, calls_used

    def _nelder_mead_polish(self, func, x0, budget, deadline):
        """Nelder-Mead polish with call counting. Returns (best_x, best_f, calls_used)."""
        if budget < 5 or time.time() >= deadline:
            return x0, func(x0), 1
        
        calls_used = 0
        best_x = x0.copy()
        best_f = func(x0)
        calls_used += 1
        
        # Counting wrapper
        count = [calls_used]
        def counted_func(x):
            nonlocal best_x, best_f
            count[0] += 1
            if time.time() >= deadline or count[0] >= budget:
                raise StopIteration
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        try:
            res = minimize(counted_func, best_x, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12,
                                   'adaptive': True})
        except StopIteration:
            pass
        except Exception:
            pass
        
        calls_used = count[0]
        return best_x, best_f, calls_used

    def __call__(self, func):
        deadline = time.time() + 320.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        
        global_best_x = np.zeros(dim)
        global_best_f = np.inf
        calls_remaining = budget
        
        # Phase 1: PSO for fast early progress (best for AOCC)
        # Use ~60% of budget for PSO
        pso_budget = int(budget * 0.6)
        if pso_budget < 10:
            pso_budget = budget
        
        # Multiple PSO starts with different initializations
        start_configs = [
            (np.zeros(dim), 0.5),           # Center start
            (rng.uniform(-2, 2, dim), 0.8), # Random moderate
            (rng.uniform(-3, 3, dim), 1.0), # Random wider
        ]
        
        # Allocate PSO budget: 50%, 30%, 20%
        pso_allocs = [0.5, 0.3, 0.2]
        
        for i, (x0, spread) in enumerate(start_configs):
            if calls_remaining <= 5 or time.time() >= deadline:
                break
            
            alloc = int(calls_remaining * pso_allocs[i])
            if alloc < 10:
                alloc = calls_remaining
            
            bx, bf, used = self._pso(func, x0, alloc, rng, deadline)
            calls_remaining -= used
            
            if bf < global_best_f:
                global_best_f = bf
                global_best_x = bx.copy()
            
            # Stop early if we've found a very good solution
            if global_best_f < 1e-4:
                break
        
        # Phase 2: Nelder-Mead polish with remaining budget
        if calls_remaining > 10 and time.time() < deadline:
            px, pf, pused = self._nelder_mead_polish(func, global_best_x, calls_remaining, deadline)
            if pf < global_best_f:
                global_best_f = pf
                global_best_x = px.copy()
        
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
