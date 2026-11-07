import numpy as np
import time

class Optimizer:
    """Hybrid PSO-CMA-ES optimizer for BBOB-style f9 (Rosenbrock rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES with the given initial state and budget."""
        dim = self.dim
        if budget <= 0:
            return x0, func(x0), 0

        # Adjust lambda for small dim
        lam = 4 + int(3 * np.log(dim))  # ~12 for dim=5
        mu = lam // 2
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, (mu_eff - 2) / (dim + 2) - 1) + cs

        x = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-20)
        B = eigvecs
        D = np.sqrt(eigvals)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = x.copy()
        best_fx = func(x)
        calls = 1
        no_improve = 0

        for gen in range(budget // lam):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            pop = []
            fvals = []
            for i in range(lam):
                if calls >= budget:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (B * D) @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                pop.append(y)
                fvals.append(fy)
                calls += 1

            if len(pop) == 0:
                break

            idx = np.argsort(fvals)
            pop = [pop[i] for i in idx]
            fvals = np.array(fvals)[idx]

            if fvals[0] < best_fx:
                best_fx = fvals[0]
                best_x = pop[0].copy()
                no_improve = 0
            else:
                no_improve += 1

            if no_improve > 15 and sigma < 1e-4:
                break

            mu_use = min(mu, len(pop))
            m = np.sum(weights[:mu_use, None] * np.array(pop[:mu_use]), axis=0)

            # Update step size
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B.T @ (m - x))
            hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 100.0)

            # Update evolution path
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) / sigma * (B.T @ (m - x))

            # Update covariance
            delta_hs = (1 - hsig) * cc * (2 - cc)
            outer = np.outer(pc, pc)
            C = (1 + c1 * delta_hs - c1 - cmu) * C + c1 * outer
            for i in range(mu_use):
                zi = (np.array(pop[i]) - x) / sigma
                C += cmu * weights[i] * np.outer(zi, zi)

            # Ensure symmetry and positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            B = eigvecs
            D = np.sqrt(eigvals)

            x = m

        return best_x, best_fx, calls

    def _pso_global(self, func, budget, rng, deadline):
        """Run PSO for global exploration."""
        dim = self.dim
        swarm_size = 20
        # Allocate a portion of the budget to PSO
        # We need to be careful not to exceed the total budget
        # Let's use roughly 40% of the budget for PSO
        pso_budget = int(budget * 0.4)
        
        if pso_budget < swarm_size * 2:
            pso_budget = swarm_size * 2

        # Initialize swarm
        particles = rng.uniform(-5, 5, size=(swarm_size, dim))
        velocities = rng.uniform(-2, 2, size=(swarm_size, dim))
        
        pbests = particles.copy()
        pbest_fvals = np.array([func(p) for p in particles])
        
        gbest = pbests[np.argmin(pbest_fvals)].copy()
        gbest_fval = np.min(pbest_fvals)
        
        calls = swarm_size
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient

        while calls < pso_budget:
            if time.time() > deadline:
                break
            
            for i in range(swarm_size):
                if calls >= pso_budget:
                    break
                
                # Update velocity
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                velocities[i] = w * velocities[i] + c1 * r1 * (pbests[i] - particles[i]) + c2 * r2 * (gbest - particles[i])
                
                # Update position
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], -5, 5)
                
                # Evaluate
                fval = func(particles[i])
                calls += 1
                
                # Update personal best
                if fval < pbest_fvals[i]:
                    pbest_fvals[i] = fval
                    pbests[i] = particles[i].copy()
                    
                    # Update global best
                    if fval < gbest_fval:
                        gbest_fval = fval
                        gbest = particles[i].copy()
                        
        return gbest, gbest_fval, calls

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        rng = np.random.default_rng(42)
        
        # Phase 1: Global Exploration with PSO
        # We give PSO a significant chunk of the budget to find a good basin
        # The remaining budget will be used by CMA-ES
        pso_budget = int(budget * 0.4)
        if pso_budget < 50:
            pso_budget = 50
            
        x_pso, fx_pso, calls_pso = self._pso_global(func, pso_budget, rng, deadline)
        
        # Phase 2: Local Refinement with CMA-ES
        remaining_budget = budget - calls_pso
        if remaining_budget > 0:
            # Start CMA-ES from the best point found by PSO
            # Use a moderate initial sigma to allow some local movement but focus on refinement
            sigma0 = 1.0
            x_cma, fx_cma, calls_cma = self._cmaes(func, x_pso, sigma0, remaining_budget, rng, deadline)
            
            # Determine the overall best
            if fx_cma < fx_pso:
                best_x = x_cma
                best_fx = fx_cma
            else:
                best_x = x_pso
                best_fx = fx_pso
        else:
            best_x = x_pso
            best_fx = fx_pso

        return best_x, best_fx
