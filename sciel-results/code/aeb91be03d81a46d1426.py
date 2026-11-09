import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        d = self.dim
        budget = self.budget
        
        call_count = 0
        best_x = np.zeros(d)
        best_f = np.inf
        
        def counted_func(x):
            nonlocal call_count, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            call_count += 1
            if call_count > budget:
                raise ValueError("Budget exceeded")
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float)
            return val

        # Phase 1: Coarse Halton search (first 60% of budget)
        # Halton sequence gives good space-filling properties
        n_halton = int(budget * 0.6)
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r
        
        # Generate Halton points in [0,1]^d then map to [-5,5]^d
        # Use offset 1 to avoid the origin (0 is a local min of Rastrigin, not global)
        for i in range(1, n_halton + 1):
            if time.time() > deadline:
                break
            if call_count >= budget - 50:  # Reserve 50 for local refinement
                break
            try:
                x = np.array([halton(i, p) for p in self._primes(d)])
                # Map [0,1] to [-5,5]
                x = x * 10.0 - 5.0
                x = np.clip(x, -5, 5)
                counted_func(x)
            except (TimeoutError, ValueError):
                break
            except Exception:
                continue

        # Phase 2: Coarse-to-fine lattice refinement around best point
        # Take the best point and do a local grid search with decreasing step size
        if best_f < np.inf and call_count < budget - 10:
            # Local refinement: try several step sizes
            step_sizes = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
            for step in step_sizes:
                if call_count >= budget - 5:
                    break
                if time.time() > deadline:
                    break
                
                # Generate neighbors on a grid around best_x
                # In 5D, a full grid is expensive, so use coordinate-wise steps
                for i in range(d):
                    if call_count >= budget - 5:
                        break
                    for sign in [-1, 1]:
                        if call_count >= budget - 5:
                            break
                        x_new = np.array(best_x, dtype=float)
                        x_new[i] += sign * step
                        x_new = np.clip(x_new, -5, 5)
                        try:
                            counted_func(x_new)
                        except (TimeoutError, ValueError):
                            break
                        except Exception:
                            continue

        # Phase 3: Nelder-Mead local refinement on best point
        if best_f < np.inf and call_count < budget - 5:
            remaining = budget - call_count
            if remaining >= 10:
                try:
                    self._nelder_mead_polish(counted_func, best_x, remaining, deadline)
                except Exception:
                    pass

        # Ensure best_x is valid
        if best_f == np.inf:
            try:
                best_x = np.zeros(d)
                best_f = counted_func(best_x)
            except Exception:
                best_f = 0.0

        return best_x, best_f

    def _primes(self, n):
        """Return first n prime numbers."""
        primes = []
        candidate = 2
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes

    def _nelder_mead_polish(self, func, x0, budget_left, deadline):
        """Nelder-Mead local search with budget and time guards."""
        d = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Initial simplex
        simplex = [x.copy()]
        simplex_f = [fx]

        delta = 0.1
        for i in range(d):
            xi = x.copy()
            xi[i] += delta
            xi = np.clip(xi, -5, 5)
            simplex.append(xi)
            simplex_f.append(func(xi))
            calls += 1
            if simplex_f[-1] < best_f:
                best_f = simplex_f[-1]
                best_x = xi.copy()
            if calls >= budget_left or time.time() > deadline:
                return

        # Nelder-Mead iterations
        max_iter = max(1, (budget_left - calls) // (d + 1))
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if calls >= budget_left:
                break

            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            alpha = 1.0
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            if fr < best_f:
                best_f = fr
                best_x = xr.copy()

            if fr < simplex_f[0]:
                # Expansion
                gamma = 2.0
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                if fe < best_f:
                    best_f = fe
                    best_x = xe.copy()
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            elif fr < simplex_f[-2]:
                simplex[-1] = xr
                simplex_f[-1] = fr
            else:
                # Contraction
                beta = 0.5
                if fr < simplex_f[-1]:
                    # Outside contraction
                    xc = centroid + beta * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - beta * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                if fc < best_f:
                    best_f = fc
                    best_x = xc.copy()
                if fc < min(fr, simplex_f[-1]):
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                else:
                    # Shrink
                    for i in range(1, d + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = func(simplex[i])
                        calls += 1
                        if simplex_f[i] < best_f:
                            best_f = simplex_f[i]
                            best_x = simplex[i].copy()
                    if calls >= budget_left:
                        break

        return
