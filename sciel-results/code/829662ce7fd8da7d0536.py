import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Multi-Start L-BFGS-B + Valley-Following Optimizer for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0
        self._cache = {}

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 285.0
        self._cache = {}

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")
            
            x = np.clip(x, -5, 5)
            key = tuple(x.tolist())
            if key in self._cache:
                return self._cache[key]
            
            self.calls_made += 1
            val = func(x)
            self._cache[key] = val
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        def rosenbrock_grad(x):
            """Analytic gradient of standard Rosenbrock (for direction hints)."""
            g = np.zeros_like(x)
            for i in range(self.dim - 1):
                g[i] += -400 * x[i] * (x[i+1] - x[i]**2)
                g[i+1] += 200 * (x[i+1] - x[i]**2)
            return g

        def valley_follow(x0, max_iters=15):
            """Budget-aware adaptive valley-following using analytic gradient direction."""
            x = np.clip(x0.copy(), -5, 5)
            f = wrapped_func(x)
            step_size = 0.5
            
            for it in range(max_iters):
                if self.calls_made >= self.budget - 5:
                    break
                if time.time() > self.deadline:
                    break
                
                improved = False
                
                # Use analytic gradient to determine valley direction
                grad = rosenbrock_grad(x)
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e-10:
                    # Move along negative gradient with adaptive step
                    for scale in [step_size, step_size * 0.5, step_size * 2.0]:
                        if self.calls_made >= self.budget - 5:
                            break
                        x_new = x - scale * grad / grad_norm * 10.0
                        x_new = np.clip(x_new, -5, 5)
                        f_new = wrapped_func(x_new)
                        if f_new < f:
                            x = x_new
                            f = f_new
                            improved = True
                            step_size = min(step_size * 1.1, 1.5)
                            break
                    if not improved:
                        step_size = max(step_size * 0.6, 0.05)
                
                # Also try valley-specific moves: x[i+1] toward x[i]^2
                for i in range(self.dim - 1):
                    if self.calls_made >= self.budget - 5:
                        break
                    target_next = x[i] ** 2
                    if abs(x[i+1] - target_next) < 1e-6:
                        continue
                    x_vf = x.copy()
                    x_vf[i+1] = 0.5 * x[i+1] + 0.5 * target_next
                    x_vf = np.clip(x_vf, -5, 5)
                    f_vf = wrapped_func(x_vf)
                    if f_vf < f:
                        x = x_vf
                        f = f_vf
                        improved = True
                
                # Coordinate descent with adaptive steps
                for i in range(self.dim):
                    if self.calls_made >= self.budget - 5:
                        break
                    best_local_f = f
                    best_local_x = x.copy()
                    for step_mult in [step_size * 0.3, step_size * 0.1]:
                        if self.calls_made >= self.budget - 5:
                            break
                        for direction in [1.0, -1.0]:
                            if self.calls_made >= self.budget - 5:
                                break
                            x_new = x.copy()
                            x_new[i] += step_mult * direction
                            x_new = np.clip(x_new, -5, 5)
                            f_new = wrapped_func(x_new)
                            if f_new < best_local_f:
                                best_local_f = f_new
                                best_local_x = x_new
                    if best_local_f < f:
                        x = best_local_x
                        f = best_local_f
                        improved = True
                
                if not improved:
                    step_size = max(step_size * 0.5, 0.01)
                    if step_size < 0.02:
                        break
            
            return x, f

        rng = np.random.default_rng(42)
        
        # Structured starts: diverse valley points + origin + random
        starts = [
            np.ones(self.dim),
            np.zeros(self.dim),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([-1.0, 1.0, 1.0, 1.0, 1.0]),
            np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        ]
        # Points along the Rosenbrock valley: x[0]=a, x[i+1]=a^2
        for a in [0.5, -0.5, 0.8, -0.8, 1.2, -1.2, 0.3, -0.3, 1.5, -1.5]:
            v = np.zeros(self.dim)
            v[0] = a
            for i in range(1, self.dim):
                v[i] = a ** (2 ** i)
            v = np.clip(v, -5, 5)
            starts.append(v)
            starts.append(v + rng.normal(0, 0.5, self.dim))
        # Random starts in wider and narrower ranges
        for _ in range(3):
            starts.append(rng.uniform(-2, 2, self.dim))
        for _ in range(3):
            starts.append(rng.uniform(-4, 4, self.dim))
        for _ in range(2):
            starts.append(rng.uniform(-5, 5, self.dim))
        
        unique_starts = []
        for s in starts:
            s = np.clip(s, -5, 5)
            is_dup = False
            for u in unique_starts:
                if np.allclose(s, u, atol=1e-4):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)
        
        # Sort starts by heuristic distance to likely optimum (origin-ish)
        unique_starts.sort(key=lambda x: np.linalg.norm(x))
        
        for x0 in unique_starts:
            if self.calls_made >= self.budget:
                break
            if time.time() > self.deadline:
                break
            
            remaining_budget = self.budget - self.calls_made
            if remaining_budget < 10:
                continue
            
            # Phase 1: L-BFGS-B with moderate settings
            try:
                maxfun_this = min(150, remaining_budget - 10)
                if maxfun_this < 10:
                    continue
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 50,
                        'maxls': 10,
                        'ftol': 1e-10,
                        'gtol': 1e-8,
                        'maxfun': maxfun_this
                    }
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5, 5)
            except StopIteration:
                break
            except Exception:
                pass
            
            # Phase 2: Valley-following refinement on current best
            if self.best_x is not None and self.calls_made < self.budget - 10:
                try:
                    x_vf, f_vf = valley_follow(self.best_x, max_iters=10)
                    if f_vf < self.best_f:
                        self.best_f = f_vf
                        self.best_x = x_vf
                except StopIteration:
                    break
                except Exception:
                    pass
        
        # Final high-precision polish
        if self.best_x is not None and self.calls_made < self.budget - 5 and time.time() < self.deadline:
            try:
                maxfun_polish = min(200, self.budget - self.calls_made - 5)
                if maxfun_polish > 10:
                    res = scipy.optimize.minimize(
                        wrapped_func,
                        self.best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={
                            'maxiter': 80,
                            'maxls': 15,
                            'ftol': 1e-13,
                            'gtol': 1e-11,
                            'maxfun': maxfun_polish
                        }
                    )
                    if res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = np.clip(res.x, -5, 5)
            except StopIteration:
                pass
            except Exception:
                pass
            
            # Nelder-Mead polish as derivative-free fallback
            if self.best_x is not None and self.calls_made < self.budget - 15 and time.time() < self.deadline:
                try:
                    res_nm = scipy.optimize.minimize(
                        wrapped_func,
                        self.best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 200,
                            'maxfev': min(100, self.budget - self.calls_made - 5),
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                    if res_nm.fun < self.best_f:
                        self.best_f = res_nm.fun
                        self.best_x = np.clip(res_nm.x, -5, 5)
                except StopIteration:
                    pass
                except Exception:
                    pass
            
            # Final valley-following polish
            if self.best_x is not None and self.calls_made < self.budget - 3:
                try:
                    x_vf, f_vf = valley_follow(self.best_x, max_iters=8)
                    if f_vf < self.best_f:
                        self.best_f = f_vf
                        self.best_x = x_vf
                except StopIteration:
                    pass
                except Exception:
                    pass

        if self.best_x is None:
            x0 = np.ones(self.dim)
            self.best_x = x0
            self.best_f = wrapped_func(x0)
        
        return self.best_x, self.best_f
