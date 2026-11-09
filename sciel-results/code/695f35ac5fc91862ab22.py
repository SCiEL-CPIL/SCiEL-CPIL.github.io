import numpy as np
import time

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 315  # Safety margin: 315s out of 330s limit
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0
        dim = self.dim
        
        # Helper: Evaluate and update best
        def evaluate(x):
            nonlocal best_x, best_f, eval_count
            x = np.clip(x, -5, 5)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Initial diverse sampling
        # 1 point at origin, 4 random points
        n_init = 5
        init_points = [np.zeros(dim)]
        for _ in range(n_init - 1):
            init_points.append(rng.uniform(-5, 5, size=dim))
            
        for x in init_points:
            if eval_count >= self.budget or time.time() > deadline - 10:
                break
            evaluate(x)
        
        # Phase 2: CMA-ES for global search
        if HAS_CMA and eval_count < self.budget - 10:
            try:
                x0 = best_x.tolist()
                sigma0 = 2.0  # Larger initial sigma for better coverage
                popsize = 10
                
                remaining_budget = self.budget - eval_count
                maxiter = max(1, remaining_budget // popsize)
                
                options = {
                    'verbose': -9,
                    'seed': 42,
                    'bounds': [-5.0] * dim + [5.0] * dim,
                    'CMA_boundaryhandler': 'reflect',
                    'popsize': popsize,
                    'maxiter': maxiter,
                    'stopitersize': 1e-10,
                    'stopconditionnumber': 1e12,
                    'stopfgap': 1e-12,
                }
                
                es = cma.CMA(x0, sigma0, options)
                
                while not es.stop() and eval_count < self.budget:
                    if time.time() > deadline - 5:
                        break
                    
                    xs = es.ask()
                    fs = []
                    for x in xs:
                        if eval_count >= self.budget:
                            break
                        if time.time() > deadline - 5:
                            break
                        fx = evaluate(x)
                        fs.append(fx)
                    
                    if len(fs) == len(xs):
                        es.tell(xs, fs)
                    else:
                        break
                        
            except Exception:
                pass
        else:
            # Fallback: (1+1)-ES if CMA is not available
            if eval_count < self.budget:
                x = best_x.copy()
                fx = best_f
                sigma = 1.0
                while eval_count < self.budget and time.time() < deadline - 5:
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = evaluate(y)
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.2
                    else:
                        sigma *= 0.9
                        if sigma < 1e-4:
                            break

        return best_x, best_f
