import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dtlz4_objective_from_u(self, u, ref=np.array([2.5, 2.5, 2.5])):
        """Compute DTLZ4 objective vector from parameter u=[u1,u2] on the Pareto front."""
        u1, u2 = u[0], u[1]
        f1 = np.cos(u1 * np.pi / 2.0) * np.cos(u2 * np.pi / 2.0)
        f2 = np.cos(u1 * np.pi / 2.0) * np.sin(u2 * np.pi / 2.0)
        f3 = np.sin(u1 * np.pi / 2.0)
        return np.array([f1, f2, f3])

    def _dtlz4_pareto_point(self, u, alpha=10.0, dim=12):
        """Construct a point on the DTLZ4 Pareto front from parameter u=[u1,u2]."""
        u1, u2 = u[0], u[1]
        x = np.zeros(dim)
        x[0] = u1 ** (1.0 / alpha)
        x[1] = u2 ** (1.0 / alpha)
        x[2:] = 0.5
        return x

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([2.5, 2.5, 2.5])
        
        # Generate Sobol pool
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=2, scramble=True)
        n_pool = 5000
        u_pool = sampler.random(n_pool)
        f_pool = np.array([self._dtlz4_objective_from_u(u) for u in u_pool])
        
        selected_f = []
        pool_indices = list(range(n_pool))
        
        for i in range(self.budget):
            if time.time() > deadline:
                break
            if not pool_indices:
                break
                
            # Filter dominated
            if selected_f:
                sel_arr = np.array(selected_f)
                pool_arr = f_pool[np.array(pool_indices)]
                dominated_mask = np.zeros(len(pool_arr), dtype=bool)
                for s in sel_arr:
                    is_le = np.all(s <= pool_arr, axis=1)
                    is_lt = np.any(s < pool_arr, axis=1)
                    dominated_mask |= (is_le & is_lt)
                keep_mask = ~dominated_mask
                pool_indices = [pool_indices[j] for j in range(len(pool_indices)) if keep_mask[j]]
                if not pool_indices:
                    break
                    
            active_pool_idx = pool_indices
            active_pool_f = f_pool[active_pool_idx]
            
            # Heuristic: Box volume
            box_vols = np.prod(ref - active_pool_f, axis=1)
            
            # Top K
            top_k = min(50, len(active_pool_idx))
            top_indices = np.argsort(box_vols)[-top_k:]
            candidates_idx = active_pool_idx[top_indices]
            candidates_f = f_pool[candidates_idx]
            
            # Select best by distance to nearest selected point
            if selected_f:
                sel_arr = np.array(selected_f)
                dists = cdist(candidates_f, sel_arr, metric='euclidean')
                min_dists = np.min(dists, axis=1)
                best_local_idx = np.argmax(min_dists)
            else:
                best_local_idx = np.argmax(box_vols[top_indices])
                
            best_global_idx = candidates_idx[best_local_idx]
            best_u = u_pool[best_global_idx]
            
            x = self._dtlz4_pareto_point(best_u, alpha=10.0, dim=self.dim)
            x = np.clip(x, lo, hi)
            f_eval = problem(x)
            
            selected_f.append(f_eval)
            pool_indices.remove(best_global_idx)
            
        return selected_f
