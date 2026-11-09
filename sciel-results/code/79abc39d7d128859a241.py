# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based multi-objective optimizer for WFG4 with NSGA-II selection and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    @staticmethod
    def _fast_non_dominated_sort(F):
        """Fast non-dominated sorting. Returns list of fronts (each is array of indices)."""
        n = F.shape[0]
        if n == 0:
            return []
        
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
        
        front = np.array([i for i in range(n) if domination_count[i] == 0])
        if len(front) == 0:
            return [np.arange(n)]
        
        fronts = [front]
        current_front = front
        while len(current_front) > 0:
            next_front = []
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(np.array(next_front))
                current_front = np.array(next_front)
            else:
                break
        
        return fronts

    @staticmethod
    def _crowding_distance(F, idx):
        """Compute crowding distance for points at indices idx in F."""
        k = len(idx)
        if k <= 2:
            return np.full(k, np.inf)
        M = F.shape[1]
        dist = np.zeros(k)
        Ff = F[idx]
        for m in range(M):
            vals = Ff[:, m]
            vmin, vmax = vals.min(), vals.max()
            if vmax - vmin < 1e-18:
                continue
            order = np.argsort(vals)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            for i in range(1, k - 1):
                dist[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / (vmax - vmin)
        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 310.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        N = min(30, max(10, budget // 6))

        # Initial random population
        X = lo + span * np.random.rand(N, dim)
        F = np.array([problem(x) for x in X])
        used = N

        # DE parameters
        F_factor = 0.8
        CR = 0.9

        def time_ok():
            return (time.time() - start_time) < deadline and used < budget

        # Phase 1: DE-based exploration
        while time_ok():
            n_offspring = min(N, budget - used)
            
            children = np.zeros((n_offspring, dim))
            Fc = np.zeros((n_offspring, n_obj))
            
            for i in range(n_offspring):
                if not time_ok():
                    break
                
                idx = np.random.choice(N, 3, replace=False)
                a, b, c = X[idx[0]], X[idx[1]], X[idx[2]]
                base = X[i % N]
                
                progress = used / budget
                F_adapt = F_factor * (1.0 - 0.3 * progress)
                
                v = a + F_adapt * (b - c)
                
                u = np.random.rand(dim) < CR
                if not np.any(u):
                    jrand = np.random.randint(dim)
                    u[jrand] = True
                
                child = np.where(u, v, base)
                child = np.clip(child, lo, hi)
                
                children[i] = child
                Fc[i] = problem(child)
                used += 1
            
            if n_offspring == 0:
                break

            Xall = np.vstack([X, children[:n_offspring]])
            Fall = np.vstack([F, Fc[:n_offspring]])
            
            fronts = self._fast_non_dominated_sort(Fall)
            
            selected = np.zeros(N, dtype=int)
            count = 0
            for fr in fronts:
                if count + len(fr) <= N:
                    selected[count:count + len(fr)] = fr
                    count += len(fr)
                else:
                    need = N - count
                    cd = self._crowding_distance(Fall, fr)
                    order = np.argsort(-cd)
                    selected[count:count + need] = fr[order[:need]]
                    count += need
                    break
            
            X = Xall[selected]
            F = Fall[selected]

        # Phase 2: Local polish - refine best solutions with small perturbations
        if time_ok() and used < budget:
            # Get the first front (best non-dominated solutions)
            fronts = self._fast_non_dominated_sort(F)
            if fronts and len(fronts[0]) > 0:
                best_idx = fronts[0]
                n_best = len(best_idx)
                
                # Try to refine each best solution with small Gaussian perturbations
                remaining_budget = budget - used
                attempts_per_solution = max(1, remaining_budget // max(1, n_best))
                
                for bi in range(n_best):
                    if used >= budget or (time.time() - start_time) > deadline:
                        break
                    
                    base_x = X[best_idx[bi]]
                    base_f = F[best_idx[bi]]
                    
                    for attempt in range(attempts_per_solution):
                        if used >= budget or (time.time() - start_time) > deadline:
                            break
                        
                        # Small Gaussian perturbation
                        sigma = 0.02 * span
                        child = base_x + sigma * np.random.randn(dim)
                        child = np.clip(child, lo, hi)
                        
                        fc = problem(child)
                        used += 1
                        
                        # Check if child dominates base or is non-dominated with it
                        if np.all(fc <= base_f) and np.any(fc < base_f):
                            # Child dominates base - replace
                            X[best_idx[bi]] = child
                            F[best_idx[bi]] = fc
                        elif not (np.all(base_f <= fc) and np.any(base_f < fc)):
                            # Non-dominated - consider replacing if it improves hypervolume
                            # Simple heuristic: replace if sum of objectives is smaller
                            if np.sum(fc) < np.sum(base_f):
                                X[best_idx[bi]] = child
                                F[best_idx[bi]] = fc

        return X
# EVOLVE-BLOCK-END
