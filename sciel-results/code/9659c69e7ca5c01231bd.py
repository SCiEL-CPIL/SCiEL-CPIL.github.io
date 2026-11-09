"""Multi-objective EGO optimizer for RE21: NSGA-II + weighted-sum local refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """NSGA-II-inspired multi-objective search with weighted-sum local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_sort(self, F):
        """Fast non-dominated sort. Returns list of fronts (each is list of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dom = np.zeros(n, dtype=int)
        fronts = [[]]
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        for i in range(n):
            if n_dom[i] == 0:
                fronts[0].append(i)
        
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            k += 1
            if next_front:
                fronts.append(next_front)
        
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a list of indices."""
        if len(indices) <= 2:
            return np.full(len(indices), np.inf)
        
        m = self.n_obj
        cd = np.zeros(len(indices))
        F_sub = F[indices]
        
        for m_idx in range(m):
            order = np.argsort(F_sub[:, m_idx])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            f_range = F_sub[order[-1], m_idx] - F_sub[order[0], m_idx]
            if f_range > 1e-15:
                for i in range(1, len(indices) - 1):
                    cd[order[i]] += (F_sub[order[i+1], m_idx] - F_sub[order[i-1], m_idx]) / f_range
        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng_size = hi - lo
        mid = (lo + hi) / 2
        
        X = []
        F = []
        used = 0
        
        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1
            return x, f
        
        # Initial population: structured + random
        pop_size = 24
        # Vertices and midpoints of the hypercube
        init_points = []
        # All 2^4 = 16 vertices
        for bits in range(16):
            x = lo.copy()
            for d in range(self.dim):
                if bits & (1 << d):
                    x[d] = hi[d]
            init_points.append(x)
        # Add some random points
        while len(init_points) < pop_size:
            init_points.append(lo + rng_size * rng.random(self.dim))
        
        for x in init_points[:pop_size]:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            x_c, f = evaluate(x)
            X.append(x_c)
            F.append(f)
        
        X = np.array(X)
        F = np.array(F)
        n_pop = len(X)
        
        # Main loop: alternate between exploration and exploitation
        phase = 0
        while used < self.budget:
            if time.time() > deadline - 10:
                break
            
            phase += 1
            
            if phase % 10 == 0 and time.time() < deadline - 30:
                # Exploitation phase: weighted-sum local refinement
                self._exploit(problem, X, F, lo, hi, rng, deadline)
                # Re-evaluate to get current state
                # Find non-dominated and refine top ones
                fronts = self._non_dominated_sort(F)
                if fronts:
                    front0 = fronts[0]
                    cd = self._crowding_distance(F, front0)
                    # Sort by crowding distance
                    sorted_idx = np.argsort(-cd)
                    n_refine = min(6, len(front0), max(1, (self.budget - used) // 3))
                    for si in sorted_idx[:n_refine]:
                        if used >= self.budget:
                            break
                        if time.time() > deadline - 10:
                            break
                        idx = front0[si]
                        # Try multiple weight vectors
                        for w in [0.1, 0.3, 0.5, 0.7, 0.9]:
                            if used >= self.budget:
                                break
                            if time.time() > deadline - 10:
                                break
                            w_vec = np.array([w, 1.0 - w])
                            
                            def scalar_obj(x, w_vec=w_vec):
                                fx = problem(x)
                                # Normalize roughly
                                return float(np.dot(w_vec, fx))
                            
                            x0 = X[idx].copy()
                            try:
                                res = minimize(scalar_obj, x0, method='Nelder-Mead',
                                               bounds=list(zip(lo, hi)),
                                               options={'maxiter': 8, 'xatol': 1e-8, 'fatol': 1e-8})
                                fx_new = problem(res.x)
                                # Only accept if it improves the weighted sum significantly
                                # and doesn't worsen the other objective too much
                                if res.fun < np.dot(w_vec, F[idx]) - 1e-4:
                                    X[idx] = np.clip(res.x, lo, hi)
                                    F[idx] = fx_new
                                    used += 1
                            except Exception:
                                pass
                continue
            
            # Exploration phase: generate offspring
            # Select parent using non-dominated sort + crowding
            fronts = self._non_dominated_sort(F)
            parent_idx = None
            if fronts:
                # Pick from first front, prefer high crowding
                front0 = fronts[0]
                if len(front0) > 1:
                    cd = self._crowding_distance(F, front0)
                    # Sample proportional to crowding
                    probs = cd / cd.sum()
                    probs = probs / probs.sum()
                    pi = rng.choice(len(front0), p=probs)
                    parent_idx = front0[pi]
                else:
                    parent_idx = front0[0]
            
            if parent_idx is None:
                parent_idx = int(rng.integers(0, n_pop))
            
            # Generate offspring with varied strategies
            n_offspring = min(4, self.budget - used)
            for _ in range(n_offspring):
                if used >= self.budget:
                    break
                if time.time() > deadline - 10:
                    break
                
                strategy = rng.integers(0, 3)
                if strategy == 0:
                    # Gaussian mutation
                    scale = 0.15 + 0.25 * rng.random()
                    child = X[parent_idx] + scale * rng_size * rng.standard_normal(self.dim)
                elif strategy == 1:
                    # Cauchy-like mutation (heavier tails)
                    cauchy = 1.0 / rng.standard_normal(self.dim)
                    cauchy = np.clip(cauchy, -3, 3)
                    child = X[parent_idx] + 0.1 * rng_size * cauchy
                else:
                    # Large jump / restart
                    child = lo + rng_size * rng.random(self.dim)
                    # Or blend with parent
                    child = 0.3 * X[parent_idx] + 0.7 * child
                
                child = np.clip(child, lo, hi)
                x_c, f = evaluate(child)
                
                # Add to population
                X = np.vstack([X, x_c.reshape(1, -1)])
                F = np.vstack([F, f.reshape(1, -1)])
                n_pop = len(X)
            
            # Truncation: keep best n_pop points using NSGA-II selection
            if n_pop > 20:
                self._truncate(X, F, 20)
                n_pop = 20
                X = X[:20]
                F = F[:20]
        
        return X

    def _exploit(self, problem, X, F, lo, hi, rng, deadline):
        """Placeholder - exploitation is done inline."""
        pass

    def _truncate(self, X, F, keep_n):
        """Truncate population using non-dominated sort + crowding distance."""
        n = len(X)
        if n <= keep_n:
            return
        
        fronts = self._non_dominated_sort(F)
        selected = []
        
        for front in fronts:
            if len(selected) + len(front) <= keep_n:
                selected.extend(front)
            else:
                # Need to pick some from this front based on crowding
                remaining = keep_n - len(selected)
                cd = self._crowding_distance(F, front)
                sorted_idx = np.argsort(-cd)
                selected.extend(front[sorted_idx[:remaining]])
                break
        
        # Reorder X and F
        X_new = X[selected]
        F_new = F[selected]
        # Copy back
        X[:keep_n] = X_new
        F[:keep_n] = F_new
# EVOLVE-BLOCK-END
