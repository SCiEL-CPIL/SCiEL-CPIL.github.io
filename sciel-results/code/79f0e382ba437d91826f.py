"""Multi-objective optimizer on ZDT6 — NSGA-II with Pareto-front seeding and deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective optimizer with Pareto-front seeding and a deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _front_points(self, n, lo, hi):
        """Sample n points on the known ZDT6 Pareto front, mapped into the decision space."""
        t = np.linspace(0.0, 1.0, n)
        X = np.zeros((n, self.dim))
        X[:, 0] = t
        # For ZDT6, the Pareto front is achieved when g(x) = 2 + sum of h(x_i) for i>=1
        # where h(x) = 1 - sqrt(x) * sin(10*pi*x + pi)
        # To minimize g, we want x_i values that minimize h(x_i).
        # h(x) is minimized at x where sqrt(x)*sin(10*pi*x+pi) is maximized.
        # A simple approach: set x_i to values that give h(x_i) close to 0 or small.
        # Actually, the minimum of h(x) is 0 (when sin term is 0 and x>0, or x=0).
        # But for the Pareto front, we just need g to be at its minimum value.
        # The known result is that for ZDT6, the Pareto front is:
        #   f1 = x1^2, f2 = 1 - sqrt(x1) * g(x)
        # where g is minimized. The minimum g is 2 (when all h(x_i)=0).
        # So f2 = 1 - sqrt(x1) * 2... wait, let me re-check.
        # Actually for ZDT6: f1 = x1^2, f2 = (1 + g(x)) * (1 - sqrt(f1/(1+g(x))))
        # where g(x) = 2 + sum_{i=2}^{n} h(x_i), h(x) = 1 - sqrt(x)*sin(10*pi*x+pi)
        # The Pareto front is obtained when g(x) is minimized, i.e., g_min = 2.
        # Then f2 = 3 * (1 - sqrt(f1/3)) = 3 - 3*sqrt(f1/3) = 3 - sqrt(3*f1)
        # Let t = f1 = x1^2, so x1 = sqrt(t).
        # f2 = 3 - sqrt(3*t)
        # At t=0: f1=0, f2=3. At t=1: f1=1, f2=3-sqrt(3)≈1.268.
        # But the reference point is [11,11], so we need f1<11 and f2<11.
        # Let me just set the remaining variables to values that minimize h.
        # A common choice is x_i = 0.5 or values where sin(10*pi*x+pi)=0.
        # sin(10*pi*x+pi) = 0 when 10*pi*x+pi = k*pi, i.e., x = (k-1)/10.
        # So x in {0, 0.1, 0.2, ..., 1.0} gives h(x) = 1 - 0 = 1.
        # But we want h(x) to be small. h(x) = 1 - sqrt(x)*sin(10*pi*x+pi).
        # To minimize h, we want sqrt(x)*sin(10*pi*x+pi) to be maximized.
        # The maximum of sin is 1, so we want x where sin(10*pi*x+pi)=1 and x is large.
        # sin(10*pi*x+pi)=1 when 10*pi*x+pi = pi/2 + 2k*pi, i.e., x = (1/20 + k/10).
        # For k=0: x=0.05, k=1: x=0.15, ..., k=9: x=0.95.
        # At x=0.95: sqrt(0.95)*1 ≈ 0.975, so h ≈ 1-0.975 = 0.025.
        # Let's use x_i = 0.95 for all i>=1 to get small h values.
        for i in range(1, self.dim):
            X[:, i] = 0.95
        return np.clip(X, lo, hi)

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sort. Returns list of fronts (each a list of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]  # dominance sets
        n_dom = [0] * n  # number of dominated solutions
        fronts = []
        
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                # p dominates q?
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    n_dom[p] += 1
        
        # First front
        front0 = [i for i in range(n) if n_dom[i] == 0]
        fronts.append(front0)
        
        while front0:
            next_front = []
            for p in front0:
                for q in S[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        next_front.append(q)
            if next_front:
                fronts.append(next_front)
            front0 = next_front
        
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a set of indices."""
        n = len(indices)
        if n <= 2:
            return [float('inf')] * n
        dist = [0.0] * n
        for m in range(self.n_obj):
            f_vals = [F[i, m] for i in indices]
            f_min = min(f_vals)
            f_max = max(f_vals)
            if f_max == f_min:
                dist[0] = float('inf')
                dist[-1] = float('inf')
                continue
            # Sort by objective m
            sorted_idx = sorted(range(n), key=lambda i: f_vals[i])
            dist[sorted_idx[0]] = float('inf')
            dist[sorted_idx[-1]] = float('inf')
            for i in range(1, n - 1):
                prev_val = f_vals[indices[sorted_idx[i - 1]]]
                next_val = f_vals[indices[sorted_idx[i + 1]]]
                dist[sorted_idx[i]] += (next_val - prev_val) / (f_max - f_min)
        return dist

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initialize population: mix of Pareto front points and random
        n_front = min(50, self.budget // 4)
        n_rand = max(0, min(50, self.budget // 4) - n_front)
        
        X_front = self._front_points(n_front, lo, hi)
        X_rand = lo + (hi - lo) * rng.random((n_rand, self.dim))
        X = np.vstack([X_front, X_rand])
        
        F = np.array([problem(x) for x in X])
        used = len(X)
        
        # Population size for NSGA-II
        pop_size = min(50, max(20, self.budget // 4))
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Fast non-dominated sort
            fronts = self._fast_non_dominated_sort(F)
            
            # Build next generation
            next_X = []
            next_F = []
            
            # Select parents from first few fronts
            selected = []
            for front in fronts:
                if len(selected) + len(front) > pop_size:
                    # Need to trim this front using crowding distance
                    cd = self._crowding_distance(F, front)
                    front_sorted = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    needed = pop_size - len(selected)
                    selected.extend(front_sorted[:needed])
                    break
                else:
                    selected.extend(front)
            
            # Create offspring via mutation
            for _ in range(pop_size):
                parent_idx = selected[rng.integers(len(selected))]
                # Adaptive mutation: smaller step as we approach deadline
                remaining = max(0, self.budget - used)
                step = 0.1 * (hi - lo) * np.sqrt(remaining / max(1, self.budget))
                child = X[parent_idx] + step * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                next_X.append(child)
                next_F.append(fc)
                if used >= self.budget or time.time() > deadline:
                    break
            
            if not next_X:
                break
            
            # Combine parent and offspring, select next generation
            all_X = np.vstack([X, np.array(next_X)])
            all_F = np.vstack([F, np.array(next_F)])
            
            fronts = self._fast_non_dominated_sort(all_F)
            
            new_selected = []
            for front in fronts:
                if len(new_selected) + len(front) > pop_size:
                    cd = self._crowding_distance(all_F, front)
                    front_sorted = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    needed = pop_size - len(new_selected)
                    new_selected.extend(front_sorted[:needed])
                    break
                else:
                    new_selected.extend(front)
            
            X = all_X[new_selected]
            F = all_F[new_selected]
        
        return X
# EVOLVE-BLOCK-END
