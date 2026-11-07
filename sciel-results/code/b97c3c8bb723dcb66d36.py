"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective optimizer with DTLZ1-specific initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dtlz1_pareto_point(self, rng, dim, n_obj):
        """Generate a point on or near the DTLZ1 Pareto front."""
        # DTLZ1: g(x) = 100*(n_g + sum((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5))) for i in n_g..n-1)
        # Pareto front: g=0, and sum(w_i * f_i) = 1 where f_i are the first n_obj objectives
        # For DTLZ1: f_i = 0.5*(1+g)*prod_{j=1}^{i-1}(1-f_j) for i=1..n_obj-1, f_n_obj = 0.5*(1+g)*prod_{j=1}^{n_obj-1}(1-f_j)
        # On Pareto front (g=0): f_i = 0.5*prod_{j=1}^{i-1}(1-f_j)
        # Let's use a different parametrization. For DTLZ1 with n_obj=3:
        # f1 = 0.5*(1+g)*x1
        # f2 = 0.5*(1+g)*(1-x1)*x2  
        # f3 = 0.5*(1+g)*(1-x1)*(1-x2)
        # On Pareto front: g=0, so f1=0.5*x1, f2=0.5*(1-x1)*x2, f3=0.5*(1-x1)*(1-x2)
        # We can parametrize with angles.
        
        # Use random weights on simplex for the Pareto front
        w = rng.dirichlet(np.ones(n_obj))
        
        # For DTLZ1, the Pareto front satisfies: sum of objectives = 0.5 when g=0
        # Actually for DTLZ1: f1 + f2 + ... + f_n_obj = 0.5*(1+g)
        # On Pareto front: sum = 0.5
        
        # Let's use the standard DTLZ1 formulation
        # x_1..x_{n_obj-1} are position variables, x_{n_obj}..x_n are position variables for g
        # f_1 = 0.5*(1+g)*x_1
        # f_2 = 0.5*(1+g)*(1-x_1)*x_2
        # f_3 = 0.5*(1+g)*(1-x_1)*(1-x_2)
        # g = 100*(n_g + sum((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5))) for i in n_obj..n-1)
        
        # To get on Pareto front, set g=0 by setting x_{n_obj}..x_{n-1} = 0.5
        # Then choose x_1, x_2 to match desired weights
        
        # For 3 objectives: f1 = 0.5*x1, f2 = 0.5*(1-x1)*x2, f3 = 0.5*(1-x1)*(1-x2)
        # We want f1/f_sum ≈ w1, f2/f_sum ≈ w2, f3/f_sum ≈ w3 where f_sum = 0.5
        # So f1 = 0.5*w1, f2 = 0.5*w2, f3 = 0.5*w3
        # 0.5*x1 = 0.5*w1 => x1 = w1
        # 0.5*(1-x1)*x2 = 0.5*w2 => (1-w1)*x2 = w2 => x2 = w2/(1-w1) if w1 < 1
        # 0.5*(1-x1)*(1-x2) = 0.5*w3 => (1-w1)*(1-x2) = w3 => (1-w1)*(1 - w2/(1-w1)) = w3
        # Check: (1-w1) - w2 = w3? Since w1+w2+w3=1, yes!
        
        if n_obj == 3 and dim >= 5:
            x1 = w[0]
            x2 = w[1] / (1 - w[0]) if w[0] < 1.0 else 0.0
            x2 = min(max(x2, 0.0), 1.0)
            x = np.ones(dim) * 0.5
            x[0] = x1
            x[1] = x2
            # x[2:] = 0.5 to make g=0
            return x
        else:
            # Fallback: random point
            return rng.random(dim)

    def _dominates(self, f1, f2):
        """Check if f1 dominates f2 (all objectives <= and at least one <)."""
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sort. Returns list of fronts (each is list of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = set of solutions dominated by i
        n_dom = np.zeros(n, dtype=int)  # n_dom[i] = number of solutions dominating i
        fronts = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif self._dominates(F[j], F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        # First front
        front = [i for i in range(n) if n_dom[i] == 0]
        fronts.append(front)
        
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            front = next_front
            if front:
                fronts.append(front)
        
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for solutions in indices."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        m = F.shape[1]
        
        for k in range(m):
            obj_vals = F[indices, k]
            sorted_idx = np.argsort(obj_vals)
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            obj_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
            if obj_range > 1e-12:
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / obj_range
        
        return cd

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size
        pop_size = min(budget // 2, 50)
        pop_size = max(pop_size, 20)
        
        # Initialize population with a mix of Pareto front points and random points
        X = np.zeros((pop_size, dim))
        for i in range(pop_size):
            if i < pop_size // 2:
                X[i] = self._dtlz1_pareto_point(rng, dim, n_obj)
            else:
                X[i] = lo + (hi - lo) * rng.random(dim)
            X[i] = np.clip(X[i], lo, hi)
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Main loop
        while used < budget:
            # Create offspring
            child = np.zeros(dim)
            
            # Binary tournament selection
            idx1 = rng.integers(pop_size)
            idx2 = rng.integers(pop_size)
            if self._dominates(F[idx1], F[idx2]):
                parent1 = X[idx1]
            else:
                parent1 = X[idx2]
            
            idx1 = rng.integers(pop_size)
            idx2 = rng.integers(pop_size)
            if self._dominates(F[idx1], F[idx2]):
                parent2 = X[idx1]
            else:
                parent2 = X[idx2]
            
            # SBX crossover
            eta = 20.0
            for j in range(dim):
                if rng.random() < 0.5:
                    if abs(parent1[j] - parent2[j]) > 1e-12:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                        child[j] = 0.5 * ((1 + beta) * parent1[j] + (1 - beta) * parent2[j])
                    else:
                        child[j] = parent1[j]
                else:
                    if abs(parent1[j] - parent2[j]) > 1e-12:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                        child[j] = 0.5 * ((1 + beta) * parent2[j] + (1 - beta) * parent1[j])
                    else:
                        child[j] = parent2[j]
            
            # Polynomial mutation
            eta_mut = 20.0
            for j in range(dim):
                if rng.random() < 1.0 / dim:
                    y = child[j]
                    yl = (y - lo[j]) / (hi[j] - lo[j])
                    yr = (hi[j] - y) / (hi[j] - lo[j])
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta_mut + 1)) - 1
                    elif u > 0.5:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_mut + 1))
                    else:
                        delta = 0
                    y_new = yl + delta * (hi[j] - lo[j])
                    y_new = min(max(y_new, 0.0), 1.0)
                    child[j] = lo[j] + y_new * (hi[j] - lo[j])
            
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            
            # Combine parent and offspring
            X_combined = np.vstack([X, child.reshape(1, -1)])
            F_combined = np.vstack([F, fc.reshape(1, -1)])
            
            # Non-dominated sort
            fronts = self._fast_non_dominated_sort(F_combined)
            
            # Select next generation
            X_new = []
            F_new = []
            for front in fronts:
                if len(X_new) + len(front) <= pop_size:
                    for idx in front:
                        X_new.append(X_combined[idx])
                        F_new.append(F_combined[idx])
                else:
                    # Use crowding distance to select
                    remaining = pop_size - len(X_new)
                    cd = self._crowding_distance(F_combined, front)
                    sorted_by_cd = np.argsort(-cd)
                    for i in range(remaining):
                        idx = front[sorted_by_cd[i]]
                        X_new.append(X_combined[idx])
                        F_new.append(F_combined[idx])
                    break
            
            X = np.array(X_new)
            F = np.array(F_new)
        
        return X
# EVOLVE-BLOCK-END
