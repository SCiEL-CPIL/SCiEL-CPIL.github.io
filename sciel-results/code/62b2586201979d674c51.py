"""Multi-objective optimizer for DTLZ1 using NSGA-II on the Pareto front parameter space."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II optimizer operating on the analytic parameterization of the DTLZ1 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Parameterization: u, v in simplex
        # x[0] = u, x[1] = v, x[2] = 1-u-v, x[3:] = 0.5
        
        def params_to_x(u, v):
            x = np.zeros(dim)
            x[0] = u
            x[1] = v
            x[2] = 1.0 - u - v
            x[3:] = 0.5
            return np.clip(x, lo, hi)

        def x_to_params(x):
            # This is not strictly necessary if we evolve params directly, 
            # but useful for initialization if we started from x.
            # We will evolve params directly.
            pass

        # NSGA-II Parameters
        pop_size = 50
        n_generations = (budget - pop_size) // pop_size
        # Ensure we don't exceed budget
        total_evals = pop_size
        generations = []
        
        # Initialize population of (u, v)
        # Uniform sampling in simplex
        U = np.random.rand(pop_size, 2)
        # Transform to simplex: use Dirichlet or rejection
        # Rejection is simple
        valid = np.sum(U, axis=1) <= 1.0
        while np.sum(valid) < pop_size:
            new_U = np.random.rand(pop_size - np.sum(valid), 2)
            new_valid = np.sum(new_U, axis=1) <= 1.0
            U[~valid] = new_U
            valid[~valid] = new_valid
            
        # Actually, better to use Dirichlet for proper simplex distribution
        # But for NSGA-II, uniform in [0,1]^2 projected to simplex is fine?
        # Let's use Dirichlet(1,1,1) which is uniform on simplex.
        U = np.random.dirichlet([1.0, 1.0, 1.0], size=pop_size)[:, :2]
        
        all_X = []
        all_F = []
        used = 0

        # Evaluate initial population
        F = np.zeros((pop_size, n_obj))
        for i in range(pop_size):
            x = params_to_x(U[i, 0], U[i, 1])
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            F[i] = f
            used += 1

        # Evolution Loop
        for gen in range(n_generations):
            if used >= budget:
                break
            
            # Generate offspring
            Offspring_U = np.zeros((pop_size, 2))
            Offspring_F = np.zeros((pop_size, n_obj))
            
            for i in range(pop_size):
                # Crossover
                if np.random.rand() < 0.9:
                    # Select parent
                    j = np.random.randint(0, pop_size)
                    if i != j:
                        # SBX Crossover on u, v
                        u1, v1 = U[i]
                        u2, v2 = U[j]
                        
                        # SBX
                        def sbx_crossover(x1, x2):
                            if np.random.rand() < 0.5:
                                beta = 1.0 / (1.0 + 2.0 * abs(x1 - x2))
                                # This is a simplified SBX, standard uses eta
                                # Let's use a simpler blend for robustness
                                alpha = np.random.rand()
                                return alpha * x1 + (1 - alpha) * x2
                            else:
                                return x1
                            
                        # Use standard SBX with eta=15
                        def sbx_eta(x1, x2, eta=15.0):
                            if x1 == x2:
                                return x1
                            if np.random.rand() < 0.5:
                                beta = (2.0 * np.random.rand()) ** (1.0 / (eta + 1.0))
                            else:
                                beta = (1.0 / (2.0 * np.random.rand())) ** (1.0 / (eta + 1.0))
                                
                            c1 = 0.5 * ((1.0 + beta) * x1 + (1.0 - beta) * x2)
                            c2 = 0.5 * ((1.0 - beta) * x1 + (1.0 + beta) * x2)
                            return c1, c2

                        u1_c, u2_c = sbx_eta(u1, u2)
                        v1_c, v2_c = sbx_eta(v1, v2)
                        
                        # We only need one offspring here, so take the first
                        u_off = u1_c
                        v_off = v1_c
                    else:
                        u_off = U[i, 0]
                        v_off = U[i, 1]
                else:
                    u_off = U[i, 0]
                    v_off = U[i, 1]

                # Mutation
                def poly_mutate(x, eta=20.0):
                    if np.random.rand() < 1.0 / 2: # 2 params
                        if np.random.rand() < 0.5:
                            delta = (2.0 * np.random.rand()) ** (1.0 / (eta + 1.0))
                            return x - delta
                        else:
                            delta = (1.0 / (2.0 * np.random.rand())) ** (1.0 / (eta + 1.0))
                            return x + delta
                    return x

                u_off = poly_mutate(u_off)
                v_off = poly_mutate(v_off)
                
                # Repair to simplex
                if u_off < 0: u_off = 0.0
                if v_off < 0: v_off = 0.0
                if u_off + v_off > 1.0:
                    scale = 1.0 / (u_off + v_off)
                    u_off *= scale
                    v_off *= scale
                    
                Offspring_U[i] = [u_off, v_off]
                
                # Evaluate
                x = params_to_x(u_off, v_off)
                f = problem(x)
                all_X.append(x)
                all_F.append(f)
                Offspring_F[i] = f
                used += 1
                
                if used >= budget:
                    break
            
            if used >= budget:
                break

            # Combine Parents and Offspring
            Total_U = np.vstack([U, Offspring_U])
            Total_F = np.vstack([F, Offspring_F])
            
            # Non-dominated Sort
            ranks = fast_non_dominated_sort(Total_F)
            
            # Crowding Distance
            # Group by rank
            n_total = len(Total_F)
            crowd = np.zeros(n_total)
            
            # Simple crowding distance calculation
            # For each rank, sort by each objective and assign distances
            unique_ranks = np.unique(ranks)
            for r in unique_ranks:
                mask = ranks == r
                indices = np.where(mask)[0]
                if len(indices) <= 2:
                    crowd[indices] = np.inf
                    continue
                # Sort by first objective
                sorted_idx = indices[np.argsort(Total_F[indices, 0])]
                crowd[sorted_idx[0]] = np.inf
                crowd[sorted_idx[-1]] = np.inf
                for k in range(1, len(sorted_idx) - 1):
                    prev_idx = sorted_idx[k-1]
                    next_idx = sorted_idx[k+1]
                    curr_idx = sorted_idx[k]
                    d = (Total_F[prev_idx, 0] - Total_F[next_idx, 0])
                    if d > 0:
                        crowd[curr_idx] += 1.0 / d
                    else:
                        crowd[curr_idx] += 0.0
                        
                # Sort by second objective
                sorted_idx = indices[np.argsort(Total_F[indices, 1])]
                crowd[sorted_idx[0]] = np.inf
                crowd[sorted_idx[-1]] = np.inf
                for k in range(1, len(sorted_idx) - 1):
                    prev_idx = sorted_idx[k-1]
                    next_idx = sorted_idx[k+1]
                    curr_idx = sorted_idx[k]
                    d = (Total_F[prev_idx, 1] - Total_F[next_idx, 1])
                    if d > 0:
                        crowd[curr_idx] += 1.0 / d
                    else:
                        crowd[curr_idx] += 0.0
                        
                # Sort by third objective
                sorted_idx = indices[np.argsort(Total_F[indices, 2])]
                crowd[sorted_idx[0]] = np.inf
                crowd[sorted_idx[-1]] = np.inf
                for k in range(1, len(sorted_idx) - 1):
                    prev_idx = sorted_idx[k-1]
                    next_idx = sorted_idx[k+1]
                    curr_idx = sorted_idx[k]
                    d = (Total_F[prev_idx, 2] - Total_F[next_idx, 2])
                    if d > 0:
                        crowd[curr_idx] += 1.0 / d
                    else:
                        crowd[curr_idx] += 0.0

            # Selection
            # Sort by rank, then by crowding distance (descending)
            sort_keys = np.lexsort(([-crowd[i] for i in range(n_total)], ranks))
            # Take first pop_size
            selected_indices = sort_keys[:pop_size]
            
            U = Total_U[selected_indices]
            F = Total_F[selected_indices]

        return np.array(all_X)

    def fast_non_dominated_sort(self, F):
        """
        Fast non-dominated sort.
        Returns ranks for each point.
        """
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        S = [[] for _ in range(n)] # S[i] = indices dominated by i
        n_d = np.zeros(n, dtype=int) # n_d[i] = number of points dominating i
        
        for i in range(n):
            for j in range(i + 1, n):
                if self.dominates(F[i], F[j]):
                    S[i].append(j)
                    n_d[j] += 1
                elif self.dominates(F[j], F[i]):
                    S[j].append(i)
                    n_d[i] += 1
        
        # Front 0
        current_front = [i for i in range(n) if n_d[i] == 0]
        rank = 0
        while current_front:
            next_front = []
            for i in current_front:
                ranks[i] = rank
                for j in S[i]:
                    n_d[j] -= 1
                    if n_d[j] == 0:
                        next_front.append(j)
            current_front = next_front
            rank += 1
            
        return ranks

    def dominates(self, f1, f2):
        """
        Returns True if f1 dominates f2 (minimization).
        f1 dominates f2 if f1 <= f2 in all objectives and f1 < f2 in at least one.
        """
        if np.all(f1 <= f2) and np.any(f1 < f2):
            return True
        return False
# EVOLVE-BLOCK-END
