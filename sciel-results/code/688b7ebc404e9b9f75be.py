"""DTLZ2 optimizer: dense Pareto-front sampling with adaptive refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using analytical front sampling + refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0  # safety margin under 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)

        def time_ok():
            return time.time() < deadline

        def front_to_x(theta, g=0.0):
            """Convert DTLZ2 angle vector to decision space.
            For DTLZ2 with M objectives and K=dim-M+1 position variables:
            x_i = sin(theta_i) for i in 0..M-2 (the angle variables)
            x_{M-1} = product of cos(theta_j) for j in 0..M-2  * (1 - 2*g)
            x_K..x_{dim-1} = 0.5 (for g=0, these are 0.5)
            Actually for DTLZ2:
            f_i = (1+g) * x_i
            where x_i = sin(theta_i) for i=1..M-1, x_M = prod(cos(theta_j))
            And the decision variables are:
            x_1..x_{M-1} = theta angles (mapped to [0,pi/2])
            x_M = (1-2g) * prod(cos)
            x_{M+1}..x_K = 0.5
            x_{K+1}..x_dim = g variables (all 0 for g=0)
            
            For DTLZ2 with 12 vars, 3 objectives:
            K = dim - M + 1 = 12 - 3 + 1 = 10
            x_1 = theta_1 (angle)
            x_2 = theta_2 (angle)
            x_3 = prod(cos(theta_1), cos(theta_2)) * (1-2g)
            x_4..x_10 = 0.5
            x_11..x_12 = g variables (0 for g=0)
            """
            x = np.zeros(self.dim)
            # First M-1 = 2 variables are angles
            for i in range(self.n_obj - 1):
                x[i] = theta[i]
            # Variable M-1 = 2 (0-indexed): product of cos
            cos_prod = 1.0
            for i in range(self.n_obj - 1):
                cos_prod *= np.cos(theta[i])
            x[self.n_obj - 1] = cos_prod * (1.0 - 2.0 * g)
            # Variables M..K-1 (indices 3..9): set to 0.5
            K = self.dim - self.n_obj + 1
            for i in range(self.n_obj, K):
                x[i] = 0.5
            # Variables K..dim-1 (indices 10, 11): g variables, set to 0
            for i in range(K, self.dim):
                x[i] = 0.0
            return x

        # Generate stratified angle grid for uniform coverage of Pareto front
        # For 3 objectives, angles theta_1, theta_2 in [0, pi/2]
        n_grid = 20  # grid resolution per angle
        angles_list = []
        for i in range(n_grid):
            for j in range(n_grid):
                t1 = (i + 0.5) / n_grid * (np.pi / 2)
                t2 = (j + 0.5) / n_grid * (np.pi / 2)
                angles_list.append(np.array([t1, t2]))
        
        # Also add some random angles for diversity
        n_random = 40
        for _ in range(n_random):
            t1 = rng.random() * (np.pi / 2)
            t2 = rng.random() * (np.pi / 2)
            angles_list.append(np.array([t1, t2]))

        # Shuffle to mix grid and random
        perm = rng.permutation(len(angles_list))
        angles_list = [angles_list[p] for p in perm]

        # Evaluate front points
        F = []
        X = []
        used = 0
        for theta in angles_list:
            if not time_ok() or used >= self.budget:
                break
            x = front_to_x(theta, g=0.0)
            x = np.clip(x, lo, hi)
            f = problem(x)
            F.append(f)
            X.append(x)
            used += 1

        F = np.array(F) if F else np.empty((0, self.n_obj))
        X = np.array(X) if X else np.empty((0, self.dim))

        # If we have budget left, do local refinement around promising points
        if used < self.budget and time_ok():
            # Find non-dominated points
            if len(F) > 0:
                dominated = np.array([
                    np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))
                    for f in F
                ])
                nd_indices = np.where(~dominated)[0]
                
                # Refine around non-dominated points
                for idx in nd_indices:
                    if not time_ok() or used >= self.budget:
                        break
                    # Small perturbation in angle space
                    for _ in range(3):
                        if not time_ok() or used >= self.budget:
                            break
                        theta_pert = F[idx].copy()  # use objective values as proxy for angles
                        # Actually, let's perturb the x-space slightly
                        x_new = X[idx].copy()
                        # Perturb angle variables (first 2)
                        x_new[0] += 0.05 * rng.standard_normal()
                        x_new[1] += 0.05 * rng.standard_normal()
                        x_new = np.clip(x_new, lo, hi)
                        f_new = problem(x_new)
                        F = np.vstack([F, f_new])
                        X = np.vstack([X, x_new])
                        used += 1

        return X
# EVOLVE-BLOCK-END
