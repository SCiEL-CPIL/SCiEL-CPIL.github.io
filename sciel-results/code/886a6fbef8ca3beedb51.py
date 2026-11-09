import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        N = 20
        if N > self.budget:
            N = self.budget
            
        # Generate weight vectors for MOEA/D
        # For 2 objectives, weights are (lambda, 1-lambda)
        lambdas = np.linspace(0, 1, N, endpoint=True).reshape(-1, 1)
        lambdas = np.hstack([lambdas, 1 - lambdas])
        
        # Initialize population
        X = np.random.uniform(lo, hi, (N, self.dim))
        F = np.array([problem(x) for x in X])
        used_evals = N
        
        # Initialize ideal point z*
        z_star = np.min(F, axis=0)
        
        # Determine neighborhood for each subproblem
        T = 5
        neighbors = []
        for i in range(N):
            dists = np.linalg.norm(lambdas - lambdas[i], axis=1)
            idx = np.argsort(dists)[:T]
            neighbors.append(idx)
            
        # MOEA/D Main Loop
        while used_evals < self.budget:
            for i in range(N):
                if used_evals >= self.budget:
                    break
                
                # Select parents from neighborhood of i
                neigh = neighbors[i]
                if len(neigh) < 2:
                    p1_idx = neigh[0]
                    p2_idx = neigh[0]
                else:
                    p1_idx = neigh[np.random.randint(len(neigh))]
                    p2_idx = neigh[np.random.randint(len(neigh))]
                    if p1_idx == p2_idx and len(neigh) > 1:
                         p2_idx = neigh[np.random.randint(len(neigh))]
                
                x1 = X[p1_idx]
                x2 = X[p2_idx]
                
                # Crossover: SBX
                c1, _ = self._sbx_crossover(x1, x2, lo, hi, 20.0)
                
                # Mutation: Polynomial
                c1 = self._poly_mutation(c1, lo, hi, 1.0/self.dim, 20.0)
                
                # Evaluate
                f1 = problem(c1)
                used_evals += 1
                
                # Update ideal point
                z_star = np.minimum(z_star, f1)
                
                # Update subproblems in neighborhood of i
                for j in neighbors[i]:
                    # Calculate Tchebycheff distance for new solution c1
                    dist_new = np.max(lambdas[j] * np.abs(f1 - z_star))
                    
                    # Calculate Tchebycheff distance for current best of j
                    dist_old = np.max(lambdas[j] * np.abs(F[j] - z_star))
                    
                    if dist_new < dist_old:
                        X[j] = c1
                        F[j] = f1
        
        return X

    def _sbx_crossover(self, x1, x2, lo, hi, eta_c):
        c1 = np.zeros_like(x1)
        c2 = np.zeros_like(x2)
        for i in range(len(x1)):
            if np.random.rand() < 0.5:
                if abs(x1[i] - x2[i]) > 1e-14:
                    if np.random.rand() <= 1.0 / (eta_c + 1):
                        beta = (2.0 * np.random.rand()) ** (1.0 / (eta_c + 1))
                    else:
                        beta = (1.0 / (2.0 * np.random.rand())) ** (1.0 / (eta_c + 1))
                    c1[i] = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
                    c2[i] = 0.5 * ((1.0 - beta) * x1[i] + (1.0 + beta) * x2[i])
                else:
                    c1[i] = x1[i]
                    c2[i] = x2[i]
            else:
                c1[i] = x2[i]
                c2[i] = x1[i]
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _poly_mutation(self, x, lo, hi, pm, eta_m):
        x_mut = x.copy()
        for i in range(len(x)):
            if np.random.rand() < pm:
                if abs(hi[i] - lo[i]) < 1e-14:
                    continue
                delta1 = (x_mut[i] - lo[i]) / (hi[i] - lo[i])
                delta2 = (hi[i] - x_mut[i]) / (hi[i] - lo[i])
                if np.random.rand() < 0.5:
                    xy = np.random.rand()
                    mut_pow = 2.0 / (eta_m + 1.0) * (1.0 + 2.0 * delta1) ** (eta_m + 1.0)
                    if 2.0 * xy <= mut_pow:
                        beta = (2.0 * xy) ** (1.0 / (eta_m + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - xy))) ** (1.0 / (eta_m + 1.0))
                    x_mut[i] = x_mut[i] - (hi[i] - lo[i]) * (beta - 1.0)
                else:
                    xy = np.random.rand()
                    mut_pow = 2.0 / (eta_m + 1.0) * (1.0 + 2.0 * delta2) ** (eta_m + 1.0)
                    if 2.0 * xy <= mut_pow:
                        beta = (2.0 * xy) ** (1.0 / (eta_m + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - xy))) ** (1.0 / (eta_m + 1.0))
                    x_mut[i] = x_mut[i] + (hi[i] - lo[i]) * (beta - 1.0)
        x_mut = np.clip(x_mut, lo, hi)
        return x_mut
