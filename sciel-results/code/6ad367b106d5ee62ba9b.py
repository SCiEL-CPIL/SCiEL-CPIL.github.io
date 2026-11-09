"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size for Bent Cigar optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 330s limit

        # Initial population
        rng = np.random.default_rng(42)
        
        # Start near center for Bent Cigar
        x0 = np.zeros(dim)
        x0[0] = 0.0  # Bent Cigar minimum is often at x[0]=0
        
        # CMA-ES parameters
        lam = 10  # population size
        mu = lam // 2  # number of parents
        
        # Initial step size - conservative for Bent Cigar
        sigma = 2.0
        
        # Evolution path
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Step sizes for evolution paths
        cs = (15 + 3 * np.log(lam - 2)) / (4 * dim + 4)
        c1 = (2 + 2 / (dim + 1.3)) / ((dim + 2) * (dim + 2) + 2)
        cmu = min(1 - c1, 2 * ((mu + 1) / (dim + 2) - 1)) / ((dim + 2) * (dim + 2) + 2)
        
        # Initial covariance matrix (identity)
        C = np.eye(dim)
        Cinv = np.eye(dim)
        Csqrt = np.eye(dim)
        
        # Eigen decomposition cache
        D = np.ones(dim)
        B = np.eye(dim)
        
        # Initial mean
        m = x0.copy()
        
        # Evaluate initial point
        fvals = []
        x = np.clip(m, -5, 5)
        f = func(x)
        best_x = x.copy()
        best_f = f
        
        # Generate initial population
        for i in range(lam):
            zi = rng.standard_normal(dim)
            xi = np.clip(m + sigma * (B @ (D * zi)), -5, 5)
            fi = func(xi)
            fvals.append((fi, xi))
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()
        
        # Sort by fitness
        fvals.sort(key=lambda t: t[0])
        
        # Weights
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights ** 2)
        
        # Main loop
        call_count = lam + 1  # 1 for initial + lam for first population
        
        for gen in range(budget - lam):
            if time.time() > deadline:
                break
            
            if call_count >= budget:
                break
            
            # Select parents
            parents = [fvals[i][1] for i in range(mu)]
            
            # Compute weighted mean
            m_new = np.sum([weights[i] * parents[i] for i in range(mu)], axis=0)
            
            # Update evolution paths
            diff = m_new - m
            if sigma > 1e-12:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Cinv @ diff) / sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) <= \
                       (1.4 + 2 / (dim + 1)) * np.sqrt((mueff + 1) / (dim + 1))
                pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1) * mueff) * diff / sigma
            
            # Update covariance matrix
            deltaD = 2 * (1 - c1 - cmu)
            C = (1 + deltaD) * C
            for i in range(mu):
                v = (parents[i] - m) / sigma
                C += cmu * weights[i] * np.outer(v, v)
            if hsig:
                C += (c1 * (2 - c1 - c1) * (1 - c1) - cmu * (2 - c1 - cmu) * (1 - c1)) * np.outer(pc, pc)
            
            # Eigen decomposition
            B, D2 = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D2, 1e-12))
            B = B  # eigenvectors
            Cinv = (B @ np.diag(1 / (D ** 2)) @ B.T)
            
            # Adapt step size
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-12, 50.0)
            
            # Update mean
            m = m_new
            
            # Generate new population
            fvals_new = []
            for i in range(lam):
                zi = rng.standard_normal(dim)
                xi = np.clip(m + sigma * (B @ (D * zi)), -5, 5)
                fi = func(xi)
                fvals_new.append((fi, xi))
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
                call_count += 1
            
            fvals = fvals_new
            fvals.sort(key=lambda t: t[0])
        
        return best_x, best_f
# EVOLVE-BLOCK-END
