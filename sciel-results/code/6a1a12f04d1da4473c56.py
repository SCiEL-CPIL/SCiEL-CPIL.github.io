    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Phase 1: Sobol-structured Latin Hypercube Sampling
        # 1. Generate Sobol points for low-discrepancy structure
        # 2. Use the ranks of Sobol coordinates to define LHS bin assignments
        # This ensures strict LHS properties (one point per bin) while
        # inheriting the superior space-filling properties of Sobol sequences.
        
        from scipy.stats import qmc

        n_lhs = budget
        # Generate Sobol sequence
        # Sobol requires power of 2 for optimal performance, but works for any N.
        # We will generate N points.
        sobol = qmc.Sobol(d=dim, scramble=False, seed=42)
        # Note: scramble=False makes it deterministic. 
        # We use the first n_lhs points.
        # For better distribution, sometimes people use the first 2^k points and subsample, 
        # but for N=200, direct generation is fine.
        sobol_points = sobol.random(n_lhs)
        
        # Convert Sobol points to LHS coordinates
        # For each dimension d, rank the Sobol coordinates.
        # The rank determines which bin the point falls into.
        lhs = np.zeros((n_lhs, dim))
        for d in range(dim):
            # Get ranks of the d-th dimension of Sobol points
            # np.argsort(np.argsort(x)) gives ranks 0 to N-1
            ranks = np.argsort(np.argsort(sobol_points[:, d]))
            # Map ranks to bin centers: (rank + 0.5) / N
            lhs[:, d] = (ranks + 0.5) / n_lhs

        # Map from [0,1]^dim to [lo, hi]^dim
        X_lhs = lo + lhs * span
        X_lhs = np.clip(X_lhs, lo, hi)

        # Evaluate all LHS points
        F_lhs = np.array([problem(x) for x in X_lhs])

        # Filter out non-finite or out-of-bounds results
        valid_mask = np.all(np.isfinite(F_lhs), axis=1)
        if not np.all(valid_mask):
            F_lhs = F_lhs[valid_mask]
            X_lhs = X_lhs[valid_mask]

        return X_lhs
