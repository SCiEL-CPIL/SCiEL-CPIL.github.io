    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        lower = -5.0
        upper = 5.0
        self.calls_made = 0

        # 1. Global Exploration with 20-point Sobol Sequence
        n_sobol = 20
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_sobol)
        sobol_points = sobol_points * (upper - lower) + lower

        best_x = None
        best_f = np.inf

        for i in range(n_sobol):
            if self.calls_made >= budget:
                break
            x = sobol_points[i]
            fx = self._call_func(func, x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # 2. Local Refinement with Nelder-Mead
        if self.calls_made < budget and best_x is not None:
            best_x, best_f = self._nelder_mead(func, best_x, lower, upper)

        return best_x, best_f
