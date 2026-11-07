    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        self.calls = 0
        self.best_x = np.zeros(dim)
        self.best_f = np.inf
        self.deadline = time.time() + 285
        
        def eval_func(x):
            if self.calls >= budget or time.time() > self.deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            self.calls += 1
            f = float(func(x))
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        # Latin Hypercube Sampling for initial points
        def lhs(dim, n):
            points = np.zeros((n, dim))
            for i in range(n):
                points[i] = (np.random.rand(dim) + i) / n
            # Permute rows to decorrelate dimensions
            points = np.random.permutation(points, axis=0)
            points = points * 10 - 5
            return points

        np.random.seed(42)
        
        # 1. Initial Exploration
        n_starts = 5
        initial_points = lhs(dim, n_starts)
        
        initial_results = []
        for x in initial_points:
            if self.calls >= budget or time.time() > self.deadline:
                break
            f = eval_func(x)
            initial_results.append((f, x))
        
        if not initial_results:
            return self.best_x, self.best_f
            
        initial_results.sort(key=lambda t: t[0])
        
        # 2. Multi-Start Nelder-Mead
        def nelder_mead(x0, max_calls):
            # Check budget
            if self.calls + dim + 1 > budget or time.time() > self.deadline:
                return
            
            # Initialize simplex
            simplex = [x0.copy()]
            f0 = float(func(x0))
            self.calls += 1
            if f0 < self.best_f:
                self.best_f = f0
                self.best_x = x0.copy()
            f_values = [f0]
            
            for i in range(dim):
                if self.calls >= budget or time.time() > self.deadline:
                    return
                xi = x0.copy()
                step = 0.1
                if abs(x0[i]) < 1e-8:
                    xi[i] = step
                else:
                    xi[i] = x0[i] * 1.1
                simplex.append(xi)
                fi = float(func(xi))
                self.calls += 1
                if fi < self.best_f:
                    self.best_f = fi
                    self.best_x = xi.copy()
                f_values.append(fi)
                
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5
            
            local_calls = 1 + dim
            while local_calls < max_calls and self.calls < budget and time.time() < self.deadline:
                # Sort
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]
                
                x_worst = simplex[-1]
                f_worst = f_values[-1]
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_reflect = x_centroid + alpha * (x_centroid - x_worst)
                x_reflect = np.clip(x_reflect, -5, 5)
                f_reflect = float(func(x_reflect))
                self.calls += 1
                local_calls += 1
                if f_reflect < self.best_f:
                    self.best_f = f_reflect
                    self.best_x = x_reflect.copy()
                
                if f_reflect < f_values[0]:
                    # Expansion
                    x_expand = x_centroid + gamma * (x_reflect - x_worst)
                    x_expand = np.clip(x_expand, -5, 5)
                    if local_calls < max_calls and self.calls < budget:
                        f_expand = float(func(x_expand))
                        self.calls += 1
                        local_calls += 1
                        if f_expand < self.best_f:
                            self.best_f = f_expand
                            self.best_x = x_expand.copy()
                        if f_expand < f_reflect:
                            simplex[-1] = x_expand
                            f_values[-1] = f_expand
                        else:
                            simplex[-1] = x_reflect
                            f_values[-1] = f_reflect
                    else:
                        simplex[-1] = x_reflect
                        f_values[-1] = f_reflect
                elif f_reflect < f_values[-2]:
                    simplex[-1] = x_reflect
                    f_values[-1] = f_reflect
                else:
                    # Contraction
                    x_contract = x_centroid + rho * (x_worst - x_centroid)
                    x_contract = np.clip(x_contract, -5, 5)
                    if local_calls < max_calls and self.calls < budget:
                        f_contract = float(func(x_contract))
                        self.calls += 1
                        local_calls += 1
                        if f_contract < self.best_f:
                            self.best_f = f_contract
                            self.best_x = x_contract.copy()
                        if f_contract < f_worst:
                            simplex[-1] = x_contract
                            f_values[-1] = f_contract
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                if local_calls < max_calls and self.calls < budget:
                                    f_values[i] = float(func(simplex[i]))
                                    self.calls += 1
                                    local_calls += 1
                                    if f_values[i] < self.best_f:
                                        self.best_f = f_values[i]
                                        self.best_x = simplex[i].copy()
                    else:
                        simplex[-1] = x_contract
                        f_values[-1] = f_contract
                        
                if max(f_values) - min(f_values) < 1e-10:
                    break
            
        # Allocate budget to starts
        # 50% to best, 25% to 2nd, 25% to 3rd
        remaining_budget = budget - self.calls
        
        b1 = int(remaining_budget * 0.5)
        b2 = int(remaining_budget * 0.25)
        b3 = remaining_budget - b1 - b2
        
        if len(initial_results) > 0:
            nelder_mead(initial_results[0][1], b1)
        if len(initial_results) > 1 and self.calls < budget:
            nelder_mead(initial_results[1][1], b2)
        if len(initial_results) > 2 and self.calls < budget:
            nelder_mead(initial_results[2][1], b3)
            
        return self.best_x, self.best_f
