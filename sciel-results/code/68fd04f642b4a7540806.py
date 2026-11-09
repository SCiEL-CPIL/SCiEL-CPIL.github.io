import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton(self, index, base):
        """Compute the Halton sequence value for a given index and base."""
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result

    def _halton_sequence(self, n, dim, offset=0):
        """Generate n points in [0,1)^dim using Halton sequence."""
        points = np.zeros((n, dim))
        for i in range(n):
            for j in range(dim):
                points[i, j] = self._halton(i + 1 + offset, j + 2)
        return points

    def _nelder_mead(self, func, x0, max_iter=500, tol=1e-10, deadline=None):
        """
        Adaptive Nelder-Mead simplex method.
        Returns (best_x, best_f, calls_used).
        """
        dim = len(x0)
        # Initialize simplex
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        for i in range(dim):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += 0.1

        fvals = np.array([func(simplex[i]) for i in range(dim + 1)])
        calls = dim + 1

        # Adaptive step size
        step = 0.5
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        best_x = simplex[np.argmin(fvals)]
        best_f = fvals.min()

        for iteration in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()

            # Check convergence
            if np.max(fvals) - np.min(fvals) < tol:
                break

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = func(xr)
            calls += 1

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = func(xe)
                calls += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                else:
                    # Inside contraction
                    xc = centroid - 0.5 * (simplex[-1] - centroid)
                fc = func(xc)
                calls += 1
                if fc <= min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls += 1
                    if deadline is not None and time.time() > deadline:
                        break

            # Adaptive step adjustment
            if calls > 0:
                progress = best_f - fvals[0]
                if progress < 1e-8:
                    step *= 0.95
                else:
                    step = min(step * 1.05, 2.0)

            if deadline is not None and time.time() > deadline:
                break

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        calls_made = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = np.array(x)
            return fx

        # Phase 1: Deterministic space-filling exploration using Halton sequence
        num_halton = 80
        halton_results = []
        
        for i in range(1, num_halton + 1):
            if calls_made >= budget - 50:
                break
            if time.time() > deadline:
                break
            
            x = np.array([self._halton(i, base) for base in range(2, dim + 2)])
            x = x * 10.0 - 5.0
            
            fx = eval_func(x)
            halton_results.append((fx, x))

        if not halton_results:
            x = np.zeros(dim)
            eval_func(x)
            return best_x, best_f

        # Sort to find the best point from Halton sampling
        halton_results.sort(key=lambda t: t[0])
        best_halton_f, best_halton_x = halton_results[0]

        # Phase 2: Multi-trajectory (1+1)-ES with dynamic budget allocation
        # Run 5 trajectories with different step sizes, allocating budget based on improvement
        
        if time.time() < deadline and calls_made < budget - 20:
            rng = np.random.default_rng(12345)
            
            # Top 3 Halton points as starting points
            num_starts = min(3, len(halton_results))
            remaining_budget = budget - calls_made
            
            # Define 5 trajectories with different step sizes
            step_sizes = [0.1, 0.3, 0.5, 1.0, 2.0]
            
            # Initialize trajectories
            trajectories = []
            for i in range(num_starts):
                x_curr = halton_results[i][1].copy()
                f_curr = halton_results[i][0]
                trajectories.append({
                    'x': x_curr,
                    'f': f_curr,
                    'sigma': step_sizes[i % len(step_sizes)],
                    'improvements': [],
                    'calls': 0
                })
            
            # Add 2 more trajectories from random points
            for i in range(2):
                x_rand = rng.uniform(-4, 4, dim)
                f_rand = eval_func(x_rand)
                trajectories.append({
                    'x': x_rand,
                    'f': f_rand,
                    'sigma': step_sizes[(num_starts + i) % len(step_sizes)],
                    'improvements': [],
                    'calls': 0
                })
            
            # Run multi-trajectory search
            max_steps_per_trajectory = remaining_budget // (len(trajectories) + 1)
            
            for step in range(max_steps_per_trajectory):
                if calls_made >= budget - 5 or time.time() > deadline:
                    break
                
                # Select trajectory with most recent improvement
                # Calculate improvement in last 20 calls
                best_traj_idx = 0
                best_improvement = -float('inf')
                
                for idx, traj in enumerate(trajectories):
                    if len(traj['improvements']) > 0:
                        recent_improvements = traj['improvements'][-20:]
                        if recent_improvements:
                            improvement = recent_improvements[0] - recent_improvements[-1]
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_traj_idx = idx
                
                # If no improvement found, use round-robin
                if best_improvement == -float('inf'):
                    best_traj_idx = step % len(trajectories)
                
                traj = trajectories[best_traj_idx]
                x_curr = traj['x']
                f_curr = traj['f']
                sigma = traj['sigma']
                
                # Generate candidate
                y = x_curr + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = eval_func(y)
                
                # Update trajectory
                traj['calls'] += 1
                
                if fy < f_curr:
                    old_f = f_curr
                    x_curr = y
                    f_curr = fy
                    traj['x'] = x_curr
                    traj['f'] = f_curr
                    traj['improvements'].append(fy - old_f)
                    # Increase step size on success
                    sigma *= 1.1
                    traj['sigma'] = sigma
                else:
                    # Decrease step size on failure
                    sigma *= 0.9
                    traj['sigma'] = sigma
                
                # Keep only last 20 improvements
                if len(traj['improvements']) > 20:
                    traj['improvements'] = traj['improvements'][-20:]
        
        # Phase 3: Nelder-Mead refinement from best point found
        if best_x is not None and calls_made < budget - 10 and time.time() < deadline - 5:
            nm_deadline = deadline - 5
            x_nm, f_nm, nm_calls = self._nelder_mead(
                func, best_x, max_iter=50, tol=1e-10, deadline=nm_deadline
            )
            calls_made += nm_calls
            if f_nm < best_f:
                best_f = f_nm
                best_x = x_nm

        # Phase 4: If budget remains, try a few more strategic points
        remaining = budget - calls_made
        if remaining > 10 and time.time() < deadline:
            n_extra = min(remaining // 2, 20)
            if n_extra > 0:
                extra_points = self._halton_sequence(n_extra, dim, offset=num_halton)
                extra_points = extra_points * 10.0 - 5.0
                for i in range(n_extra):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    x = extra_points[i]
                    fx = eval_func(x)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                        # Quick NM refinement if we found something better
                        if calls_made < budget - 20 and time.time() < deadline - 10:
                            x_nm2, f_nm2, nm_calls2 = self._nelder_mead(
                                func, best_x, max_iter=30, tol=1e-10, deadline=deadline - 10
                            )
                            calls_made += nm_calls2
                            if f_nm2 < best_f:
                                best_f = f_nm2
                                best_x = x_nm2

        return best_x, best_f
