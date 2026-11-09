import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        budget = self.budget
        dim = self.dim

        calls_made = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        all_points = []

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError()
            if calls_made >= budget:
                raise ValueError("Budget exceeded")
            calls_made += 1
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            all_points.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def halton_sequence(n, dim):
            """Generate n Halton sequence points in [0,1)^dim."""
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            points = []
            for i in range(1, n + 1):
                point = []
                for d in range(dim):
                    prime = primes[d % len(primes)]
                    f = 1.0
                    ip = i
                    while ip > 0:
                        f /= prime
                        ip = int(ip / prime)
                    point.append(f)
                points.append(point)
            return np.array(points)

        # Phase 1: Coarse stratified sampling for global basin discovery
        n_stratified = 30
        # Use stratified sampling: divide each dimension into 3 regions
        # and pick one point per stratum
        rng = np.random.RandomState(42)
        stratified_points = []
        for i in range(n_stratified):
            # Assign to a stratum based on index
            stratum_idx = i % (3 ** dim)
            point = np.zeros(dim)
            temp = stratum_idx
            for d in range(dim):
                region = temp % 3
                temp //= 3
                # Region 0: [-5, -5/3), Region 1: [-5/3, 5/3), Region 2: [5/3, 5]
                lower = -5.0 + region * (10.0 / 3.0)
                upper = lower + (10.0 / 3.0)
                point[d] = rng.uniform(lower, upper)
            stratified_points.append(point)

        # Also add a few Halton points for better coverage
        halton_pts = halton_sequence(10, dim)
        halton_pts_mapped = -5.0 + 10.0 * halton_pts
        stratified_points.extend(halton_pts_mapped.tolist())

        for pt in stratified_points:
            if time.time() > deadline - 20 or calls_made >= budget - 50:
                break
            eval_func(pt)

        # Phase 2: Multi-fidelity (1+1)-ES with 5 different step sizes
        # Each trajectory runs in parallel conceptually, but we interleave them
        # Dynamically allocate budget based on recent improvement

        step_sizes = [0.1, 0.3, 0.8, 1.5, 4.0]
        n_trajectories = 5

        # Initialize trajectories from best diverse points
        all_points.sort(key=lambda p: p[0])
        start_points = []
        min_dist = 2.0
        for f, x in all_points:
            if len(start_points) >= n_trajectories:
                break
            if all(np.linalg.norm(x - sp) > min_dist for sp in start_points):
                start_points.append(x)

        # Fill if needed
        while len(start_points) < n_trajectories and len(all_points) > len(start_points):
            f, x = all_points[len(start_points)]
            if all(np.linalg.norm(x - sp) > 0.5 for sp in start_points):
                start_points.append(x)

        # If still not enough, use best_x
        while len(start_points) < n_trajectories:
            start_points.append(best_x.copy())

        # Track each trajectory's state
        trajectories = []
        for i in range(n_trajectories):
            trajectories.append({
                'x': start_points[i].copy(),
                'f': float(func(start_points[i])) if calls_made < budget else best_f,
                'step_size': step_sizes[i],
                'recent_improvements': [],
                'calls': 0
            })

        # Re-evaluate to get actual fitness values (counted in budget)
        for i in range(n_trajectories):
            if calls_made >= budget or time.time() > deadline - 15:
                break
            try:
                fx = eval_func(trajectories[i]['x'])
                trajectories[i]['f'] = fx
            except:
                trajectories[i]['f'] = best_f

        # Main loop: dynamically allocate budget to trajectories
        # based on their recent improvement
        max_es_steps = 50  # Maximum steps per trajectory
        es_step_count = 0

        while es_step_count < max_es_steps:
            if calls_made >= budget or time.time() > deadline - 10:
                break

            # Score each trajectory based on recent improvements
            scores = []
            for t in trajectories:
                if len(t['recent_improvements']) >= 5:
                    # Average of last 5 improvements (positive = good)
                    score = np.mean(t['recent_improvements'][-5:])
                elif len(t['recent_improvements']) > 0:
                    score = np.mean(t['recent_improvements'])
                else:
                    score = 0.0
                scores.append(score)

            # Allocate next evaluation to trajectory with highest score
            # Add some randomness to avoid getting stuck
            if np.random.random() < 0.1:
                idx = np.random.randint(n_trajectories)
            else:
                idx = np.argmax(scores)

            t = trajectories[idx]

            # Generate a new point using (1+1)-ES
            direction = np.random.randn(dim)
            direction /= (np.linalg.norm(direction) + 1e-10)
            new_x = t['x'] + t['step_size'] * direction
            new_x = np.clip(new_x, -5, 5)

            if calls_made >= budget or time.time() > deadline - 10:
                break

            try:
                new_f = eval_func(new_x)
            except:
                break

            improvement = t['f'] - new_f
            t['recent_improvements'].append(improvement)
            t['calls'] += 1

            if new_f < t['f']:
                t['x'] = new_x.copy()
                t['f'] = new_f
                # Increase step size slightly on success
                t['step_size'] = min(t['step_size'] * 1.1, 5.0)
            else:
                # Decrease step size on failure
                t['step_size'] = max(t['step_size'] * 0.8, 0.01)

            es_step_count += 1

        # Phase 3: Final refinement with Nelder-Mead on best point
        remaining = budget - calls_made
        if remaining > 10 and time.time() < deadline - 5:
            try:
                minimize(
                    eval_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(remaining - 2, 50),
                        'xatol': 1e-7,
                        'fatol': 1e-9,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        return best_x, best_f
