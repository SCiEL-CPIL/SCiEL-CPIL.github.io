"""Black-box optimizer for BBOB-style Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: basin-hopping on integer lattice + adaptive coordinate descent."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        calls = 0

        def check_time():
            return time.time() < deadline

        def evaluate(x):
            nonlocal calls
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            return fx

        best_x = None
        best_f = float('inf')

        # Phase 1: Evaluate all 25 integer-coordinate centers in [-4,4]^5
        # Rastrigin has local minima at integer coordinates (after shift/rotation)
        # The hidden shift keeps the global minimum within [-4,4]^5
        integer_centers = []
        for v1 in range(-4, 5):
            for v2 in range(-4, 5):
                for v3 in range(-4, 5):
                    for v4 in range(-4, 5):
                        for v5 in range(-4, 5):
                            integer_centers.append(np.array([v1, v2, v3, v4, v5], dtype=float))

        # Also add the origin and some half-integer points as backup
        backup_points = [np.zeros(self.dim)]
        for d in range(self.dim):
            for val in [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]:
                pt = np.zeros(self.dim)
                pt[d] = val
                backup_points.append(pt)

        all_candidates = integer_centers + backup_points

        # Deduplicate
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            key = tuple(np.round(c, 10))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)

        # Evaluate all candidates
        for c in unique_candidates:
            if calls >= self.budget or not check_time():
                break
            fc = evaluate(c)
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = evaluate(best_x)

        # Phase 2: Basin-hopping - refine around the best integer center
        # Use coordinate descent with adaptive step halving
        x = best_x.copy()
        fx = best_f

        # First, try a fine grid around the best center (±0.5 in each coordinate)
        # This helps if the minimum is slightly offset from the integer point
        step_sizes = [0.5, 0.25, 0.125, 0.0625, 0.03125]
        
        for step in step_sizes:
            if calls >= self.budget or not check_time():
                break
            
            # Coordinate descent: try moving each coordinate by ±step
            for coord_iter in range(3):  # multiple passes for convergence
                if calls >= self.budget or not check_time():
                    break
                for d in range(self.dim):
                    for direction in [-1.0, 1.0]:
                        y = x.copy()
                        y[d] += direction * step
                        y = np.clip(y, -5.0, 5.0)
                        fy = evaluate(y)
                        if fy < fx:
                            x = y
                            fx = fy
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()

        # Phase 3: (1+1)-ES with adaptive sigma for fine refinement
        x = best_x.copy()
        fx = best_f
        sigma = 0.1  # Start small
        success_count = 0
        total_trials = 0
        window = 15

        while calls < self.budget and check_time():
            # Adaptive sigma: 1/5 rule
            if total_trials >= window:
                success_rate = success_count / total_trials
                if success_rate > 0.25:
                    sigma *= 1.2
                elif success_rate < 0.15:
                    sigma *= 0.8
                success_count = 0
                total_trials = 0

            # Clamp sigma to reasonable range
            sigma = np.clip(sigma, 0.001, 1.0)

            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5.0, 5.0)
            fy = evaluate(y)
            total_trials += 1

            if fy < fx:
                x = y
                fx = fy
                success_count += 1
                best_x = x.copy()
                best_f = fx
            else:
                # Occasional lattice jumps to escape local minima
                if total_trials % 50 == 0 and calls < self.budget - 50:
                    # Try nearby integer lattice points
                    jump_base = np.round(x).astype(int).astype(float)
                    d = rng.integers(self.dim)
                    offset = rng.choice([-1.0, 1.0])
                    jump = jump_base.copy()
                    jump[d] += offset
                    jump = np.clip(jump, -5.0, 5.0)
                    fj = evaluate(jump)
                    if fj < fx:
                        x = jump
                        fx = fj
                        best_x = x.copy()
                        best_f = fx
                        sigma = 0.2

        # Phase 4: Final high-precision polish
        if calls < self.budget and check_time():
            x = best_x.copy()
            fx = best_f
            sigma_fine = 0.01
            for _ in range(30):
                if calls >= self.budget or not check_time():
                    break
                y = x + sigma_fine * rng.standard_normal(self.dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                if fy < fx:
                    x = y
                    fx = fy
                    best_x = x.copy()
                    best_f = fx
                sigma_fine *= 0.7

        return best_x, best_f
# EVOLVE-BLOCK-END
