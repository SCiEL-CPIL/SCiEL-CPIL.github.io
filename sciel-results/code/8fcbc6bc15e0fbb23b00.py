import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s limit

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12

    dt = T / n_exercises
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    df = np.exp(-r * dt)

    # Uniform linear grid — better for resolving the put exercise boundary kink
    S_min = 0.0
    S_max = 150.0
    n_grid = 40000
    S_grid = np.linspace(S_min, S_max, n_grid)

    # V at maturity (step 12)
    V_next = np.maximum(K - S_grid, 0.0)

    # boundaries[j] for j in 1..12
    boundaries = [0.0] * (n_exercises + 1)
    boundaries[n_exercises] = K  # At maturity, exercise if S <= K

    # Store value functions for later polish
    V_funcs = [None] * (n_exercises + 1)
    V_funcs[n_exercises] = V_next.copy()

    # Backward induction from step 11 down to 1
    for j in range(n_exercises - 1, 0, -1):
        if time.time() > deadline:
            # Emergency break: return what we have.
            for jj in range(j, 0, -1):
                boundaries[jj] = boundaries[n_exercises]
            break

        intrinsic = np.maximum(K - S_grid, 0.0)

        # Continuation value
        Su = S_grid * u
        Sd = S_grid * d

        # Interpolate V_next
        V_up = np.interp(Su, S_grid, V_next)
        V_down = np.interp(Sd, S_grid, V_next)

        continuation = df * (p * V_up + (1 - p) * V_down)

        V_current = np.maximum(intrinsic, continuation)

        # Find boundary
        diff = intrinsic - continuation

        exercise_indices = np.where(diff > 0)[0]

        if len(exercise_indices) > 0:
            last_idx = exercise_indices[-1]
            if last_idx < n_grid - 1:
                d0 = diff[last_idx]
                d1 = diff[last_idx + 1]
                if d1 != d0:
                    t_interp = d0 / (d0 - d1)
                    boundaries[j] = S_grid[last_idx] + t_interp * (S_grid[last_idx + 1] - S_grid[last_idx])
                else:
                    boundaries[j] = S_grid[last_idx]
            else:
                boundaries[j] = S_grid[-1]
        else:
            boundaries[j] = S_min

        V_funcs[j] = V_current.copy()
        V_next = V_current

    # High-precision polish pass: refine each boundary using a local fine grid
    if time.time() < deadline - 10.0:
        for j in range(1, n_exercises):
            if time.time() > deadline:
                break
            b_coarse = boundaries[j]
            if b_coarse <= 0:
                continue

            # Local fine grid around the coarse boundary
            # Window: narrow, e.g., +/- 0.5 around b_coarse
            lo = max(0.0, b_coarse - 0.5)
            hi = min(S_max, b_coarse + 0.5)
            if lo >= hi:
                continue
            n_local = 50000
            S_local = np.linspace(lo, hi, n_local)

            # Get V_next from step j+1 (stored)
            V_next_j1 = V_funcs[j + 1]

            intrinsic_local = np.maximum(K - S_local, 0.0)

            Su_local = S_local * u
            Sd_local = S_local * d

            V_up_local = np.interp(Su_local, S_grid, V_next_j1)
            V_down_local = np.interp(Sd_local, S_grid, V_next_j1)

            continuation_local = df * (p * V_up_local + (1 - p) * V_down_local)

            diff_local = intrinsic_local - continuation_local

            # Find sign change
            exercise_local = np.where(diff_local > 0)[0]
            if len(exercise_local) > 0:
                last_idx = exercise_local[-1]
                if last_idx < n_local - 1:
                    d0 = diff_local[last_idx]
                    d1 = diff_local[last_idx + 1]
                    if d1 != d0:
                        t_interp = d0 / (d0 - d1)
                        boundaries[j] = S_local[last_idx] + t_interp * (S_local[last_idx + 1] - S_local[last_idx])
                    else:
                        boundaries[j] = S_local[last_idx]
                else:
                    boundaries[j] = S_local[-1]
            else:
                boundaries[j] = lo

    class ExercisePolicy:
        def __init__(self, boundaries, K):
            self.boundaries = boundaries
            self.K = K

        def exercise(self, current_state, time):
            current_state = np.asarray(current_state, dtype=np.float64)
            if isinstance(time, (int, float)):
                idx = int(time)
                idx = max(1, min(12, idx))
                b = self.boundaries[idx]
                return current_state <= b
            else:
                times = np.asarray(time)
                times_clipped = np.clip(times, 1, 12).astype(int)
                b = np.array([self.boundaries[t] for t in times_clipped])
                return current_state <= b

    return ExercisePolicy(boundaries, K)
