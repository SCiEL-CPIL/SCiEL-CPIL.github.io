import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s limit

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12

    # Use a fine grid for accurate boundary calculation
    # 2400 steps ensures the 12 exercise dates align with grid points (2400/12 = 200)
    n_fine_steps = 2400
    dt = T / n_fine_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    df = np.exp(-r * dt)

    # Grid for S: use a hybrid grid - fine near K, coarser in tails
    # This gives better resolution where the boundary is most sensitive
    n_grid = 6000
    # Create a non-uniform grid: denser around K
    S_min = 0.01
    S_max = 250.0
    # Use a transformation to concentrate points near K
    # Linear in log-space around K, linear in tails
    x = np.linspace(0, 1, n_grid)
    # Blend: use sqrt transformation to concentrate near low S
    S_grid = S_min + (S_max - S_min) * x
    # Actually, let's use a piecewise approach:
    # Fine grid from 0.01 to 2*K, coarser from 2*K to S_max
    n_fine = 4500
    n_coarse = n_grid - n_fine
    S_fine = np.linspace(S_min, 2.0 * K, n_fine)
    S_coarse = np.linspace(2.0 * K + (2.0*K - S_min)/(n_fine-1), S_max, n_coarse)
    S_grid = np.concatenate([S_fine, S_coarse])
    # Ensure sorted
    S_grid = np.sort(S_grid)

    # V at maturity (step n_fine_steps)
    V = np.maximum(K - S_grid, 0.0)

    boundaries = [0.0] * (n_exercises + 1)
    boundaries[n_exercises] = K  # At maturity, exercise if S <= K

    # Backward induction
    # Exercise dates are at fine steps: 200, 400, ..., 2400
    # We iterate backwards from n_fine_steps - 1 down to 1
    for k in range(n_fine_steps - 1, 0, -1):
        if time.time() > deadline:
            break

        intrinsic = np.maximum(K - S_grid, 0.0)

        # Continuation value
        Su = S_grid * u
        Sd = S_grid * d

        # Interpolate V
        V_up = np.interp(Su, S_grid, V)
        V_down = np.interp(Sd, S_grid, V)

        continuation = df * (p * V_up + (1 - p) * V_down)

        V = np.maximum(intrinsic, continuation)

        # Check if this step corresponds to an exercise date
        if k % 200 == 0:
            j = k // 200  # j is 1..11
            if 1 <= j <= n_exercises - 1:
                diff = intrinsic - continuation
                # Find the boundary: last index where diff > 0
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

    # If we have time, do a refinement pass with a locally denser grid around each boundary
    if time.time() < deadline - 5.0:
        for j in range(1, n_exercises):
            if boundaries[j] <= 0:
                continue
            # Refine around the estimated boundary
            b_est = boundaries[j]
            local_min = max(S_min, b_est * 0.5)
            local_max = min(S_max, b_est * 1.5)
            n_local = 2000
            S_local = np.linspace(local_min, local_max, n_local)

            # Recompute V on local grid using the full backward induction from maturity
            # But we only need the value at exercise date j
            # Re-do backward induction but track only the local grid values
            V_local = np.maximum(K - S_local, 0.0)
            u_local = u
            d_local = d
            p_local = p
            df_local = df

            for kk in range(n_fine_steps - 1, 0, -1):
                if time.time() > deadline:
                    break
                intrinsic_local = np.maximum(K - S_local, 0.0)
                Su_l = S_local * u_local
                Sd_l = S_local * d_local

                # Interpolate from full grid to local grid
                V_up_l = np.interp(Su_l, S_grid, V)
                V_down_l = np.interp(Sd_l, S_grid, V)
                continuation_l = df_local * (p_local * V_up_l + (1 - p_local) * V_down_l)
                V_local = np.maximum(intrinsic_local, continuation_l)

                if kk == j * 200:
                    diff_l = intrinsic_local - continuation_l
                    ex_idx = np.where(diff_l > 0)[0]
                    if len(ex_idx) > 0:
                        li = ex_idx[-1]
                        if li < n_local - 1:
                            d0 = diff_l[li]
                            d1 = diff_l[li + 1]
                            if d1 != d0:
                                t_i = d0 / (d0 - d1)
                                boundaries[j] = S_local[li] + t_i * (S_local[li + 1] - S_local[li])
                            else:
                                boundaries[j] = S_local[li]
                        else:
                            boundaries[j] = S_local[-1]
                    else:
                        boundaries[j] = local_min

    class ExercisePolicy:
        def __init__(self, boundaries, K):
            self.boundaries = boundaries
            self.K = K

        def exercise(self, current_state, time):
            current_state = np.asarray(current_state)
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
