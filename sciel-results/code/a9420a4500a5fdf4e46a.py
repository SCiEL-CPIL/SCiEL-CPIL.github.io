"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    def get_rect_dims(t):
        t = float(np.clip(t, 0.1, 0.9))
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        return w, h

    def evaluate(state, beta=1000.0):
        """
        Returns score = sum_r - beta * penalty.
        Higher is better. Valid solutions have penalty=0.
        """
        t = state[0]
        w, h = get_rect_dims(t)
        x = state[1::3]
        y = state[2::3]
        radii = state[3::3]

        # Clamp radii to positive
        radii = np.maximum(radii, 1e-6)

        sum_r = np.sum(radii)

        penalty = 0.0

        # Boundary violations (linear)
        viol_left = np.maximum(0.0, radii - x)
        viol_right = np.maximum(0.0, x + radii - w)
        viol_bottom = np.maximum(0.0, radii - y)
        viol_top = np.maximum(0.0, y + radii - h)
        penalty += np.sum(viol_left) + np.sum(viol_right) + np.sum(viol_bottom) + np.sum(viol_top)

        # Overlap violations (vectorized, only upper triangle)
        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        min_dist = radii[idx_i] + radii[idx_j]

        overlap_mask = dist_sq < min_dist * min_dist
        if np.any(overlap_mask):
            dist = np.sqrt(dist_sq[overlap_mask])
            overlaps = min_dist[overlap_mask] - dist
            penalty += np.sum(overlaps)

        score = sum_r - beta * penalty
        return score

    def check_valid(state, eps=1e-7):
        t = state[0]
        w, h = get_rect_dims(t)
        x = state[1::3]
        y = state[2::3]
        radii = state[3::3]

        if np.any(radii < 1e-6):
            return False

        if np.any(x - radii < -eps) or np.any(x + radii > w + eps) or \
           np.any(y - radii < -eps) or np.any(y + radii > h + eps):
            return False

        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        min_dist = radii[idx_i] + radii[idx_j] - eps
        if np.any(dist_sq < min_dist * min_dist):
            return False
        return True

    def make_initial_hexagonal(t_init=0.5, jitter=0.0):
        w, h = get_rect_dims(t_init)
        state = np.zeros(1 + 3 * N)
        state[0] = t_init
        r_est = np.sqrt(0.9069 * w * h / (N * np.pi))
        r_current = r_est
        positions = []

        for attempt in range(30):
            positions = []
            y = r_current
            row_idx = 0
            while y <= h - r_current + 1e-9:
                if row_idx % 2 == 0:
                    x_start = r_current
                else:
                    x_start = r_current + r_current
                x = x_start
                while x <= w - r_current + 1e-9:
                    positions.append((x, y, r_current))
                    x += 2.0 * r_current
                y += np.sqrt(3) * r_current
                row_idx += 1
            if len(positions) >= N:
                break
            else:
                r_current *= 0.93

        if len(positions) < N:
            # Fallback: grid
            side = 5
            idx = 0
            positions = []
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    gx = (i + 0.5) / side * w
                    gy = (j + 0.5) / side * h
                    r = 0.5 / side * min(w, h) * 0.9
                    positions.append((gx, gy, r))
                    idx += 1

        for i in range(N):
            if i < len(positions):
                px, py, pr = positions[i]
                if jitter > 0:
                    px += random.uniform(-jitter, jitter)
                    py += random.uniform(-jitter, jitter)
                state[1 + 3 * i] = px
                state[1 + 3 * i + 1] = py
                state[1 + 3 * i + 2] = pr
            else:
                state[1 + 3 * i] = w / 2
                state[1 + 3 * i + 1] = h / 2
                state[1 + 3 * i + 2] = 0.01
        return state

    def make_initial_grid(t_init=0.5, jitter=0.0):
        w, h = get_rect_dims(t_init)
        state = np.zeros(1 + 3 * N)
        state[0] = t_init

        best_positions = None
        best_mean_r = 0.0
        for rows in range(3, 8):
            cols = int(np.ceil(N / rows))
            if rows * cols < N:
                continue
            positions = []
            for i in range(rows):
                for j in range(cols):
                    if len(positions) >= N:
                        break
                    gx = (j + 0.5) / cols * w
                    gy = (i + 0.5) / rows * h
                    r = 0.5 * min(w / cols, h / rows) * 0.9
                    positions.append((gx, gy, r))
            if len(positions) >= N:
                mean_r = np.mean([p[2] for p in positions])
                if mean_r > best_mean_r:
                    best_mean_r = mean_r
                    best_positions = positions

        if best_positions is None:
            side = 5
            positions = []
            for i in range(side):
                for j in range(side):
                    if i * side + j >= N:
                        break
                    gx = (i + 0.5) / side * w
                    gy = (j + 0.5) / side * h
                    r = 0.5 / side * min(w, h) * 0.9
                    positions.append((gx, gy, r))
            best_positions = positions

        for i in range(N):
            px, py, pr = best_positions[i]
            if jitter > 0:
                px += random.uniform(-jitter, jitter)
                py += random.uniform(-jitter, jitter)
            state[1 + 3 * i] = px
            state[1 + 3 * i + 1] = py
            state[1 + 3 * i + 2] = pr
        return state

    def run_slsqp(state, beta, max_iter=200):
        """Run SLSQP with given beta penalty weight."""
        def obj(s):
            return -evaluate(s, beta)

        # Bounds: t in [0.1, 0.9], x in [0, w], y in [0, h], r in [1e-6, min(w,h)/2]
        # Since w,h depend on t, use loose bounds
        bounds = [(0.1, 0.9)] + [(0.0, 2.0)] * N + [(0.0, 2.0)] * N + [(1e-6, 1.0)] * N

        result = minimize(obj, state, method='SLSQP', bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': 1e-10})
        if result.success or hasattr(result, 'x'):
            return result.x
        return state

    def aspect_refine(state, n_steps=30):
        """Fine scan over t to find best aspect ratio."""
        best_state = state.copy()
        best_score = evaluate(state, 10000.0)
        t0 = state[0]
        for i in range(1, n_steps + 1):
            dt = (i / n_steps) * 0.05
            for sign in [1, -1]:
                t_new = t0 + sign * dt
                if t_new < 0.1 or t_new > 0.9:
                    continue
                s = state.copy()
                s[0] = t_new
                # Quick re-optimize with low max_iter
                s = run_slsqp(s, 10000.0, max_iter=50)
                sc = evaluate(s, 10000.0)
                if sc > best_score:
                    best_score = sc
                    best_state = s
        return best_state

    def sa_aspect_diversify(state, n_iter=200, T0=0.05):
        """Lightweight SA over t only to escape local minima."""
        best_state = state.copy()
        best_score = evaluate(state, 10000.0)
        current = state.copy()
        T = T0
        for it in range(n_iter):
            if time.time() > deadline:
                break
            dt = np.random.normal(0, 0.03)
            t_new = float(np.clip(current[0] + dt, 0.1, 0.9))
            s = current.copy()
            s[0] = t_new
            sc = evaluate(s, 10000.0)
            delta = sc - best_score
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                current = s
                if sc > best_score:
                    best_score = sc
                    best_state = s
            T *= 0.99
            if T < 0.005:
                T = 0.005
        return best_state

    # Main optimization loop
    best_state = None
    best_score = -np.inf

    # Try multiple initializations
    initializations = []
    # Hexagonal at various aspect ratios
    for t_init in [0.5, 0.45, 0.55, 0.4, 0.6]:
        initializations.append(make_initial_hexagonal(t_init, jitter=0.02))
    # Grid at a couple of aspect ratios
    for t_init in [0.5, 0.55]:
        initializations.append(make_initial_grid(t_init, jitter=0.01))

    # Multi-phase beta schedule
    beta_phases = [100.0, 500.0, 2000.0, 5000.0, 10000.0]

    for init_state in initializations:
        if time.time() > deadline:
            break

        current = init_state.copy()

        for beta in beta_phases:
            if time.time() > deadline:
                break
            # Run SLSQP
            current = run_slsqp(current, beta, max_iter=300)

            # Aspect refinement
            current = aspect_refine(current, n_steps=20)

            sc = evaluate(current, 10000.0)
            if sc > best_score:
                best_score = sc
                best_state = current.copy()

        # SA diversification over aspect ratio
        if time.time() < deadline - 20:
            current = sa_aspect_diversify(current, n_iter=150)
            # Re-polish after SA
            current = run_slsqp(current, 5000.0, max_iter=100)
            current = aspect_refine(current, n_steps=15)
            sc = evaluate(current, 10000.0)
            if sc > best_score:
                best_score = sc
                best_state = current.copy()

    # Final polish with highest beta
    if best_state is not None and time.time() < deadline - 10:
        final = run_slsqp(best_state, 10000.0, max_iter=200)
        final = aspect_refine(final, n_steps=25)
        sc = evaluate(final, 10000.0)
        if sc > best_score:
            best_score = sc
            best_state = final

    # Ensure validity: if best_state is invalid, project it
    if best_state is None:
        # Fallback: use the first initialization
        best_state = initializations[0]

    if not check_valid(best_state):
        # Try to fix by slightly reducing radii
        fixed = best_state.copy()
        for _ in range(200):
            if check_valid(fixed):
                break
            t = fixed[0]
            w, h = get_rect_dims(t)
            x = fixed[1::3]
            y = fixed[2::3]
            r = fixed[3::3]

            # Find max violation
            viol = 0.0
            viol = max(viol, np.max(0, r - x))
            viol = max(viol, np.max(0, x + r - w))
            viol = max(viol, np.max(0, r - y))
            viol = max(viol, np.max(0, y + r - h))

            # Overlap violations
            idx_i, idx_j = np.triu_indices(N, k=1)
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_d = r[idx_i] + r[idx_j]
            overlaps = np.maximum(0, min_d - dist)
            if np.any(overlaps > 0):
                viol = max(viol, np.max(overlaps) / 2)

            if viol > 1e-9:
                r -= viol / 2
                r = np.maximum(r, 1e-6)
                fixed[3::3] = r
            else:
                break

        if check_valid(fixed):
            best_state = fixed
        else:
            # Last resort: use a very safe grid
            best_state = make_initial_grid(0.5, jitter=0.0)
            # Shrink radii to ensure validity
            for _ in range(100):
                if check_valid(best_state):
                    break
                t = best_state[0]
                w, h = get_rect_dims(t)
                x = best_state[1::3]
                y = best_state[2::3]
                r = best_state[3::3]
                r *= 0.95
                best_state[3::3] = r

    # Extract output
    x = best_state[1::3]
    y = best_state[2::3]
    r = best_state[3::3]

    out = np.column_stack((x, y, r))
    return out

# EVOLVE-BLOCK-END
