import numpy as np
from scipy.optimize import minimize
import time
import random

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def get_dims(W):
        W = np.clip(W, 0.2, 1.8)
        H = 2.0 - W
        return W, H

    def unpack(state):
        W = state[0]
        coords = state[1:].reshape(N, 3)
        return W, coords

    def pack(W, coords):
        state = np.zeros(1 + 3 * N)
        state[0] = W
        state[1:] = coords.flatten()
        return state

    def evaluate(state):
        W = state[0]
        W, H = get_dims(W)
        coords = state[1:].reshape(N, 3)
        x = coords[:, 0]
        y = coords[:, 1]
        r = coords[:, 2]

        penalty = 0.0
        sum_r = np.sum(r)

        # Boundary violations
        viol = 0.0
        viol += np.sum(np.maximum(0, r - x))
        viol += np.sum(np.maximum(0, x + r - W))
        viol += np.sum(np.maximum(0, r - y))
        viol += np.sum(np.maximum(0, y + r - H))
        penalty += 500.0 * viol

        # Overlap violations
        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        min_dist = r[idx_i] + r[idx_j]
        mask = dist_sq < min_dist * min_dist
        if np.any(mask):
            dist = np.sqrt(dist_sq[mask])
            overlaps = min_dist[mask] - dist
            penalty += 1000.0 * np.sum(overlaps)

        # Small radius penalty
        small = np.maximum(0, 0.01 - r)
        penalty += 10.0 * np.sum(small)

        score = sum_r - penalty
        return score, penalty

    def check_valid(state, eps=1e-7):
        W = state[0]
        W, H = get_dims(W)
        coords = state[1:].reshape(N, 3)
        x = coords[:, 0]
        y = coords[:, 1]
        r = coords[:, 2]

        if np.any(r < 1e-6):
            return False

        if np.any(x - r < -eps) or np.any(x + r > W + eps):
            return False
        if np.any(y - r < -eps) or np.any(y + r > H + eps):
            return False

        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        min_dist = r[idx_i] + r[idx_j] - eps
        if np.any(dist_sq < min_dist * min_dist):
            return False

        return True

    def make_hex(W_init=1.0, H_init=1.0, jitter=0.0):
        W, H = get_dims(W_init)
        state = np.zeros(1 + 3 * N)
        state[0] = W_init

        # Estimate radius from density
        r_est = np.sqrt(0.9069 * W * H / (N * np.pi))
        r = r_est
        positions = []

        for attempt in range(30):
            positions = []
            y = r
            row_idx = 0
            while y <= H - r + 1e-9:
                if row_idx % 2 == 0:
                    x_start = r
                else:
                    x_start = r + r
                x = x_start
                while x <= W - r + 1e-9:
                    positions.append((x, y, r))
                    x += 2.0 * r
                y += np.sqrt(3) * r
                row_idx += 1
            if len(positions) >= N:
                break
            r *= 0.95

        if len(positions) < N:
            # Fallback to grid
            positions = []
            side = 5
            for i in range(side):
                for j in range(side):
                    if len(positions) >= N:
                        break
                    gx = (i + 0.5) / side * W
                    gy = (j + 0.5) / side * H
                    r_g = 0.5 / side * min(W, H) * 0.9
                    positions.append((gx, gy, r_g))

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
                state[1 + 3 * i] = W / 2
                state[1 + 3 * i + 1] = H / 2
                state[1 + 3 * i + 2] = 0.01

        return state

    def make_grid(W_init=1.0, H_init=1.0, jitter=0.0):
        W, H = get_dims(W_init)
        state = np.zeros(1 + 3 * N)
        state[0] = W_init

        best_positions = None
        for rows in range(3, 8):
            cols = int(np.ceil(N / rows))
            if rows * cols < N:
                continue
            positions = []
            for i in range(rows):
                for j in range(cols):
                    if len(positions) >= N:
                        break
                    gx = (j + 0.5) / cols * W
                    gy = (i + 0.5) / rows * H
                    r_g = 0.5 * min(W / cols, H / rows) * 0.85
                    positions.append((gx, gy, r_g))
            if len(positions) >= N:
                mean_r = np.mean([p[2] for p in positions])
                if best_positions is None or mean_r > np.mean([p[2] for p in best_positions]):
                    best_positions = positions

        if best_positions is None:
            side = 5
            best_positions = []
            for i in range(side):
                for j in range(side):
                    if len(best_positions) >= N:
                        break
                    gx = (i + 0.5) / side * W
                    gy = (j + 0.5) / side * H
                    r_g = 0.5 / side * min(W, H) * 0.9
                    best_positions.append((gx, gy, r_g))

        for i in range(N):
            px, py, pr = best_positions[i]
            if jitter > 0:
                px += random.uniform(-jitter, jitter)
                py += random.uniform(-jitter, jitter)
            state[1 + 3 * i] = px
            state[1 + 3 * i + 1] = py
            state[1 + 3 * i + 2] = pr

        return state

    def make_random(W_init=1.0, H_init=1.0):
        W, H = get_dims(W_init)
        state = np.zeros(1 + 3 * N)
        state[0] = W_init

        positions = []
        r_est = np.sqrt(0.9069 * W * H / (N * np.pi))
        max_attempts = 2000

        for i in range(N):
            placed = False
            for attempt in range(max_attempts):
                r = r_est * random.uniform(0.7, 1.3)
                px = random.uniform(r, W - r)
                py = random.uniform(r, H - r)
                # Check overlap with existing
                ok = True
                for j, (qx, qy, qr) in enumerate(positions):
                    dx = px - qx
                    dy = py - qy
                    if dx * dx + dy * dy < (r + qr) ** 2:
                        ok = False
                        break
                if ok:
                    positions.append((px, py, r))
                    placed = True
                    break
            if not placed:
                # Place at center with small radius
                positions.append((W / 2 + random.uniform(-0.1, 0.1),
                                  H / 2 + random.uniform(-0.1, 0.1),
                                  r_est * 0.5))

        for i in range(N):
            px, py, pr = positions[i]
            state[1 + 3 * i] = px
            state[1 + 3 * i + 1] = py
            state[1 + 3 * i + 2] = pr

        return state

    def objective(state):
        score, penalty = evaluate(state)
        return -score

    # Generate multiple initial configurations
    configs = []
    W_values = [0.8, 1.0, 1.2, 1.4]

    for W_init in W_values:
        configs.append(make_hex(W_init, 2.0 - W_init, jitter=0.0))
        configs.append(make_grid(W_init, 2.0 - W_init, jitter=0.0))

    configs.append(make_random(1.0, 1.0))
    configs.append(make_random(0.9, 1.1))
    configs.append(make_random(1.1, 0.9))

    # Add jittered versions
    for W_init in [1.0, 1.2]:
        configs.append(make_hex(W_init, 2.0 - W_init, jitter=0.03))
        configs.append(make_grid(W_init, 2.0 - W_init, jitter=0.03))

    best_score = -np.inf
    best_state = None

    # Phase 1: Nelder-Mead coarse optimization
    for idx, p0 in enumerate(configs):
        if time.time() > deadline - 10:
            break

        try:
            res = minimize(objective, p0, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8,
                                    'adaptive': True})
            score, penalty = evaluate(res.x)
            if score > best_score:
                best_score = score
                best_state = res.x.copy()
        except Exception:
            pass

    # Phase 2: Gradient-based polish with L-BFGS-B on best
    if best_state is not None and time.time() < deadline - 20:
        try:
            # Define bounds
            W, H = get_dims(best_state[0])
            bounds = [(0.2, 1.8)]
            for i in range(N):
                bounds.append((0.0, 1.9))  # x
                bounds.append((0.0, 1.9))  # y
                bounds.append((1e-6, 0.5))  # r

            res = minimize(objective, best_state, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            score, penalty = evaluate(res.x)
            if score > best_score:
                best_score = score
                best_state = res.x.copy()
        except Exception:
            pass

    # Phase 3: Try a few more restarts from perturbed best if time allows
    if best_state is not None and time.time() < deadline - 15:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            perturbed = best_state.copy()
            # Small random perturbation
            for i in range(1, 1 + 3 * N):
                perturbed[i] += random.gauss(0, 0.01)
            # Clamp
            W_p, H_p = get_dims(perturbed[0])
            for i in range(N):
                perturbed[1 + 3 * i] = np.clip(perturbed[1 + 3 * i], 0, W_p)
                perturbed[1 + 3 * i + 1] = np.clip(perturbed[1 + 3 * i + 1], 0, H_p)
                perturbed[1 + 3 * i + 2] = np.clip(perturbed[1 + 3 * i + 2], 1e-6, 0.5)

            try:
                res = minimize(objective, perturbed, method='Nelder-Mead',
                               options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-8,
                                        'adaptive': True})
                score, penalty = evaluate(res.x)
                if score > best_score:
                    best_score = score
                    best_state = res.x.copy()
            except Exception:
                pass

    # Final validation and cleanup
    if best_state is None:
        # Fallback: simple grid
        W = 1.0
        H = 1.0
        side = 5
        result = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                gx = (i + 0.5) / side * W
                gy = (j + 0.5) / side * H
                r = 0.5 / side * min(W, H) * 0.9
                result.append([gx, gy, r])
                count += 1
        return np.array(result)

    # Validate and fix if needed
    if not check_valid(best_state):
        # Try to fix by reducing radii
        W, H = get_dims(best_state[0])
        coords = best_state[1:].reshape(N, 3)
        # Reduce all radii by 5%
        coords[:, 2] *= 0.95
        best_state[1:] = coords.flatten()

        if not check_valid(best_state):
            # More aggressive fix
            coords[:, 2] *= 0.8
            best_state[1:] = coords.flatten()

            if not check_valid(best_state):
                # Fallback
                W = 1.0
                H = 1.0
                side = 5
                result = []
                count = 0
                for i in range(side):
                    for j in range(side):
                        if count >= N:
                            break
                        gx = (i + 0.5) / side * W
                        gy = (j + 0.5) / side * H
                        r = 0.5 / side * min(W, H) * 0.9
                        result.append([gx, gy, r])
                        count += 1
                return np.array(result)

    # Extract final result
    W, H = get_dims(best_state[0])
    coords = best_state[1:].reshape(N, 3)

    # Clamp to valid range
    x = np.clip(coords[:, 0], 0, W)
    y = np.clip(coords[:, 1], 0, H)
    r = np.clip(coords[:, 2], 1e-6, min(W, H) / 2)

    result = np.column_stack((x, y, r))

    # Final validity check - if invalid, reduce radii
    if not check_valid(pack(W, result)):
        result[:, 2] *= 0.9
        if not check_valid(pack(W, result)):
            result[:, 2] *= 0.9
            if not check_valid(pack(W, result)):
                # Last resort fallback
                W = 1.0
                H = 1.0
                side = 5
                result = []
                count = 0
                for i in range(side):
                    for j in range(side):
                        if count >= N:
                            break
                        gx = (i + 0.5) / side * W
                        gy = (j + 0.5) / side * H
                        r = 0.5 / side * min(W, H) * 0.9
                        result.append([gx, gy, r])
                        count += 1
                result = np.array(result)

    return result
