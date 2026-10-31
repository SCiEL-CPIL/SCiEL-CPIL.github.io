"""Circle packing in perimeter-4 rectangle: Vectorized gradient descent with penalty objective."""
import numpy as np
import time
import math

N = 21
_IU = np.triu_indices(N, k=1)  # precomputed upper-triangle indices


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # ---------- helpers ----------
    def check_validity(x, y, r, w, h, tol=1e-8):
        if w <= 0.01 or h <= 0.01:
            return False, 0.0
        if np.any(r < 0):
            return False, 0.0
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum * r_sum - tol:
                    return False, 0.0
        return True, float(np.sum(r))

    def expand_all(x, y, r, w, h, max_rounds=40):
        """Push every radius to its geometric maximum, iterating to convergence."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        for _ in range(max_rounds):
            improved = False
            for i in range(N):
                # Max radius from walls
                r_wall = min(x[i], w - x[i], y[i], h - y[i])
                r_max = r_wall
                # Max radius from other circles
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = math.sqrt(dx * dx + dy * dy)
                    r_allowed = d - r[j]
                    if r_allowed < r_max:
                        r_max = r_allowed
                r_max = max(0.0, r_max)
                if r_max > r[i] + 1e-9:
                    r[i] = r_max
                    improved = True
            if not improved:
                break
        return x, y, r

    def gradient_descent(x0, y0, r0, w0, time_budget=20.0):
        """Vectorized gradient descent on penalty objective."""
        n = N
        # State: x, y, r, w
        x = x0.copy().astype(np.float64)
        y = y0.copy().astype(np.float64)
        r = r0.copy().astype(np.float64)
        w = float(w0)
        h = 2.0 - w

        # Parameters
        lambda_rep = 500.0   # overlap penalty weight
        lambda_gap = 100.0   # gap penalty weight (encourage close packing)
        lambda_wall = 500.0  # wall penalty weight
        lambda_radius = 1.0  # radius reward weight
        gap_margin = 0.02    # desired gap between circles
        
        lr = 0.005           # learning rate
        lr_decay = 0.9995
        momentum = 0.9
        
        # Momentum buffers
        vx = np.zeros(n)
        vy = np.zeros(n)
        vr = np.zeros(n)
        vw = 0.0

        best_x, best_y, best_r, best_w = x.copy(), y.copy(), r.copy(), w
        best_obj = 1e10
        current_obj = 1e10

        iu_i, iu_j = _IU

        while time.time() < start_time + time_budget:
            # Compute objective and gradients
            obj = 0.0
            gx = np.zeros(n)
            gy = np.zeros(n)
            gr = np.zeros(n)
            gw = 0.0

            # Circle-circle interactions
            for i, j in zip(iu_i, iu_j):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                d = math.sqrt(d2) if d2 > 1e-12 else 1e-6
                r_sum = r[i] + r[j]
                diff = d - r_sum  # positive if gap, negative if overlap
                
                # Penalty: quadratic for overlap, linear for gap (to encourage tight packing)
                if diff < 0:
                    # Overlap: strong penalty
                    penalty = lambda_rep * diff * diff
                    dpen_dd = 2 * lambda_rep * diff
                else:
                    # Gap: mild penalty to encourage closeness
                    penalty = lambda_gap * diff * diff
                    dpen_dd = 2 * lambda_gap * diff
                
                obj += penalty
                
                # Gradient w.r.t. distance
                if d > 1e-8:
                    dx_dd = dx / d
                    dy_dd = dy / d
                    dr_i_dd = 1.0
                    dr_j_dd = 1.0
                else:
                    dx_dd, dy_dd, dr_i_dd, dr_j_dd = 0, 0, 0, 0
                
                # Chain rule: d(obj)/d(x_i) = dpen/dd * dx_dd
                gx[i] += dpen_dd * dx_dd
                gy[i] += dpen_dd * dy_dd
                gx[j] -= dpen_dd * dx_dd
                gy[j] -= dpen_dd * dy_dd
                gr[i] += dpen_dd * dr_i_dd
                gr[j] += dpen_dd * dr_j_dd

            # Wall interactions
            for i in range(n):
                # Left wall
                diff_l = x[i] - r[i]  # should be >= 0
                if diff_l < 0:
                    obj += lambda_wall * diff_l * diff_l
                    gx[i] += 2 * lambda_wall * diff_l
                    gr[i] += -2 * lambda_wall * diff_l
                # Right wall
                diff_r = (w - x[i]) - r[i]  # should be >= 0
                if diff_r < 0:
                    obj += lambda_wall * diff_r * diff_r
                    gx[i] -= 2 * lambda_wall * diff_r
                    gr[i] += -2 * lambda_wall * diff_r
                    gw += 2 * lambda_wall * diff_r
                # Bottom wall
                diff_b = y[i] - r[i]  # should be >= 0
                if diff_b < 0:
                    obj += lambda_wall * diff_b * diff_b
                    gy[i] += 2 * lambda_wall * diff_b
                    gr[i] += -2 * lambda_wall * diff_b
                # Top wall
                diff_t = (h - y[i]) - r[i]  # should be >= 0
                if diff_t < 0:
                    obj += lambda_wall * diff_t * diff_t
                    gy[i] -= 2 * lambda_wall * diff_t
                    gr[i] += -2 * lambda_wall * diff_t

            # Radius reward: we want to maximize sum(r), so minimize -sum(r)
            obj -= lambda_radius * np.sum(r)
            gr -= lambda_radius * np.ones(n)

            # Update with momentum
            vx = momentum * vx - lr * gx
            vy = momentum * vy - lr * gy
            vr = momentum * vr - lr * gr
            vw = momentum * vw - lr * gw

            x += vx
            y += vy
            r += vr
            w += vw

            # Clamp to valid ranges
            h = 2.0 - w
            if w < 0.1:
                w = 0.1
            if h < 0.1:
                w = 1.9
            h = 2.0 - w
            
            r = np.maximum(r, 0.01)
            # Keep centers within bounds (with margin)
            x = np.clip(x, 0.01, w - 0.01)
            y = np.clip(y, 0.01, h - 0.01)

            # Decay learning rate
            lr *= lr_decay

            # Track best
            if obj < best_obj:
                best_obj = obj
                best_x, best_y, best_r, best_w = x.copy(), y.copy(), r.copy(), w

        return best_x, best_y, best_r, best_w, 2.0 - best_w

    def get_initial_guesses():
        guesses = []
        # Hexagonal lattice initialization
        side = int(math.ceil(math.sqrt(N)))
        aspect_ratios = [1.0, 0.85, 1.15, 0.75, 1.25, 0.65, 1.4, 0.6, 1.6, 0.55, 0.9, 1.1]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            # Hexagonal packing
            r0 = 0.08
            dx = 2 * r0
            dy = dx * math.sqrt(3) / 2
            for i in range(N):
                row = i // int(math.ceil(w0 / dx))
                col = i % int(math.ceil(w0 / dx))
                cx = (col + 0.5) * dx
                cy = (row + 0.5) * dy + (0.5 * dy if row % 2 else 0)
                if cx + r0 > w0:
                    cx = w0 - r0
                if cy + r0 > h0:
                    cy = h0 - r0
                x[i] = cx
                y[i] = cy
                r[i] = r0
            guesses.append((x, y, r, w0))

        # Random starts
        rng = np.random.default_rng(123)
        for _ in range(8):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.09)
                x[i] = rng.uniform(r0, w0 - r0)
                y[i] = rng.uniform(r0, h0 - r0)
                r[i] = r0
            guesses.append((x, y, r, w0))

        return guesses

    # ---------- Main search loop ----------
    best_value = 0.0
    best_result = None

    guesses = get_initial_guesses()

    for idx, (x0, y0, r0, w0) in enumerate(guesses):
        if time.time() > deadline - 15:
            break

        # Allocate time for GD based on remaining budget
        remaining = deadline - time.time()
        gd_time = min(15.0, remaining - 10.0)
        if gd_time < 3.0:
            break

        x, y, r, w, h = gradient_descent(x0, y0, r0, w0, time_budget=gd_time)

        # Expand radii to maximum
        x, y, r = expand_all(x, y, r, w, h, max_rounds=50)

        # Check validity
        valid, val = check_validity(x, y, r, w, h)
        if valid and val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w, h)

    # ---------- Final polish ----------
    if best_result is not None and time.time() < deadline - 5:
        x, y, r, w, h = best_result
        # One more expansion pass
        x, y, r = expand_all(x, y, r, w, h, max_rounds=50)
        valid, val = check_validity(x, y, r, w, h)
        if valid and val > best_value:
            best_value = val
            best_result = (x.copy(), y.copy(), r.copy(), w, h)

    if best_result is None:
        # Fallback: simple grid
        side = int(math.ceil(math.sqrt(N)))
        cells = []
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side
            cy = (row + 0.5) / side
            r0 = 0.5 / side - 0.005
            cells.append((cx, cy, r0))
        return np.array(cells)

    x, y, r, w, h = best_result
    return np.column_stack((x, y, r))
