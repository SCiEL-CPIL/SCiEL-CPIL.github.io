"""Circle packing in perimeter-4 rectangle: Soft-body relaxation + contact-graph local search."""
import numpy as np
import time

N = 21


def _is_valid(x, y, r, W, H, tol=1e-8):
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(x - r < -tol) or np.any(y - r < -tol):
        return False
    if np.any(x + r > W + tol) or np.any(y + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _grow_radii(x, y, r, W, H, time_budget, min_gain=1e-10):
    """Binary-search for the largest uniform scale factor s such that r*s is valid."""
    lo = 1.0
    hi = 1.0
    # First, find an upper bound by exponential search
    for _ in range(60):
        if time.time() > time_budget:
            break
        trial = r * hi
        if _is_valid(x, y, trial, W, H):
            lo = hi
            hi *= 1.1
            if hi > 10.0:
                break
        else:
            break
    if hi <= 1.0:
        return r.copy()
    # Binary search
    for _ in range(50):
        if time.time() > time_budget:
            break
        mid = (lo + hi) / 2.0
        if _is_valid(x, y, r * mid, W, H):
            lo = mid
        else:
            hi = mid
    if lo - 1.0 > min_gain:
        return r * lo
    return r.copy()


def _soft_body_relax(x, y, r, W, H, time_budget, n_steps=20000):
    """
    Kinetic soft-body: all radii grow at a constant rate; positions are pushed apart
    by repulsion proportional to (r_i + r_j - d_ij) when overlapping, plus wall forces.
    This mimics inflating balloons in a box.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    W = float(W)
    H = float(H)

    grow_rate = 0.00008  # radius growth per step
    pos_dt = 0.003
    wall_k = 50.0
    pair_k = 12.0
    min_r = 0.005

    for step in range(n_steps):
        if time.time() > time_budget:
            break

        # Grow all radii
        r += grow_rate
        # Clamp to minimum
        r = np.maximum(r, min_r)

        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)

        # Wall repulsion
        left_v = np.maximum(r - x, 0.0)
        right_v = np.maximum(x + r - W, 0.0)
        bottom_v = np.maximum(r - y, 0.0)
        top_v = np.maximum(y + r - H, 0.0)
        fx -= left_v * wall_k
        fx += right_v * wall_k
        fy -= bottom_v * wall_k
        fy += top_v * wall_k

        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                rsum = r[i] + r[j]
                if d < rsum:
                    if d < 1e-12:
                        dx = 0.01
                        dy = 0.01
                        d = np.sqrt(dx * dx + dy * dy)
                    overlap = rsum - d
                    f = overlap * pair_k
                    ux = dx / d
                    uy = dy / d
                    fx[i] += ux * f
                    fy[i] += uy * f
                    fx[j] -= ux * f
                    fy[j] -= uy * f

        # Update positions
        x += pos_dt * fx
        y += pos_dt * fy

        # Clamp to walls (hard constraint)
        for i in range(N):
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

    return x, y, r


def _soft_body_with_width(x, y, r, W, H, time_budget, n_steps=15000):
    """Soft-body relaxation where width is also a free variable (perimeter=4 constraint)."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    W = float(W)
    H = float(H)

    grow_rate = 0.00006
    pos_dt = 0.002
    wall_k = 40.0
    pair_k = 10.0
    w_dt = 0.0001  # width adjustment rate
    min_r = 0.005

    # Target: maximize sum(r) subject to perimeter=4
    # We'll let W drift: if circles are more "width-limited" than "height-limited", increase W
    for step in range(n_steps):
        if time.time() > time_budget:
            break

        r += grow_rate
        r = np.maximum(r, min_r)

        fx = np.zeros(N)
        fy = np.zeros(N)
        fW = 0.0

        # Wall repulsion
        left_v = np.maximum(r - x, 0.0)
        right_v = np.maximum(x + r - W, 0.0)
        bottom_v = np.maximum(r - y, 0.0)
        top_v = np.maximum(y + r - H, 0.0)
        fx -= left_v * wall_k
        fx += right_v * wall_k
        fy -= bottom_v * wall_k
        fy += top_v * wall_k

        # Width pressure: if right wall violations > left, push W right (increase W)
        fW += (np.sum(right_v) - np.sum(left_v)) * 0.1
        # Height pressure: if top > bottom, we'd want H larger => W smaller
        fW -= (np.sum(top_v) - np.sum(bottom_v)) * 0.1

        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                rsum = r[i] + r[j]
                if d < rsum:
                    if d < 1e-12:
                        dx = 0.01
                        dy = 0.01
                        d = np.sqrt(dx * dx + dy * dy)
                    overlap = rsum - d
                    f = overlap * pair_k
                    ux = dx / d
                    uy = dy / d
                    fx[i] += ux * f
                    fy[i] += uy * f
                    fx[j] -= ux * f
                    fy[j] -= uy * f

        x += pos_dt * fx
        y += pos_dt * fy
        W += w_dt * fW
        W = np.clip(W, 0.3, 1.7)
        H = 2.0 - W

        for i in range(N):
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(x[i], r[i], W - r[i])  # will fix below
            y[i] = np.clip(y[i], r[i], H - r[i])

    return x, y, r, W, H


def _contact_local_search(x, y, r, W, H, time_budget, n_rounds=50):
    """
    Contact-graph local search:
    1. Identify the circle that is most "squeezed" (smallest max possible radius).
    2. Try to move it to a better position.
    3. After each move, grow all radii via binary search.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    W = float(W)
    H = float(H)

    for rnd in range(n_rounds):
        if time.time() > time_budget:
            break

        # Compute max possible radius for each circle
        max_r = np.zeros(N)
        for i in range(N):
            mr = min(x[i], W - x[i], y[i], H - y[i])
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                allowed = d - r[j]
                if allowed < mr:
                    mr = allowed
            max_r[i] = max(0.0, mr)

        # Find the most constrained circle (smallest max_r)
        i_star = int(np.argmin(max_r))

        # Try to move circle i_star to improve its max_r
        # Try moving in 8 directions + random
        best_move = None
        best_maxr = max_r[i_star]

        # 8 directions
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
        for ang in angles:
            for step in [0.05, 0.03, 0.02, 0.01, 0.005]:
                nx = x[i_star] + step * np.cos(ang)
                ny = y[i_star] + step * np.sin(ang)
                # Check validity
                if nx < r[i_star] or nx > W - r[i_star]:
                    continue
                if ny < r[i_star] or ny > H - r[i_star]:
                    continue
                valid = True
                for j in range(N):
                    if i_star == j:
                        continue
                    dx = nx - x[j]
                    dy = ny - y[j]
                    if dx*dx + dy*dy < (r[i_star] + r[j])**2 - 1e-9:
                        valid = False
                        break
                if not valid:
                    continue
                # Compute new max_r for this circle
                mr = min(nx, W - nx, ny, H - ny)
                for j in range(N):
                    if i_star == j:
                        continue
                    dx = nx - x[j]
                    dy = ny - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    allowed = d - r[j]
                    if allowed < mr:
                        mr = allowed
                if mr > best_maxr + 1e-6:
                    best_maxr = mr
                    best_move = (nx, ny)

        # Also try random moves
        for _ in range(20):
            ang = np.random.uniform(0, 2*np.pi)
            for step in [0.04, 0.02, 0.01]:
                nx = x[i_star] + step * np.cos(ang)
                ny = y[i_star] + step * np.sin(ang)
                if nx < r[i_star] or nx > W - r[i_star]:
                    continue
                if ny < r[i_star] or ny > H - r[i_star]:
                    continue
                valid = True
                for j in range(N):
                    if i_star == j:
                        continue
                    dx = nx - x[j]
                    dy = ny - y[j]
                    if dx*dx + dy*dy < (r[i_star] + r[j])**2 - 1e-9:
                        valid = False
                        break
                if not valid:
                    continue
                mr = min(nx, W - nx, ny, H - ny)
                for j in range(N):
                    if i_star == j:
                        continue
                    dx = nx - x[j]
                    dy = ny - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    allowed = d - r[j]
                    if allowed < mr:
                        mr = allowed
                if mr > best_maxr + 1e-6:
                    best_maxr = mr
                    best_move = (nx, ny)

        if best_move is not None:
            x[i_star] = best_move[0]
            y[i_star] = best_move[1]

        # Grow all radii
        r = _grow_radii(x, y, r, W, H, time_budget)

    return x, y, r


def _width_optimization(x, y, r, time_budget, n_trials=30):
    """Try different widths (perimeter=4) and pick the best."""
    best_val = 0.0
    best_state = None

    # Current width
    W0 = 2.0 - (2.0 - 1.0)  # placeholder
    # We need to figure out current W from context; pass it in
    # Actually, let's just try a range of widths
    for W in np.linspace(0.5, 1.5, n_trials):
        if time.time() > time_budget:
            break
        H = 2.0 - W
        # Scale positions to fit in [0,W] x [0,H]
        # We need to know the "natural" aspect of the packing
        # For now, just check if current positions fit
        x_s = x.copy()
        y_s = y.copy()
        # Clamp
        x_s = np.clip(x_s, r, W - r)
        y_s = np.clip(y_s, r, H - r)
        if not _is_valid(x_s, y_s, r, W, H):
            continue
        val = np.sum(r)
        if val > best_val:
            best_val = val
            best_state = (x_s.copy(), y_s.copy(), r.copy(), W, H)

    if best_state is None:
        return None
    return best_state


def _hexagonal_init(W, H, n_rows, n_cols):
    """Generate hexagonal packing initial positions."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)

    spacing_x = W / n_cols
    spacing_y = H / n_rows
    r0 = min(spacing_x, spacing_y) * 0.48

    count = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if count >= N:
                break
            cx = (col + 0.5) * spacing_x
            cy = (row + 0.5) * spacing_y
            # Hex offset
            if row % 2 == 1:
                cx += spacing_x * 0.3
            x[count] = np.clip(cx, r0, W - r0)
            y[count] = np.clip(cy, r0, H - r0)
            r[count] = r0
            count += 1

    # Fill remaining with small circles in corners/gaps
    while count < N:
        x[count] = W / 2 + (count - N//2) * 0.05
        y[count] = H / 2 + (count - N//2) * 0.05
        r[count] = 0.03
        count += 1

    return x, y, r


def _random_init(W, H, rng, min_r=0.02, max_r=0.1):
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    for i in range(N):
        r[i] = rng.uniform(min_r, max_r)
        x[i] = rng.uniform(r[i], W - r[i])
        y[i] = rng.uniform(r[i], H - r[i])
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_value = 0.0
    best_x = None
    best_y = None
    best_r = None
    best_W = 1.0
    best_H = 1.0

    rng = np.random.default_rng(12345)

    # Phase 1: Multi-start soft-body relaxation
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.65, 1.5, 0.6, 1.3, 0.75]
    inits = []

    for W0 in aspect_ratios:
        H0 = 2.0 - W0
        if H0 < 0.3:
            continue
        # Hexagonal init
        n_cols = int(np.sqrt(N * W0 / H0))
        n_rows = int(np.ceil(N / n_cols))
        x, y, r = _hexagonal_init(W0, H0, n_rows, n_cols)
        inits.append((x, y, r, W0, H0))

    # Random inits
    for _ in range(8):
        W0 = rng.uniform(0.6, 1.4)
        H0 = 2.0 - W0
        x, y, r = _random_init(W0, H0, rng)
        inits.append((x, y, r, W0, H0))

    for idx, (x0, y0, r0, W0, H0) in enumerate(inits):
        if time.time() > deadline - 30:
            break

        t_budget = deadline - (time.time() - start_time) - 25
        if t_budget < 5:
            break

        # Soft-body relaxation
        x1, y1, r1 = _soft_body_relax(x0, y0, r0, W0, H0, start_time + t_budget)

        # Grow radii
        r1 = _grow_radii(x1, y1, r1, W0, H0, start_time + t_budget)

        # Contact local search
        t_budget2 = deadline - (time.time() - start_time) - 15
        if t_budget2 > 5:
            x2, y2, r2 = _contact_local_search(x1, y1, r1, W0, H0, start_time + t_budget2)
        else:
            x2, y2, r2 = x1, y1, r1

        # Final grow
        r2 = _grow_radii(x2, y2, r2, W0, H0, start_time + t_budget2)

        if _is_valid(x2, y2, r2, W0, H0):
            val = float(np.sum(r2))
            if val > best_value:
                best_value = val
                best_x = x2.copy()
                best_y = y2.copy()
                best_r = r2.copy()
                best_W = W0
                best_H = H0

    # Phase 2: Width optimization on best result
    if best_x is not None and time.time() < deadline - 20:
        t_budget = deadline - (time.time() - start_time) - 15
        if t_budget > 5:
            for W_try in np.linspace(max(0.4, best_W - 0.3), min(1.6, best_W + 0.3), 15):
                if time.time() > start_time + t_budget:
                    break
                H_try = 2.0 - W_try
                # Rescale positions
                x_s = best_x * (W_try / best_W)
                y_s = best_y * (H_try / best_H)
                # Clamp
                x_s = np.clip(x_s, best_r, W_try - best_r)
                y_s = np.clip(y_s, best_r, H_try - best_r)
                if _is_valid(x_s, y_s, best_r, W_try, H_try):
                    # Grow
                    r_grown = _grow_radii(x_s, y_s, best_r, W_try, H_try, start_time + t_budget)
                    if _is_valid(x_s, y_s, r_grown, W_try, H_try):
                        val = float(np.sum(r_grown))
                        if val > best_value:
                            best_value = val
                            best_x = x_s.copy()
                            best_y = y_s.copy()
                            best_r = r_grown.copy()
                            best_W = W_try
                            best_H = H_try

    # Phase 3: Final intensive local search on best
    if best_x is not None and time.time() < deadline - 8:
        t_budget = deadline - (time.time() - start_time) - 5
        if t_budget > 3:
            x3, y3, r3 = _contact_local_search(best_x, best_y, best_r, best_W, best_H, start_time + t_budget, n_rounds=100)
            r3 = _grow_radii(x3, y3, r3, best_W, best_H, start_time + t_budget)
            if _is_valid(x3, y3, r3, best_W, best_H):
                val = float(np.sum(r3))
                if val > best_value:
                    best_value = val
                    best_x = x3.copy()
                    best_y = y3.copy()
                    best_r = r3.copy()

    # Fallback
    if best_x is None:
        W = 1.0
        H = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final safety shrink
    for _ in range(20):
        r_trial = best_r * (1 - 1e-8)
        if _is_valid(best_x, best_y, r_trial, best_W, best_H):
            best_r = r_trial
        else:
            break

    return np.column_stack((best_x, best_y, best_r))
