"""Circle packing in perimeter-4 rectangle: force-directed pressure simulation."""
import numpy as np
import time

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_value = 0.0
    best_result = None

    def check_validity(x, y, r, w, h, tol=1e-9):
        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        if np.any(r <= 0):
            return False, 0.0
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d2 = dx * dx + dy * dy
        rs = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        if np.any(d2[iu] < rs[iu] ** 2 - tol):
            return False, 0.0
        return True, float(np.sum(r))

    def simulate(w0, h0, time_budget, seed=0):
        """Run force-directed pressure simulation for a given rectangle."""
        rng = np.random.default_rng(seed)
        
        # Grid-based initialization
        aspect = w0 / h0
        cols = max(1, int(np.sqrt(N * aspect)))
        rows = max(1, int(np.ceil(N / cols)))
        
        x = np.empty(N)
        y = np.empty(N)
        r = np.empty(N)
        
        # Place circles in a grid
        dx_cell = w0 / cols
        dy_cell = h0 / rows
        idx = 0
        for i in range(cols):
            for j in range(rows):
                if idx >= N:
                    break
                cx = (i + 0.5) * dx_cell
                cy = (j + 0.5) * dy_cell
                # Add jitter
                x[idx] = cx + rng.uniform(-0.1 * dx_cell, 0.1 * dx_cell)
                y[idx] = cy + rng.uniform(-0.1 * dy_cell, 0.1 * dy_cell)
                # Initial radius: half the min cell dimension, scaled down
                r[idx] = 0.4 * min(dx_cell, dy_cell)
                idx += 1
        
        # Ensure initial radii are small enough to not overlap significantly
        # and within bounds
        r = np.clip(r, 0.01, 0.1)
        x = np.clip(x, r, w0 - r)
        y = np.clip(y, r, h0 - r)

        vx = np.zeros(N)
        vy = np.zeros(N)
        dt = 0.005
        damping = 0.85
        iu = np.triu_indices(N, k=1)

        def forces(x, y, r, inflate):
            fx = np.zeros(N)
            fy = np.zeros(N)
            # Wall repulsion
            fx -= np.maximum(r - x, 0.0) * 100.0
            fx += np.maximum(x + r - w0, 0.0) * 100.0
            fy -= np.maximum(r - y, 0.0) * 100.0
            fy += np.maximum(y + r - h0, 0.0) * 100.0
            
            # Weak center attraction to keep them in bounds but allow corner packing
            fx += (w0 * 0.5 - x) * 0.1
            fy += (h0 * 0.5 - y) * 0.1
            
            # Pairwise repulsion
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d2 = dx * dx + dy * dy
            rs = r[:, None] + r[None, :]
            dp = np.sqrt(np.maximum(d2[iu], 1e-12))
            overlap = np.maximum(rs[iu] - dp, 0.0)
            # Force proportional to overlap
            fp = overlap * 50.0
            ux = dx[iu] / dp
            uy = dy[iu] / dp
            for k in range(len(iu[0])):
                a = iu[0][k]; b = iu[1][k]
                fx[a] += ux[k] * fp[k]; fx[b] -= ux[k] * fp[k]
                fy[a] += uy[k] * fp[k]; fy[b] -= uy[k] * fp[k]
            return fx, fy

        def step(x, y, r, vx, vy, dt, damping):
            fx, fy = forces(x, y, r, False)
            vx = vx * damping + fx * dt
            vy = vy * damping + fy * dt
            x = x + vx * dt
            y = y + vy * dt
            x = np.clip(x, r, w0 - r)
            y = np.clip(y, r, h0 - r)
            return x, y, vx, vy

        # Phase A: Settle positions
        for it in range(1000):
            x, y, vx, vy = step(x, y, r, vx, vy, dt, damping)
            if time.time() > start_time + time_budget:
                break

        # Phase B: Inflate radii gradually
        base = r.copy()
        best_r = base.copy()
        best_x = x.copy()
        best_y = y.copy()
        best_val = float(np.sum(base))
        
        max_phases = 30
        for phase in range(max_phases):
            target_scale = 1.0 + 0.05 * phase
            for sub in range(100):
                if time.time() > start_time + time_budget:
                    break
                # Interpolate scale
                scale = 1.0 + (target_scale - 1.0) * (sub + 1) / 100.0
                r = base * scale
                
                # Add small random jitter to escape local minima
                jitter = 0.001 * (1.0 - phase / max_phases)
                x += rng.normal(0, jitter, N)
                y += rng.normal(0, jitter, N)
                
                x, y, vx, vy = step(x, y, r, vx, vy, dt, damping)
                
                # Check validity and update best
                valid, val = check_validity(x, y, r, w0, h0)
                if valid and val > best_val:
                    best_val = val
                    best_r = r.copy()
                    best_x = x.copy()
                    best_y = y.copy()
                    
        return best_x, best_y, best_r, w0, h0

    def expand_uniform(x, y, r, w, h, time_budget):
        """Binary-search uniform radius scale that stays valid."""
        lo = 1.0
        hi = 1.50
        # find a feasible hi
        for _ in range(30):
            mid = (lo + hi) / 2.0
            if check_validity(x, y, r * mid, w, h)[0]:
                lo = mid
            else:
                hi = mid
            if time.time() > start_time + time_budget:
                break
        return r * lo

    def relocate_polish(x, y, r, w, h, time_budget):
        """Nudge each circle to its max-radius position, then re-expand."""
        x = x.copy(); y = y.copy(); r = r.copy()
        iu = np.triu_indices(N, k=1)
        for rnd in range(15):
            if time.time() > start_time + time_budget:
                break
            improved = False
            for i in range(N):
                # max radius at current position
                r_wall = min(x[i], w - x[i], y[i], h - y[i])
                rmax = r_wall
                for j in range(N):
                    if i == j:
                        continue
                    d = np.hypot(x[i] - x[j], y[i] - y[j])
                    rmax = min(rmax, d - r[j])
                rmax = max(0.0, rmax)
                if rmax > r[i] + 1e-6:
                    r[i] = rmax
                    improved = True
                
                # Try to move circle to increase its max possible radius
                # or to allow others to grow
                if r[i] < rmax - 1e-5:
                    # Try moving in 8 directions
                    step_size = 0.02
                    for dx, dy in [(step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size),
                                   (step_size/np.sqrt(2), step_size/np.sqrt(2)), (-step_size/np.sqrt(2), step_size/np.sqrt(2)),
                                   (step_size/np.sqrt(2), -step_size/np.sqrt(2)), (-step_size/np.sqrt(2), -step_size/np.sqrt(2))]:
                        nx = x[i] + dx
                        ny = y[i] + dy
                        if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                            continue
                        ok = True
                        for j in range(N):
                            if i == j:
                                continue
                            d2 = (nx - x[j]) ** 2 + (ny - y[j]) ** 2
                            if d2 < (r[i] + r[j]) ** 2 - 1e-9:
                                ok = False
                                break
                        if ok:
                            x[i] = nx
                            y[i] = ny
                            improved = True
                            break
            if not improved:
                break
        return x, y, r

    # Sweep a range of aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 2.0, 0.4]
    candidates = []
    for k, w0 in enumerate(aspect_ratios):
        h0 = 2.0 - w0
        if h0 < 0.35 or w0 < 0.35:
            continue
        remaining = deadline - (time.time() - start_time)
        if remaining < 5.0:
            break
        budget = min(4.0, remaining - 2.0)
        try:
            x, y, r, w, h = simulate(w0, h0, budget, seed=1000 + k)
            # uniform expansion
            r = expand_uniform(x, y, r, w, h, deadline - (time.time() - start_time) - 3)
            # relocate polish
            x, y, r = relocate_polish(x, y, r, w, h, deadline - (time.time() - start_time) - 3)
            r = expand_uniform(x, y, r, w, h, deadline - (time.time() - start_time) - 2)
            valid, val = check_validity(x, y, r, w, h)
            if valid:
                candidates.append((val, x.copy(), y.copy(), r.copy(), w, h))
                if val > best_value:
                    best_value = val
                    best_result = (x.copy(), y.copy(), r.copy(), w, h)
        except Exception:
            continue

    # Fine-tune aspect ratio for the best candidate
    if best_result is not None:
        val, x, y, r, w, h = best_result
        # Try small variations of w
        for dw in [-0.05, -0.02, 0.0, 0.02, 0.05]:
            new_w = w + dw
            new_h = 2.0 - new_w
            if new_w < 0.35 or new_h < 0.35:
                continue
            # Scale positions and radii to fit new rectangle? 
            # No, re-simulate briefly or just try to fit.
            # Re-simulating is safer.
            remaining = deadline - (time.time() - start_time)
            if remaining < 2.0:
                break
            budget = min(1.5, remaining - 1.0)
            try:
                # Use the previous best as a seed for initialization? 
                # simulate() uses grid init. We can pass a flag to use current x,y,r as init.
                # For now, just run simulate with new dims.
                nx, ny, nr, nw, nh = simulate(new_w, new_h, budget, seed=2000)
                nr = expand_uniform(nx, ny, nr, nw, nh, deadline - (time.time() - start_time) - 2)
                nx, ny, nr = relocate_polish(nx, ny, nr, nw, nh, deadline - (time.time() - start_time) - 2)
                nr = expand_uniform(nx, ny, nr, nw, nh, deadline - (time.time() - start_time) - 1)
                valid, nval = check_validity(nx, ny, nr, nw, nh)
                if valid and nval > best_value:
                    best_value = nval
                    best_result = (nx.copy(), ny.copy(), nr.copy(), nw, nh)
            except Exception:
                continue

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result
    # Final safety shrink
    for _ in range(8):
        rt = r * (1 - 1e-9)
        if check_validity(x, y, rt, w, h)[0]:
            r = rt
        else:
            break

    return np.column_stack((x, y, r))
