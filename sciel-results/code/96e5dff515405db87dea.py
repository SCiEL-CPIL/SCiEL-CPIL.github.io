"""Circle packing in perimeter-4 rectangle: Combined Grid, Force-Relaxation, and SLSQP."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    best_result = None
    best_score = -1.0

    def validate_and_update(positions, radii, w, h):
        """Validate packing and update best if valid."""
        nonlocal best_result, best_score
        if len(positions) != N:
            return
        if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
            return
        if np.any(np.isnan(radii)) or np.any(np.isinf(radii)):
            return
        if np.any(radii < -1e-9):
            return
        # Check boundaries
        for i in range(N):
            if positions[i, 0] - radii[i] < -1e-9:
                return
            if positions[i, 0] + radii[i] > w + 1e-9:
                return
            if positions[i, 1] - radii[i] < -1e-9:
                return
            if positions[i, 1] + radii[i] > h + 1e-9:
                return
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j] - 1e-9:
                    return
        score = float(np.sum(radii))
        if score > best_score:
            best_score = score
            best_result = np.column_stack([positions, radii]).copy()

    def compute_grid_positions(rows, cols, w, h):
        """Generate grid positions for N circles."""
        if rows * cols < N:
            return None
        positions = np.zeros((N, 2))
        for i in range(N):
            row = i // cols
            col = i % cols
            if row == rows - 1:
                count_in_last_row = N - (rows - 1) * cols
                if count_in_last_row < cols:
                    k = count_in_last_row
                    x = (col + 0.5) * w / k
                else:
                    x = (col + 0.5) * w / cols
            else:
                x = (col + 0.5) * w / cols
            y = (row + 0.5) * h / rows
            positions[i] = [x, y]
        return positions

    def force_relaxation(positions, radii, w, h, steps=200, lr_pos=0.05, lr_rad=0.002, seed=42):
        """
        Force-directed relaxation to maximize sum of radii.
        1. Try to grow radii.
        2. Push apart overlapping circles.
        3. Push circles inside boundaries.
        """
        rng = np.random.RandomState(seed)
        pos = positions.copy()
        rad = radii.copy()
        
        # Ensure initial validity (shrink if needed)
        for _ in range(10):
            valid = True
            for i in range(N):
                if pos[i, 0] - rad[i] < 0 or pos[i, 0] + rad[i] > w or pos[i, 1] - rad[i] < 0 or pos[i, 1] + rad[i] > h:
                    rad[i] *= 0.9
                    valid = False
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < rad[i] + rad[j]:
                        rad[i] *= 0.9
                        rad[j] *= 0.9
                        valid = False
            if valid:
                break

        for step in range(steps):
            if time.time() > deadline - 5:
                break
            
            # 1. Attempt to grow radii (slightly random to avoid symmetry traps)
            growth = lr_rad * (1 + 0.1 * rng.randn(N))
            rad += growth
            
            # 2. Resolve Boundary Violations
            for i in range(N):
                # Left
                if pos[i, 0] - rad[i] < 0:
                    pos[i, 0] = rad[i]
                # Right
                if pos[i, 0] + rad[i] > w:
                    pos[i, 0] = w - rad[i]
                # Bottom
                if pos[i, 1] - rad[i] < 0:
                    pos[i, 1] = rad[i]
                # Top
                if pos[i, 1] + rad[i] > h:
                    pos[i, 1] = h - rad[i]
                
                # If still invalid (e.g., w < 2*rad), shrink
                if pos[i, 0] - rad[i] < 0 or pos[i, 0] + rad[i] > w or pos[i, 1] - rad[i] < 0 or pos[i, 1] + rad[i] > h:
                    rad[i] *= 0.95

            # 3. Resolve Overlaps
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    min_dist = rad[i] + rad[j]
                    
                    if dist < min_dist:
                        if dist < 1e-6:
                            # Random nudge
                            dx = rng.uniform(-1, 1)
                            dy = rng.uniform(-1, 1)
                            dist = math.sqrt(dx*dx + dy*dy)
                        
                        # Overlap amount
                        overlap = min_dist - dist
                        # Push apart along normal
                        nx = dx / dist
                        ny = dy / dist
                        
                        # Move each half of the overlap
                        push = overlap * 0.5 * lr_pos * 10 # Scale force
                        pos[i, 0] += nx * push
                        pos[i, 1] += ny * push
                        pos[j, 0] -= nx * push
                        pos[j, 1] -= ny * push
                        
                        # If still overlapping significantly, shrink radii
                        if overlap > min_dist * 0.1:
                            rad[i] *= 0.98
                            rad[j] *= 0.98

        # Final cleanup: ensure strict validity
        for _ in range(5):
            valid = True
            for i in range(N):
                if pos[i, 0] - rad[i] < 0 or pos[i, 0] + rad[i] > w or pos[i, 1] - rad[i] < 0 or pos[i, 1] + rad[i] > h:
                    rad[i] *= 0.95
                    valid = False
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < rad[i] + rad[j]:
                        rad[i] *= 0.95
                        rad[j] *= 0.95
                        valid = False
            if valid:
                break

        return pos, rad

    def run_slsqp(w_init, x0, y0, r0, max_iter=100):
        """Polish with SLSQP."""
        h = 2.0 - w_init
        if h <= 0.1 or w_init <= 0.1:
            return

        n_var = 3 * N + 1

        def objective(v):
            xs = v[0:3*N:3]
            ys = v[1:3*N:3]
            rs = v[2:3*N:3]
            w = v[3*N]
            h = 2.0 - w
            if h <= 0.1 or w <= 0.1:
                return 1e10
            
            penalty = 0.0
            # Boundary
            for i in range(N):
                if xs[i] - rs[i] < 0:
                    penalty += 100 * (xs[i] - rs[i])**2
                if xs[i] + rs[i] > w:
                    penalty += 100 * (xs[i] + rs[i] - w)**2
                if ys[i] - rs[i] < 0:
                    penalty += 100 * (ys[i] - rs[i])**2
                if ys[i] + rs[i] > h:
                    penalty += 100 * (ys[i] + rs[i] - h)**2
            
            # Overlap
            for i in range(N):
                for j in range(i + 1, N):
                    dist = math.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                    min_d = rs[i] + rs[j]
                    if dist < min_d:
                        penalty += 100 * (min_d - dist)**2
            
            return -np.sum(rs) + penalty

        bounds = []
        for i in range(N):
            bounds.append((0, 2.0))
            bounds.append((0, 2.0))
            bounds.append((0, 1.0))
        bounds.append((0.3, 1.7))

        v0 = np.zeros(n_var)
        for i in range(N):
            v0[i*3] = x0[i]
            v0[i*3+1] = y0[i]
            v0[i*3+2] = r0[i]
        v0[3*N] = w_init

        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': 1e-10})
            if res.success or res.fun < objective(v0):
                xs = res.x[0:3*N:3]
                ys = res.x[1:3*N:3]
                rs = res.x[2:3*N:3]
                w = res.x[3*N]
                h = 2.0 - w
                positions = np.column_stack([xs, ys])
                radii = rs
                validate_and_update(positions, radii, w, h)
        except Exception:
            pass

    # Phase 1: Grid-based packings with various aspect ratios
    # We try a variety of grid layouts to find good initial configurations
    candidate_configs = []
    for rows in range(1, 8):
        for cols in range(1, 8):
            if rows * cols >= N:
                candidate_configs.append((rows, cols))
    
    # Add some specific aspect ratios by adjusting w
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]

    for rows, cols in candidate_configs:
        if time.time() > deadline - 30:
            break
            
        for ar in aspect_ratios:
            # w + h = 2, w/h = ar => w = 2*ar/(1+ar), h = 2/(1+ar)
            w = 2.0 * ar / (1.0 + ar)
            h = 2.0 / (1.0 + ar)
            
            positions = compute_grid_positions(rows, cols, w, h)
            if positions is None:
                continue
            
            # Calculate max possible radius for this grid
            min_r = float('inf')
            for i in range(N):
                min_r = min(min_r, positions[i, 0], w - positions[i, 0],
                           positions[i, 1], h - positions[i, 1])
            for i in range(N):
                for j in range(i + 1, N):
                    dist = math.sqrt((positions[i, 0] - positions[j, 0])**2 + (positions[i, 1] - positions[j, 1])**2)
                    min_r = min(min_r, dist / 2.0)
            
            radii = np.full(N, min_r * 0.9) # Start slightly smaller to allow growth
            
            # Validate initial grid
            validate_and_update(positions, radii, w, h)
            
            # Run force relaxation
            if time.time() > deadline - 10:
                continue
                
            pos_rel, rad_rel = force_relaxation(positions, radii, w, h, steps=150, lr_pos=0.05, lr_rad=0.002, seed=rows*10+cols)
            validate_and_update(pos_rel, rad_rel, w, h)
            
            # Run SLSQP polish on the relaxed result
            if time.time() > deadline - 5:
                continue
            run_slsqp(w, pos_rel, rad_rel, rad_rel, max_iter=50)

    # Phase 2: Random restarts if time permits
    if time.time() < deadline - 10:
        np.random.seed(123)
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            w = np.random.uniform(0.8, 1.2)
            h = 2.0 - w
            positions = np.random.uniform(0.1, w - 0.1, (N, 2))
            positions[:, 1] = np.random.uniform(0.1, h - 0.1, N)
            radii = np.full(N, 0.05)
            
            pos_rel, rad_rel = force_relaxation(positions, radii, w, h, steps=100, lr_pos=0.05, lr_rad=0.005, seed=trial)
            validate_and_update(pos_rel, rad_rel, w, h)
            
            if time.time() > deadline - 2:
                break
            run_slsqp(w, pos_rel, rad_rel, rad_rel, max_iter=50)

    # Fallback: return a valid grid configuration if nothing better found
    if best_result is None:
        side = int(math.ceil(math.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
        best_result = np.array(cells)
        best_score = _validate_and_score(best_result) if '_validate_and_score' in globals() else float(np.sum(best_result[:, 2]))

    return best_result
