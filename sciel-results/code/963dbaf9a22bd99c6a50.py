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

    def force_relaxation(positions, radii, w, h, steps=100, lr_pos=0.05, lr_rad=0.005):
        """
        Simple force-directed relaxation to maximize sum of radii.
        1. Try to grow radii.
        2. Push apart overlapping circles.
        3. Push circles inside boundaries.
        """
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
            # 1. Attempt to grow radii
            rad += lr_rad
            
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
                            dx = np.random.uniform(-1, 1)
                            dy = np.random.uniform(-1, 1)
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

    def run_slsqp(w_init, x0, y0, r0):
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
        bounds.append((0.2, 1.8))

        v0 = np.zeros(n_var)
        for i in range(N):
            v0[i*3] = x0[i]
            v0[i*3+1] = y0[i]
            v0[i*3+2] = r0[i]
        v0[3*N] = w_init

        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-10})
            
            if res.fun < objective(v0):
                xs = res.x[0:3*N:3]
                ys = res.x[1:3*N:3]
                rs = res.x[2:3*N:3]
                w = res.x[3*N]
                h = 2.0 - w
                validate_and_update(xs, rs, w, h)
        except Exception:
            pass

    # Phase 1: Generate diverse starts and relax
    # Aspect ratios: w/(w+h) = w/2. w in [0.2, 1.8]
    aspect_ratios = np.linspace(0.3, 1.7, 15)
    
    # Grids to try for each aspect ratio
    grid_configs = [
        (3, 7), # 21
        (4, 6), # 24 -> 21
        (5, 5), # 25 -> 21
        (2, 11),# 22 -> 21
        (1, 21),# 21
        (7, 3), # 21
        (6, 4), # 24
        (11, 2),# 22
        (21, 1) # 21
    ]

    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
        
        w = ar
        h = 2.0 - w
        
        # Try a few grid layouts for this aspect ratio
        # Select grids that are "close" to the aspect ratio
        # ar = w/h. If ar is close to 1, square grids are good.
        # If ar is large (wide), rows < cols.
        
        for rows, cols in grid_configs:
            if time.time() > deadline - 5:
                break
            
            # Quick check: if grid aspect is very different from rect aspect, skip?
            # Actually, force relaxation can fix it, but it's better to start close.
            grid_ar = cols / rows
            if abs(grid_ar - (w/h)) > 2.0:
                continue

            positions = compute_grid_positions(rows, cols, w, h)
            if positions is None:
                continue
            
            # Initialize with small radii
            # Estimate max possible radius from grid spacing
            min_spacing = min(w/cols, h/rows)
            r_init = np.full(N, min_spacing * 0.4)
            
            # Relax
            pos_rel, rad_rel = force_relaxation(positions, r_init, w, h, steps=50)
            
            # Validate relaxed result
            validate_and_update(pos_rel, rad_rel, w, h)
            
            # Polish with SLSQP
            if time.time() < deadline - 20:
                run_slsqp(w, pos_rel[:, 0], pos_rel[:, 1], rad_rel)

    # Fallback
    if best_result is None:
        # Simple square packing
        side = int(math.ceil(math.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
        return np.array(cells)

    return best_result
