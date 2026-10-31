<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 115.0  # Safety margin before 150s limit
    best_result = None
    best_score = 0.0

    def validate_and_score(x, y, r, w, h, tol=1e-9):
        """Check if circles are valid and return sum of radii."""
        if np.any(r <= 0):
            return 0.0
        for i in range(N):
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if d < r[i] + r[j] - tol:
                    return 0.0
        return float(np.sum(r))

    def generate_initial_layout(w, h, seed_offset=0):
        """Generate a staggered grid initial layout."""
        rng = np.random.RandomState(42 + seed_offset)
        n_cols = max(1, int(np.round(np.sqrt(N * w / h))))
        n_rows = max(1, int(np.ceil(N / n_cols)))
        while n_cols * n_rows < N:
            if n_cols <= n_rows:
                n_cols += 1
            else:
                n_rows += 1
        xs = []
        ys = []
        for i in range(N):
            col = i % n_cols
            row = i // n_cols
            x = (col + 0.5) * w / n_cols
            y = (row + 0.5) * h / n_rows
            if row % 2 == 1:
                x += 0.25 * w / n_cols
                if x > w - w / n_cols * 0.25:
                    x -= 0.5 * w / n_cols
            xs.append(x)
            ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        # Add small random perturbation
        xs += rng.uniform(-0.02, 0.02, N)
        ys += rng.uniform(-0.02, 0.02, N)
        # Clip to stay in bounds
        xs = np.clip(xs, 0.05, w - 0.05)
        ys = np.clip(ys, 0.05, h - 0.05)
        # Compute initial radius
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                min_dist = min(min_dist, d)
        for i in range(N):
            min_dist = min(min_dist, xs[i], w - xs[i], ys[i], h - ys[i])
        r0 = 0.35 * min_dist
        radii = np.full(N, r0)
        return xs, ys, radii

    def soft_penalty_objective(vars_, w, h, penalty_weight):
        """Soft-body penalty objective: -sum(r) + penalty for violations."""
        x = vars_[0::3]
        y = vars_[1::3]
        r = vars_[2::3]
        
        # Base objective: maximize sum of radii
        obj = -np.sum(r)
        
        # Wall violation penalty (quadratic)
        wall_pen = 0.0
        for i in range(N):
            if x[i] < r[i]:
                wall_pen += (r[i] - x[i])**2
            if x[i] > w - r[i]:
                wall_pen += (x[i] - (w - r[i]))**2
            if y[i] < r[i]:
                wall_pen += (r[i] - y[i])**2
            if y[i] > h - r[i]:
                wall_pen += (y[i] - (h - r[i]))**2
        
        # Circle-circle overlap penalty (quadratic)
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                min_d = r[i] + r[j]
                if d < min_d:
                    overlap_pen += (min_d - d)**2
        
        # Negative radius penalty
        neg_r_pen = 0.0
        for i in range(N):
            if r[i] < 0:
                neg_r_pen += r[i]**2
        
        return obj + penalty_weight * (wall_pen + overlap_pen + neg_r_pen)

    def soft_penalty_grad(vars_, w, h, penalty_weight):
        """Gradient of the soft penalty objective."""
        x = vars_[0::3].copy()
        y = vars_[1::3].copy()
        r = vars_[2::3].copy()
        
        grad = np.zeros(3 * N)
        
        # Base objective gradient: -1 for each radius
        for i in range(N):
            grad[3*i + 2] = -1.0
        
        # Wall violation gradients
        for i in range(N):
            # x[i] < r[i]
            if x[i] < r[i]:
                viol = r[i] - x[i]
                grad[3*i] += -2 * penalty_weight * viol
                grad[3*i + 2] += 2 * penalty_weight * viol
            # x[i] > w - r[i]
            if x[i] > w - r[i]:
                viol = x[i] - (w - r[i])
                grad[3*i] += 2 * penalty_weight * viol
                grad[3*i + 2] += 2 * penalty_weight * viol
            # y[i] < r[i]
            if y[i] < r[i]:
                viol = r[i] - y[i]
                grad[3*i + 1] += -2 * penalty_weight * viol
                grad[3*i + 2] += 2 * penalty_weight * viol
            # y[i] > h - r[i]
            if y[i] > h - r[i]:
                viol = y[i] - (h - r[i])
                grad[3*i + 1] += 2 * penalty_weight * viol
                grad[3*i + 2] += 2 * penalty_weight * viol
        
        # Circle-circle overlap gradients
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    viol = min_d - d
                    # Gradient wrt x[i]
                    grad[3*i] += 2 * penalty_weight * viol * dx / d
                    grad[3*j] += -2 * penalty_weight * viol * dx / d
                    # Gradient wrt y[i]
                    grad[3*i + 1] += 2 * penalty_weight * viol * dy / d
                    grad[3*j + 1] += -2 * penalty_weight * viol * dy / d
                    # Gradient wrt r[i] and r[j]
                    grad[3*i + 2] += 2 * penalty_weight * viol
                    grad[3*j + 2] += 2 * penalty_weight * viol
        
        # Negative radius gradients
        for i in range(N):
            if r[i] < 0:
                grad[3*i + 2] += 2 * penalty_weight * r[i]
        
        return grad

    # Try different aspect ratios with soft-body optimization
    aspect_ratios = [1.0, 1.2, 1.5, 2.0, 0.8, 0.6, 1.3, 1.7, 2.5, 0.5, 1.1, 1.4, 1.8, 0.7, 2.2]
    
    for ar_idx, ar in enumerate(aspect_ratios):
        if time.time() > deadline - 15:
            break

        w = 2.0 / (1.0 + ar)
        h = 2.0 - w

        # Multiple restarts with different initial conditions
        for restart in range(3):
            if time.time() > deadline - 10:
                break

            xs, ys, radii = generate_initial_layout(w, h, seed_offset=ar_idx * 10 + restart)

            x0 = np.zeros(3 * N)
            for i in range(N):
                x0[3*i] = xs[i]
                x0[3*i+1] = ys[i]
                x0[3*i+2] = radii[i]

            # Simulated annealing style: start with low penalty, increase
            penalty_schedule = [1.0, 10.0, 50.0, 200.0, 500.0]
            
            current_vars = x0.copy()
            for pw in penalty_schedule:
                if time.time() > deadline - 8:
                    break
                
                try:
                    result = minimize(
                        soft_penalty_objective, current_vars,
                        jac=soft_penalty_grad,
                        args=(w, h, pw),
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-8}
                    )
                    current_vars = result.x
                except Exception:
                    continue

            # Validate final result
            x = current_vars[0::3]
            y = current_vars[1::3]
            r = current_vars[2::3]
            
            # Clip radii to be positive
            r = np.maximum(r, 0.0)
            
            score = validate_and_score(x, y, r, w, h)
            
            if score > best_score:
                best_score = score
                best_result = np.column_stack([x, y, r])

    # Fallback: simple grid in a square of perimeter 4 (w=h=1)
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_result
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    deadline = time.time() + 115.0  # Safety margin before 150s limit
    best_result = None
    best_score = 0.0

    def validate_and_score(x, y, r, w, h, tol=1e-9):
        """Check if circles are valid and return sum of radii."""
        if np.any(r <= 0):
            return 0.0
        for i in range(N):
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if d < r[i] + r[j] - tol:
                    return 0.0
        return float(np.sum(r))

    def compute_radii_from_grid(cx, cy, w, h, grid_res):
        """
        Given circle centers on a grid, compute the maximum feasible radius
        for each circle based on distance to walls and other circles.
        """
        radii = np.zeros(N)
        for i in range(N):
            # Distance to walls
            r_wall = min(cx[i], w - cx[i], cy[i], h - cy[i])
            # Distance to other circles
            r_circles = np.inf
            for j in range(N):
                if i == j:
                    continue
                d = np.sqrt((cx[i] - cx[j])**2 + (cy[i] - cy[j])**2)
                # If circles touch, each can have at most d/2 radius
                # But we want to maximize sum, so use d/2 as upper bound
                r_circles = min(r_circles, d / 2.0)
            radii[i] = min(r_wall, r_circles)
        return radii

    def grid_simulated_annealing(w, h, grid_res, T_init, T_final, n_steps, rng):
        """
        Simulated annealing on a discrete grid.
        Circles are placed at grid cell centers.
        Moves: pick a circle and move it to a random nearby grid cell.
        """
        # Grid cell size
        cell_w = w / grid_res
        cell_h = h / grid_res
        
        # Initial placement: random distinct grid cells
        all_cells = [(i, j) for i in range(grid_res) for j in range(grid_res)]
        placed_cells = rng.sample(all_cells, N)
        cx = np.array([(i + 0.5) * cell_w for i, j in placed_cells])
        cy = np.array([(j + 0.5) * cell_h for i, j in placed_cells])
        
        # Track which cell each circle occupies
        cell_occupied = np.zeros((grid_res, grid_res), dtype=bool)
        for i in range(N):
            ci = int(cx[i] / cell_w)
            cj = int(cy[i] / cell_h)
            cell_occupied[ci, cj] = True
        
        radii = compute_radii_from_grid(cx, cy, w, h, grid_res)
        current_score = np.sum(radii)
        
        best_cx = cx.copy()
        best_cy = cy.copy()
        best_radii = radii.copy()
        best_local_score = current_score
        
        for step in range(n_steps):
            if time.time() > deadline - 5:
                break
            
            # Temperature schedule (geometric cooling)
            T = T_init * (T_final / T_init) ** (step / max(1, n_steps - 1))
            
            # Pick a random circle to move
            i = rng.randint(N)
            ci = int(cx[i] / cell_w)
            cj = int(cy[i] / cell_h)
            
            # Pick a random nearby cell (within radius 2)
            move_r = 2
            neighbors = []
            for di in range(-move_r, move_r + 1):
                for dj in range(-move_r, move_r + 1):
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < grid_res and 0 <= nj < grid_res:
                        if not cell_occupied[ni, nj] or (ni == ci and nj == cj):
                            neighbors.append((ni, nj))
            
            if not neighbors:
                continue
            
            ni, nj = neighbors[rng.randint(len(neighbors))]
            
            if ni == ci and nj == cj:
                continue
            
            # Compute new score
            old_cx_i = cx[i]
            old_cy_i = cy[i]
            cx[i] = (ni + 0.5) * cell_w
            cy[i] = (nj + 0.5) * cell_h
            
            new_radii = compute_radii_from_grid(cx, cy, w, h, grid_res)
            new_score = np.sum(new_radii)
            
            delta = new_score - current_score
            
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                # Accept move
                cell_occupied[ci, cj] = False
                cell_occupied[ni, nj] = True
                current_score = new_score
                radii = new_radii
                
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_cx = cx.copy()
                    best_cy = cy.copy()
                    best_radii = radii.copy()
            else:
                # Reject move
                cx[i] = old_cx_i
                cy[i] = old_cy_i
        
        return best_cx, best_cy, best_radii, best_local_score

    def local_refine(cx, cy, w, h, n_iters=50):
        """
        Local refinement: iteratively adjust each circle's position slightly
        to increase its radius (and thus the sum).
        """
        cx = cx.copy()
        cy = cy.copy()
        
        for _ in range(n_iters):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(N):
                # Try small perturbations
                for dx in [-0.01, 0, 0.01]:
                    for dy in [-0.01, 0, 0.01]:
                        if dx == 0 and dy == 0:
                            continue
                        nx = cx[i] + dx
                        ny = cy[i] + dy
                        if nx < 0 or nx > w or ny < 0 or ny > h:
                            continue
                        
                        old_cx_i = cx[i]
                        old_cy_i = cy[i]
                        cx[i] = nx
                        cy[i] = ny
                        
                        new_radii = compute_radii_from_grid(cx, cy, w, h, grid_res=1)
                        new_score = np.sum(new_radii)
                        
                        old_radii = compute_radii_from_grid(cx, cy, w, h, grid_res=1)
                        old_score = np.sum(old_radii)
                        
                        # Restore for comparison
                        cx[i] = old_cx_i
                        cy[i] = old_cy_i
                        old_radii = compute_radii_from_grid(cx, cy, w, h, grid_res=1)
                        old_score = np.sum(old_radii)
                        
                        # Re-apply new position
                        cx[i] = nx
                        cy[i] = ny
                        new_radii = compute_radii_from_grid(cx, cy, w, h, grid_res=1)
                        new_score = np.sum(new_radii)
                        
                        if new_score > old_score + 1e-10:
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        
        radii = compute_radii_from_grid(cx, cy, w, h, grid_res=1)
        return cx, cy, radii

    # Try different aspect ratios and grid resolutions
    aspect_ratios = [1.0, 1.2, 1.5, 2.0, 0.8, 0.6, 1.3, 1.7, 2.5, 0.5, 1.1, 1.4, 1.8, 0.7, 2.2]
    grid_resolutions = [8, 10, 12, 14]
    
    rng = np.random.RandomState(42)
    
    for ar_idx, ar in enumerate(aspect_ratios):
        if time.time() > deadline - 20:
            break

        w = 2.0 / (1.0 + ar)
        h = 2.0 - w

        for grid_res in grid_resolutions:
            if time.time() > deadline - 15:
                break
            
            # Simulated annealing parameters
            T_init = 0.5
            T_final = 0.001
            n_steps = 5000
            
            cx, cy, radii, local_score = grid_simulated_annealing(
                w, h, grid_res, T_init, T_final, n_steps, rng
            )
            
            # Local refinement
            cx_ref, cy_ref, radii_ref = local_refine(cx, cy, w, h, n_iters=30)
            
            score = validate_and_score(cx_ref, cy_ref, radii_ref, w, h)
            
            if score > best_score:
                best_score = score
                best_result = np.column_stack([cx_ref, cy_ref, radii_ref])

    # Fallback: simple grid in a square of perimeter 4 (w=h=1)
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
