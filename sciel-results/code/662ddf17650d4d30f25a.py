"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time

N = 26

def construct():
    deadline = time.time() + 115  # safety margin before 120s kill
    rng = np.random.default_rng(42)
    
    # --- Phase 1: Initial layout ---
    # Use a structured grid with some perturbation, then optimize
    # Start with a 5x5 grid (25 cells) + 1 extra, or 6x5=30 pick 26
    # Better: use a hexagonal-ish pattern
    
    # Create initial positions on a grid
    cols = 6
    rows = 5
    total = cols * rows
    x_grid, y_grid = np.meshgrid(
        np.linspace(0.5/cols, 1 - 0.5/cols, cols),
        np.linspace(0.5/rows, 1 - 0.5/rows, rows)
    )
    x0 = x_grid.flatten()[:N]
    y0 = y_grid.flatten()[:N]
    
    # Initial radii: small, based on grid spacing
    r0 = np.full(N, 0.5 / cols - 0.02)
    
    # --- Phase 2: Simulated Annealing ---
    def compute_penalty(x, y, r, penalty_w=10.0):
        """Compute total penalty: overlap + boundary violations."""
        # Boundary violations
        pen = 0.0
        for i in range(N):
            if x[i] < r[i]:
                pen += penalty_w * (r[i] - x[i]) ** 2
            if x[i] > 1 - r[i]:
                pen += penalty_w * (x[i] - (1 - r[i])) ** 2
            if y[i] < r[i]:
                pen += penalty_w * (r[i] - y[i]) ** 2
            if y[i] > 1 - r[i]:
                pen += penalty_w * (y[i] - (1 - r[i])) ** 2
        # Overlap violations
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    pen += penalty_w * (min_dist - dist) ** 2
        return pen
    
    def objective(x, y, r, penalty_w=10.0):
        """Objective: sum of radii - penalty."""
        return np.sum(r) - compute_penalty(x, y, r, penalty_w)
    
    # Annealing parameters
    T_start = 0.05
    T_end = 0.0001
    n_steps = 50000
    
    x = x0.copy()
    y = y0.copy()
    r = r0.copy()
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_obj = objective(x, y, r, 10.0)
    
    for step in range(n_steps):
        if time.time() > deadline - 30:
            break
        
        # Temperature schedule
        T = T_start * (T_end / T_start) ** (step / n_steps)
        
        # Propose move: pick random circle, perturb position and/or radius
        i = rng.integers(N)
        
        # Move type: 0=position x, 1=position y, 2=radius
        move_type = rng.integers(3)
        
        old_x, old_y, old_r = x[i], y[i], r[i]
        
        if move_type == 0:
            x[i] += rng.normal(0, 0.02)
            x[i] = np.clip(x[i], 0, 1)
        elif move_type == 1:
            y[i] += rng.normal(0, 0.02)
            y[i] = np.clip(y[i], 0, 1)
        else:
            r[i] += rng.normal(0, 0.01)
            r[i] = np.clip(r[i], 0.005, 0.5)
        
        old_obj = objective(x, y, r, 10.0)
        new_obj = objective(x, y, r, 10.0)
        delta = new_obj - old_obj
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            # Accept
            if new_obj > best_obj:
                best_obj = new_obj
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        else:
            # Reject
            x[i], y[i], r[i] = old_x, old_y, old_r
    
    x, y, r = best_x, best_y, best_r
    
    # --- Phase 3: Local polish ---
    # Try to increase radii while maintaining feasibility
    for polish_iter in range(200):
        if time.time() > deadline - 10:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline - 10:
                break
            # Try to increase radius of circle i
            old_r = r[i]
            new_r = old_r * 1.02
            if new_r > 0.5:
                new_r = 0.5
            r[i] = new_r
            if compute_penalty(x, y, r, 1.0) < 1e-8:
                improved = True
            else:
                r[i] = old_r
        
        if not improved:
            break
    
    # --- Phase 4: Final validation and repair ---
    # Ensure all circles are within bounds and non-overlapping
    # If any violations remain, shrink radii slightly
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        pen = compute_penalty(x, y, r, 1.0)
        if pen < 1e-10:
            break
        # Find worst-violating circle and shrink it
        max_viol = 0
        worst_i = 0
        for i in range(N):
            # Check boundary
            b_pen = 0
            if x[i] < r[i]:
                b_pen += (r[i] - x[i]) ** 2
            if x[i] > 1 - r[i]:
                b_pen += (x[i] - (1 - r[i])) ** 2
            if y[i] < r[i]:
                b_pen += (r[i] - y[i]) ** 2
            if y[i] > 1 - r[i]:
                b_pen += (y[i] - (1 - r[i])) ** 2
            # Check overlaps
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j]:
                    b_pen += (r[i] + r[j] - dist) ** 2
            if b_pen > max_viol:
                max_viol = b_pen
                worst_i = i
        r[worst_i] *= 0.98
        # Also move it toward center if near boundary
        if x[worst_i] < r[worst_i]:
            x[worst_i] = r[worst_i] + 0.001
        if x[worst_i] > 1 - r[worst_i]:
            x[worst_i] = 1 - r[worst_i] - 0.001
        if y[worst_i] < r[worst_i]:
            y[worst_i] = r[worst_i] + 0.001
        if y[worst_i] > 1 - r[worst_i]:
            y[worst_i] = 1 - r[worst_i] - 0.001
    
    # Final safety: clip to valid range
    for i in range(N):
        r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i])
        r[i] = max(r[i], 0.001)
    
    result = np.column_stack([x, y, r])
    return result
