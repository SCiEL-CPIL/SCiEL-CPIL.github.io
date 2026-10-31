"""Circle packing in perimeter-4 rectangle: Simulated Annealing with Robust Repair."""
import time
import numpy as np

N = 21
DEADLINE_SECONDS = 140.0  # Safety margin


def construct():
    start_time = time.time()
    best_value = -np.inf
    best_result = None

    def elapsed():
        return time.time() - start_time

    def is_feasible(circles, width, height, tol=1e-8):
        """Check if circles are within bounds and do not overlap."""
        if circles.shape[0] != N:
            return False
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]

        if np.any(rs < 1e-6):
            return False
        if np.any(xs < -tol) or np.any(ys < -tol):
            return False
        if np.any(xs + rs > width + tol) or np.any(ys + rs > height + tol):
            return False

        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dists_sq = dx**2 + dy**2
        min_dists = rs[:, None] + rs[None, :]
        mask = ~np.eye(N, dtype=bool)
        
        # Check for overlaps: dist^2 < (r_i + r_j)^2
        # We use a small tolerance to avoid floating point issues
        if np.any(dists_sq[mask] < (min_dists[mask] - tol)**2):
            return False

        return True

    def sum_radii(circles):
        return np.sum(circles[:, 2])

    def repair(circles, width, height, max_iters=200):
        """
        Deterministically repair overlaps and boundary violations.
        Returns a feasible configuration if possible.
        """
        circles = circles.copy()
        xs, ys, rs = circles[:, 0].copy(), circles[:, 1].copy(), circles[:, 2].copy()
        
        # Ensure radii are positive
        rs = np.maximum(rs, 1e-4)
        
        # Initial clamp to boundaries
        xs = np.clip(xs, rs, width - rs)
        ys = np.clip(ys, rs, height - rs)
        
        for _ in range(max_iters):
            if elapsed() > DEADLINE_SECONDS:
                break
                
            # Check feasibility
            if is_feasible(np.column_stack((xs, ys, rs)), width, height):
                circles[:, 0] = xs
                circles[:, 1] = ys
                circles[:, 2] = rs
                return circles

            # Resolve overlaps
            # We will iterate through pairs and push apart if overlapping
            # To be efficient, we can compute all pairwise distances
            dx = xs[:, None] - xs[None, :]
            dy = ys[:, None] - ys[None, :]
            dists = np.sqrt(dx**2 + dy**2)
            min_dists = rs[:, None] + rs[None, :]
            mask = ~np.eye(N, dtype=bool)
            
            # Find violations
            violations = min_dists - dists
            violations[~mask] = 0
            
            if np.max(violations) < 1e-9:
                # No overlaps, just clamp and return
                xs = np.clip(xs, rs, width - rs)
                ys = np.clip(ys, rs, height - rs)
                circles[:, 0] = xs
                circles[:, 1] = ys
                circles[:, 2] = rs
                return circles

            # Apply corrections
            # We use a simple gradient-like step for overlaps
            # Force on i from j is proportional to overlap * unit_vector
            forces_x = np.zeros(N)
            forces_y = np.zeros(N)
            
            # Only process pairs with significant violation to save time
            # Get indices of violating pairs
            viol_idx = np.where(violations > 1e-8)
            if len(viol_idx[0]) == 0:
                break
                
            for i, j in zip(viol_idx[0], viol_idx[1]):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 1e-9:
                    # Random direction if centers coincide
                    angle = np.random.uniform(0, 2*np.pi)
                    ux, uy = np.cos(angle), np.sin(angle)
                else:
                    ux, uy = dx[i, j] / d, dy[i, j] / d
                    
                # Push apart by half the overlap
                push = violations[i, j] * 0.5
                forces_x[i] -= ux * push
                forces_y[i] -= uy * push
                forces_x[j] += ux * push
                forces_y[j] += uy * push
            
            # Apply forces with a step size
            step = 0.5
            xs += forces_x * step
            ys += forces_y * step
            
            # Clamp to boundaries
            xs = np.clip(xs, rs, width - rs)
            ys = np.clip(ys, rs, height - rs)

        # Final clamp and check
        xs = np.clip(xs, rs, width - rs)
        ys = np.clip(ys, rs, height - rs)
        circles[:, 0] = xs
        circles[:, 1] = ys
        circles[:, 2] = rs
        
        # If still not feasible, shrink radii slightly to force feasibility
        if not is_feasible(circles, width, height):
            # This is a fallback: shrink radii until feasible
            scale = 1.0
            for _ in range(50):
                scale *= 0.95
                new_rs = rs * scale
                new_xs = np.clip(xs, new_rs, width - new_rs)
                new_ys = np.clip(ys, new_rs, height - new_rs)
                if is_feasible(np.column_stack((new_xs, new_ys, new_rs)), width, height):
                    circles[:, 0] = new_xs
                    circles[:, 1] = new_ys
                    circles[:, 2] = new_rs
                    break
        
        return circles

    def generate_seed(width, height):
        """Generate a random initial configuration."""
        # Start with a grid-like structure but add randomness
        cols = int(np.ceil(np.sqrt(N * width / height)))
        rows = int(np.ceil(N / cols))
        
        cw = width / cols
        ch = height / rows
        
        circles = np.zeros((N, 3))
        idx = 0
        for j in range(rows):
            for i in range(cols):
                if idx >= N:
                    break
                # Add some jitter
                jitter_x = np.random.uniform(-0.2, 0.2) * cw
                jitter_y = np.random.uniform(-0.2, 0.2) * ch
                cx = (i + 0.5) * cw + jitter_x
                cy = (j + 0.5) * ch + jitter_y
                # Estimate radius based on cell size
                r = 0.5 * min(cw, ch) * 0.9
                circles[idx] = [cx, cy, r]
                idx += 1
        
        # Ensure at least N circles
        while idx < N:
            circles[idx] = [np.random.uniform(0.1, width-0.1), np.random.uniform(0.1, height-0.1), 0.05]
            idx += 1
            
        return circles

    def simulated_annealing(width, height, initial_temp=0.1, cooling_rate=0.9995, max_iters=5000):
        """Run Simulated Annealing to optimize circle positions and radii."""
        circles = generate_seed(width, height)
        # Initial repair to start from a feasible state if possible
        circles = repair(circles, width, height)
        
        current_value = sum_radii(circles) if is_feasible(circles, width, height) else -np.inf
        best_circles = circles.copy()
        best_value = current_value
        
        temp = initial_temp
        step_size = 0.05
        
        for it in range(max_iters):
            if elapsed() > DEADLINE_SECONDS:
                break
                
            # Generate new candidate
            new_circles = circles.copy()
            
            # Randomly perturb positions and radii
            # Positions
            mask_pos = np.random.random(N) < 0.5
            new_circles[mask_pos, 0] += np.random.normal(0, step_size, N)[mask_pos]
            new_circles[mask_pos, 1] += np.random.normal(0, step_size, N)[mask_pos]
            
            # Radii
            mask_rad = np.random.random(N) < 0.3
            new_circles[mask_rad, 2] *= (1 + np.random.normal(0, 0.1, N)[mask_rad])
            
            # Clamp radii
            max_r = min(width, height) / 2.0
            new_circles[:, 2] = np.clip(new_circles[:, 2], 1e-4, max_r)
            
            # Clamp positions to valid range given radii (approx)
            new_circles[:, 0] = np.clip(new_circles[:, 0], new_circles[:, 2], width - new_circles[:, 2])
            new_circles[:, 1] = np.clip(new_circles[:, 1], new_circles[:, 2], height - new_circles[:, 2])
            
            # Repair the new candidate
            new_circles = repair(new_circles, width, height)
            
            if is_feasible(new_circles, width, height):
                new_value = sum_radii(new_circles)
            else:
                new_value = -np.inf
            
            # Accept or reject
            if new_value > current_value:
                accept = True
            else:
                delta = new_value - current_value
                if delta < 0:
                    prob = np.exp(delta / temp)
                    accept = np.random.random() < prob
                else:
                    accept = True
            
            if accept:
                circles = new_circles
                current_value = new_value
                if current_value > best_value:
                    best_value = current_value
                    best_circles = circles.copy()
            
            # Cool down
            temp *= cooling_rate
            # Adapt step size
            if accept:
                step_size *= 1.01
            else:
                step_size *= 0.99
            step_size = np.clip(step_size, 0.001, 0.2)

        return best_circles, best_value

    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 1.1, 0.9]
    
    for aspect in aspect_ratios:
        if elapsed() > DEADLINE_SECONDS - 5:
            break

        height = 2.0 / (1.0 + aspect)
        width = 2.0 * aspect / (1.0 + aspect)

        if width < 0.2 or height < 0.2:
            continue

        # Run SA
        try:
            result_circles, result_value = simulated_annealing(width, height)
            
            if is_feasible(result_circles, width, height):
                if result_value > best_value:
                    best_value = result_value
                    best_result = result_circles.copy()
        except Exception:
            continue

    # Fallback if no valid result
    if best_result is None:
        width, height = 1.0, 1.0
        # Simple grid
        side = int(np.ceil(np.sqrt(N)))
        cw = width / side
        ch = height / side
        r = 0.5 * min(cw, ch) * 0.9
        best_result = np.zeros((N, 3))
        idx = 0
        for j in range(side):
            for i in range(side):
                if idx >= N:
                    break
                best_result[idx] = [(i + 0.5) * cw, (j + 0.5) * ch, r]
                idx += 1
        # Ensure feasibility of fallback
        best_result = repair(best_result, width, height)
        if not is_feasible(best_result, width, height):
             # Last resort: tiny circles
             r = 0.01
             best_result = np.zeros((N, 3))
             for i in range(N):
                 best_result[i] = [r + i * (width - 2*r) / (N-1), height/2, r]

    return best_result
