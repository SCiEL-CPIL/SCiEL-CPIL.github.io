"""
Circle packing in perimeter-4 rectangle: 21 circles.
Uses a Physics-based Force-Directed Relaxation approach:
1. Initialize circles in a grid or hex lattice.
2. Use molecular dynamics (repulsive forces) to resolve overlaps and fit into the rectangle.
3. Iteratively grow radii while maintaining validity via local force relaxation.
4. Optimize aspect ratio by simulating pressure on the walls.
"""
import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def check_validity(x, y, r, w, tol=1e-9):
        h = 2.0 - w
        if w <= 0.1 or h <= 0.1:
            return False, 0.0
        if np.any(r <= 0):
            return False, 0.0
        
        # Boundary checks
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
            
        # Overlap checks
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum * r_sum - tol:
                    return False, 0.0
        return True, float(np.sum(r))

    def relax_forces(x, y, r, w, iterations=200, dt=0.01):
        """
        Force-directed relaxation to resolve overlaps and boundary violations.
        Returns updated x, y.
        """
        h = 2.0 - w
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        
        # Adaptive step size
        current_dt = dt
        
        for _ in range(iterations):
            if time.time() > deadline - 1.0:
                break
                
            fx = np.zeros(N)
            fy = np.zeros(N)
            max_force = 0.0

            # 1. Wall Forces
            # Push circles back if they are outside
            for i in range(N):
                if x[i] < r[i]:
                    f = (r[i] - x[i]) * 50.0
                    fx[i] += f
                if x[i] > w - r[i]:
                    f = (x[i] - (w - r[i])) * 50.0
                    fx[i] -= f
                if y[i] < r[i]:
                    f = (r[i] - y[i]) * 50.0
                    fy[i] += f
                if y[i] > h - r[i]:
                    f = (y[i] - (h - r[i])) * 50.0
                    fy[i] -= f

            # 2. Pairwise Repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d2 = dx * dx + dy * dy
                    d = np.sqrt(d2) if d2 > 1e-12 else 1e-12
                    r_sum = r[i] + r[j]
                    if d < r_sum:
                        # Spring force proportional to overlap
                        overlap = r_sum - d
                        force_mag = overlap * 10.0
                        if d < 1e-9:
                            # Random direction if perfectly overlapping
                            ux = np.random.uniform(-1, 1)
                            uy = np.random.uniform(-1, 1)
                            norm = np.sqrt(ux*ux + uy*uy)
                            if norm > 0:
                                ux /= norm
                                uy /= norm
                        else:
                            ux = dx / d
                            uy = dy / d
                        fx[i] += ux * force_mag
                        fy[i] += uy * force_mag
                        fx[j] -= ux * force_mag
                        fy[j] -= uy * force_mag
                        max_force = max(max_force, force_mag)

            # Update positions
            # Damping helps stability
            x += current_dt * fx
            y += current_dt * fy
            
            # Clamp to bounds to prevent explosion
            for i in range(N):
                x[i] = np.clip(x[i], 0, w)
                y[i] = np.clip(y[i], 0, h)
                
            # Adaptive timestep: reduce if forces are high
            if max_force > 0.5:
                current_dt = max(0.001, current_dt * 0.8)
            else:
                current_dt = min(0.05, current_dt * 1.05)

        return x, y

    def optimize_radius(x, y, r, w, time_limit=5.0):
        """
        Try to increase radii.
        Strategy: Iteratively try to scale all radii by a small factor,
        then relax positions. If valid, keep.
        """
        h = 2.0 - w
        best_r = r.copy()
        best_val = np.sum(r)
        
        # Initial relaxation
        x, y = relax_forces(x, y, r, w, iterations=100)
        valid, val = check_validity(x, y, r, w)
        if not valid:
             return None

        while time.time() < time_limit:
            # Try to grow radii by 0.1%
            new_r = r * 1.001
            # Relax with new radii
            x_new, y_new = relax_forces(x, y, new_r, w, iterations=150)
            
            valid, val_new = check_validity(x_new, y_new, new_r, w)
            if valid:
                r = new_r
                x = x_new
                y = y_new
                best_r = r.copy()
                best_val = val_new
            else:
                # Try smaller growth
                new_r = r * 1.0002
                x_new, y_new = relax_forces(x, y, new_r, w, iterations=100)
                valid, val_new = check_validity(x_new, y_new, new_r, w)
                if valid:
                    r = new_r
                    x = x_new
                    y = y_new
                    best_r = r.copy()
                    best_val = val_new
                else:
                    break
                    
        return x, y, best_r, w

    def optimize_aspect_ratio(x, y, r, w, time_limit=10.0):
        """
        Optimize width w.
        We can try different w values and see which allows larger radii.
        """
        best_result = (x.copy(), y.copy(), r.copy(), w, np.sum(r))
        
        # Explore a range of aspect ratios
        # The optimal aspect ratio depends on the packing.
        # Let's try perturbing w and re-optimizing radii.
        current_best_val = np.sum(r)
        
        for _ in range(20):
            if time.time() > time_limit:
                break
                
            # Perturb w
            dw = np.random.uniform(-0.1, 0.1)
            new_w = np.clip(w + dw, 0.5, 1.5)
            
            # Scale positions to fit new width roughly
            scale_x = new_w / w
            # Keep y scale same for now, or scale y to maintain aspect? 
            # Actually, h = 2 - w. If w increases, h decreases.
            # Simple approach: just project positions into new box and relax.
            x_new = x * scale_x
            y_new = y * ( (2.0 - new_w) / (2.0 - w) )
            
            # Clamp
            for i in range(N):
                x_new[i] = np.clip(x_new[i], r[i], new_w - r[i])
                y_new[i] = np.clip(y_new[i], r[i], (2.0 - new_w) - r[i])
            
            # Relax and grow radii
            res = optimize_radius(x_new, y_new, r.copy(), new_w, time_limit=2.0)
            if res is not None:
                rx, ry, rr, rw = res
                val = np.sum(rr)
                if val > current_best_val:
                    current_best_val = val
                    best_result = (rx, ry, rr, rw, val)
                    w = rw
                    x = rx
                    y = ry
                    r = rr
                    # Restart search around new best
                    continue
            # If no improvement, maybe try a smaller step or different direction
            # For simplicity, we just accept the perturbation if it's close
            if abs(val - current_best_val) < 1e-6:
                 pass

        return best_result[0], best_result[1], best_result[2], best_result[3]

    def run_single_trial(w_init, time_budget):
        """
        Run a full optimization cycle for a given initial aspect ratio.
        """
        h_init = 2.0 - w_init
        
        # 1. Initialization: Grid
        # Estimate radius
        # Area per circle ~ (w*h)/N
        # r ~ sqrt(area/(pi)) * packing_efficiency
        r_est = np.sqrt((w_init * h_init) / N / np.pi) * 0.9
        r_est = min(r_est, 0.2) # Cap to avoid absurdly large initial guesses
        
        # Grid layout
        cols = int(np.sqrt(N * w_init / h_init))
        if cols < 1: cols = 1
        rows = int(np.ceil(N / cols))
        
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            row, col = divmod(i, cols)
            # Center of cell
            cx = (col + 0.5) * (w_init / cols)
            cy = (row + 0.5) * (h_init / rows)
            x[i] = cx
            y[i] = cy
            # Initial radius slightly smaller than cell size to allow movement
            cell_w = w_init / cols
            cell_h = h_init / rows
            r[i] = min(cell_w, cell_h) * 0.45
            
        # 2. Initial Relaxation
        x, y = relax_forces(x, y, r, w_init, iterations=500)
        
        # 3. Grow Radii
        res = optimize_radius(x, y, r, w_init, time_limit=time_budget * 0.5)
        if res is None:
             return None
        x, y, r, w = res
        val = np.sum(r)
        
        # 4. Optimize Aspect Ratio
        if time.time() < deadline - 5.0:
             x, y, r, w = optimize_aspect_ratio(x, y, r, w, time_limit=time_budget * 0.4)
             val = np.sum(r)
             
        # 5. Final Polish
        x, y = relax_forces(x, y, r, w, iterations=200)
        valid, val_final = check_validity(x, y, r, w)
        if valid:
            return x, y, r, w, val_final
        else:
            return None

    best_value = 0.0
    best_result = None

    # Try multiple initial aspect ratios
    # The optimal aspect ratio is unknown, so we sample a range.
    # For N=21, square (1.0) or slightly rectangular are common.
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 1.6, 0.6]
    
    # Sort by likelihood (square first)
    aspect_ratios.sort(key=lambda x: abs(x - 1.0))
    
    for w0 in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
            
        time_budget = min(15.0, deadline - time.time() - 5.0)
        if time_budget < 2.0:
            break
            
        result = run_single_trial(w0, time_budget)
        
        if result is not None:
            x, y, r, w, val = result
            if val > best_value:
                best_value = val
                best_result = (x, y, r, w)
                
    # If we have a result, do a final high-precision polish
    if best_result is not None:
        x, y, r, w = best_result
        
        # Final aggressive radius growth with very small steps
        # and very strict relaxation
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            new_r = r * 1.0001
            x_new, y_new = relax_forces(x, y, new_r, w, iterations=300)
            valid, val_new = check_validity(x_new, y_new, new_r, w)
            if valid:
                x, y, r, w = x_new, y_new, new_r, w
                best_value = val_new
            else:
                break
                
        # Final validity check and safety shrink
        valid, val = check_validity(x, y, r, w)
        if valid:
             # Shrink slightly to ensure strict inequality for evaluator
             r_final = r * (1 - 1e-9)
             return np.column_stack((x, y, r_final))
        else:
             # Fallback to previous best if polish failed
             return np.column_stack((best_result[0], best_result[1], best_result[2] * (1 - 1e-9)))

    # Fallback: Simple grid in unit square (Perimeter 4 -> 1x1)
    # This should not happen if any trial succeeded
    cols = 5
    rows = 5
    cells = []
    for i in range(N):
        row, col = divmod(i, cols)
        cx = (col + 0.5) / cols
        cy = (row + 0.5) / rows
        r0 = min(1.0 / cols, 1.0 / rows) * 0.45
        cells.append((cx, cy, r0))
    return np.array(cells)

if __name__ == "__main__":
    res = construct()
    print(res.shape)
