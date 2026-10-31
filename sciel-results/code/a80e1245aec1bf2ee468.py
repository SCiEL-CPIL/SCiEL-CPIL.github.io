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

    def run_simulation(w, h, time_budget, seed=0):
        """
        Run force-directed simulation.
        Returns x, y, r.
        """
        rng = np.random.default_rng(seed)
        
        # Initialize positions in a grid-like pattern with jitter to avoid initial overlaps
        # We want to pack N circles. A grid of size roughly sqrt(N) x sqrt(N) is a good start.
        cols = int(np.ceil(np.sqrt(N * w / h)))
        rows = int(np.ceil(N / cols))
        
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Initial radius estimate: area / N
        r_init = np.sqrt(w * h / N) * 0.4 # conservative start
        r[:] = r_init
        
        # Grid initialization
        for i in range(N):
            row = i // cols
            col = i % cols
            # Add jitter
            x[i] = (col + 0.5 + rng.uniform(-0.3, 0.3)) * (w / cols)
            y[i] = (row + 0.5 + rng.uniform(-0.3, 0.3)) * (h / rows)
            # Ensure inside
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])

        vx = np.zeros(N)
        vy = np.zeros(N)
        
        dt = 0.01
        damping = 0.9
        iu = np.triu_indices(N, k=1)
        
        # Simulation loop
        # We will gradually increase radii to fill space
        max_iters = 2000
        iter_count = 0
        
        # Force calculation helper
        def calc_forces(x, y, r):
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Wall repulsion: strong force pushing back if outside
            # If x < r, force is positive (push right)
            # If x > w-r, force is negative (push left)
            fx += np.maximum(r - x, 0.0) * 50.0
            fx -= np.maximum(x + r - w, 0.0) * 50.0
            fy += np.maximum(r - y, 0.0) * 50.0
            fy -= np.maximum(y + r - h, 0.0) * 50.0
            
            # Pairwise repulsion
            # Only compute for pairs that are close or overlapping to save time? 
            # For N=21, O(N^2) is 210 pairs, which is very fast. No need for spatial hashing.
            for k in range(len(iu[0])):
                a = iu[0][k]
                b = iu[1][k]
                dx = x[a] - x[b]
                dy = y[a] - y[b]
                d2 = dx*dx + dy*dy
                d = np.sqrt(d2) if d2 > 1e-12 else 1e-6
                rs = r[a] + r[b]
                if d < rs:
                    # Overlap
                    # Force proportional to overlap
                    overlap = rs - d
                    f = 50.0 * overlap
                    fx[a] += (dx/d) * f
                    fy[a] += (dy/d) * f
                    fx[b] -= (dx/d) * f
                    fy[b] -= (dy/d) * f
            
            # Attraction to center (weak) to prevent drifting to corners if not constrained
            # Actually, for packing, we usually want them pushed to boundaries.
            # The wall repulsion handles the "stay inside" part.
            # We don't need a center attraction, we need them to spread out.
            # The repulsion handles spreading.
            
            return fx, fy

        # Phase 1: Relax positions with fixed small radius
        for i in range(100):
            if time.time() > start_time + time_budget:
                break
            fx, fy = calc_forces(x, y, r)
            vx = vx * damping + fx * dt
            vy = vy * damping + fy * dt
            x += vx * dt
            y += vy * dt
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

        # Phase 2: Inflate radii
        # We want to maximize sum of radii.
        # We can treat radius as a variable that grows if there is space.
        # Simple approach: increase all radii by a small factor each step, 
        # and let the forces resolve overlaps by moving circles.
        # If we can't resolve overlaps, we might need to shrink, but usually 
        # in packing, we just push until they touch.
        
        # Better approach: Binary search on a global scale factor?
        # Or just iterative inflation.
        
        scale = 1.0
        base_r = r.copy()
        
        # Try to inflate
        for step in range(500):
            if time.time() > start_time + time_budget:
                break
            
            # Increase target radius
            scale += 0.02
            r = base_r * scale
            
            # Relax positions to accommodate new radii
            # Use smaller dt for stability during inflation
            dt_sub = 0.005
            damping_sub = 0.85
            for sub in range(50):
                if time.time() > start_time + time_budget:
                    break
                fx, fy = calc_forces(x, y, r)
                vx = vx * damping_sub + fx * dt_sub
                vy = vy * damping_sub + fy * dt_sub
                x += vx * dt_sub
                y += vy * dt_sub
                x = np.clip(x, r, w - r)
                y = np.clip(y, r, h - r)
                
                # Check if valid
                # If we are still heavily overlapping, the forces will push them apart.
                # If they can't be pushed apart (constrained), the simulation will get stuck.
                # To detect "stuck", we can check if the max overlap is decreasing.
                # For now, we assume the force is strong enough.
            
        return x, y, r

    def polish(x, y, r, w, h, time_budget):
        """
        Local search to improve the packing.
        1. For each circle, try to increase its radius.
        2. If radius increase causes overlap, try moving the circle to free space.
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        iu = np.triu_indices(N, k=1)
        
        for rnd in range(20):
            if time.time() > start_time + time_budget:
                break
            
            improved = False
            for i in range(N):
                # Calculate max possible radius for circle i at current position
                r_max = min(x[i], w - x[i], y[i], h - y[i])
                
                # Check against other circles
                for j in range(N):
                    if i == j:
                        continue
                    d = np.hypot(x[i] - x[j], y[i] - y[j])
                    r_max = min(r_max, d - r[j])
                
                r_max = max(0.0, r_max)
                
                # If we can grow, grow it
                if r_max > r[i] + 1e-5:
                    r[i] = r_max
                    improved = True
                    # After growing, we might have created overlaps with neighbors?
                    # No, r_max is calculated to be exactly touching or less.
                    # So it should be valid.
                
                # Try to move circle i to a better position
                # We want to maximize its radius.
                # Try small moves in 8 directions
                best_move = None
                best_r = r[i]
                
                # Current position is best_r
                # Try neighbors
                for dx in [-0.02, -0.01, 0.01, 0.02]:
                    for dy in [-0.02, -0.01, 0.01, 0.02]:
                        if dx == 0 and dy == 0:
                            continue
                        nx = x[i] + dx
                        ny = y[i] + dy
                        
                        # Check walls
                        r_wall = min(nx, w - nx, ny, h - ny)
                        if r_wall < r[i] - 1e-5:
                            continue
                        
                        # Check neighbors
                        r_new_max = r_wall
                        for j in range(N):
                            if i == j:
                                continue
                            d = np.hypot(nx - x[j], ny - y[j])
                            r_new_max = min(r_new_max, d - r[j])
                        
                        if r_new_max > best_r + 1e-5:
                            best_r = r_new_max
                            best_move = (nx, ny)
                
                if best_move:
                    x[i], y[i] = best_move
                    r[i] = best_r
                    improved = True
            
            if not improved:
                break
        
        return x, y, r

    # Sweep aspect ratios
    # Perimeter = 2(w+h) = 4 => w+h=2
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8]
    
    for k, w0 in enumerate(aspect_ratios):
        h0 = 2.0 - w0
        if h0 < 0.4 or w0 < 0.4:
            continue
        
        remaining = deadline - (time.time() - start_time)
        if remaining < 5.0:
            break
        
        budget = min(8.0, remaining - 2.0)
        
        try:
            x, y, r = run_simulation(w0, h0, budget, seed=100 + k)
            
            # Polish
            x, y, r = polish(x, y, r, w0, h0, deadline - (time.time() - start_time) - 2)
            
            # Final validity check
            valid, val = check_validity(x, y, r, w0, h0)
            if valid and val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w0, h0)
                
        except Exception as e:
            continue

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result
    
    # Final safety shrink
    for _ in range(10):
        rt = r * (1 - 1e-9)
        if check_validity(x, y, rt, w, h)[0]:
            r = rt
        else:
            break

    return np.column_stack((x, y, r))
