"""Rewritten program for circle packing in perimeter-4 rectangle."""
import numpy as np
import time
import random

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_sum_r = 0.0
    best_config = None
    best_W = 1.0
    best_H = 1.0
    
    def is_valid(circles, W, H, tol=1e-6):
        """Check if packing is valid: no overlaps, all inside rectangle."""
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            if x < -tol or x > W + tol or y < -tol or y > H + tol:
                return False
            if x < r - tol or x > W - r + tol:
                return False
            if y < r - tol or y > H - r + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist_sq = dx * dx + dy * dy
                min_dist = circles[i][2] + circles[j][2]
                if dist_sq < min_dist * min_dist - tol:
                    return False
        return True
    
    def make_hilbert_curve(n, order=8):
        """Generate Hilbert curve coordinates for n points in unit square."""
        # Use a simple iterative Hilbert curve mapping
        # For simplicity, use a deterministic space-filling approach
        # that approximates a Hilbert curve ordering
        size = 1 << order
        coords = []
        
        # Generate Hilbert curve using the standard algorithm
        def hilbert_d2xy(h, size):
            """Convert Hilbert index to (x, y) coordinates."""
            x = 0
            y = 0
            for s in range(1, size, 2):
                rx = 1 & (h // 2)
                ry = 1 & (h ^ rx)
                if ry == 0:
                    if rx == 1:
                        x = s - 1 - x
                        y = s - 1 - y
                    x, y = y, x
                x += s * rx
                y += s * ry
            return x, y
        
        # Map n points along the Hilbert curve
        for i in range(n):
            h = int(i * (size - 1) / max(n - 1, 1))
            x, y = hilbert_d2xy(h, size)
            coords.append(((x + 0.5) / size, (y + 0.5) / size))
        
        return coords
    
    def z_order_curve(n, order=8):
        """Generate Z-order (Morton) curve coordinates."""
        size = 1 << order
        coords = []
        
        def z2xy(z, size):
            """Convert Z-order index to (x, y)."""
            x = 0
            y = 0
            for s in range(size):
                bit_x = (z >> (2 * s)) & 1
                bit_y = (z >> (2 * s + 1)) & 1
                x |= bit_x << s
                y |= bit_y << s
            return x, y
        
        for i in range(n):
            z = int(i * (size * size - 1) / max(n - 1, 1))
            x, y = z2xy(z, size)
            coords.append(((x + 0.5) / size, (y + 0.5) / size))
        
        return coords
    
    def init_from_curve(curve, W, H, base_r=0.08):
        """Initialize circles using space-filling curve positions."""
        circles = []
        for (u, v) in curve:
            x = u * W
            y = v * H
            r = base_r
            circles.append((x, y, r))
        return circles
    
    def relax_packing(circles, W, H, max_iter=3000, growth_rate=0.0005, lr=0.1, time_limit=None):
        """
        Relax packing with radius growth and force-based separation.
        Returns (circles, success, time_elapsed).
        """
        xs = np.array([c[0] for c in circles], dtype=np.float64)
        ys = np.array([c[1] for c in circles], dtype=np.float64)
        rs = np.array([c[2] for c in circles], dtype=np.float64)
        n = len(xs)
        
        t0 = time.time()
        
        for it in range(max_iter):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # Try to grow radii
            rs_new = rs * (1.0 + growth_rate)
            
            # Compute forces
            fx = np.zeros(n)
            fy = np.zeros(n)
            
            # Pairwise repulsion
            for i in range(n):
                for j in range(i + 1, n):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist_sq = dx * dx + dy * dy
                    dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-9
                    min_dist = rs_new[i] + rs_new[j]
                    
                    if dist < min_dist:
                        overlap = min_dist - dist
                        if dist > 1e-8:
                            nx = dx / dist
                            ny = dy / dist
                            fx[i] += nx * overlap
                            fy[i] += ny * overlap
                            fx[j] -= nx * overlap
                            fy[j] -= ny * overlap
                        else:
                            angle = random.uniform(0, 2 * np.pi)
                            fx[i] += np.cos(angle) * overlap
                            fy[i] += np.sin(angle) * overlap
                            fx[j] -= np.cos(angle) * overlap
                            fy[j] -= np.sin(angle) * overlap
            
            # Wall constraints
            for i in range(n):
                if xs[i] < rs_new[i]:
                    fx[i] += (rs_new[i] - xs[i]) * 5.0
                if xs[i] > W - rs_new[i]:
                    fx[i] -= (xs[i] - (W - rs_new[i])) * 5.0
                if ys[i] < rs_new[i]:
                    fy[i] += (rs_new[i] - ys[i]) * 5.0
                if ys[i] > H - rs_new[i]:
                    fy[i] -= (ys[i] - (H - rs_new[i])) * 5.0
            
            # Update positions
            xs_new = xs + lr * fx
            ys_new = ys + lr * fy
            
            # Project to bounds
            xs_new = np.clip(xs_new, rs_new, W - rs_new)
            ys_new = np.clip(ys_new, rs_new, H - rs_new)
            
            # Check validity
            valid = True
            for i in range(n):
                if xs_new[i] < rs_new[i] - 1e-8 or xs_new[i] > W - rs_new[i] + 1e-8:
                    valid = False
                    break
                if ys_new[i] < rs_new[i] - 1e-8 or ys_new[i] > H - rs_new[i] + 1e-8:
                    valid = False
                    break
            if valid:
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = xs_new[i] - xs_new[j]
                        dy = ys_new[i] - ys_new[j]
                        dist_sq = dx * dx + dy * dy
                        min_dist = rs_new[i] + rs_new[j]
                        if dist_sq < min_dist * min_dist - 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                xs = xs_new
                ys = ys_new
                rs = rs_new
            else:
                # Revert and slow down growth
                growth_rate *= 0.95
                if growth_rate < 1e-6:
                    break
        
        elapsed = time.time() - t0
        circles_out = list(zip(xs, ys, rs))
        success = is_valid(circles_out, W, H, tol=1e-5)
        return circles_out, success, elapsed
    
    def local_search(circles, W, H, time_limit=None):
        """
        Perform local search: try to grow each circle individually,
        then do a global relaxation.
        """
        xs = np.array([c[0] for c in circles], dtype=np.float64)
        ys = np.array([c[1] for c in circles], dtype=np.float64)
        rs = np.array([c[2] for c in circles], dtype=np.float64)
        n = len(xs)
        
        t0 = time.time()
        
        # Try to grow each circle individually
        for _ in range(200):
            if time_limit is not None and time.time() > time_limit:
                break
            
            for i in range(n):
                # Try growing circle i
                old_r = rs[i]
                rs[i] = old_r * 1.002
                old_x = xs[i]
                old_y = ys[i]
                
                # Small position adjustment
                if random.random() < 0.3:
                    angle = random.uniform(0, 2 * np.pi)
                    step = 0.01 * old_r
                    xs[i] = np.clip(xs[i] + step * np.cos(angle), rs[i], W - rs[i])
                    ys[i] = np.clip(ys[i] + step * np.sin(angle), rs[i], H - rs[i])
                
                # Check validity
                valid = True
                for j in range(n):
                    if i == j:
                        continue
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist_sq = dx * dx + dy * dy
                    min_dist = rs[i] + rs[j]
                    if dist_sq < min_dist * min_dist - 1e-8:
                        valid = False
                        break
                if xs[i] < rs[i] - 1e-8 or xs[i] > W - rs[i] + 1e-8:
                    valid = False
                if ys[i] < rs[i] - 1e-8 or ys[i] > H - rs[i] + 1e-8:
                    valid = False
                
                if not valid:
                    rs[i] = old_r
                    xs[i] = old_x
                    ys[i] = old_y
        
        circles_out = list(zip(xs, ys, rs))
        elapsed = time.time() - t0
        return circles_out, elapsed
    
    # Main optimization loop
    random.seed(42)
    np.random.seed(42)
    
    # Try different aspect ratios
    aspect_ratios = [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]
    
    for W_ratio in aspect_ratios:
        if time.time() > deadline - 30:
            break
        
        W = W_ratio
        H = 2.0 - W_ratio
        if W <= 0.1 or H <= 0.1:
            continue
        
        # Try different initialization strategies
        inits = []
        
        # Hilbert curve initialization
        hilbert_pts = make_hilbert_curve(N, order=8)
        inits.append(init_from_curve(hilbert_pts, W, H, base_r=0.07))
        
        # Z-order curve initialization
        zorder_pts = z_order_curve(N, order=8)
        inits.append(init_from_curve(zorder_pts, W, H, base_r=0.07))
        
        # Random initialization
        for _ in range(3):
            random_circles = []
            for _ in range(N):
                x = random.uniform(0.1, W - 0.1)
                y = random.uniform(0.1, H - 0.1)
                r = random.uniform(0.04, 0.1)
                random_circles.append((x, y, r))
            inits.append(random_circles)
        
        # Grid initialization
        cols = 5
        rows = 5
        grid_circles = []
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                x = (i + 0.5) * W / cols
                y = (j + 0.5) * H / rows
                r = min(W / cols, H / rows) * 0.4
                grid_circles.append((x, y, r))
                count += 1
        if len(grid_circles) == N:
            inits.append(grid_circles)
        
        for init_circles in inits:
            if time.time() > deadline - 15:
                break
            
            # Relax with radius growth
            relaxed, success, elapsed = relax_packing(
                init_circles, W, H,
                max_iter=2000,
                growth_rate=0.0005,
                lr=0.1,
                time_limit=deadline - 5
            )
            
            if not success:
                continue
            
            # Local search
            local, elapsed2 = local_search(relaxed, W, H, time_limit=deadline - 5)
            
            if not is_valid(local, W, H, tol=1e-5):
                continue
            
            total_r = sum(c[2] for c in local)
            if total_r > best_sum_r:
                best_sum_r = total_r
                best_config = np.array(local)
                best_W = W
                best_H = H
    
    if best_config is not None and len(best_config) == N:
        return best_config
    
    # Fallback: simple grid
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.8
            cells.append((x, y, r))
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
