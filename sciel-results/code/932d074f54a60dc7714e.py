import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    best_sum = 0.0
    best_config = None
    
    # Helper to check validity and get score
    def evaluate(config, w, h):
        # config: (N, 3) array of [x, y, r]
        # Check wall constraints
        for i in range(N):
            x, y, r = config[i]
            if x - r < -1e-6 or x + r > w + 1e-6 or y - r < -1e-6 or y + r > h + 1e-6:
                return 0.0, False
        # Check circle-circle constraints
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = config[i]
                xj, yj, rj = config[j]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                rad_sum = ri + rj
                if dist_sq < rad_sum**2 - 1e-6:
                    return 0.0, False
        return np.sum(config[:, 2]), True

    # Helper to generate hexagonal grid
    def generate_hex_grid(n_circles):
        # Generate a hexagonal pattern of n_circles
        # Rows will have sizes like 4, 5, 4, 4, 4 or similar
        # Let's just generate a large enough hex grid and take the first n
        # Or specifically construct rows
        rows = []
        remaining = n_circles
        # Try to balance rows
        # Hex packing: row i has k, k+1, k, k+1... or k, k, k...
        # Let's try a simple approach: find number of rows R
        # Average per row ~ sqrt(N)
        R = int(np.sqrt(n_circles))
        # Adjust R to fit
        while True:
            base = n_circles // R
            rem = n_circles % R
            # Distribute
            row_sizes = [base + (1 if i < rem else 0) for i in range(R)]
            if max(row_sizes) - min(row_sizes) <= 1:
                break
            R += 1
            if R > 20: break
        
        # If R is too small, increase it
        # Actually, for hex packing, alternating rows have 1 more circle
        # Let's try a fixed structure for 21:
        # 4, 5, 4, 4, 4 (sum 21)
        # 5, 4, 5, 4, 3 (sum 21)
        # Let's just generate a dense hex cluster
        
        # Standard hex coordinates
        # Use axial coordinates or just cartesian
        # Let's generate points in a hexagonal lattice
        # r = 1
        # dx = 2, dy = sqrt(3)
        
        points = []
        # Generate enough points
        max_rows = 10
        for row in range(max_rows):
            # Offset row
            offset = 0 if row % 2 == 0 else 1
            for col in range(max_rows):
                x = col * 2 + offset
                y = row * np.sqrt(3)
                points.append((x, y))
                
        # Sort by distance from center? Or just take first N?
        # Taking first N in row-major order gives a rectangle-ish block
        # Better to take a central cluster
        # Calculate centroid of first 100 points
        # Let's just take the first N points in a hexagonal order
        # A better way: generate all points in a large hex area, sort by distance to (0,0), take N closest.
        
        all_points = []
        for row in range(-10, 11):
            offset = 0 if row % 2 == 0 else 1
            for col in range(-10, 11):
                x = col * 2 + offset
                y = row * np.sqrt(3)
                all_points.append((x, y))
        
        # Sort by distance to origin
        all_points.sort(key=lambda p: p[0]**2 + p[1]**2)
        selected = all_points[:n_circles]
        
        # Normalize to center at (0,0)
        xs = [p[0] for p in selected]
        ys = [p[1] for p in selected]
        cx = np.mean(xs)
        cy = np.mean(ys)
        centered = [(x - cx, y - cy) for x, y in selected]
        
        return np.array(centered)

    base_grid = generate_hex_grid(N)
    
    # Optimize aspect ratio
    # h is height, w = 2 - h
    # We want to scale the grid to fit in w x h
    # The grid has a bounding box. Let's find the scale factor.
    
    for h in np.linspace(0.4, 1.6, 100):
        w = 2.0 - h
        if w <= 0: continue
        
        # Current grid coordinates
        xs = base_grid[:, 0]
        ys = base_grid[:, 1]
        
        # Bounding box of grid
        min_x, max_x = np.min(xs), np.max(xs)
        min_y, max_y = np.min(ys), np.max(ys)
        
        width_grid = max_x - min_x
        height_grid = max_y - min_y
        
        if width_grid == 0 or height_grid == 0: continue
        
        # Scale factor to fit in w, h
        # We need to fit the grid (with radius r) into w, h
        # Let's assume unit radius for now to find scale
        # The effective width of the packing is width_grid + 2*r
        # The effective height is height_grid + 2*r
        # We want to maximize r such that:
        # scale * (width_grid + 2*r) <= w  => scale * width_grid + 2*scale*r <= w
        # scale * (height_grid + 2*r) <= h => scale * height_grid + 2*scale*r <= h
        
        # Let s be the scale factor applied to the grid coordinates.
        # The radii will be s * r_unit. Let r_unit = 1.
        # So radii = s.
        # Constraints:
        # s * width_grid + 2*s <= w  => s * (width_grid + 2) <= w
        # s * height_grid + 2*s <= h => s * (height_grid + 2) <= h
        
        s1 = w / (width_grid + 2.0)
        s2 = h / (height_grid + 2.0)
        s = min(s1, s2)
        
        # Apply scale
        scaled_x = xs * s
        scaled_y = ys * s
        r = s
        
        # Center the scaled grid in the rectangle
        min_sx, max_sx = np.min(scaled_x), np.max(scaled_x)
        min_sy, max_sy = np.min(scaled_y), np.max(scaled_y)
        
        # We can shift the grid to optimize packing, but for now, center it
        # Shift so that the bounding box is centered in w x h
        # Actually, we can shift to maximize the minimum distance to walls
        # The available space in x is w - (max_sx - min_sx)
        # The available space in y is h - (max_sy - min_sy)
        
        # Let's just center it
        cx_rect = w / 2.0
        cy_rect = h / 2.0
        
        cx_grid = (min_sx + max_sx) / 2.0
        cy_grid = (min_sy + max_sy) / 2.0
        
        config = np.zeros((N, 3))
        config[:, 0] = scaled_x + (cx_rect - cx_grid)
        config[:, 1] = scaled_y + (cy_rect - cy_grid)
        config[:, 2] = r
        
        # Local Optimization: Push circles to maximize min distance
        # Simple iterative relaxation
        for _ in range(50):
            if time.time() - start_time > deadline - 5:
                break
            
            improved = False
            # Calculate current min distance to constraints
            # We want to increase r. 
            # Let's try to move circles to create more space.
            
            # For each circle, find the direction that increases the minimum clearance
            # This is complex. Let's try a simpler "jiggle"
            
            # Check if we can increase r
            # Current r is s.
            # Let's try to increase r by a small epsilon
            # If we increase r, we might violate constraints.
            # Instead, let's move circles to reduce the bounding box or increase inter-circle distances.
            
            # Move circles towards the center of their local free space
            # This is effectively gradient ascent on the minimum distance function.
            
            # Simple heuristic: 
            # 1. Check overlaps. If any, push apart.
            # 2. Check walls. If any, push in.
            # 3. If no violations, increase r slightly.
            
            violations = False
            # Check walls
            for i in range(N):
                x, y, rad = config[i]
                if x - rad < 0:
                    config[i, 0] = rad
                    violations = True
                elif x + rad > w:
                    config[i, 0] = w - rad
                    violations = True
                if y - rad < 0:
                    config[i, 1] = rad
                    violations = True
                elif y + rad > h:
                    config[i, 1] = h - rad
                    violations = True
            
            # Check circles
            for i in range(N):
                for j in range(i+1, N):
                    dx = config[i, 0] - config[j, 0]
                    dy = config[i, 1] - config[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = config[i, 2] + config[j, 2]
                    if dist < min_dist:
                        # Push apart
                        if dist < 1e-6:
                            config[i, 0] += 1e-3
                            config[j, 0] -= 1e-3
                        else:
                            push = (min_dist - dist) / 2.0
                            nx = dx / dist
                            ny = dy / dist
                            config[i, 0] += nx * push
                            config[i, 1] += ny * push
                            config[j, 0] -= nx * push
                            config[j, 1] -= ny * push
                        violations = True
            
            if not violations:
                # Try to increase radius
                # Check if we can increase r by 1e-4
                new_r = config[0, 2] * 1.001
                # Apply new r to all
                old_r = config[0, 2]
                if old_r > 0:
                    factor = new_r / old_r
                    # Scale positions? No, just increase r.
                    # If we increase r, we might violate constraints.
                    # Let's just check if increasing r causes violation.
                    # For simplicity, let's just increase r if no violations were found in the previous step
                    # and hope the geometry allows it.
                    # A safer approach: 
                    # Calculate the maximum possible r given current positions.
                    # r_max = min( min_i (x_i, w-x_i, y_i, h-y_i), min_{i<j} dist_ij / 2 )
                    # If r_max > current r, set r = r_max * 0.99
                    min_wall = np.inf
                    for i in range(N):
                        min_wall = min(min_wall, config[i, 0], w - config[i, 0], config[i, 1], h - config[i, 1])
                    
                    min_circ = np.inf
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((config[i, 0]-config[j, 0])**2 + (config[i, 1]-config[j, 1])**2)
                            min_circ = min(min_circ, dist / 2.0)
                    
                    potential_r = min(min_wall, min_circ)
                    if potential_r > config[0, 2]:
                        # Update all radii
                        config[:, 2] = potential_r * 0.99
                        improved = True
                        # After updating radii, we might have violations if we pushed too hard? 
                        # No, we calculated potential_r based on current positions.
                        # But increasing r might cause new violations if positions are not optimal.
                        # The relaxation loop above will fix positions in the next iteration.
            
            if not improved and not violations:
                # Stagnant? Break or continue
                pass

        score, valid = evaluate(config, w, h)
        if valid and score > best_sum:
            best_sum = score
            best_config = config.copy()
            best_w = w
            best_h = h

    if best_config is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_config

