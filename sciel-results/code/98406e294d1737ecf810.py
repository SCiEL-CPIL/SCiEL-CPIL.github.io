"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_sum_r = 0.0
    best_config = None
    best_W = 1.0
    best_H = 1.0
    
    # Strategy: Sequential greedy packing (Descartes-like iterative circle addition)
    # 1. Pick aspect ratio W, H (W+H=2)
    # 2. Place circles one by one, each at maximal radius that fits in free space
    # 3. To escape local minima, randomize insertion order and initial "anchors"
    # 4. After placing all 21, do a local refinement: try to grow each circle slightly
    #    while pushing neighbors, using a force-based relaxation with radius growth.
    
    def place_circle_greedy(circles, W, H, max_attempts=500):
        """
        Place a new circle at maximal radius that doesn't overlap existing circles
        and fits in [0,W]x[0,H].
        circles: list of (x, y, r)
        Returns (x, y, r) or None if no space.
        """
        # Sample candidate positions on a grid + random jitter
        # The maximal radius at a point (px, py) is:
        #   min over walls: px, W-px, py, H-py
        #   min over existing circles i: dist((px,py),(xi,yi)) - ri
        # We want to maximize this over (px, py).
        # Use random sampling + local hill climbing.
        
        best_r = 0.0
        best_pos = None
        
        # Random sampling
        for attempt in range(max_attempts):
            if attempt < 100:
                # Grid-like sampling
                px = np.random.uniform(0, W)
                py = np.random.uniform(0, H)
            else:
                # Random
                px = np.random.uniform(0, W)
                py = np.random.uniform(0, H)
            
            # Compute max radius at this point
            r_candidate = min(px, W - px, py, H - py)
            for (xi, yi, ri) in circles:
                dx = px - xi
                dy = py - yi
                dist = np.sqrt(dx*dx + dy*dy)
                r_candidate = min(r_candidate, dist - ri)
            
            if r_candidate > best_r:
                best_r = r_candidate
                best_pos = (px, py)
        
        if best_pos is None or best_r <= 1e-6:
            return None
        
        # Local hill climbing to refine position
        px, py = best_pos
        for step in range(50):
            improved = False
            for dx in np.linspace(-0.05, 0.05, 5):
                for dy in np.linspace(-0.05, 0.05, 5):
                    np_ = (px + dx, py + dy)
                    if np_[0] < 0 or np_[0] > W or np_[1] < 0 or np_[1] > H:
                        continue
                    r_new = min(np_[0], W - np_[0], np_[1], H - np_[1])
                    for (xi, yi, ri) in circles:
                        ddx = np_[0] - xi
                        ddy = np_[1] - yi
                        dist = np.sqrt(ddx*ddx + ddy*ddy)
                        r_new = min(r_new, dist - ri)
                    if r_new > best_r + 1e-8:
                        best_r = r_new
                        best_pos = np_
                        px, py = np_
                        improved = True
            if not improved:
                break
        
        return (best_pos[0], best_pos[1], best_r)
    
    def refine_packing(circles, W, H, iterations=2000):
        """
        Refine the packing by trying to grow all circles simultaneously.
        Uses force-based relaxation: if circles overlap, push apart; 
        also try to increase radii.
        """
        xs = np.array([c[0] for c in circles])
        ys = np.array([c[1] for c in circles])
        rs = np.array([c[2] for c in circles])
        n = len(xs)
        
        # Try to grow radii: scale them up slightly each iteration
        # and relax positions.
        growth_rate = 0.001
        lr = 0.05
        
        for it in range(iterations):
            if time.time() > deadline - 5:
                break
            
            # Grow radii
            rs = rs * (1.0 + growth_rate)
            
            # Force calculation
            fx = np.zeros(n)
            fy = np.zeros(n)
            
            # Pairwise repulsion
            for i in range(n):
                for j in range(i+1, n):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist_sq = dx*dx + dy*dy
                    dist = np.sqrt(dist_sq) if dist_sq > 0 else 1e-9
                    min_dist = rs[i] + rs[j]
                    
                    if dist < min_dist:
                        overlap = min_dist - dist
                        if dist > 1e-6:
                            nx = dx / dist
                            ny = dy / dist
                            fx[i] += nx * overlap
                            fy[i] += ny * overlap
                            fx[j] -= nx * overlap
                            fy[j] -= ny * overlap
                        else:
                            angle = np.random.uniform(0, 2*np.pi)
                            fx[i] += np.cos(angle) * overlap
                            fy[i] += np.sin(angle) * overlap
                            fx[j] -= np.cos(angle) * overlap
                            fy[j] -= np.sin(angle) * overlap
            
            # Wall constraints
            for i in range(n):
                if xs[i] < rs[i]:
                    fx[i] += (rs[i] - xs[i]) * 10.0
                if xs[i] > W - rs[i]:
                    fx[i] -= (xs[i] - (W - rs[i])) * 10.0
                if ys[i] < rs[i]:
                    fy[i] += (rs[i] - ys[i]) * 10.0
                if ys[i] > H - rs[i]:
                    fy[i] -= (ys[i] - (H - rs[i])) * 10.0
            
            # Update
            xs += lr * fx
            ys += lr * fy
            
            # Project to bounds
            xs = np.clip(xs, rs, W - rs)
            ys = np.clip(ys, rs, H - rs)
            
            # Check validity
            valid = True
            for i in range(n):
                if xs[i] < rs[i] - 1e-5 or xs[i] > W - rs[i] + 1e-5:
                    valid = False
                if ys[i] < rs[i] - 1e-5 or ys[i] > H - rs[i] + 1e-5:
                    valid = False
            for i in range(n):
                for j in range(i+1, n):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist_sq = dx*dx + dy*dy
                    min_dist = rs[i] + rs[j]
                    if dist_sq < min_dist*min_dist - 1e-5:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                # Revert growth
                rs = rs / (1.0 + growth_rate)
                # Try to reduce overlap
                # Keep positions, just revert radii
                continue
        
        return list(zip(xs, ys, rs))
    
    def is_valid(circles, W, H):
        """Check if packing is valid."""
        n = len(circles)
        for i in range(n):
            xi, yi, ri = circles[i]
            if xi < ri - 1e-5 or xi > W - ri + 1e-5:
                return False
            if yi < ri - 1e-5 or yi > H - ri + 1e-5:
                return False
        for i in range(n):
            for j in range(i+1, n):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist_sq = dx*dx + dy*dy
                min_dist = circles[i][2] + circles[j][2]
                if dist_sq < min_dist*min_dist - 1e-5:
                    return False
        return True
    
    # Main loop: try different aspect ratios and random seeds
    np.random.seed(42)
    
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
    num_restarts = 5
    
    for W_ratio in aspect_ratios:
        if time.time() > deadline - 30:
            break
        
        W = W_ratio
        H = 2.0 - W_ratio
        if W <= 0 or H <= 0:
            continue
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + restart * 1000 + int(W_ratio * 100))
            
            circles = []
            
            # Place circles sequentially
            for k in range(N):
                if time.time() > deadline - 5:
                    break
                
                # For the first few circles, use strategic placement
                if k == 0:
                    # Place first circle in center
                    r0 = min(W, H) / 4.0
                    circles.append((W/2, H/2, r0))
                elif k < 4:
                    # Place in corners
                    corners = [(W/4, H/4), (3*W/4, H/4), (W/4, 3*H/4), (3*W/4, 3*H/4)]
                    cx, cy = corners[k-1]
                    # Compute max radius at this corner
                    r_c = min(cx, W-cx, cy, H-cy)
                    for (xi, yi, ri) in circles:
                        dx = cx - xi
                        dy = cy - yi
                        dist = np.sqrt(dx*dx + dy*dy)
                        r_c = min(r_c, dist - ri)
                    if r_c > 1e-4:
                        circles.append((cx, cy, r_c))
                    else:
                        # Try greedy
                        new_c = place_circle_greedy(circles, W, H, max_attempts=200)
                        if new_c:
                            circles.append(new_c)
                        else:
                            break
                else:
                    new_c = place_circle_greedy(circles, W, H, max_attempts=300)
                    if new_c:
                        circles.append(new_c)
                    else:
                        break
            
            if len(circles) < N:
                continue
            
            # Refine
            refined = refine_packing(circles, W, H, iterations=1000)
            
            if not is_valid(refined, W, H):
                continue
            
            total_r = sum(c[2] for c in refined)
            if total_r > best_sum_r:
                best_sum_r = total_r
                best_config = np.array(refined)
                best_W = W
                best_H = H
    
    if best_config is not None:
        return best_config
    
    # Fallback: Grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
