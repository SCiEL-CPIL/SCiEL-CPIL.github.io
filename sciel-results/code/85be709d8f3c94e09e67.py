"""
Circle packing in perimeter-4 rectangle.
Uses Simulated Annealing with hard constraint repair and "grow" moves.
"""
import numpy as np
import time
import random

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit (or 120s if strict, but 145 is safer for "150s wall-clock" claim in prompt)
    
    best_value = -np.inf
    best_result = None
    best_w = 1.0
    best_h = 1.0

    # Helper: Check if configuration is valid (inside box, no overlaps)
    def is_valid(circles, w, h, eps=1e-7):
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            if x - r < -eps or x + r > w + eps or y - r < -eps or y + r > h + eps:
                return False
        for i in range(n):
            xi, yi, ri = circles[i]
            for j in range(i + 1, n):
                xj, yj, rj = circles[j]
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                min_dist = ri + rj
                if dist_sq < min_dist * min_dist - eps:
                    return False
        return True

    def get_value(circles):
        return float(np.sum(circles[:, 2]))

    # Helper: Repair overlaps by pushing circles apart
    # Returns True if repair was successful (no overlaps, inside box)
    def repair(circles, w, h, max_iters=100):
        n = len(circles)
        for _ in range(max_iters):
            moved = False
            for i in range(n):
                xi, yi, ri = circles[i]
                # Check bounds
                if xi - ri < 0:
                    xi = ri
                    moved = True
                if xi + ri > w:
                    xi = w - ri
                    moved = True
                if yi - ri < 0:
                    yi = ri
                    moved = True
                if yi + ri > h:
                    yi = h - ri
                    moved = True
                circles[i] = [xi, yi, ri]
                
                # Check overlaps
                for j in range(n):
                    if i == j: continue
                    xj, yj, rj = circles[j]
                    dx = xi - xj
                    dy = yi - yj
                    dist_sq = dx*dx + dy*dy
                    min_dist = ri + rj
                    if dist_sq < min_dist * min_dist:
                        dist = np.sqrt(dist_sq)
                        if dist < 1e-9:
                            # Random push
                            angle = random.uniform(0, 2*np.pi)
                            push = min_dist / 2.0
                            dx = np.cos(angle) * push
                            dy = np.sin(angle) * push
                        else:
                            # Push along vector
                            overlap = min_dist - dist
                            dx = (dx / dist) * overlap * 0.5
                            dy = (dy / dist) * overlap * 0.5
                        
                        # Move i and j apart
                        circles[i][0] += dx
                        circles[i][1] += dy
                        circles[j][0] -= dx
                        circles[j][1] -= dy
                        moved = True
            if not moved:
                return True
        return is_valid(circles, w, h)

    def simulate_annealing(w, h, time_limit):
        # Initialize with hex-like grid
        # Estimate radius
        # Area = w*h. N circles. ~ pi*r^2 * N <= w*h -> r ~ sqrt(w*h/(N*pi))
        # But packing efficiency is lower. Let's start small.
        r_est = np.sqrt(w * h / (N * np.pi)) * 0.8
        
        # Create initial grid
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        
        circles = np.zeros((N, 3))
        for i in range(N):
            col = i % a
            row = i // a
            # Add jitter to break symmetry
            x = (col + 0.5) * w / a + random.uniform(-0.1, 0.1) * r_est
            y = (row + 0.5) * h / b + random.uniform(-0.1, 0.1) * r_est
            r = r_est * 0.5
            circles[i] = [x, y, r]
        
        # Initial repair
        repair(circles, w, h)
        
        if not is_valid(circles, w, h):
            # Fallback: very small circles
            r_small = min(w, h) / (N + 2) * 0.5
            for i in range(N):
                circles[i] = [random.uniform(r_small, w-r_small), 
                              random.uniform(r_small, h-r_small), 
                              r_small]
            repair(circles, w, h)
        
        current_val = get_value(circles)
        best_local_val = current_val
        best_local_circles = circles.copy()
        
        T = 1.0
        T_min = 1e-6
        alpha = 0.999 # Cooling rate
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            if T < T_min:
                break
                
            # Choose a move
            move_type = random.random()
            idx = random.randint(0, N - 1)
            new_circles = circles.copy()
            
            if move_type < 0.5:
                # Move center
                dx = random.gauss(0, 0.1 * r_est)
                dy = random.gauss(0, 0.1 * r_est)
                new_circles[idx][0] += dx
                new_circles[idx][1] += dy
            elif move_type < 0.8:
                # Grow radius
                # Try to grow by a small amount
                grow_amt = random.uniform(0.01, 0.1) * r_est
                new_circles[idx][2] += grow_amt
            else:
                # Shrink radius (less common, helps escape traps)
                shrink_amt = random.uniform(0.01, 0.1) * r_est
                new_circles[idx][2] = max(1e-6, new_circles[idx][2] - shrink_amt)
            
            # Repair
            repair(new_circles, w, h)
            
            if is_valid(new_circles, w, h):
                new_val = get_value(new_circles)
                delta = new_val - current_val
                
                if delta > 0 or random.random() < np.exp(delta / T):
                    circles = new_circles
                    current_val = new_val
                    if current_val > best_local_val:
                        best_local_val = current_val
                        best_local_circles = circles.copy()
            
            T *= alpha
            
        return best_local_circles, best_local_val

    # Define aspect ratios to try
    # Perimeter 4 -> w + h = 2.
    # Aspect ratio = w/h.
    # w = 2 * ar / (1 + ar), h = 2 / (1 + ar)
    aspects = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]
    
    # Allocate time roughly evenly, but give more to likely optimal ratios (near 1.0)
    # For N=21, a 3x7 or 4x5 grid is likely.
    # 3x7: w/h ~ 7/3 * (1/sqrt(3)) ~ 1.35? Or just 7/3=2.33?
    # Hex packing is denser.
    # Let's just run all.
    
    n_aspects = len(aspects)
    time_per_aspect = (deadline - start_time) / n_aspects
    
    # Ensure time_per_aspect is reasonable
    if time_per_aspect < 5:
        # Too many aspects, reduce list
        aspects = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7]
        n_aspects = len(aspects)
        time_per_aspect = (deadline - start_time) / n_aspects

    for ar in aspects:
        if time.time() > deadline - 2:
            break
            
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        # Run SA
        circles, val = simulate_annealing(w, h, time_per_aspect)
        
        if val > best_value:
            best_value = val
            best_result = circles.copy()
            best_w = w
            best_h = h

    # If no valid result found (should not happen), return a trivial valid one
    if best_result is None:
        w, h = 1.0, 1.0
        r = 0.01
        best_result = np.zeros((N, 3))
        for i in range(N):
            best_result[i] = [0.5, 0.5, r]
        # This is invalid (overlapping), so we need a valid fallback
        # Grid of tiny circles
        a = 5
        b = 5
        w, h = 1.0, 1.0
        r = 0.05
        for i in range(N):
            col = i % a
            row = i // a
            best_result[i] = [(col+0.5)*w/a, (row+0.5)*h/b, r]
        # Verify
        if not is_valid(best_result, w, h):
             # Even smaller
             r = 0.01
             for i in range(N):
                 best_result[i][2] = r
             # Still might overlap if N > 25 in 1x1 with 0.01?
             # 1x1 / 0.01^2 = 10000 capacity. So 21 is fine.
             # Just need to place them.
             for i in range(N):
                 best_result[i][0] = (i % 5 + 0.5) * 0.2
                 best_result[i][1] = (i // 5 + 0.5) * 0.2

    return best_result

# Entry point
if __name__ == "__main__":
    res = construct()
    print(f"Result shape: {res.shape}")
    print(f"Sum of radii: {np.sum(res[:, 2])}")
