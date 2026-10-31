import numpy as np
import time

N = 21
DEADLINE = 140.0

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def check_time():
        return elapsed() < DEADLINE
    
    def compute_score(circles):
        return np.sum(circles[:, 2])
    
    def max_radius_for_circle(i, circles, w, h, tol=1e-9):
        x, y = circles[i, 0], circles[i, 1]
        r_max = min(x, w - x, y, h - y)
        for j in range(len(circles)):
            if i == j:
                continue
            dx = x - circles[j, 0]
            dy = y - circles[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            r_candidate = dist - circles[j, 2]
            if r_candidate < r_max:
                r_max = r_candidate
        return max(0.0, r_max)
    
    def expand_radii(circles, w, h, tol=1e-9):
        new_circles = circles.copy()
        for i in range(len(new_circles)):
            new_circles[i, 2] = max_radius_for_circle(i, new_circles, w, h, tol)
        return new_circles
    
    def local_search(circles, w, h, n_outer=20, n_inner=10):
        circles = circles.copy()
        for outer in range(n_outer):
            if not check_time():
                break
            improved = False
            for i in range(len(circles)):
                x, y, r = circles[i]
                best_x, best_y, best_r = x, y, r
                # Try multiple directions and step sizes
                for angle in range(0, 360, 15):
                    for step in [0.05, 0.03, 0.015, 0.007]:
                        nx = x + step * np.cos(np.radians(angle))
                        ny = y + step * np.sin(np.radians(angle))
                        if nx < 0 or nx > w or ny < 0 or ny > h:
                            continue
                        circles[i, 0] = nx
                        circles[i, 1] = ny
                        new_r = max_radius_for_circle(i, circles, w, h)
                        if new_r > best_r + 1e-10:
                            best_x, best_y, best_r = nx, ny, new_r
                if best_r > r + 1e-10:
                    circles[i, 0] = best_x
                    circles[i, 1] = best_y
                    circles[i, 2] = best_r
                    improved = True
                else:
                    circles[i, 0] = x
                    circles[i, 1] = y
            if not improved:
                break
        return circles
    
    def optimize_for_aspect(w, h, init_unit_circles):
        circles = init_unit_circles.copy()
        circles[:, 0] *= w
        circles[:, 1] *= h
        
        # Initial radius expansion
        circles = expand_radii(circles, w, h)
        
        # Iterative local search
        for _ in range(15):
            if not check_time():
                break
            circles = local_search(circles, w, h, n_outer=5, n_inner=5)
            circles = expand_radii(circles, w, h)
        
        return circles
    
    # Initial layout: 5x5 grid in unit square
    side = 5
    init_unit = []
    for i in range(side):
        for j in range(side):
            init_unit.append([(i + 0.5) / side, (j + 0.5) / side, 0.0])
    init_unit = np.array(init_unit[:N])
    
    best_circles = None
    best_score = -1
    best_w, best_h = 1.0, 1.0
    
    # Try various aspect ratios
    aspects = [
        (1.0, 1.0),
        (1.1, 0.9), (0.9, 1.1),
        (1.2, 0.8), (0.8, 1.2),
        (1.3, 0.7), (0.7, 1.3),
        (1.4, 0.6), (0.6, 1.4),
        (1.5, 0.5), (0.5, 1.5),
        (1.6, 0.4), (0.4, 1.6),
        (1.7, 0.3), (0.3, 1.7),
        (1.8, 0.2), (0.2, 1.8),
    ]
    
    for w, h in aspects:
        if not check_time():
            break
        circles = optimize_for_aspect(w, h, init_unit)
        score = compute_score(circles)
        if score > best_score:
            best_score = score
            best_circles = circles.copy()
            best_w, best_h = w, h
    
    # Final polish
    if best_circles is not None and check_time():
        for _ in range(20):
            if not check_time():
                break
            best_circles = expand_radii(best_circles, best_w, best_h, tol=1e-12)
    
    if best_circles is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_circles
