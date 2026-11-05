import time
import random
from collections import Counter
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin
    
    N = 30
    RANGE = 30
    
    def get_max_multiplicity(points):
        """
        Returns the maximum multiplicity of any doubled area.
        Uses standard loop for accuracy and simplicity, called rarely.
        """
        n = len(points)
        counts = Counter()
        for i in range(n-2):
            x1, y1 = points[i]
            for j in range(i+1, n-1):
                x2, y2 = points[j]
                dx1 = x2 - x1
                dy1 = y2 - y1
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    d = dx1 * (y3 - y1) - dy1 * (x3 - x1)
                    if d != 0:
                        counts[abs(d)] += 1
        if not counts:
            return 0
        return max(counts.values())

    def count_specific_area_np(points, A):
        """
        Counts triangles with absolute doubled area exactly A using NumPy.
        """
        if A == 0:
            return 0
        pts = np.array(points)
        x = pts[:, 0]
        y = pts[:, 1]
        n = len(points)
        total_count = 0
        for i in range(n-2):
            xi, yi = x[i], y[i]
            for j in range(i+1, n-1):
                dx = x[j] - xi
                dy = y[j] - yi
                # Calculate determinant for all k
                d_vals = dx * (y - yi) - dy * (x - xi)
                # Mask out i and j
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                mask[j] = False
                # Count matches
                total_count += np.sum(np.abs(d_vals[mask]) == A)
        return total_count // 3

    def gen_grid(w, h, offset_x=0, offset_y=0):
        pts = []
        for i in range(w):
            for j in range(h):
                pts.append((i + offset_x, j + offset_y))
        return pts

    # Initial configurations
    inits = []
    inits.append(gen_grid(6, 5, -2, -2))
    inits.append(gen_grid(5, 6, -2, -2))
    inits.append(gen_grid(10, 3, -4, -1))
    inits.append(gen_grid(3, 10, -1, -4))
    
    random.seed(42)
    for _ in range(3):
        pts = []
        while len(pts) < N:
            p = (random.randint(-RANGE, RANGE), random.randint(-RANGE, RANGE))
            if p not in pts:
                pts.append(p)
        inits.append(pts)

    best_score = 0
    best_points = inits[0]
    
    target_areas = [1, 2, 4, 8, 16, 32, 64]
    
    # Time management
    # We will run SA for each (init, target) pair.
    # With NumPy, count_specific_area is fast (~0.1ms?).
    # So we can do many more steps.
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
            
        current_points = list(init_pts)
        if len(set(current_points)) != N:
            continue
        for p in current_points:
            if p[0] < -RANGE or p[0] > RANGE or p[1] < -RANGE or p[1] > RANGE:
                break
        else:
            for A in target_areas:
                if time.time() > deadline - 5.0:
                    break
                
                # Simulated Annealing
                T = 5.0
                min_T = 0.01
                cooling = 0.9995 # Slower cooling
                current_score = count_specific_area_np(current_points, A)
                best_local_score = current_score
                best_local_points = list(current_points)
                
                no_improve = 0
                max_no_improve = 20000 # Much higher limit due to speed
                
                while time.time() < deadline - 5.0 and no_improve < max_no_improve:
                    idx = random.randint(0, N-1)
                    old_pt = current_points[idx]
                    
                    if random.random() < 0.8:
                        dx = random.randint(-2, 2)
                        dy = random.randint(-2, 2)
                        if dx == 0 and dy == 0:
                            dx = 1
                        new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                        if not (-RANGE <= new_pt[0] <= RANGE and -RANGE <= new_pt[1] <= RANGE):
                            continue
                        if new_pt in current_points:
                            continue
                    else:
                        new_pt = (random.randint(-RANGE, RANGE), random.randint(-RANGE, RANGE))
                        if new_pt in current_points:
                            continue
                        
                    current_points[idx] = new_pt
                    new_score = count_specific_area_np(current_points, A)
                    
                    delta = new_score - current_score
                    
                    if delta > 0 or random.random() < np.exp(delta / T):
                        current_score = new_score
                        if new_score > best_local_score:
                            best_local_score = new_score
                            best_local_points = list(current_points)
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        current_points[idx] = old_pt
                        no_improve += 1
                        
                    T *= cooling
                    if T < min_T:
                        T = min_T

                final_score = get_max_multiplicity(best_local_points)
                
                if final_score > best_score:
                    best_score = final_score
                    best_points = best_local_points

    if len(best_points) != N or len(set(best_points)) != N:
         return gen_grid(6, 5, -2, -2)
    for p in best_points:
        if p[0] < -30 or p[0] > 30 or p[1] < -30 or p[1] > 30:
            return gen_grid(6, 5, -2, -2)

    return best_points
