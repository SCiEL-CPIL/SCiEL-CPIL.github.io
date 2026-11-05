import time
import random
from collections import Counter
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s safety margin for evaluation overhead
    
    N = 30
    RANGE = 30
    
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
    
    # Add a few random starts
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
    
    # Target areas to optimize for
    # Small areas are common in grids.
    target_areas = [1, 2, 4, 8, 16, 32]
    
    def get_max_multiplicity(points):
        """
        Returns the maximum multiplicity of any doubled area.
        Used for final evaluation of candidate sets.
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

    def compute_initial_count(points, A):
        """
        Computes the count of triangles with area A for the initial set.
        """
        n = len(points)
        count = 0
        for i in range(n-2):
            x1, y1 = points[i]
            for j in range(i+1, n-1):
                x2, y2 = points[j]
                dx1 = x2 - x1
                dy1 = y2 - y1
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    d = dx1 * (y3 - y1) - dy1 * (x3 - x1)
                    if abs(d) == A:
                        count += 1
        return count

    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 2.0:
            break
            
        current_points = list(init_pts)
        if len(set(current_points)) != N:
            continue
        valid = True
        for p in current_points:
            if p[0] < -RANGE or p[0] > RANGE or p[1] < -RANGE or p[1] > RANGE:
                valid = False
                break
        if not valid:
            continue
            
        for A in target_areas:
            if time.time() > deadline - 2.0:
                break
                
            # Simulated Annealing to maximize count of area A
            T = 5.0
            min_T = 0.01
            cooling = 0.999
            current_score = compute_initial_count(current_points, A)
            best_local_score = current_score
            best_local_points = list(current_points)
            
            no_improve = 0
            max_no_improve = 5000 # Steps without improvement
            
            while time.time() < deadline - 2.0 and no_improve < max_no_improve:
                idx = random.randint(0, N-1)
                old_pt = current_points[idx]
                
                # Generate move
                # 80% local move, 20% random global move
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
                    
                # Incremental update: Calculate delta in count for area A
                # Triangles involving idx are affected.
                # Old contribution: Triangles (idx, i, j) where i,j != idx
                # New contribution: Triangles (new_pt, i, j) where i,j != idx
                
                delta = 0
                old_x, old_y = old_pt
                new_x, new_y = new_pt
                
                for i in range(N):
                    if i == idx:
                        continue
                    x1, y1 = current_points[i]
                    # Triangles (idx, i, j)
                    # We need to check all pairs (i, j) with j != idx, i
                    # To avoid double counting in the delta logic, we can just
                    # sum over all j != idx, i for the old point and subtract,
                    # then sum for new point and add.
                    
                    # Actually, it's easier to just iterate all j != idx, i
                    # and compute the area for (old_pt, p_i, p_j) and (new_pt, p_i, p_j)
                    
                    for j in range(N):
                        if j == idx or j == i:
                            continue
                        x2, y2 = current_points[j]
                        
                        # Area with old_pt
                        d_old = (x1 - old_x) * (y2 - old_y) - (y1 - old_y) * (x2 - old_x)
                        if abs(d_old) == A:
                            delta -= 1
                            
                        # Area with new_pt
                        d_new = (x1 - new_x) * (y2 - new_y) - (y1 - new_y) * (x2 - new_x)
                        if abs(d_new) == A:
                            delta += 1
                
                new_score = current_score + delta
                
                if delta > 0 or random.random() < np.exp(delta / T):
                    current_points[idx] = new_pt
                    current_score = new_score
                    if new_score > best_local_score:
                        best_local_score = new_score
                        best_local_points = list(current_points)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Revert
                    current_points[idx] = old_pt
                    no_improve += 1
                    
                T *= cooling
                if T < min_T:
                    T = min_T

            # After SA for this target area, evaluate the best local points
            # with the FULL metric (max multiplicity of ANY area)
            final_score = get_max_multiplicity(best_local_points)
            
            if final_score > best_score:
                best_score = final_score
                best_points = best_local_points

    # Final validation
    if len(best_points) != N or len(set(best_points)) != N:
         return gen_grid(6, 5, -2, -2)
    for p in best_points:
        if p[0] < -30 or p[0] > 30 or p[1] < -30 or p[1] > 30:
            return gen_grid(6, 5, -2, -2)

    return best_points
