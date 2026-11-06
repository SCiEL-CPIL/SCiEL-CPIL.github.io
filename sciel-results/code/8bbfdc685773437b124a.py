import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s buffer for safety
    
    # Initial configuration: 6x5 grid centered in [-30, 30]
    # Using a grid that fits well and has high symmetry
    x_coords = [-10, -6, -2, 2, 6, 10]
    y_coords = [-8, -4, 0, 4, 8]
    initial_pts = [(x, y) for x in x_coords for y in y_coords]
    
    # Helper to compute areas for all triples
    def compute_areas(points):
        areas = []
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i+1, n):
                x2, y2 = points[j]
                dx1 = x2 - x1
                dy1 = y2 - y1
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    dx2 = x3 - x1
                    dy2 = y3 - y1
                    area = abs(dx1 * dy2 - dy1 * dx2)
                    if area > 0:
                        areas.append(area)
        return areas

    def build_counter(points):
        areas = compute_areas(points)
        return Counter(areas)

    # Initialize
    current_pts = [p for p in initial_pts]
    current_counter = build_counter(current_pts)
    current_score = max(current_counter.values()) if current_counter else 0
    
    best_pts = [p for p in current_pts]
    best_score = current_score
    
    occupied = set(current_pts)
    all_coords = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    # Precompute indices for faster access? No, N is small.
    
    # Function to evaluate a move incrementally
    def evaluate_move(points, idx, new_p, counter):
        old_p = points[idx]
        if old_p == new_p:
            return counter, max(counter.values()) if counter else 0
        
        n = len(points)
        # Collect indices of other points
        others = [i for i in range(n) if i != idx]
        
        old_x, old_y = old_p
        new_x, new_y = new_p
        
        deltas = {}
        
        # Calculate areas for all pairs of others with old_p and new_p
        for i in range(len(others)):
            a = others[i]
            xa, ya = points[a]
            for j in range(i + 1, len(others)):
                b = others[j]
                xb, yb = points[b]
                
                # Old area
                dxa = xa - old_x
                dya = ya - old_y
                dxb = xb - old_x
                dyb = yb - old_y
                old_area = abs(dxa * dyb - dya * dxb)
                
                # New area
                dxa_n = xa - new_x
                dya_n = ya - new_y
                dxb_n = xb - new_x
                dyb_n = yb - new_y
                new_area = abs(dxa_n * dyb_n - dya_n * dxb_n)
                
                if old_area > 0:
                    deltas[old_area] = deltas.get(old_area, 0) - 1
                if new_area > 0:
                    deltas[new_area] = deltas.get(new_area, 0) + 1
        
        # Apply deltas to a copy of the counter
        new_counter = counter.copy()
        for area, delta in deltas.items():
            if delta > 0:
                new_counter[area] += delta
            else:
                new_counter[area] += delta
                if new_counter[area] <= 0:
                    if new_counter[area] == 0:
                        del new_counter[area]
                    else:
                        # Should not happen if logic is correct, but handle gracefully
                        del new_counter[area]
        
        new_score = max(new_counter.values()) if new_counter else 0
        return new_counter, new_score

    stuck_count = 0
    max_stuck = 2000
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_p = current_pts[idx]
        
        # Pick a random empty point
        # To avoid infinite loop if all points are occupied (impossible with 30 pts in 61x61 grid)
        while True:
            new_p = random.choice(all_coords)
            if new_p not in occupied:
                break
        
        # Evaluate move
        new_counter, new_score = evaluate_move(current_pts, idx, new_p, current_counter)
        
        if new_score > current_score:
            # Accept move
            current_pts[idx] = new_p
            occupied.remove(old_p)
            occupied.add(new_p)
            current_counter = new_counter
            current_score = new_score
            stuck_count = 0
            
            if new_score > best_score:
                best_score = new_score
                best_pts = [p for p in current_pts]
        else:
            # Revert
            stuck_count += 1
            if stuck_count > max_stuck:
                # Restart from best state with random perturbation
                current_pts = [p for p in best_pts]
                occupied = set(current_pts)
                current_counter = build_counter(current_pts)
                current_score = best_score
                stuck_count = 0
                
                # Perturb: move a random point to a random empty spot
                if random.random() < 0.5:
                    idx_p = random.randint(0, 29)
                    old_p_p = current_pts[idx_p]
                    while True:
                        new_p_p = random.choice(all_coords)
                        if new_p_p not in occupied:
                            break
                    new_counter_p, _ = evaluate_move(current_pts, idx_p, new_p_p, current_counter)
                    current_pts[idx_p] = new_p_p
                    occupied.remove(old_p_p)
                    occupied.add(new_p_p)
                    current_counter = new_counter_p
                    # Do not update best_score here, just state

    return best_pts
