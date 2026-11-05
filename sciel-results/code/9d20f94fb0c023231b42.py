import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 30
    RANGE = 30
    
    # Helper to calculate area
    def area(p1, p2, p3):
        return abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
    
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
    inits.append(gen_grid(15, 2, -6, 0))
    inits.append(gen_grid(2, 15, 0, -6))
    
    # Add some random ones
    for _ in range(5):
        random.seed(random.randint(0, 10000))
        pts = []
        while len(pts) < N:
            p = (random.randint(-RANGE, RANGE), random.randint(-RANGE, RANGE))
            if p not in pts:
                pts.append(p)
        inits.append(pts)

    best_overall_score = 0
    best_points = inits[0]
    
    for init_pts in inits:
        if time.time() > deadline - 5.0:
            break
            
        current_points = list(init_pts)
        if len(set(current_points)) != N:
            continue
        for p in current_points:
            if p[0] < -RANGE or p[0] > RANGE or p[1] < -RANGE or p[1] > RANGE:
                break
        else:
            # Compute initial areas
            areas = Counter()
            for i in range(N-2):
                x1, y1 = current_points[i]
                for j in range(i+1, N-1):
                    x2, y2 = current_points[j]
                    for k in range(j+1, N):
                        x3, y3 = current_points[k]
                        d = area((x1,y1), (x2,y2), (x3,y3))
                        if d > 0:
                            areas[d] += 1
            
            curr_max = max(areas.values()) if areas else 0
            curr_target_area = next((a for a, c in areas.items() if c == curr_max), 0)
            
            if curr_max > best_overall_score:
                best_overall_score = curr_max
                best_points = list(current_points)
            
            # Local Search with incremental updates
            no_improve = 0
            max_no_improve = 5000
            T = 0.5 # Temperature parameter for acceptance of equal/worse moves?
            
            while time.time() < deadline - 2.0 and no_improve < max_no_improve:
                idx = random.randint(0, N-1)
                old_pt = current_points[idx]
                
                # Generate candidate moves
                moves = []
                # Local moves
                for dx in [-2, -1, 0, 1, 2]:
                    for dy in [-2, -1, 0, 1, 2]:
                        if dx == 0 and dy == 0: continue
                        new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                        if -RANGE <= new_pt[0] <= RANGE and -RANGE <= new_pt[1] <= RANGE:
                            if new_pt not in current_points:
                                moves.append(new_pt)
                
                # Global moves
                for _ in range(3):
                    new_pt = (random.randint(-RANGE, RANGE), random.randint(-RANGE, RANGE))
                    if new_pt not in current_points and new_pt != old_pt:
                        moves.append(new_pt)
                
                if not moves:
                    no_improve += 1
                    continue
                    
                random.shuffle(moves)
                new_pt = moves[0] # Try only the first shuffled move to save time? 
                # Or try a few?
                # Given the speedup, we can try 1-2 moves per iteration.
                
                improved = False
                
                # Apply move
                current_points[idx] = new_pt
                
                # Update areas
                # We need to subtract old areas and add new areas for triangles involving idx
                # Triangles involving idx are formed by idx and any pair (j, k) from the other N-1 points
                # Let's iterate j from 0 to N-1, k from j+1 to N-1, skipping idx
                
                # To make it faster, we can pre-compute the list of other points?
                # No, just loop.
                
                # Store changes to revert if necessary
                # Or just revert by re-doing the loop with old_pt
                
                # 1. Subtract old contributions
                for j in range(N):
                    if j == idx: continue
                    x2, y2 = current_points[j]
                    for k in range(j+1, N):
                        if k == idx: continue
                        x3, y3 = current_points[k]
                        d_old = area(old_pt, (x2,y2), (x3,y3))
                        if d_old > 0:
                            areas[d_old] -= 1
                            if areas[d_old] == 0:
                                del areas[d_old] # Keep counter clean? Maybe not needed for max()
                
                # 2. Add new contributions
                for j in range(N):
                    if j == idx: continue
                    x2, y2 = current_points[j]
                    for k in range(j+1, N):
                        if k == idx: continue
                        x3, y3 = current_points[k]
                        d_new = area(new_pt, (x2,y2), (x3,y3))
                        if d_new > 0:
                            areas[d_new] += 1
                
                # 3. Evaluate new max
                new_max = max(areas.values()) if areas else 0
                
                # Acceptance criteria
                if new_max > curr_max:
                    # Accept
                    curr_max = new_max
                    no_improve = 0
                    if curr_max > best_overall_score:
                        best_overall_score = curr_max
                        best_points = list(current_points)
                        improved = True
                elif new_max == curr_max:
                    # Accept with some probability to escape local minima
                    if random.random() < 0.1:
                        no_improve += 1
                        # Keep the move
                    else:
                        # Revert
                        current_points[idx] = old_pt
                        # Revert areas
                        for j in range(N):
                            if j == idx: continue
                            x2, y2 = current_points[j]
                            for k in range(j+1, N):
                                if k == idx: continue
                                x3, y3 = current_points[k]
                                d_new = area(new_pt, (x2,y2), (x3,y3))
                                if d_new > 0:
                                    areas[d_new] -= 1
                                    if areas[d_new] == 0:
                                        del areas[d_new]
                                d_old = area(old_pt, (x2,y2), (x3,y3))
                                if d_old > 0:
                                    areas[d_old] += 1
                        no_improve += 1
                else:
                    # Revert
                    current_points[idx] = old_pt
                    # Revert areas
                    for j in range(N):
                        if j == idx: continue
                        x2, y2 = current_points[j]
                        for k in range(j+1, N):
                            if k == idx: continue
                            x3, y3 = current_points[k]
                            d_new = area(new_pt, (x2,y2), (x3,y3))
                            if d_new > 0:
                                areas[d_new] -= 1
                                if areas[d_new] == 0:
                                    del areas[d_new]
                            d_old = area(old_pt, (x2,y2), (x3,y3))
                            if d_old > 0:
                                areas[d_old] += 1
                    no_improve += 1

    # Final validation
    if len(best_points) != N or len(set(best_points)) != N:
         return gen_grid(6, 5, -2, -2)
    for p in best_points:
        if p[0] < -30 or p[0] > 30 or p[1] < -30 or p[1] > 30:
            return gen_grid(6, 5, -2, -2)

    return best_points
