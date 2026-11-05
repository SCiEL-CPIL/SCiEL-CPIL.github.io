import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s buffer for 60s limit
    
    N = 30
    RANGE = 30
    
    # Helper to generate a hexagonal patch of n points
    def gen_hex_patch(n, center=(0, 0)):
        pts = []
        cx, cy = center
        # Add center
        pts.append((cx, cy))
        # Add rings
        r = 1
        while len(pts) < n:
            # 6 sides of hexagon
            # Directions: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) in axial coords?
            # Let's use simple cartesian offsets for a "loose" hex or just a square grid rotated?
            # A true hex lattice in cartesian integer coords is tricky without irrational scales.
            # The task requires INTEGER coordinates.
            # A "hexagonal" arrangement in integer grid is usually a rhombus or a parallelogram grid.
            # Let's stick to rectangular grids and "staggered" grids which are integer-friendly.
            
            # Staggered grid (like a brick wall)
            # Row 0: x = 0, 1, 2, ...
            # Row 1: x = 0.5, 1.5 ... -> Not integer.
            # So we can't have a perfect hex lattice in Z^2.
            # We can have a "triangular" lattice subset?
            # Points (x, y) such that x+y is even? No, that's a checkerboard.
            # The best integer approximations to hex lattices are rectangular grids or "diagonal" grids.
            
            # Let's generate a rectangular grid and a "diagonal" grid (rotated 45 deg approx).
            pass
        
        # Since we can't have a perfect hex lattice in Z^2, we rely on the search to find the optimal integer configuration.
        # However, we can generate "quasi-hex" configurations:
        # e.g., rows of alternating lengths.
        pts = []
        # Row lengths: 4, 5, 5, 5, 5, 4, 2? Sum=30?
        # Let's try a compact cluster.
        # 6x5 grid is 30.
        # 5x6 grid is 30.
        # Let's generate a few variations of compact grids.
        return pts

    # Better Initial Configurations
    def gen_grid(w, h, offset_x=0, offset_y=0):
        pts = []
        for i in range(w):
            for j in range(h):
                pts.append((i + offset_x, j + offset_y))
        return pts

    def gen_staggered_grid(w, h, offset_x=0, offset_y=0):
        # Creates a grid where odd rows are shifted by 1
        pts = []
        for j in range(h):
            for i in range(w):
                x = i + offset_x
                y = j + offset_y
                if j % 2 == 1:
                    x += 1
                pts.append((x, y))
        return pts

    inits = []
    # Standard grids
    inits.append(gen_grid(6, 5, -2, -2))
    inits.append(gen_grid(5, 6, -2, -2))
    inits.append(gen_grid(10, 3, -4, -1))
    inits.append(gen_grid(3, 10, -1, -4))
    
    # Staggered grids (might create more equal areas due to symmetry)
    inits.append(gen_staggered_grid(6, 5, -2, -2))
    inits.append(gen_staggered_grid(5, 6, -2, -2))
    
    # Random compact clusters
    for _ in range(5):
        pts = []
        while len(pts) < N:
            p = (random.randint(-5, 5), random.randint(-5, 5))
            if p not in pts:
                pts.append(p)
        inits.append(pts)

    best_overall_score = 0
    best_points = inits[0]
    
    def init_state(points):
        area_counts = Counter()
        freq_counts = Counter()
        p = points
        for i in range(N-2):
            xi, yi = p[i]
            for j in range(i+1, N-1):
                xj, yj = p[j]
                dx1 = xj - xi
                dy1 = yj - yi
                for k in range(j+1, N):
                    xk, yk = p[k]
                    dx2 = xk - xi
                    dy2 = yk - yi
                    det = dx1 * dy2 - dy1 * dx2
                    if det < 0:
                        det = -det
                    if det > 0:
                        area_counts[det] += 1
        
        for area, cnt in area_counts.items():
            freq_counts[cnt] += 1
            
        max_score = max(freq_counts.keys()) if freq_counts else 0
        return area_counts, freq_counts, max_score

    def apply_move(points, idx, old_pt, new_pt, area_counts, freq_counts, max_score):
        # Apply move and update counts
        points[idx] = new_pt
        
        delta_area_counts = {}
        
        # Remove triangles with old_pt
        # Note: points[idx] is now new_pt, so we use old_pt explicitly for "old" calcs
        for j in range(N):
            if j == idx: continue
            xj, yj = points[j]
            for k in range(j+1, N):
                if k == idx: continue
                xk, yk = points[k]
                
                # Old area
                det_old = (xj - old_pt[0]) * (yk - old_pt[1]) - (xk - old_pt[0]) * (yj - old_pt[1])
                if det_old < 0: det_old = -det_old
                if det_old > 0:
                    delta_area_counts[det_old] = delta_area_counts.get(det_old, 0) - 1
                
                # New area
                det_new = (xj - new_pt[0]) * (yk - new_pt[1]) - (xk - new_pt[0]) * (yj - new_pt[1])
                if det_new < 0: det_new = -det_new
                if det_new > 0:
                    delta_area_counts[det_new] = delta_area_counts.get(det_new, 0) + 1
        
        new_max_score = max_score
        for area, delta in delta_area_counts.items():
            if delta == 0:
                continue
                
            old_count = area_counts.get(area, 0)
            new_count = old_count + delta
            
            # Update freq_counts
            if old_count > 0:
                freq_counts[old_count] -= 1
                if freq_counts[old_count] == 0:
                    del freq_counts[old_count]
            
            if new_count > 0:
                freq_counts[new_count] = freq_counts.get(new_count, 0) + 1
            # If new_count <= 0, it's removed from area_counts, so no freq update needed for 0 count
            
            # Update area_counts
            if new_count > 0:
                area_counts[area] = new_count
            else:
                if area in area_counts:
                    del area_counts[area]
            
            if new_count > new_max_score:
                new_max_score = new_count
        
        # If max_score count dropped to 0, recalculate
        if max_score in freq_counts and freq_counts[max_score] == 0:
            if freq_counts:
                new_max_score = max(freq_counts.keys())
            else:
                new_max_score = 0
        elif new_max_score < max_score:
            # This case is handled by the check above if the specific area that had max_score was reduced.
            # But what if max_score was 10, and we reduced one area from 10 to 9, but another area still has 10?
            # Then freq_counts[10] is still > 0. So max_score stays 10.
            # The check `if max_score in freq_counts and freq_counts[max_score] == 0` covers the case where NO area has max_score anymore.
            pass

        return new_max_score

    def revert_move(points, idx, new_pt, old_pt, area_counts, freq_counts, max_score):
        # Revert is just applying the move from new_pt to old_pt
        # But we must be careful: apply_move assumes points[idx] is the NEW point.
        # Currently points[idx] is new_pt. We want to go to old_pt.
        # So we call apply_move(points, idx, new_pt, old_pt, ...)
        # This will remove triangles with new_pt (which are currently in the count)
        # and add triangles with old_pt.
        # This is correct.
        return apply_move(points, idx, new_pt, old_pt, area_counts, freq_counts, max_score)

    # Main Search
    for init_pts in inits:
        if time.time() > deadline - 5.0:
            break
            
        # Validate init
        if len(set(init_pts)) != N:
            continue
        valid = True
        for p in init_pts:
            if p[0] < -RANGE or p[0] > RANGE or p[1] < -RANGE or p[1] > RANGE:
                valid = False
                break
        if not valid:
            continue
            
        current_points = init_pts[:]
        area_counts, freq_counts, current_score = init_state(current_points)
        
        if current_score > best_overall_score:
            best_overall_score = current_score
            best_points = current_points[:]
            
        # Simulated Annealing
        T = 1.0
        no_improve = 0
        max_no_improve = 2000
        step = 0
        max_steps = 50000 # Cap steps to avoid infinite loop if time check fails
        
        while time.time() < deadline - 2.0 and no_improve < max_no_improve and step < max_steps:
            step += 1
            idx = random.randint(0, N-1)
            old_pt = current_points[idx]
            
            # Generate moves
            # Mix of local and global
            moves = []
            
            # Local moves
            step_size = 1 if random.random() < 0.7 else 2
            for dx in range(-step_size, step_size+1):
                for dy in range(-step_size, step_size+1):
                    if dx == 0 and dy == 0: continue
                    new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                    if -RANGE <= new_pt[0] <= RANGE and -RANGE <= new_pt[1] <= RANGE:
                        # Check distinctness
                        valid = True
                        for j in range(N):
                            if j != idx and current_points[j] == new_pt:
                                valid = False
                                break
                        if valid:
                            moves.append(new_pt)
                            
            # Global move occasionally
            if random.random() < 0.1:
                new_pt = (random.randint(-RANGE, RANGE), random.randint(-RANGE, RANGE))
                valid = True
                for j in range(N):
                    if j != idx and current_points[j] == new_pt:
                        valid = False
                        break
                if valid:
                    moves.append(new_pt)
                    
            if not moves:
                no_improve += 1
                continue
                
            random.shuffle(moves)
            improved = False
            
            for new_pt in moves:
                new_score = apply_move(current_points, idx, old_pt, new_pt, area_counts, freq_counts, current_score)
                delta_score = new_score - current_score
                
                if delta_score > 0:
                    accept = True
                else:
                    # SA Acceptance
                    # If delta_score == 0, accept with prob 0.5 to explore
                    # If delta_score < 0, accept with prob exp(delta_score / T)
                    if delta_score == 0:
                        accept = random.random() < 0.5
                    else:
                        # delta_score is integer, usually -1, -2, etc.
                        # T starts at 1.0, cools down.
                        accept = random.random() < (2.71828 ** (delta_score / T))
                
                if accept:
                    current_score = new_score
                    improved = True
                    no_improve = 0
                    if current_score > best_overall_score:
                        best_overall_score = current_score
                        best_points = current_points[:]
                    break
                else:
                    # Revert
                    revert_move(current_points, idx, new_pt, old_pt, area_counts, freq_counts, current_score)
                    # Note: revert_move updates the counters back.
                    # The score variable current_score remains unchanged (correct).
                    
            if not improved:
                no_improve += 1
                
            # Cooling
            T *= 0.995
            if T < 0.05:
                T = 0.05
    
    # Final Validation
    if len(best_points) != N or len(set(best_points)) != N:
        return gen_grid(6, 5, -2, -2)
    for p in best_points:
        if p[0] < -30 or p[0] > 30 or p[1] < -30 or p[1] > 30:
            return gen_grid(6, 5, -2, -2)

    return best_points
