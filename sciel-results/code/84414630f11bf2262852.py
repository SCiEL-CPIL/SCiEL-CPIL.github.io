"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import Counter

def _doubled_areas(points):
    """Compute all positive doubled areas for all triples."""
    n = len(points)
    areas = []
    for i in range(n):
        ax, ay = points[i]
        for j in range(i+1, n):
            bx, by = points[j]
            dx1 = bx - ax
            dy1 = by - ay
            for k in range(j+1, n):
                cx, cy = points[k]
                dx2 = cx - ax
                dy2 = cy - ay
                det = abs(dx1 * dy2 - dy1 * dx2)
                if det > 0:
                    areas.append(det)
    return areas

def _area_counter(points):
    """Return Counter of positive doubled areas."""
    areas = _doubled_areas(points)
    return Counter(areas)

def _max_multiplicity(points):
    """Return the maximum count of any single positive doubled area."""
    areas = _doubled_areas(points)
    if not areas:
        return 0
    return Counter(areas).most_common(1)[0][1]

def _valid(points):
    """Check 30 distinct integer points in [-30,30]^2."""
    if len(points) != 30:
        return False
    if len(set(points)) != 30:
        return False
    for x, y in points:
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if x < -30 or x > 30 or y < -30 or y > 30:
            return False
    return True

def _score(points):
    if not _valid(points):
        return 0
    return _max_multiplicity(points)

def _fast_score(points, prev_counter=None, moved_idx=None, old_point=None, new_point=None):
    """
    Fast scoring: if we only moved one point, we can update the area counter
    by removing areas involving the old point and adding areas with the new point.
    Returns (max_multiplicity, area_counter).
    """
    if prev_counter is None or moved_idx is None:
        counter = _area_counter(points)
        if not counter:
            return 0, counter
        return max(counter.values()), counter
    
    # Incremental update: remove triples containing moved_idx with old_point,
    # add triples containing moved_idx with new_point
    counter = dict(prev_counter)
    n = len(points)
    idx = moved_idx
    
    # Remove old contributions
    ox, oy = old_point
    for i in range(n):
        if i == idx:
            continue
        px, py = points[i]
        for j in range(i+1, n):
            if j == idx:
                continue
            qx, qy = points[j]
            det = abs((px - ox) * (qy - oy) - (py - oy) * (qx - ox))
            if det > 0:
                counter[det] = counter.get(det, 0) - 1
                if counter[det] <= 0:
                    del counter[det]
    
    # Add new contributions
    nx, ny = new_point
    for i in range(n):
        if i == idx:
            continue
        px, py = points[i]
        for j in range(i+1, n):
            if j == idx:
                continue
            qx, qy = points[j]
            det = abs((px - nx) * (qy - ny) - (py - ny) * (qx - nx))
            if det > 0:
                counter[det] = counter.get(det, 0) + 1
    
    if not counter:
        return 0, counter
    return max(counter.values()), counter

def _structured_candidates():
    """Generate various structured grid-like candidates."""
    candidates = []
    
    # Standard grids: m x n with m*n = 30
    for m, n in [(6,5), (5,6), (10,3), (3,10), (15,2), (2,15), (30,1), (1,30)]:
        pts = []
        for i in range(m):
            for j in range(n):
                x = i - m // 2
                y = j - n // 2
                pts.append((x, y))
        if _valid(pts):
            candidates.append(pts)
    
    # Grids with different spacings
    for m, n in [(6,5), (5,6), (10,3)]:
        for sx in [1, 2, 3, 4, 5]:
            for sy in [1, 2, 3, 4, 5]:
                pts = []
                valid = True
                for i in range(m):
                    for j in range(n):
                        x = i * sx - (m * sx) // 2
                        y = j * sy - (n * sy) // 2
                        if x < -30 or x > 30 or y < -30 or y > 30:
                            valid = False
                            break
                        pts.append((x, y))
                    if not valid:
                        break
                if valid and _valid(pts):
                    candidates.append(pts)
    
    # Interleaved: points on lines
    for offset in range(0, 6):
        pts = []
        for line in range(3):
            for k in range(10):
                x = k * 6 - 30 + offset
                y = line * 10 - 10
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        if _valid(pts):
            candidates.append(pts)
    
    # Staggered rows (hexagonal-like)
    for dx in [1, 2, 3, 4]:
        for dy in [1, 2, 3, 4]:
            pts = []
            valid = True
            for row in range(5):
                for col in range(6):
                    x = col * dx + (row % 2) * (dx // 2 + 1) - 15
                    y = row * dy - 10
                    if x < -30 or x > 30 or y < -30 or y > 30:
                        valid = False
                        break
                    pts.append((x, y))
                if not valid:
                    break
            if valid and _valid(pts):
                candidates.append(pts)
    
    # Arithmetic progression in both directions
    for step_x in [1, 2, 3, 4, 5]:
        for step_y in [1, 2, 3, 4, 5]:
            for start_x in range(-25, 26, 3):
                for start_y in range(-25, 26, 3):
                    pts = []
                    valid = True
                    for i in range(30):
                        row = i // 6
                        col = i % 6
                        x = start_x + col * step_x
                        y = start_y + row * step_y
                        if x < -30 or x > 30 or y < -30 or y > 30:
                            valid = False
                            break
                        pts.append((x, y))
                    if valid and _valid(pts):
                        candidates.append(pts)
    
    # Points concentrated on few parallel lines (for repeated cross-products)
    # 6 lines of 5 points each
    for spacing in [2, 3, 4, 5, 6, 8]:
        for y_offset in range(-20, 21, 4):
            pts = []
            valid = True
            for line in range(6):
                y = y_offset + line * spacing
                if y < -30 or y > 30:
                    valid = False
                    break
                for k in range(5):
                    x = k * 12 - 24
                    if x < -30 or x > 30:
                        valid = False
                        break
                    pts.append((x, y))
                if not valid:
                    break
            if valid and _valid(pts):
                candidates.append(pts)
    
    # 5 lines of 6 points each
    for spacing in [2, 3, 4, 5, 6, 8]:
        for y_offset in range(-20, 21, 4):
            pts = []
            valid = True
            for line in range(5):
                y = y_offset + line * spacing
                if y < -30 or y > 30:
                    valid = False
                    break
                for k in range(6):
                    x = k * 10 - 25
                    if x < -30 or x > 30:
                        valid = False
                        break
                    pts.append((x, y))
                if not valid:
                    break
            if valid and _valid(pts):
                candidates.append(pts)
    
    # Diagonal line clusters
    for slope in [1, 2, -1, -2]:
        for intercept in range(-20, 21, 3):
            pts = []
            for line_idx in range(5):
                b = intercept + line_idx * 6
                for k in range(6):
                    x = k * 5 - 12
                    y = slope * x + b
                    if -30 <= x <= 30 and -30 <= y <= 30 and isinstance(y, int):
                        pts.append((x, y))
            if len(pts) == 30 and _valid(pts):
                candidates.append(pts)
    
    # Concentric ring approximations (integer points near circles)
    for center_x in [-5, 0, 5]:
        for center_y in [-5, 0, 5]:
            pts = []
            for r in [4, 8, 12, 16, 20]:
                for angle_deg in range(0, 360, 60):
                    import math
                    rad = math.radians(angle_deg)
                    x = center_x + round(r * math.cos(rad))
                    y = center_y + round(r * math.sin(rad))
                    if -30 <= x <= 30 and -30 <= y <= 30:
                        pts.append((x, y))
            # Deduplicate and check
            seen = set()
            unique_pts = []
            for p in pts:
                if p not in seen:
                    seen.add(p)
                    unique_pts.append(p)
            if len(unique_pts) >= 30:
                # Take first 30
                candidates.append(unique_pts[:30])
            elif len(unique_pts) == 30:
                candidates.append(unique_pts)
    
    return candidates

def _local_search(points, max_iterations=5000, time_limit=60.0):
    """Hill-climbing local search with incremental scoring and target-area focus."""
    start_time = time.time()
    best_points = list(points)
    best_score = _score(best_points)
    best_counter = _area_counter(best_points)
    
    if best_score == 0:
        return best_points
    
    # Track the target area (the one with max multiplicity)
    target_area = best_counter.most_common(1)[0][0]
    
    # Precompute set of occupied points for O(1) lookup
    occupied = set(best_points)
    
    # Neighborhood sizes to try in sequence
    neighborhoods = [
        [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4) if (dx, dy) != (0, 0)],
        [(dx, dy) for dx in range(-6, 7) for dy in range(-6, 7) if (dx, dy) != (0, 0)],
        [(dx, dy) for dx in range(-10, 11) for dy in range(-10, 11) if (dx, dy) != (0, 0)],
        [(dx, dy) for dx in range(-15, 16) for dy in range(-15, 16) if (dx, dy) != (0, 0)],
    ]
    
    no_improve_count = 0
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit - 3:
            break
        
        improved = False
        # Cycle through neighborhood sizes, biasing toward smaller ones
        nh_size = iteration % len(neighborhoods)
        if time.time() - start_time > time_limit - 10:
            nh_size = min(nh_size, 1)  # Use smaller neighborhoods near deadline
        
        for idx in range(30):
            if time.time() - start_time > time_limit - 3:
                break
            
            px, py = best_points[idx]
            for dx, dy in neighborhoods[nh_size]:
                nx, ny = px + dx, py + dy
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue
                
                new_point = (nx, ny)
                if new_point in occupied:
                    continue
                
                # Fast incremental score
                new_points = list(best_points)
                new_points[idx] = new_point
                s, new_counter = _fast_score(new_points, best_counter, idx, (px, py), new_point)
                
                if s > best_score:
                    best_score = s
                    best_points = new_points
                    best_counter = new_counter
                    target_area = best_counter.most_common(1)[0][0]
                    occupied.discard((px, py))
                    occupied.add(new_point)
                    improved = True
                    no_improve_count = 0
                    break
                elif s == best_score:
                    # Tie-breaking: prefer moves that increase target_area count
                    old_target_count = best_counter.get(target_area, 0)
                    new_target_count = new_counter.get(target_area, 0)
                    if new_target_count > old_target_count:
                        best_points = new_points
                        best_counter = new_counter
                        occupied.discard((px, py))
                        occupied.add(new_point)
                        improved = True
                        no_improve_count = 0
                        break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            
            # If stuck, try swapping two points
            if no_improve_count >= 2 and time.time() - start_time < time_limit - 5:
                swap_improved = False
                for idx1 in range(30):
                    if time.time() - start_time > time_limit - 3:
                        break
                    for idx2 in range(idx1 + 1, 30):
                        if time.time() - start_time > time_limit - 3:
                            break
                        p1 = best_points[idx1]
                        p2 = best_points[idx2]
                        new_points = list(best_points)
                        new_points[idx1] = p2
                        new_points[idx2] = p1
                        
                        # Check validity
                        if len(set(new_points)) != 30:
                            continue
                        if any(x < -30 or x > 30 or y < -30 or y > 30 for x, y in new_points):
                            continue
                        
                        s = _score(new_points)
                        if s > best_score:
                            best_score = s
                            best_points = new_points
                            best_counter = _area_counter(best_points)
                            target_area = best_counter.most_common(1)[0][0]
                            occupied = set(best_points)
                            improved = True
                            swap_improved = True
                            no_improve_count = 0
                            break
                    if swap_improved:
                        break
            
            # If still stuck, try random perturbations
            if not improved and time.time() - start_time < time_limit - 8:
                random.seed(iteration * 7 + 13)
                for _ in range(8):
                    trial = list(best_points)
                    num_moves = random.randint(2, 5)
                    for _ in range(num_moves):
                        idx = random.randint(0, 29)
                        px, py = trial[idx]
                        nx = px + random.randint(-8, 8)
                        ny = py + random.randint(-8, 8)
                        if -30 <= nx <= 30 and -30 <= ny <= 30:
                            trial[idx] = (nx, ny)
                    if len(set(trial)) == 30:
                        s = _score(trial)
                        if s > best_score:
                            best_score = s
                            best_points = trial
                            best_counter = _area_counter(best_points)
                            target_area = best_counter.most_common(1)[0][0]
                            occupied = set(best_points)
                            improved = True
                            no_improve_count = 0
                            break
            
            if not improved and no_improve_count >= 15:
                break
    
    return best_points

def construct():
    """Construct 30 points maximizing the multiplicity of a single doubled area."""
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for the 90s limit
    
    candidates = _structured_candidates()
    
    # Score all candidates
    best_points = None
    best_score = 0
    for cand in candidates:
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
    
    if best_points is None:
        # Fallback to seed
        return [(x, y) for x in range(6) for y in range(5)]
    
    # Run local search on the best candidate with generous time
    remaining = deadline - time.time() - 5
    if remaining > 5:
        best_points = _local_search(best_points, max_iterations=10000, time_limit=remaining)
    
    # If we have time, try local search on other top candidates
    scored_candidates = []
    for cand in candidates:
        s = _score(cand)
        scored_candidates.append((s, cand))
    scored_candidates.sort(key=lambda x: -x[0])
    
    for s, cand in scored_candidates[:10]:
        if time.time() > deadline - 8:
            break
        if s >= best_score:
            continue
        time_budget = min(deadline - time.time() - 5, 10)
        if time_budget < 3:
            break
        result = _local_search(cand, max_iterations=3000, time_limit=time_budget)
        rs = _score(result)
        if rs > best_score:
            best_score = rs
            best_points = result
    
    return best_points
# EVOLVE-BLOCK-END
