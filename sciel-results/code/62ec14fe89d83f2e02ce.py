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
    
    # Incremental update
    counter = Counter(prev_counter)
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
                counter[det] -= 1
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
                counter[det] += 1
    
    if not counter:
        return 0, counter
    return max(counter.values()), counter

def _structured_candidates():
    """Generate various structured grid-like candidates."""
    candidates = []
    
    # Standard grids
    for m, n in [(6,5), (5,6), (10,3), (3,10), (15,2), (2,15)]:
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
        for sx in [1, 2, 3]:
            for sy in [1, 2, 3]:
                pts = []
                for i in range(m):
                    for j in range(n):
                        x = i * sx - (m * sx) // 2
                        y = j * sy - (n * sy) // 2
                        pts.append((x, y))
                if _valid(pts):
                    candidates.append(pts)
    
    # Staggered rows
    for dx in [1, 2, 3]:
        for dy in [1, 2, 3]:
            pts = []
            for row in range(5):
                for col in range(6):
                    x = col * dx + (row % 2) * (dx // 2 + 1) - 15
                    y = row * dy - 10
                    pts.append((x, y))
            if _valid(pts):
                candidates.append(pts)
    
    # Hexagonal-like patterns
    for spacing in [2, 3, 4]:
        pts = []
        for row in range(6):
            for col in range(5):
                x = col * spacing + (row % 2) * (spacing // 2) - 10
                y = row * spacing - 12
                pts.append((x, y))
        if _valid(pts):
            candidates.append(pts)
    
    # Concentric rings
    for r in [5, 8, 12]:
        pts = []
        for angle in range(30):
            import math
            a = 2 * math.pi * angle / 30
            x = round(r * math.cos(a))
            y = round(r * math.sin(a))
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
        if len(pts) == 30 and _valid(pts):
            candidates.append(pts)
    
    return candidates

def _local_search(points, max_iterations=5000, time_limit=60.0):
    """Hill-climbing local search with incremental scoring and target-area focus."""
    start_time = time.time()
    best_points = list(points)
    best_score = _score(best_points)
    best_counter = _area_counter(best_points)
    
    if best_score == 0:
        return best_points
    
    target_area = best_counter.most_common(1)[0][0]
    occupied = set(best_points)
    
    neighborhoods = [
        [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4) if (dx, dy) != (0, 0)],
        [(dx, dy) for dx in range(-6, 7) for dy in range(-6, 7) if (dx, dy) != (0, 0)],
        [(dx, dy) for dx in range(-12, 13) for dy in range(-12, 13) if (dx, dy) != (0, 0)],
    ]
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit - 3:
            break
        
        improved = False
        nh_size = iteration % 3
        if time.time() - start_time > time_limit - 10:
            nh_size = min(nh_size, 1)
        
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
                    break
                elif s == best_score:
                    old_target_count = best_counter.get(target_area, 0)
                    new_target_count = new_counter.get(target_area, 0)
                    if new_target_count > old_target_count:
                        best_points = new_points
                        best_counter = new_counter
                        occupied.discard((px, py))
                        occupied.add(new_point)
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Try swapping two points
            if time.time() - start_time < time_limit - 5:
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
                            break
                    if improved:
                        break
            
            if not improved:
                # Random perturbation
                if time.time() - start_time < time_limit - 10:
                    random.seed(iteration * 7 + 13)
                    for _ in range(3):
                        trial = list(best_points)
                        for _ in range(2):
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
                                break
                    if improved:
                        continue
            
            if not improved:
                break
    
    return best_points

def construct():
    """Construct 30 points maximizing the multiplicity of a single doubled area."""
    start_time = time.time()
    deadline = start_time + 85
    
    candidates = _structured_candidates()
    
    best_points = None
    best_score = 0
    for cand in candidates:
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
    
    if best_points is None:
        return [(x, y) for x in range(6) for y in range(5)]
    
    remaining = deadline - time.time() - 5
    if remaining > 5:
        best_points = _local_search(best_points, max_iterations=8000, time_limit=remaining)
    
    # Try other candidates
    scored_candidates = []
    for cand in candidates:
        s = _score(cand)
        scored_candidates.append((s, cand))
    scored_candidates.sort(key=lambda x: -x[0])
    
    for s, cand in scored_candidates[:6]:
        if time.time() > deadline - 8:
            break
        if s >= best_score:
            continue
        time_budget = min(deadline - time.time() - 5, 10)
        if time_budget < 3:
            break
        result = _local_search(cand, max_iterations=1500, time_limit=time_budget)
        rs = _score(result)
        if rs > best_score:
            best_score = rs
            best_points = result
    
    return best_points
# EVOLVE-BLOCK-END
