"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter

_START_TIME = time.time()
TIME_LIMIT = 85  # safety margin before 90s kill

def _doubled_areas(points):
    """Compute all positive doubled areas (absolute determinants) for triples."""
    n = len(points)
    areas = []
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            for k in range(j+1, n):
                xk, yk = points[k]
                det = abs(xi*(yj-yk) + xj*(yk-yi) + xk*(yi-yj))
                if det > 0:
                    areas.append(det)
    return areas

def _max_multiplicity(points):
    """Return the largest count of any single positive doubled area."""
    areas = _doubled_areas(points)
    if not areas:
        return 0
    c = Counter(areas)
    return c.most_common(1)[0][1]

def _valid_points(points):
    """Check points are 30 distinct integer pairs in [-30,30]^2."""
    if len(points) != 30:
        return False
    seen = set()
    for x, y in points:
        if not (-30 <= x <= 30 and -30 <= y <= 30):
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    return True

def _grid_points(x0, y0, dx, dy, nx, ny):
    """Generate a grid of points."""
    pts = []
    for i in range(nx):
        for j in range(ny):
            x = x0 + i * dx
            y = y0 + j * dy
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
    return pts

def _lattice_points(a, b, c, d, nx, ny):
    """Generate points on a lattice: (a*i + c*j, b*i + d*j)."""
    pts = []
    for i in range(nx):
        for j in range(ny):
            x = a * i + c * j
            y = b * i + d * j
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
    return pts

def _generate_initial_candidates():
    """Generate diverse initial point configurations."""
    candidates = []
    
    # Various grid configurations
    for dx in range(1, 7):
        for dy in range(1, 7):
            for x0 in range(-15, 16, dx):
                for y0 in range(-15, 16, dy):
                    nx = min(30, (60) // dx + 1)
                    ny = min(30, (60) // dy + 1)
                    pts = _grid_points(x0, y0, dx, dy, nx, ny)
                    if len(pts) >= 30:
                        # Take first 30 or random 30
                        if len(pts) > 30:
                            random.shuffle(pts)
                            pts = pts[:30]
                        if len(pts) == 30 and _valid_points(pts):
                            candidates.append(pts)
    
    # Lattice-based configurations
    for a in range(1, 5):
        for b in range(1, 5):
            for c in range(-3, 4):
                for d in range(1, 5):
                    if a * d - b * c == 0:
                        continue
                    for x0 in range(-10, 11, 2):
                        for y0 in range(-10, 11, 2):
                            pts = []
                            for i in range(8):
                                for j in range(8):
                                    x = x0 + a * i + c * j
                                    y = y0 + b * i + d * j
                                    if -30 <= x <= 30 and -30 <= y <= 30:
                                        pts.append((x, y))
                            if len(pts) >= 30:
                                random.shuffle(pts)
                                pts = pts[:30]
                                if _valid_points(pts):
                                    candidates.append(pts)
    
    # Random points
    for _ in range(50):
        pts = []
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in pts:
                pts.append((x, y))
        candidates.append(pts)
    
    return candidates

def _local_search(points, time_budget):
    """Improve points via local search: move individual points."""
    best_pts = list(points)
    best_score = _max_multiplicity(best_pts)
    
    deadline = time.time() + time_budget
    no_improve_count = 0
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_x, old_y = best_pts[idx]
        
        # Try moving to a nearby position
        found_improvement = False
        for _ in range(20):
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            new_x = old_x + dx
            new_y = old_y + dy
            if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                continue
            if (new_x, new_y) in best_pts[:idx] + best_pts[idx+1:]:
                continue
            
            trial = best_pts[:]
            trial[idx] = (new_x, new_y)
            score = _max_multiplicity(trial)
            
            if score > best_score:
                best_pts = trial
                best_score = score
                no_improve_count = 0
                found_improvement = True
                break
            elif score == best_score and random.random() < 0.3:
                # Accept equal with some probability for exploration
                best_pts = trial
                no_improve_count = 0
        
        if not found_improvement:
            no_improve_count += 1
            if no_improve_count > 50:
                break
    
    return best_pts, best_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    random.seed(42)
    
    candidates = _generate_initial_candidates()
    
    # Evaluate all candidates
    best_pts = None
    best_score = 0
    
    for cand in candidates:
        if time.time() > _START_TIME + TIME_LIMIT - 5:
            break
        score = _max_multiplicity(cand)
        if score > best_score:
            best_score = score
            best_pts = cand
    
    if best_pts is None:
        # Fallback to simple grid
        return [(x, y) for x in range(6) for y in range(5)]
    
    # Local search on the best candidate
    remaining_time = TIME_LIMIT - (time.time() - _START_TIME) - 2
    if remaining_time > 1:
        best_pts, best_score = _local_search(best_pts, remaining_time)
    
    # Final validation
    if not _valid_points(best_pts):
        return [(x, y) for x in range(6) for y in range(5)]
    
    return best_pts
# EVOLVE-BLOCK-END
