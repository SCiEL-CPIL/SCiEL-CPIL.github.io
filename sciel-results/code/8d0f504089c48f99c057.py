import random
import time
from math import gcd

def _get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = -(A * x1 + B * y1)
    
    g = gcd(abs(A), abs(B))
    if g == 0:
        g = abs(C)
    if g != 0:
        A //= g
        B //= g
        C //= g
        
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def _build_line_index(points):
    n = len(points)
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _get_line_key(points[i], points[j])
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    return line_points

def _evaluate_from_index(points, line_points):
    valid = True
    score = 0
    for key, idx_set in line_points.items():
        cnt = len(idx_set)
        if cnt >= 5:
            valid = False
            break
        elif cnt == 4:
            score += 1
    return valid, score

def _incremental_update(points, line_points, old_idx, old_pt, new_pt):
    n = len(points)
    lines_to_remove = []
    for key, idx_set in line_points.items():
        if old_idx in idx_set:
            lines_to_remove.append(key)
    
    for key in lines_to_remove:
        idx_set = line_points[key]
        idx_set.discard(old_idx)
        if len(idx_set) <= 1:
            del line_points[key]
    
    for j in range(n):
        if j == old_idx:
            continue
        key = _get_line_key(new_pt, points[j])
        if key not in line_points:
            line_points[key] = set()
        line_points[key].add(old_idx)
        line_points[key].add(j)
    
    return line_points

def _generate_informed_moves(points, line_points, idx, rng):
    old_x, old_y = points[idx]
    moves = []
    
    # 1) Small perturbations
    for _ in range(5):
        dx = rng.randint(-5, 5)
        dy = rng.randint(-5, 5)
        moves.append([old_x + dx, old_y + dy])
        
    # 2) Align with 3-point lines
    for key, idx_set in list(line_points.items()):
        if len(idx_set) == 3 and idx not in idx_set:
            pts_on_line = [points[i] for i in idx_set]
            A, B, C = key
            for t in range(-10, 11):
                if B != 0:
                    num = -C - A * t
                    if num % B == 0:
                        y_val = num // B
                        if abs(t) <= 1000 and abs(y_val) <= 1000:
                            cand = [t, y_val]
                            is_distinct = True
                            for i, p in enumerate(points):
                                if i != idx and p[0] == t and p[1] == y_val:
                                    is_distinct = False
                                    break
                            if is_distinct:
                                moves.append(cand)
                elif A != 0:
                    num = -C - B * t
                    if num % A == 0:
                        x_val = num // A
                        if abs(x_val) <= 1000 and abs(t) <= 1000:
                            cand = [x_val, t]
                            is_distinct = True
                            for i, p in enumerate(points):
                                if i != idx and p[0] == x_val and p[1] == t:
                                    is_distinct = False
                                    break
                            if is_distinct:
                                moves.append(cand)
                if len(moves) > 20:
                    break
        if len(moves) > 20:
            break
            
    # 3) Random jumps
    for _ in range(5):
        moves.append([rng.randint(-20, 20), rng.randint(-20, 20)])
        
    return moves

def _anneal_from_start(points, line_points, start_score, deadline, rng, initial_temp=1.0):
    current_score = start_score
    best_score = start_score
    best_points = [p[:] for p in points]
    
    temp = initial_temp
    cool_rate = 0.99
    
    while time.time() < deadline and temp > 0.01:
        idx = rng.randint(0, 15)
        old_pt = [points[idx][0], points[idx][1]]
        
        candidates = _generate_informed_moves(points, line_points, idx, rng)
        
        if not candidates:
            temp *= cool_rate
            continue
        
        new_pt = candidates[rng.randint(0, len(candidates) - 1)]
        
        if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
            temp *= cool_rate
            continue
        
        is_distinct = True
        for i, p in enumerate(points):
            if i != idx and p[0] == new_pt[0] and p[1] == new_pt[1]:
                is_distinct = False
                break
        if not is_distinct:
            temp *= cool_rate
            continue
        
        points[idx] = new_pt
        new_line_points = _incremental_update(points, line_points, idx, old_pt, new_pt)
        is_valid, new_score = _evaluate_from_index(points, new_line_points)
        
        if is_valid and new_score >= current_score:
            accept = True
        elif is_valid and new_score < current_score:
            delta = current_score - new_score
            prob = 2.718281828 ** (-delta / max(temp, 0.001))
            accept = rng.random() < prob
        else:
            accept = temp > 5.0 and rng.random() < 0.01
            
        if accept:
            current_score = new_score
            line_points = new_line_points
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
        else:
            points[idx] = old_pt
            line_points = _build_line_index(points)
            is_valid, current_score = _evaluate_from_index(points, line_points)
            
        temp *= cool_rate
        
    return best_points, best_score, points, line_points, current_score

def _affine_plane_grid():
    """Generate 16 points from affine plane of order 4 structure on integer grid.
    Uses points (x, y) where x, y in {0, 1, 4, 8} - this creates a 4x4 grid
    with spacing that preserves the affine plane line structure."""
    coords = [0, 1, 4, 8]
    points = []
    for x in coords:
        for y in coords:
            points.append([x, y])
    return points

def _perturbed_affine_grid(rng, scale=1):
    """Generate a perturbed affine plane grid to explore nearby configurations."""
    coords = [0, 1, 4, 8]
    points = []
    for x in coords:
        for y in coords:
            # Small random perturbations
            px = x + rng.randint(-scale, scale)
            py = y + rng.randint(-scale, scale)
            # Ensure bounds
            px = max(-1000, min(1000, px))
            py = max(-1000, min(1000, py))
            points.append([px, py])
    
    # Ensure distinctness
    unique_points = []
    for p in points:
        if p not in unique_points:
            unique_points.append(p)
        else:
            # If duplicate, nudge
            p = [p[0] + 1, p[1]]
            if p in unique_points:
                p = [p[0], p[1] + 1]
            unique_points.append(p)
    
    return unique_points[:16]

def _maxsat_local_search(points, line_points, deadline, rng):
    """Stochastic local search with Max-SAT-inspired clause weighting.
    
    Hard clauses: no 5+ collinear (penalty = large negative)
    Soft clauses: reward 4-collinear (positive weight)
    
    Uses simulated annealing with clause-based move evaluation."""
    current_score = 0
    current_valid = True
    best_score = 0
    best_points = [p[:] for p in points]
    
    # Initial evaluation
    current_valid, current_score = _evaluate_from_index(points, line_points)
    best_score = current_score
    
    temp = 3.0
    cool_rate = 0.995
    hard_penalty = 100  # Large penalty for 5+ collinear
    
    while time.time() < deadline and temp > 0.01:
        idx = rng.randint(0, 15)
        old_pt = [points[idx][0], points[idx][1]]
        
        # Generate moves with emphasis on line-forming positions
        candidates = _generate_informed_moves(points, line_points, idx, rng)
        
        # Add moves that specifically target creating 4-point lines
        # Find 3-point lines and try to add 4th point
        for key, idx_set in list(line_points.items()):
            if len(idx_set) == 3 and idx not in idx_set:
                A, B, C = key
                # Find integer points on this line
                if B != 0:
                    # Line: A*x + B*y + C = 0, solve for y given x
                    for x_test in range(max(-1000, points[idx][0] - 15), 
                                       min(1000, points[idx][0] + 15) + 1):
                        num = -C - A * x_test
                        if num % B == 0:
                            y_val = num // B
                            if abs(y_val) <= 1000:
                                cand = [x_test, y_val]
                                if cand != old_pt and cand not in points:
                                    candidates.append(cand)
                elif A != 0:
                    # Solve for x given y
                    for y_test in range(max(-1000, points[idx][1] - 15), 
                                       min(1000, points[idx][1] + 15) + 1):
                        num = -C - B * y_test
                        if num % A == 0:
                            x_val = num // A
                            if abs(x_val) <= 1000:
                                cand = [x_val, y_test]
                                if cand != old_pt and cand not in points:
                                    candidates.append(cand)
                
                if len(candidates) > 30:
                    break
        
        if not candidates:
            temp *= cool_rate
            continue
        
        new_pt = candidates[rng.randint(0, len(candidates) - 1)]
        
        if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
            temp *= cool_rate
            continue
        
        is_distinct = True
        for i, p in enumerate(points):
            if i != idx and p[0] == new_pt[0] and p[1] == new_pt[1]:
                is_distinct = False
                break
        if not is_distinct:
            temp *= cool_rate
            continue
        
        points[idx] = new_pt
        new_line_points = _incremental_update(points, line_points, idx, old_pt, new_pt)
        is_valid, new_score = _evaluate_from_index(points, new_line_points)
        
        # Max-SAT style: compute clause satisfaction
        if not is_valid:
            # Hard clause violated - large penalty
            new_clause_score = -hard_penalty
        else:
            new_clause_score = new_score
        
        if current_valid:
            current_clause_score = current_score
        else:
            current_clause_score = -hard_penalty
        
        delta = current_clause_score - new_clause_score
        
        if new_clause_score >= current_clause_score:
            accept = True
        else:
            prob = 2.718281828 ** (-delta / max(temp, 0.001))
            accept = rng.random() < prob
        
        if accept:
            current_score = new_score
            current_valid = is_valid
            line_points = new_line_points
            if is_valid and new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
        else:
            points[idx] = old_pt
            line_points = _build_line_index(points)
            current_valid, current_score = _evaluate_from_index(points, line_points)
            
        temp *= cool_rate
        
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    rng = random.Random(42)
    
    best_points = None
    best_score = 0
    
    # Strategy 1: Pure affine plane grid (known to give 20 four-point lines)
    points = _affine_plane_grid()
    line_points = _build_line_index(points)
    is_valid, current_score = _evaluate_from_index(points, line_points)
    if is_valid:
        best_points = [p[:] for p in points]
        best_score = current_score
    
    # Strategy 2: Perturbed affine grids with local search
    for scale in [1, 2, 3]:
        if time.time() >= deadline:
            break
        
        for restart in range(3):
            if time.time() >= deadline:
                break
            
            pts = _perturbed_affine_grid(rng, scale)
            pline_points = _build_line_index(pts)
            is_valid, p_score = _evaluate_from_index(pts, pline_points)
            
            if not is_valid:
                continue
            
            phase_deadline = min(deadline, time.time() + 4)
            new_best_points, new_best_score = _maxsat_local_search(
                pts, pline_points, phase_deadline, rng
            )
            
            if new_best_score > best_score:
                best_score = new_best_score
                best_points = [p[:] for p in new_best_points]
    
    # Strategy 3: Standard annealing from affine grid
    if time.time() < deadline:
        pts = [p[:] for p in _affine_plane_grid()]
        pline_points = _build_line_index(pts)
        is_valid, p_score = _evaluate_from_index(pts, pline_points)
        
        if is_valid:
            phase_deadline = min(deadline, time.time() + 10)
            new_best_points, new_best_score, _, _, _ = _anneal_from_start(
                pts, pline_points, p_score, phase_deadline, rng, initial_temp=2.0
            )
            
            if new_best_score > best_score:
                best_score = new_best_score
                best_points = [p[:] for p in new_best_points]
    
    # Strategy 4: Random starts with standard annealing
    for restart in range(10):
        if time.time() >= deadline:
            break
        
        pts = []
        while len(pts) < 16:
            p = [rng.randint(-50, 50), rng.randint(-50, 50)]
            if p not in pts:
                pts.append(p)
                
        pline_points = _build_line_index(pts)
        is_valid, p_score = _evaluate_from_index(pts, pline_points)
        
        if not is_valid:
            continue
            
        phase_deadline = min(deadline, time.time() + 3)
        new_best_points, new_best_score, _, _, _ = _anneal_from_start(
            pts, pline_points, p_score, phase_deadline, rng, initial_temp=2.0
        )
        
        if new_best_score > best_score:
            best_score = new_best_score
            best_points = [p[:] for p in new_best_points]
    
    # Fallback
    if best_points is None:
        return _affine_plane_grid()
        
    return best_points
