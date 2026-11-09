"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations
from fractions import Fraction

def _line_key(p1, p2):
    """Compute normalized line key (a, b, c) for line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    a = dy
    b = -dx
    c = -a * x1 - b * y1
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def _build_line_members(points):
    """Build line membership dict and count ordinary lines."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    ordinary = sum(1 for members in lines.values() if len(members) == 2)
    return lines, ordinary

def _delta_count_fast(points, line_members, ordinary_count, idx, new_pos):
    """
    Incrementally compute new ordinary count when moving point idx to new_pos.
    Returns (new_ordinary, new_line_members) or (None, None) if invalid.
    """
    if abs(new_pos[0]) > 10**9 or abs(new_pos[1]) > 10**9:
        return None, None

    # Check collision
    for j in range(12):
        if j != idx and points[j][0] == new_pos[0] and points[j][1] == new_pos[1]:
            return None, None

    # Check not all collinear after move
    new_pts = [p[:] for p in points]
    new_pts[idx] = list(new_pos)
    if not _valid(new_pts):
        return None, None

    # Incremental update: remove lines involving idx, add new lines
    # First, collect all lines that contain idx
    old_lines_with_idx = {}
    for key, members in line_members.items():
        if idx in members:
            old_lines_with_idx[key] = set(members)

    # Compute delta for each old line containing idx
    # For a line with k members including idx: removing idx gives k-1 members
    # If k==2: line disappears (was ordinary, -1)
    # If k==3: line becomes 2 (was not ordinary, becomes ordinary, +1)
    # If k>=4: line becomes k-1 (still not ordinary, no change)
    delta = 0
    for key, members in old_lines_with_idx.items():
        k = len(members)
        if k == 2:
            delta -= 1  # ordinary line removed
        elif k == 3:
            delta += 1  # new ordinary line created

    # Now add new lines from idx to each other point
    # For each other point j, compute line key between new_pos and points[j]
    new_line_keys = {}
    for j in range(12):
        if j == idx:
            continue
        key = _line_key(new_pos, points[j])
        if key not in new_line_keys:
            new_line_keys[key] = set()
        new_line_keys[key].add(j)

    # For each new line, compute its total membership
    # It includes idx + all j's mapped to this key
    # Plus any other points already on this line (from line_members, excluding idx)
    for key, js in new_line_keys.items():
        # Total members = 1 (idx) + len(js) + (other points on this line not in js and not idx)
        # Other points on this line: from old line_members, if key exists
        other_members = set()
        if key in line_members:
            for m in line_members[key]:
                if m != idx and m not in js:
                    other_members.add(m)
        total = 1 + len(js) + len(other_members)
        # Before: this line had (len(line_members[key]) - 1 if idx was in it else len(line_members[key])) members
        # Actually we need to check: was this line in old line_members?
        if key in line_members:
            old_total = len(line_members[key])
            # idx was in this line (since it's in old_lines_with_idx if idx was there)
            # But wait: idx might not have been on this line before
            # If idx was NOT on this line before: old had old_total members, new has total
            # If idx WAS on this line before: old had old_total members (including idx), 
            #   but we already handled the removal delta above
            # Let me reconsider...
            
            # The old line had old_total members. We removed idx from it (handled in delta above).
            # Now we're adding idx back plus potentially new connections.
            # Actually the issue is: some of the js might have been on this line before.
            # Let me just compute: new total members for this line
            # = idx + all points j such that line(new_pos, points[j]) == key
            # = 1 + len(js)
            # But some of those j might have already been on this line with other points.
            # The total set of points on this line in the new config:
            new_members = {idx} | js
            # Plus any point m (m != idx) that was on this line before AND is still on it
            # A point m is still on this line if line(points[m], new_pos) == key
            # which is equivalent to m being on the line through new_pos with direction key
            # This is complex. Let me use a simpler approach:
            # For each other point m, check if it's on the new line
            # But that's O(n) per line which is fine for n=12
            pass
        
        # Simpler: just check all 11 other points for collinearity with new_pos
        # This is O(11) per candidate line, and we have at most 11 candidate lines
        pass

    # Actually, let me use a cleaner incremental approach:
    # For the new configuration, the lines through idx are determined by new_pos and each other point.
    # For lines NOT through idx, they remain unchanged.
    # So: new_ordinary = old_ordinary - (ordinary lines through idx in old) + (ordinary lines through idx in new)
    #     + (change in ordinary status of lines NOT through idx, which is 0 since they're unchanged)
    
    # Lines through idx in old config:
    old_ordinary_through_idx = 0
    for key, members in old_lines_with_idx.items():
        if len(members) == 2:
            old_ordinary_through_idx += 1

    # Lines through idx in new config:
    # Group other points by line key with new_pos
    new_lines_through_idx = {}
    for j in range(12):
        if j == idx:
            continue
        key = _line_key(new_pos, points[j])
        if key not in new_lines_through_idx:
            new_lines_through_idx[key] = set()
        new_lines_through_idx[key].add(j)

    # For each new line through idx, total members = 1 + |js| + (other points on this line)
    # Other points on this line: points m (m != idx) where line(points[m], new_pos) == key
    # But that's the same as: m is on the line defined by key
    # A point (xm, ym) is on line (a,b,c) if a*xm + b*ym + c == 0
    new_ordinary_through_idx = 0
    for key, js in new_lines_through_idx.items():
        a, b, c = key
        total = 1 + len(js)  # idx + the js points
        # Check other points
        for m in range(12):
            if m == idx or m in js:
                continue
            xm, ym = points[m]
            if a * xm + b * ym + c == 0:
                total += 1
        if total == 2:
            new_ordinary_through_idx += 1

    new_ordinary = ordinary_count - old_ordinary_through_idx + new_ordinary_through_idx

    # Build new line members dict
    new_line_members = {}
    # Copy lines not through idx
    for key, members in line_members.items():
        if idx not in members:
            new_line_members[key] = set(members)
    # Add lines through idx in new config
    for key, js in new_lines_through_idx.items():
        a, b, c = key
        members = {idx} | js
        for m in range(12):
            if m == idx or m in js:
                continue
            xm, ym = points[m]
            if a * xm + b * ym + c == 0:
                members.add(m)
        new_line_members[key] = members

    return new_ordinary, new_line_members

def _count_ordinary(points):
    """Count the number of lines containing exactly 2 of the given points."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = 0
            lines[key] += 1
    
    ordinary = 0
    for key, cnt in lines.items():
        if cnt == 2:
            ordinary += 1
    return ordinary

def _valid(points):
    """Check if points are valid: 12 distinct integer points, |x|,|y| <= 10^9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if not isinstance(p, list) or len(p) != 2:
            return False
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    
    # Check not all collinear
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    if cross == 0:
        for i in range(3, 12):
            xi, yi = points[i]
            cross_i = (x2 - x1) * (yi - y1) - (y2 - y1) * (xi - x1)
            if cross_i != 0:
                return True
        return False
    return True

def _get_line_members(points):
    """Return dict mapping line key -> list of point indices on that line."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return lines

def _delta_count(points, line_members, ordinary_count, idx, new_pos):
    """
    Compute the new ordinary line count when moving point idx to new_pos.
    Uses incremental update for speed.
    Returns the new total count, or None if invalid.
    """
    new_ordinary, _ = _delta_count_fast(points, line_members, ordinary_count, idx, new_pos)
    return new_ordinary

def _delta_swap(points, line_members, ordinary_count, idx1, idx2):
    """
    Compute the new ordinary line count when swapping positions of points idx1 and idx2.
    Uses incremental update for speed.
    Returns the new total count, or None if invalid.
    """
    # Check collision after swap
    if points[idx1][0] == points[idx2][0] and points[idx1][1] == points[idx2][1]:
        return None
    
    new_points = [p[:] for p in points]
    new_points[idx1], new_points[idx2] = points[idx2][:], points[idx1][:]
    
    # Check validity
    if not _valid(new_points):
        return None
    
    # Use incremental: move idx1 to old idx2 pos, then idx2 to old idx1 pos
    # But simpler: just do two sequential moves
    old_pos1 = points[idx1][:]
    old_pos2 = points[idx2][:]
    
    # Move idx1 to old_pos2
    new_ordinary, new_line_members = _delta_count_fast(points, line_members, ordinary_count, idx1, old_pos2)
    if new_ordinary is None:
        return None
    
    # Move idx2 to old_pos1
    new_ordinary, new_line_members = _delta_count_fast(new_points, new_line_members, new_ordinary, idx2, old_pos1)
    if new_ordinary is None:
        return None
    
    return new_ordinary

def _generate_strategic_candidates(points, idx, limit=300):
    """
    Generate candidate positions for point idx that are likely to create collinearity.
    Focuses on positions that extend existing lines or create new 3+ point lines.
    """
    x, y = points[idx]
    candidates = set()
    others = [j for j in range(12) if j != idx]
    
    # 1. For each pair of other points (j, k), find integer points on line jk
    #    that are close to current position or create long lines
    for j in others:
        for k in others:
            if j >= k:
                continue
            xj, yj = points[j]
            xk, yk = points[k]
            dx = xk - xj
            dy = yk - yj
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            
            # Find t values where point is near current position
            # Point on line: (xj + t*dx_n, yj + t*dy_n)
            # Minimize distance to (x, y)
            # Project (x,y) onto line
            if dx_n != 0:
                t_approx = (x - xj) / dx_n
            elif dy_n != 0:
                t_approx = (y - yj) / dy_n
            else:
                continue
            
            # Try t values near projection and also small t values for long lines
            t_set = set()
            for t in range(int(t_approx) - 5, int(t_approx) + 6):
                t_set.add(t)
            # Also try t values that would place point near other points (to extend lines)
            for m in others:
                if m == j or m == k:
                    continue
                xm, ym = points[m]
                if dx_n != 0:
                    t_m = (xm - xj) / dx_n
                    for t in range(int(t_m) - 2, int(t_m) + 3):
                        t_set.add(t)
                elif dy_n != 0:
                    t_m = (ym - yj) / dy_n
                    for t in range(int(t_m) - 2, int(t_m) + 3):
                        t_set.add(t)
            
            for t in t_set:
                nx = xj + t * dx_n
                ny = yj + t * dy_n
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    # Check it's not the same as existing points
                    is_same = False
                    for m in range(12):
                        if m != idx and points[m][0] == nx and points[m][1] == ny:
                            is_same = True
                            break
                    if not is_same:
                        candidates.add((nx, ny))
    
    # 2. Small perturbations
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                is_same = False
                for m in range(12):
                    if m != idx and points[m][0] == nx and points[m][1] == ny:
                        is_same = True
                        break
                if not is_same:
                    candidates.add((nx, ny))
    
    # 3. Random candidates
    for _ in range(20):
        nx = x + random.randint(-20, 20)
        ny = y + random.randint(-20, 20)
        if abs(nx) <= 10**9 and abs(ny) <= 10**9:
            is_same = False
            for m in range(12):
                if m != idx and points[m][0] == nx and points[m][1] == ny:
                    is_same = True
                    break
            if not is_same:
                candidates.add((nx, ny))
    
    return list(candidates)[:limit]

def _hexagonal_initial():
    """
    Generate 12 points on a hexagonal lattice pattern.
    Uses the regular hexagon vertices plus center and edge midpoints.
    """
    points = []
    # Center
    points.append([0, 0])
    
    # 6 vertices of hexagon (radius 2 in hex lattice)
    hex_verts = [(2,0), (1,2), (-1,2), (-2,0), (-1,-2), (1,-2)]
    for i, j in hex_verts:
        x = 2*i + j
        y = j
        points.append([x, y])
    
    # 3 more points to make 12: edge midpoints or additional ring
    extra = [(1,0), (0,1), (-1,1)]
    for i, j in extra:
        x = 2*i + j
        y = j
        points.append([x, y])
    
    return points

def _affine_grid_initial():
    """
    Generate 12 points from a 3x4 grid with affine transformation.
    A 3x4 grid has 12 points with good collinearity structure.
    Apply random affine transforms to explore different configurations.
    """
    base = [[i, j] for i in range(4) for j in range(3)]
    # Apply random affine transform: (x', y') = (a*x + b*y + e, c*x + d*y + f)
    a, b, c, d = random.randint(1, 5), random.randint(-3, 3), random.randint(-3, 3), random.randint(1, 5)
    e, f = random.randint(-10, 10), random.randint(-10, 10)
    pts = []
    for x, y in base:
        pts.append([a*x + b*y + e, c*x + d*y + f])
    return pts

def _pappus_initial():
    """
    Generate 12 points from Pappus's hexagon theorem configuration.
    Pappus gives 9 points with 3 lines of 4, but we can extend.
    """
    # Two lines with 3 points each, plus 6 intersection points
    # Line 1: y=0, points at x=0,1,2
    # Line 2: y=1, points at x=0,1,2
    # Cross connections create more collinearities
    pts = [
        [0, 0], [1, 0], [2, 0],  # Line 1
        [0, 1], [1, 1], [2, 1],  # Line 2
        [0, 2], [1, 2], [2, 2],  # Line 3
        [0, 3], [1, 3], [2, 3],  # Line 4
    ]
    return pts

def _sa_search(points, line_members, count, deadline, best_points, best_count):
    """
    Simulated annealing with line snapping.
    Returns (best_points, best_count, best_line_members).
    """
    current = [p[:] for p in points]
    cur_lm, cur_cnt = _build_line_members(current)
    
    if cur_cnt < best_count:
        best_count = cur_cnt
        best_points = [p[:] for p in current]
    
    T = 5.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() < deadline and T > T_min:
        # Pick a random point to move
        i = random.randint(0, 11)
        
        # Generate a candidate position
        x, y = current[i]
        cand_type = random.random()
        
        if cand_type < 0.5:
            # Snap to a line through two other points
            others = [j for j in range(12) if j != i]
            j = random.choice(others)
            k = random.choice([m for m in others if m != j])
            xj, yj = current[j]
            xk, yk = current[k]
            dx = xk - xj
            dy = yk - yj
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                nx, ny = x + random.randint(-5, 5), y + random.randint(-5, 5)
            else:
                dx_n = dx // g
                dy_n = dy // g
                # Find t that puts point near current position
                if dx_n != 0:
                    t_approx = (x - xj) / dx_n
                elif dy_n != 0:
                    t_approx = (y - yj) / dy_n
                else:
                    nx, ny = x + random.randint(-5, 5), y + random.randint(-5, 5)
                    t_approx = None
                if t_approx is not None:
                    t = int(round(t_approx)) + random.randint(-3, 3)
                    nx = xj + t * dx_n
                    ny = yj + t * dy_n
                else:
                    nx, ny = x + random.randint(-5, 5), y + random.randint(-5, 5)
        elif cand_type < 0.75:
            # Snap to intersection of two lines
            others = [j for j in range(12) if j != i]
            if len(others) >= 4:
                j1, k1 = random.sample(others, 2)
                j2, k2 = random.sample([m for m in others if m not in (j1, k1)], 2)
                # Line 1 through j1, k1
                xj1, yj1 = current[j1]
                xk1, yk1 = current[k1]
                # Line 2 through j2, k2
                xj2, yj2 = current[j2]
                xk2, yk2 = current[k2]
                # Intersection of lines
                # Line 1: a1*x + b1*y + c1 = 0
                dx1 = xk1 - xj1
                dy1 = yk1 - yj1
                a1, b1 = dy1, -dx1
                c1 = -a1 * xj1 - b1 * yj1
                # Line 2: a2*x + b2*y + c2 = 0
                dx2 = xk2 - xj2
                dy2 = yk2 - yj2
                a2, b2 = dy2, -dx2
                c2 = -a2 * xj2 - b2 * yj2
                # Solve
                det = a1 * b2 - a2 * b1
                if det != 0:
                    nx = (b1 * c2 - b2 * c1) / det
                    ny = (a2 * c1 - a1 * c2) / det
                    nx = int(round(nx))
                    ny = int(round(ny))
                else:
                    nx, ny = x + random.randint(-5, 5), y + random.randint(-5, 5)
            else:
                nx, ny = x + random.randint(-5, 5), y + random.randint(-5, 5)
        else:
            # Random perturbation
            nx = x + random.randint(-10, 10)
            ny = y + random.randint(-10, 10)
        
        if abs(nx) > 10**9 or abs(ny) > 10**9:
            T *= cooling
            continue
        
        # Check collision
        valid = True
        for m in range(12):
            if m != i and current[m][0] == nx and current[m][1] == ny:
                valid = False
                break
        if not valid:
            T *= cooling
            continue
        
        new_count = _delta_count(current, cur_lm, cur_cnt, i, (nx, ny))
        if new_count is None:
            T *= cooling
            continue
        
        delta = new_count - cur_cnt
        if delta < 0 or random.random() < math.exp(-delta / T):
            current[i] = [nx, ny]
            cur_cnt = new_count
            cur_lm, cur_cnt = _build_line_members(current)
            
            if cur_cnt < best_count:
                best_count = cur_cnt
                best_points = [p[:] for p in current]
                if best_count <= 6:
                    return best_points, best_count, cur_lm
        
        T *= cooling
    
    return best_points, best_count, cur_lm

def _two_point_move(points, line_members, count, i, j, deadline):
    """
    Try coordinated moves of two points simultaneously.
    Returns (new_points, new_count, new_line_members) if improved, else None.
    """
    best_result = None
    best_cnt = count
    
    # Generate candidates for point i
    cand_i = _generate_strategic_candidates(points, i, limit=50)
    cand_j = _generate_strategic_candidates(points, j, limit=50)
    
    # Try combinations
    for ni in cand_i[:20]:
        if time.time() > deadline:
            break
        for nj in cand_j[:20]:
            if time.time() > deadline:
                break
            
            # Check collisions
            if ni[0] == nj[0] and ni[1] == nj[1]:
                continue
            if any(points[m][0] == ni[0] and points[m][1] == ni[1] for m in range(12) if m != i and m != j):
                continue
            if any(points[m][0] == nj[0] and points[m][1] == nj[1] for m in range(12) if m != i and m != j):
                continue
            
            # Apply both moves
            new_pts = [p[:] for p in points]
            new_pts[i] = list(ni)
            new_pts[j] = list(nj)
            
            if not _valid(new_pts):
                continue
            
            new_lm, new_c = _build_line_members(new_pts)
            if new_c < best_cnt:
                best_cnt = new_c
                best_result = (new_pts, new_c, new_lm)
    
    return best_result

def construct():
    """
    Construct 12 integer points with a minimized number of ordinary lines.
    Uses fast local search with incremental line membership updates.
    """
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    random.seed(42)
    
    # Initial candidates - expanded with more structured configurations
    candidates_list = [
        [[i, 0] for i in range(11)] + [[0, 1]],  # 11 on line, 1 off
        [[x, y] for x in range(4) for y in range(3)],  # 4x3 grid
        [[x, y] for x in range(6) for y in range(2)],  # 6x2 grid
        [[x, y] for x in range(3) for y in range(4)],  # 3x4 grid
        _hexagonal_initial(),  # Hexagonal lattice
        _affine_grid_initial(),  # Affine-transformed grid
        _pappus_initial(),  # Pappus-like configuration
    ]
    
    # Add more structured initial configurations
    # 3x3 grid plus 3 points on a line
    grid33 = [[x, y] for x in range(3) for y in range(3)]
    grid33 += [[4, 0], [4, 1], [4, 2]]
    candidates_list.append(grid33)
    
    # 2x6 grid
    candidates_list.append([[x, y] for x in range(6) for y in range(2)])
    
    # 12 points: 4 on each of 3 parallel lines
    for offset in range(3):
        row = [[i, offset*2] for i in range(4)]
        candidates_list.append(row + [[i, offset*2+1] for i in range(4)] + [[i, offset*2+2] for i in range(4)])
    
    # Configuration with multiple 3-point lines
    # 3 lines of 4 points each, sharing a common point
    common = [0, 0]
    line1 = [[1, 0], [2, 0], [3, 0]]
    line2 = [[0, 1], [0, 2], [0, 3]]
    line3 = [[1, 1], [2, 2], [3, 3]]
    candidates_list.append([common] + line1 + line2 + line3)
    
    # Affine transforms of 3x4 grid
    for seed_val in range(5):
        random.seed(seed_val + 100)
        candidates_list.append(_affine_grid_initial())
    random.seed(42)
    
    best_points = None
    best_count = float('inf')
    best_line_members = None
    
    for pts in candidates_list:
        if not _valid(pts):
            continue
        line_members, c = _build_line_members(pts)
        if c < best_count:
            best_count = c
            best_points = [p[:] for p in pts]
            best_line_members = line_members
    
    current_points = best_points
    current_count = best_count
    current_line_members = best_line_members
    
    # Local search with strategic moves, swaps, 2-point moves, and SA
    stuck_count = 0
    sa_used = False
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try moving individual points
        for i in range(12):
            if time.time() > deadline:
                break
            
            # Generate strategic candidates
            cand_positions = _generate_strategic_candidates(current_points, i, limit=200)
            
            best_move = None
            best_move_count = current_count
            
            for nx, ny in cand_positions:
                if time.time() > deadline:
                    break
                
                new_count = _delta_count(current_points, current_line_members, current_count, i, (nx, ny))
                if new_count is not None and new_count < best_move_count:
                    best_move_count = new_count
                    best_move = (i, nx, ny, new_count)
            
            if best_move:
                idx, nx, ny, new_count = best_move
                current_points[idx] = [nx, ny]
                current_count = new_count
                # Rebuild line members after move
                current_line_members, current_count = _build_line_members(current_points)
                improved = True
                stuck_count = 0
                
                if current_count <= 6:
                    return current_points
        
        # Phase 2: Try swapping pairs of points
        for i in range(12):
            if time.time() > deadline:
                break
            for j in range(i + 1, 12):
                if time.time() > deadline:
                    break
                
                new_count = _delta_swap(current_points, current_line_members, current_count, i, j)
                if new_count is not None and new_count < current_count:
                    # Perform the swap
                    current_points[i], current_points[j] = current_points[j][:], current_points[i][:]
                    current_count = new_count
                    # Rebuild line members after swap
                    current_line_members, current_count = _build_line_members(current_points)
                    improved = True
                    stuck_count = 0
                    
                    if current_count <= 6:
                        return current_points
        
        # Phase 3: Try 2-point coordinated moves
        if not improved:
            for i in range(12):
                if time.time() > deadline:
                    break
                for j in range(i + 1, 12):
                    if time.time() > deadline:
                        break
                    result = _two_point_move(current_points, current_line_members, current_count, i, j, deadline)
                    if result is not None:
                        current_points, current_count, current_line_members = result
                        improved = True
                        stuck_count = 0
                        if current_count <= 6:
                            return current_points
        
        if not improved:
            stuck_count += 1
            
            # Try larger perturbations
            for i in range(12):
                if time.time() > deadline:
                    break
                x, y = current_points[i]
                for _ in range(30):
                    nx = x + random.randint(-50, 50)
                    ny = y + random.randint(-50, 50)
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    new_count = _delta_count(current_points, current_line_members, current_count, i, (nx, ny))
                    if new_count is not None and new_count < current_count:
                        current_points[i] = [nx, ny]
                        current_count = new_count
                        current_line_members, current_count = _build_line_members(current_points)
                        improved = True
                        stuck_count = 0
                        break
                if improved:
                    break
            
            # Phase 4: Simulated annealing when stuck
            if not improved and stuck_count >= 2:
                sa_deadline = min(deadline, time.time() + 20)  # Limit SA to 20s per invocation
                best_points, best_count, best_line_members = _sa_search(
                    current_points, current_line_members, current_count,
                    sa_deadline, current_points, current_count
                )
                if best_count < current_count:
                    current_points = [p[:] for p in best_points]
                    current_count = best_count
                    current_line_members, current_count = _build_line_members(current_points)
                    stuck_count = 0
                    if current_count <= 6:
                        return current_points
                sa_used = True
            
            if not improved and stuck_count >= 5:
                # Random restart with hexagonal bias
                new_pts = []
                for _ in range(12):
                    ri = random.randint(-15, 15)
                    rj = random.randint(-15, 15)
                    new_pts.append([2*ri + rj, rj])
                if _valid(new_pts):
                    line_m, cnt = _build_line_members(new_pts)
                    if cnt < current_count:
                        current_points = new_pts
                        current_count = cnt
                        current_line_members = line_m
                        stuck_count = 0
                else:
                    break
    
    return current_points

# EVOLVE-BLOCK-END
