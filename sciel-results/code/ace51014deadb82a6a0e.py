import numpy as np
import time
import random
import math
from collections import Counter, defaultdict
from itertools import combinations

def _canon_line(A, B, C):
    """Canonicalize a line (A, B, C) -> (A, B, C) such that gcd(|A|,|B|,|C|)=1 and first non-zero is positive."""
    if A == 0 and B == 0:
        return None
    g = math.gcd(abs(A), math.gcd(abs(B), abs(C)))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count incidences exactly."""
    I = 0
    for x, y in points:
        for A, B, C in lines:
            if A * x + B * y + C == 0:
                I += 1
    return I

def _build_candidate_lines(points, max_coord=20):
    """Build all candidate lines that pass through at least 2 points."""
    lines_dict = {}  # line -> set of point indices
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            # Line through (xi, yi) and (xj, yj)
            # A = yj - yi, B = xi - xj, C = xj*yi - xi*yj
            A = yj - yi
            B = xi - xj
            C = xj * yi - xi * yj
            cl = _canon_line(A, B, C)
            if cl:
                if cl not in lines_dict:
                    lines_dict[cl] = set()
                lines_dict[cl].add(i)
                lines_dict[cl].add(j)
    return lines_dict

def _select_top_lines(lines_dict, k=27):
    """Select top k lines by number of incident points."""
    sorted_lines = sorted(lines_dict.items(), key=lambda x: -len(x[1]))
    return [l for l, pts in sorted_lines[:k]]

def _seed_grid_v1():
    """Grid 6x9."""
    points = []
    for x in range(1, 7):
        for y in range(1, 10):
            points.append((x, y))
    lines_dict = _build_candidate_lines(points)
    lines = _select_top_lines(lines_dict, 27)
    return points, lines

def _seed_grid_v2():
    """Grid 3x18."""
    points = []
    for x in range(1, 4):
        for y in range(1, 19):
            points.append((x, y))
    lines_dict = _build_candidate_lines(points)
    lines = _select_top_lines(lines_dict, 27)
    return points, lines

def _seed_grid_v3():
    """Grid 2x27."""
    points = []
    for x in range(1, 3):
        for y in range(1, 28):
            points.append((x, y))
    lines_dict = _build_candidate_lines(points)
    lines = _select_top_lines(lines_dict, 27)
    return points, lines

def _seed_grid_v4():
    """Grid 9x6."""
    points = []
    for x in range(1, 10):
        for y in range(1, 7):
            points.append((x, y))
    lines_dict = _build_candidate_lines(points)
    lines = _select_top_lines(lines_dict, 27)
    return points, lines

def _seed_grid_v5():
    """Grid 18x3."""
    points = []
    for x in range(1, 19):
        for y in range(1, 4):
            points.append((x, y))
    lines_dict = _build_candidate_lines(points)
    lines = _select_top_lines(lines_dict, 27)
    return points, lines

def _seed_grid_v6():
    """Grid 27x2."""
    points = []
    for x in range(1, 28):
        for y in range(1, 3):
            points.append((x, y))
    lines_dict = _build_candidate_lines(points)
    lines = _select_top_lines(lines_dict, 27)
    return points, lines

def _seed_arithmetic():
    """Arithmetic progression points."""
    points = []
    for i in range(54):
        points.append((i, i * 2))
    lines_dict = _build_candidate_lines(points)
    lines = _select_top_lines(lines_dict, 27)
    return points, lines

def _seed_random():
    """Random points."""
    points = set()
    while len(points) < 54:
        x = random.randint(-15, 15)
        y = random.randint(-15, 15)
        points.add((x, y))
    points = list(points)
    lines_dict = _build_candidate_lines(points)
    lines = _select_top_lines(lines_dict, 27)
    return points, lines

def _local_search(points, lines, max_iters=20000, deadline=None):
    """Local search with point and line moves."""
    points = list(points)
    lines = list(lines)
    
    # Build incidence matrix
    M = np.zeros((54, 27), dtype=np.int8)
    for i in range(54):
        x, y = points[i]
        for j in range(27):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                M[i, j] = 1
    
    current_I = int(np.sum(M))
    best_I = current_I
    best_points = [tuple(p) for p in points]
    best_lines = [tuple(l) for l in lines]
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        if random.random() < 0.6:
            # Move a point
            i = random.randint(0, 53)
            old_x, old_y = points[i]
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            if dx == 0 and dy == 0:
                continue
            new_x, new_y = old_x + dx, old_y + dy
            new_pt = (new_x, new_y)
            
            if new_pt in points:
                continue
            
            delta = 0
            for j in range(27):
                A, B, C = lines[j]
                old_inc = M[i, j]
                new_inc = 1 if (A * new_x + B * new_y + C == 0) else 0
                delta += new_inc - old_inc
                
            if delta > 0 or random.random() < math.exp(delta / 0.5):
                points[i] = new_pt
                for j in range(27):
                    A, B, C = lines[j]
                    M[i, j] = 1 if (A * new_x + B * new_y + C == 0) else 0
                current_I += delta
                if current_I > best_I:
                    best_I = current_I
                    best_points = [tuple(p) for p in points]
                    best_lines = [tuple(l) for l in lines]
        else:
            # Change a line
            j = random.randint(0, 26)
            old_A, old_B, old_C = lines[j]
            
            # Try to find a line through 2 random points
            i1 = random.randint(0, 53)
            i2 = random.randint(0, 53)
            if i1 == i2:
                continue
            x1, y1 = points[i1]
            x2, y2 = points[i2]
            new_A = y2 - y1
            new_B = x1 - x2
            new_C = x2 * y1 - x1 * y2
            
            if new_A == 0 and new_B == 0:
                continue
            
            cl = _canon_line(new_A, new_B, new_C)
            if not cl:
                continue
            
            if cl in lines:
                continue
            
            new_A, new_B, new_C = cl
            
            delta = 0
            for i in range(54):
                x, y = points[i]
                old_inc = M[i, j]
                new_inc = 1 if (new_A * x + new_B * y + new_C == 0) else 0
                delta += new_inc - old_inc
                
            if delta > 0 or random.random() < math.exp(delta / 0.5):
                lines[j] = cl
                for i in range(54):
                    x, y = points[i]
                    M[i, j] = 1 if (new_A * x + new_B * y + new_C == 0) else 0
                current_I += delta
                if current_I > best_I:
                    best_I = current_I
                    best_points = [tuple(p) for p in points]
                    best_lines = [tuple(l) for l in lines]
                    
    return best_points, best_lines, best_I

def _polish(points, lines, deadline):
    """Polish by trying all point moves and line moves."""
    points = list(points)
    lines = list(lines)
    
    M = np.zeros((54, 27), dtype=np.int8)
    for i in range(54):
        x, y = points[i]
        for j in range(27):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                M[i, j] = 1
    
    current_I = int(np.sum(M))
    best_I = current_I
    best_points = [tuple(p) for p in points]
    best_lines = [tuple(l) for l in lines]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try moving each point
        for i in range(54):
            if time.time() > deadline:
                break
            x, y = points[i]
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in points:
                        continue
                    
                    delta = 0
                    for j in range(27):
                        A, B, C = lines[j]
                        old_inc = M[i, j]
                        new_inc = 1 if (A * nx + B * ny + C == 0) else 0
                        delta += new_inc - old_inc
                        
                    if delta > 0:
                        points[i] = (nx, ny)
                        for j in range(27):
                            A, B, C = lines[j]
                            M[i, j] = 1 if (A * nx + B * ny + C == 0) else 0
                        current_I += delta
                        improved = True
                        if current_I > best_I:
                            best_I = current_I
                            best_points = [tuple(p) for p in points]
                            best_lines = [tuple(l) for l in lines]
                        break
                else:
                    continue
                break
        
        # Try changing each line
        for j in range(27):
            if time.time() > deadline:
                break
            # Try lines through pairs of points
            best_delta = 0
            best_line = None
            for i1 in range(54):
                for i2 in range(i1 + 1, 54):
                    x1, y1 = points[i1]
                    x2, y2 = points[i2]
                    A = y2 - y1
                    B = x1 - x2
                    C = x2 * y1 - x1 * y2
                    cl = _canon_line(A, B, C)
                    if not cl or cl in lines:
                        continue
                    
                    delta = 0
                    for i in range(54):
                        x, y = points[i]
                        old_inc = M[i, j]
                        new_inc = 1 if (cl[0] * x + cl[1] * y + cl[2] == 0) else 0
                        delta += new_inc - old_inc
                        
                    if delta > best_delta:
                        best_delta = delta
                        best_line = cl
                        
            if best_line and best_delta > 0:
                lines[j] = best_line
                for i in range(54):
                    x, y = points[i]
                    M[i, j] = 1 if (best_line[0] * x + best_line[1] * y + best_line[2] == 0) else 0
                current_I += best_delta
                improved = True
                if current_I > best_I:
                    best_I = current_I
                    best_points = [tuple(p) for p in points]
                    best_lines = [tuple(l) for l in lines]
                    
    return best_points, best_lines, best_I

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines.
    """
    start_time = time.time()
    deadline = start_time + 590  # 40s safety margin
    
    best_I = 0
    best_points = None
    best_lines = None
    
    # Generate seeds
    seeds = [
        _seed_grid_v1(),
        _seed_grid_v2(),
        _seed_grid_v3(),
        _seed_grid_v4(),
        _seed_grid_v5(),
        _seed_grid_v6(),
        _seed_arithmetic(),
        _seed_random(),
        _seed_random(),
        _seed_random()
    ]
    
    # Evaluate each seed
    for i, (pts, lns) in enumerate(seeds):
        if time.time() > deadline - 60:
            break
        I = _count_incidences(pts, lns)
        if I > best_I:
            best_I = I
            best_points = pts
            best_lines = lns
    
    # Run local search on top seeds
    top_seeds = []
    for pts, lns in seeds:
        I = _count_incidences(pts, lns)
        top_seeds.append((I, pts, lns))
    top_seeds.sort(key=lambda x: -x[0])
    
    for I, pts, lns in top_seeds[:5]:
        if time.time() > deadline - 30:
            break
        res_pts, res_lns, res_I = _local_search(pts, lns, max_iters=10000, deadline=deadline - 30)
        if res_I > best_I:
            best_I = res_I
            best_points = res_pts
            best_lines = res_lns
    
    # Polish the best solution
    if best_points and time.time() < deadline - 10:
        res_pts, res_lns, res_I = _polish(best_points, best_lines, deadline - 5)
        if res_I > best_I:
            best_I = res_I
            best_points = res_pts
            best_lines = res_lns
    
    # Convert to numpy arrays
    points_arr = np.array(best_points, dtype=np.int64)
    lines_arr = np.array(best_lines, dtype=np.int64)
    
    return points_arr, lines_arr
