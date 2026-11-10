# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
from itertools import combinations
import random

def _canon_line(A, B, C):
    """Canonicalize a line (A, B, C) by dividing by gcd and ensuring first nonzero is positive."""
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return None
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count exact incidences using int64 arithmetic."""
    count = 0
    for (x, y) in points:
        for (A, B, C) in lines:
            if A * x + B * y + C == 0:
                count += 1
    return count

def _grid_diagonal_initialization():
    """
    Construct a high-incidence configuration using a grid + diagonals approach.
    
    Strategy: Place 54 points on a structured grid where many points are collinear.
    A 9x6 grid gives 54 points. Lines can be rows, columns, and diagonals.
    
    For a 9x6 grid (x in 0..8, y in 0..5):
    - 6 horizontal lines (rows): y = 0,1,2,3,4,5 -> each has 9 points
    - 9 vertical lines (columns): x = 0,1,...,8 -> each has 6 points  
    - Diagonals: various slopes
    
    We need exactly 27 lines. Let's use:
    - All 6 horizontal lines (9 points each) = 54 incidences
    - All 9 vertical lines (6 points each) = 54 incidences
    - 12 diagonal lines with various slopes
    
    Total from rows+cols: 108 incidences
    Plus diagonal contributions.
    
    Actually, let me think more carefully. With 27 lines and 54 points:
    If we use a 9x6 grid:
    - 6 rows: 6*9 = 54
    - 9 cols: 9*6 = 54  
    - That's 15 lines, 108 incidences
    - 12 more lines for diagonals
    
    Diagonals in a 9x6 grid with slope 1:
    y = x + c for various c. Points where 0<=x<9, 0<=y<6.
    c ranges from -8 to 5.
    c=0: (0,0),(1,1),...,(5,5) -> 6 points
    c=1: (0,1),...,(4,5) -> 5 points
    c=-1: (1,0),...,(5,4) -> 5 points
    etc.
    
    Let me just construct it and count.
    """
    # 9x6 grid
    nx, ny = 9, 6
    points = []
    for x in range(nx):
        for y in range(ny):
            points.append((x, y))
    
    # Select 27 lines: 6 rows + 9 cols + 12 diagonals
    lines = []
    
    # 6 horizontal lines: y = k -> 0*x + 1*y - k = 0
    for k in range(ny):
        lines.append((0, 1, -k))
    
    # 9 vertical lines: x = k -> 1*x + 0*y - k = 0
    for k in range(nx):
        lines.append((1, 0, -k))
    
    # 12 diagonal lines with slope 1: y = x + c -> -1*x + 1*y - c = 0
    # c from -5 to 6 (to get good coverage)
    # Let's pick c values that give the most points
    diag_scores = []
    for c in range(-nx, ny):
        # Count points on y = x + c
        count = 0
        for x in range(nx):
            y = x + c
            if 0 <= y < ny:
                count += 1
        if count >= 2:
            diag_scores.append((count, c))
    
    # Sort by count descending, take top 12
    diag_scores.sort(key=lambda x: (-x[0], x[1]))
    for count, c in diag_scores[:12]:
        lines.append((-1, 1, -c))
    
    # We should have 6 + 9 + 12 = 27 lines
    assert len(lines) == 27, f"Expected 27 lines, got {len(lines)}"
    
    # Canonicalize and deduplicate
    canon_lines = []
    seen = set()
    for l in lines:
        c = _canon_line(*l)
        if c is not None and c not in seen:
            seen.add(c)
            canon_lines.append(c)
    
    # If we lost some due to duplicates, we need to add more
    while len(canon_lines) < 27:
        # Add more diagonals with slope -1: y = -x + c -> 1*x + 1*y - c = 0
        c = len(canon_lines) - 15  # offset
        l = (1, 1, -c)
        cl = _canon_line(*l)
        if cl is not None and cl not in seen:
            seen.add(cl)
            canon_lines.append(cl)
        c += 1
    
    points_arr = np.array(points, dtype=np.int64)
    lines_arr = np.array(canon_lines[:27], dtype=np.int64)
    
    return points_arr, lines_arr

def _local_search(points, lines, max_time=300):
    """
    Local search: try moving individual points or swapping lines to improve incidence count.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    points = list(points)
    lines = list(lines)
    I = _count_incidences(points, lines)
    best_I = I
    best_points = points[:]
    best_lines = lines[:]
    
    # Precompute for each point, which lines it's on
    def point_incidence(pt):
        x, y = pt
        return sum(1 for (A, B, C) in lines if A * x + B * y + C == 0)
    
    # Point movement local search
    random.seed(42)
    no_improve_count = 0
    
    while time.time() < deadline - 60:
        improved = False
        
        # Try moving each point to a nearby location
        for idx in range(len(points)):
            if time.time() > deadline - 60:
                break
            
            x, y = points[idx]
            current_inc = point_incidence((x, y))
            
            # Try nearby integer points in a radius
            best_new_pt = None
            best_new_inc = current_inc
            
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = (x + dx, y + dy)
                    
                    # Check for duplicates
                    is_dup = False
                    for j, p in enumerate(points):
                        if j != idx and p == new_pt:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    
                    new_inc = 0
                    for (A, B, C) in lines:
                        if A * new_pt[0] + B * new_pt[1] + C == 0:
                            new_inc += 1
                    
                    if new_inc > best_new_inc:
                        best_new_inc = new_inc
                        best_new_pt = new_pt
            
            if best_new_pt is not None and best_new_inc > current_inc:
                points[idx] = best_new_pt
                I += best_new_inc - current_inc
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 2:
                # Try line swaps
                improved2 = False
                for i in range(len(lines)):
                    if time.time() > deadline - 60:
                        break
                    for j in range(i + 1, len(lines)):
                        # Try swapping line i with a different line
                        # Generate candidate lines through pairs of points
                        if len(points) < 2:
                            continue
                        p1, p2 = points[0], points[1]
                        # Line through p1 and p2
                        A = p2[1] - p1[1]
                        B = p1[0] - p2[0]
                        C = -(A * p1[0] + B * p1[1])
                        if A == 0 and B == 0:
                            continue
                        cl = _canon_line(A, B, C)
                        if cl is None:
                            continue
                        if cl in lines:
                            continue
                        
                        # Count incidences if we replace line i with cl
                        old_inc = 0
                        for p in points:
                            if lines[i][0] * p[0] + lines[i][1] * p[1] + lines[i][2] == 0:
                                old_inc += 1
                        
                        new_inc = 0
                        for p in points:
                            if cl[0] * p[0] + cl[1] * p[1] + cl[2] == 0:
                                new_inc += 1
                        
                        if new_inc > old_inc:
                            lines[i] = cl
                            I += new_inc - old_inc
                            improved2 = True
                            break
                    if improved2:
                        break
                
                if not improved2:
                    break
                no_improve_count = 0
        
        if I > best_I:
            best_I = I
            best_points = points[:]
            best_lines = lines[:]
    
    return np.array(best_points, dtype=np.int64), np.array(best_lines, dtype=np.int64), best_I

def _alternating_optimization(points, lines, max_time=200):
    """
    Alternating optimization:
    1. Given points, find best 27 lines
    2. Given lines, find best 54 points
    Repeat until convergence or time limit.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    points = list(points)
    lines = list(lines)
    I = _count_incidences(points, lines)
    best_I = I
    best_points = points[:]
    best_lines = lines[:]
    
    while time.time() < deadline - 30:
        # Step 1: Given points, find best lines
        # Generate candidate lines from all pairs of points
        candidate_lines = set()
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                p1, p2 = points[i], points[j]
                A = p2[1] - p1[1]
                B = p1[0] - p2[0]
                C = -(A * p1[0] + B * p1[1])
                if A == 0 and B == 0:
                    continue
                cl = _canon_line(A, B, C)
                if cl is not None:
                    candidate_lines.add(cl)
        
        # Also add some structured lines
        for c in range(-20, 21):
            for A, B in [(0, 1), (1, 0), (1, 1), (-1, 1), (1, -1)]:
                cl = _canon_line(A, B, -c)
                if cl is not None:
                    candidate_lines.add(cl)
        
        candidate_lines = list(candidate_lines)
        
        # Greedily select 27 lines that maximize total incidences
        selected_lines = []
        used = set()
        
        for _ in range(27):
            best_line = None
            best_score = -1
            
            for l in candidate_lines:
                if l in used:
                    continue
                A, B, C = l
                score = 0
                for p in points:
                    if A * p[0] + B * p[1] + C == 0:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_line = l
            
            if best_line is None:
                break
            selected_lines.append(best_line)
            used.add(best_line)
        
        if len(selected_lines) < 27:
            # Fill with random lines
            while len(selected_lines) < 27:
                A = random.randint(-5, 5)
                B = random.randint(-5, 5)
                C = random.randint(-10, 10)
                if A == 0 and B == 0:
                    continue
                cl = _canon_line(A, B, C)
                if cl is not None and cl not in used:
                    selected_lines.append(cl)
                    used.add(cl)
        
        lines = selected_lines
        
        # Step 2: Given lines, find best points
        # Generate candidate points from line intersections and nearby points
        candidate_points = set()
        
        # Intersections of line pairs
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                A1, B1, C1 = lines[i]
                A2, B2, C2 = lines[j]
                det = A1 * B2 - A2 * B1
                if det == 0:
                    continue
                x_num = B1 * C2 - B2 * C1
                y_num = A2 * C1 - A1 * C2
                if x_num % det == 0 and y_num % det == 0:
                    x = x_num // det
                    y = y_num // det
                    if abs(x) <= 100 and abs(y) <= 100:
                        candidate_points.add((x, y))
        
        # Also add grid points near existing points
        for p in points:
            x, y = p
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    candidate_points.add((x + dx, y + dy))
        
        candidate_points = list(candidate_points)
        
        # Greedily select 54 points that maximize total incidences
        selected_points = []
        used_pts = set()
        
        for _ in range(54):
            best_pt = None
            best_score = -1
            
            for p in candidate_points:
                if p in used_pts:
                    continue
                x, y = p
                score = 0
                for (A, B, C) in lines:
                    if A * x + B * y + C == 0:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_pt = p
            
            if best_pt is None:
                break
            selected_points.append(best_pt)
            used_pts.add(best_pt)
        
        if len(selected_points) < 54:
            # Fill with nearby points
            while len(selected_points) < 54:
                x = random.randint(-20, 20)
                y = random.randint(-20, 20)
                if (x, y) not in used_pts:
                    selected_points.append((x, y))
                    used_pts.add((x, y))
        
        points = selected_points
        
        # Evaluate
        I = _count_incidences(points, lines)
        if I > best_I:
            best_I = I
            best_points = points[:]
            best_lines = lines[:]
    
    return np.array(best_points, dtype=np.int64), np.array(best_lines, dtype=np.int64), best_I

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences using alternating optimization with
    grid+diagonal initialization.
    """
    start_time = time.time()
    deadline = 600.0  # 600s limit
    
    # Phase 1: Grid + diagonal initialization
    points_arr, lines_arr = _grid_diagonal_initialization()
    points = points_arr.tolist()
    lines = lines_arr.tolist()
    I = _count_incidences(points, lines)
    best_I = I
    best_points = points[:]
    best_lines = lines[:]
    
    # Phase 2: Local search
    elapsed = time.time() - start_time
    if elapsed < deadline - 120:
        points_arr, lines_arr, I = _local_search(points, lines, max_time=120)
        points = points_arr.tolist()
        lines = lines_arr.tolist()
        if I > best_I:
            best_I = I
            best_points = points[:]
            best_lines = lines[:]
    
    # Phase 3: Alternating optimization
    elapsed = time.time() - start_time
    if elapsed < deadline - 180:
        points_arr, lines_arr, I = _alternating_optimization(points, lines, max_time=180)
        points = points_arr.tolist()
        lines = lines_arr.tolist()
        if I > best_I:
            best_I = I
            best_points = points[:]
            best_lines = lines[:]
    
    # Phase 4: Final local search
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        points_arr, lines_arr, I = _local_search(points, lines, max_time=60)
        points = points_arr.tolist()
        lines = lines_arr.tolist()
        if I > best_I:
            best_I = I
            best_points = points[:]
            best_lines = lines[:]
    
    # Final conversion
    points_arr = np.array(best_points, dtype=np.int64)
    lines_arr = np.array(best_lines, dtype=np.int64)
    
    # Verify shapes
    assert points_arr.shape == (54, 2), f"Points shape: {points_arr.shape}"
    assert lines_arr.shape == (27, 3), f"Lines shape: {lines_arr.shape}"
    
    return points_arr, lines_arr

# EVOLVE-BLOCK-END
