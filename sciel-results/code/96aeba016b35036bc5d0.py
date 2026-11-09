import numpy as np
import time
import random
from math import gcd
from itertools import product, combinations

def normalize_line(A, B, C):
    """Normalize line coefficients to ensure uniqueness and integer validity."""
    if A == 0 and B == 0:
        return None
    
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        g = 1
    A //= g
    B //= g
    C //= g
    
    # Make first non-zero entry positive
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def count_incidences(points, lines):
    """
    points: (n, 2) array
    lines: (m, 3) array (A, B, C)
    Returns number of (p, l) such that A*x + B*y + C == 0
    """
    if len(points) == 0 or len(lines) == 0:
        return 0
    P = points.astype(np.int64)
    L = lines.astype(np.int64)
    
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    
    vals = A * px + B * py + C
    count = np.sum(vals == 0)
    return int(count)

def get_best_points_from_lines(lines, max_candidates=5000):
    """
    Generate candidate points from intersections of lines and select the best 54.
    """
    candidates = set()
    L = lines
    m = len(L)
    
    # Add intersections of all pairs of lines
    for i in range(m):
        for j in range(i + 1, m):
            A1, B1, C1 = L[i]
            A2, B2, C2 = L[j]
            
            det = A1 * B2 - A2 * B1
            if det == 0:
                continue # Parallel or same line (normalized ensures same line is unlikely unless duplicate, but we check)
                
            x_num = B1 * C2 - B2 * C1
            y_num = A1 * C2 - A2 * C1
            
            # Check if integer
            if x_num % det == 0 and y_num % det == 0:
                x = x_num // det
                y = y_num // det
                # Limit coordinate range to avoid huge numbers which might cause issues or just be "bad" for density
                if -1000 <= x <= 1000 and -1000 <= y <= 1000:
                    candidates.add((x, y))
    
    # If we have fewer than 54 candidates, add some grid points near the centroid
    if len(candidates) < 54:
        if candidates:
            xs = [p[0] for p in candidates]
            ys = [p[1] for p in candidates]
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
        else:
            cx, cy = 0, 0
            
        # Add grid points
        for dx in range(-10, 11):
            for dy in range(-10, 11):
                candidates.add((cx + dx, cy + dy))
                if len(candidates) >= 54:
                    break
            if len(candidates) >= 54:
                break

    # Score all candidates
    cand_list = np.array(list(candidates), dtype=np.int64)
    if len(cand_list) == 0:
        return np.zeros((54, 2), dtype=np.int64)
        
    px = cand_list[:, 0].reshape(-1, 1)
    py = cand_list[:, 1].reshape(-1, 1)
    
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    
    vals = A * px + B * py + C
    counts = np.sum(vals == 0, axis=1)
    
    # Get top 54
    top_indices = np.argsort(counts)[::-1][:54]
    best_points = cand_list[top_indices]
    
    return best_points

def generate_grid_lines(k, b_max):
    """
    Generate lines y = a*x + b for a in 1..k, b in 1..b_max.
    Total lines = k * b_max.
    """
    lines = []
    for a in range(1, k + 1):
        for b in range(1, b_max + 1):
            lines.append((a, -1, b))
    return np.array(lines, dtype=np.int64)

def generate_pencil_lines(center, slopes, offsets):
    """
    Generate lines passing through (center_x + ox, center_y + oy) with given slopes.
    Line: y - y0 = m(x - x0) => m*x - y + (y0 - m*x0) = 0
    """
    lines = []
    cx, cy = center
    for m in slopes:
        for ox, oy in offsets:
            x0 = cx + ox
            y0 = cy + oy
            # m*x - y + (y0 - m*x0) = 0
            A = m
            B = -1
            C = y0 - m * x0
            n_line = normalize_line(A, B, C)
            if n_line:
                lines.append(n_line)
    return np.array(lines, dtype=np.int64)

def generate_bipartite_lines(k, m):
    """
    Lines x = i for i in 1..k and y = j for j in 1..m.
    Total lines = k + m.
    """
    lines = []
    for i in range(1, k + 1):
        lines.append((1, 0, -i))
    for j in range(1, m + 1):
        lines.append((0, 1, -j))
    return np.array(lines, dtype=np.int64)

def generate_diagonal_lines(k, m, shift=0):
    """
    Lines x + y = c and x - y = c.
    """
    lines = []
    # x + y = c
    for c in range(shift, shift + k):
        lines.append((1, 1, -c))
    # x - y = c
    for c in range(shift, shift + m):
        lines.append((1, -1, -c))
    return np.array(lines, dtype=np.int64)

def local_search_lines(lines, points, time_limit, seed=42):
    """
    Refine lines given fixed points? No, refine both.
    Strategy: Try swapping lines to increase incidences with the current points,
    then re-optimize points.
    """
    random.seed(seed)
    best_I = count_incidences(points, lines)
    best_lines = lines.copy()
    best_points = points.copy()
    
    start_time = time.time()
    no_improve = 0
    
    while time.time() - start_time < time_limit:
        if no_improve > 100:
            break
            
        # Pick a random line to replace
        idx = random.randint(0, len(lines) - 1)
        old_line = lines[idx]
        
        # Generate a new candidate line
        # Try lines defined by pairs of points in the current set
        if len(points) >= 2:
            p1_idx, p2_idx = random.sample(range(len(points)), 2)
            p1 = points[p1_idx]
            p2 = points[p2_idx]
            
            # Line through p1, p2
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # Normalize (dx, -dy, dx*p1[1] - dy*p1[0])?
            # A x + B y + C = 0
            # dx (x - x1) - dy (y - y1) = 0
            # dx x - dy y + (dy x1 - dx y1) = 0
            A = dx
            B = -dy
            C = dy * p1[0] - dx * p1[1]
            
            new_line = normalize_line(A, B, C)
        else:
            new_line = None
            
        if new_line is None or new_line in [tuple(l) for l in lines]:
            no_improve += 1
            continue
            
        # Try replacing
        lines[idx] = new_line
        
        # Re-optimize points for new lines
        new_points = get_best_points_from_lines(lines)
        new_I = count_incidences(new_points, lines)
        
        if new_I > best_I:
            best_I = new_I
            best_lines = lines.copy()
            best_points = new_points.copy()
            no_improve = 0
        else:
            # Revert
            lines[idx] = old_line
            no_improve += 1
            
    return best_lines, best_points, best_I

def szemeredi_trotter_incidences():
    deadline = time.time() + 600 # 600s limit, keep margin
    best_I_global = -1
    best_points_global = None
    best_lines_global = None
    
    # Strategy 1: Grids
    # 3x18, 2x27, 1x54, 6x9, 9x6
    grid_configs = [
        (3, 18, 3, 9),   # k=3 slopes, b_max=9 -> 27 lines. Points 3x18=54
        (2, 27, 2, 13),  # k=2 slopes, b_max=13 -> 26 lines. Add 1?
        (1, 54, 1, 54),  # k=1 slope, b_max=54 -> 54 lines. Too many.
        (6, 9, 6, 9),    # k=6 slopes, b_max=9 -> 54 lines. Too many.
        (9, 6, 9, 6),    # k=9 slopes, b_max=6 -> 54 lines. Too many.
        (3, 9, 3, 9),    # 27 lines.
        (2, 14, 2, 14),  # 28 lines -> take 27.
        (4, 7, 4, 7),    # 28 lines -> take 27.
        (5, 6, 5, 6),    # 30 lines -> take 27.
    ]
    
    for k, b_max, pk, pm in grid_configs:
        if time.time() > deadline - 30:
            break
            
        lines = generate_grid_lines(k, b_max)
        if len(lines) > 27:
            lines = lines[:27]
        elif len(lines) < 27:
            # Pad with random lines
            while len(lines) < 27:
                A = random.randint(-10, 10)
                B = random.randint(-10, 10)
                if A == 0 and B == 0: continue
                C = random.randint(-50, 50)
                nl = normalize_line(A, B, C)
                if nl and nl not in [tuple(l) for l in lines]:
                    lines = np.vstack([lines, [nl]])
        
        points = get_best_points_from_lines(lines)
        I = count_incidences(points, lines)
        
        if I > best_I_global:
            best_I_global = I
            best_points_global = points
            best_lines_global = lines
            
        # Local search on this configuration
        if time.time() < deadline - 60:
            bl, bp, bi = local_search_lines(lines, points, time_limit=10, seed=hash((k, b_max)))
            if bi > best_I_global:
                best_I_global = bi
                best_points_global = bp
                best_lines_global = bl

    # Strategy 2: Pencils
    # Lines through (0,0) with various slopes
    # If we use slopes 1..27, all pass through (0,0).
    # Points: intersections are (0,0). Only 1 point.
    # We need distinct points.
    # So use "Near Pencils": Lines through (0,0) and (1,0).
    # Or lines through a small set of hubs.
    
    hub_configs = [
        [(0,0)],
        [(0,0), (1,0)],
        [(0,0), (0,1)],
        [(0,0), (1,1)],
        [(0,0), (1,0), (0,1)],
    ]
    
    for hubs in hub_configs:
        if time.time() > deadline - 30:
            break
            
        lines = []
        # Assign lines to hubs
        # For each hub, generate lines with small integer slopes passing through it
        slopes = list(range(-10, 11)) + [100, -100, 200, -200] # Add steep slopes
        random.shuffle(slopes)
        
        for i, hub in enumerate(hubs):
            num_lines_for_hub = 27 // len(hubs) + (1 if i < 27 % len(hubs) else 0)
            for m in slopes[:num_lines_for_hub]:
                # y - y0 = m(x - x0)
                # m x - y + (y0 - m x0) = 0
                A = m
                B = -1
                C = hub[1] - m * hub[0]
                nl = normalize_line(A, B, C)
                if nl and nl not in [tuple(l) for l in lines]:
                    lines.append(nl)
        
        # Pad if necessary
        while len(lines) < 27:
            A = random.randint(-10, 10)
            B = random.randint(-10, 10)
            if A == 0 and B == 0: continue
            C = random.randint(-50, 50)
            nl = normalize_line(A, B, C)
            if nl and nl not in [tuple(l) for l in lines]:
                lines.append(nl)
        
        if len(lines) > 27:
            lines = lines[:27]
            
        lines = np.array(lines, dtype=np.int64)
        points = get_best_points_from_lines(lines)
        I = count_incidences(points, lines)
        
        if I > best_I_global:
            best_I_global = I
            best_points_global = points
            best_lines_global = lines
            
        if time.time() < deadline - 60:
            bl, bp, bi = local_search_lines(lines, points, time_limit=10, seed=hash(tuple(hubs)))
            if bi > best_I_global:
                best_I_global = bi
                best_points_global = bp
                best_lines_global = bl

    # Strategy 3: Bipartite / Grid Lines
    # x = 1..k, y = 1..m
    for k in range(1, 28):
        m = 27 - k
        if m < 1: continue
        if time.time() > deadline - 10: break
        
        lines = generate_bipartite_lines(k, m)
        points = get_best_points_from_lines(lines)
        I = count_incidences(points, lines)
        
        if I > best_I_global:
            best_I_global = I
            best_points_global = points
            best_lines_global = lines

    # Strategy 4: Diagonals
    for k in range(1, 15):
        m = 27 - k
        if m < 1: continue
        if time.time() > deadline - 10: break
        
        lines = generate_diagonal_lines(k, m)
        points = get_best_points_from_lines(lines)
        I = count_incidences(points, lines)
        
        if I > best_I_global:
            best_I_global = I
            best_points_global = points
            best_lines_global = lines

    # Final Safety Check
    if best_points_global is None or best_lines_global is None:
        # Fallback to a simple grid
        lines = generate_grid_lines(3, 9)
        points = get_best_points_from_lines(lines)
        best_points_global = points
        best_lines_global = lines
        
    # Validate Points
    unique_points, indices = np.unique(best_points_global, axis=0, return_index=True)
    if len(unique_points) != 54:
        # This shouldn't happen if get_best_points_from_lines works correctly, but just in case
        # Fallback
        lines = generate_grid_lines(3, 9)
        points = get_best_points_from_lines(lines)
        best_points_global = points
        best_lines_global = lines
    else:
        best_points_global = unique_points[indices]
        
    # Validate Lines
    unique_lines = []
    for line in best_lines_global:
        n_line = normalize_line(line[0], line[1], line[2])
        if n_line and n_line not in unique_lines:
            unique_lines.append(n_line)
            
    if len(unique_lines) < 27:
        # Fallback
        lines = generate_grid_lines(3, 9)
        points = get_best_points_from_lines(lines)
        best_points_global = points
        best_lines_global = lines
    else:
        best_lines_global = np.array(unique_lines[:27], dtype=np.int64)

    return best_points_global, best_lines_global

# EVOLVE-BLOCK-END
