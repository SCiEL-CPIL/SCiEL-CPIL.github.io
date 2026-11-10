# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd

def _normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) to a canonical form."""
    if A == 0 and B == 0:
        return (0, 0, 0)
    
    g = gcd(abs(A), abs(B))
    g = gcd(g, abs(C))
    
    if g > 1:
        A //= g
        B //= g
        C //= g
        
    if A < 0:
        A = -A
        B = -B
        C = -C
    elif A == 0 and B < 0:
        B = -B
        C = -C
        
    return (A, B, C)

def _count_incidences(points, lines):
    """
    Count total incidences between points and lines.
    """
    if len(points) == 0 or len(lines) == 0:
        return 0
    total = 0
    for A, B, C in lines:
        vals = A * points[:, 0] + B * points[:, 1] + C
        total += np.sum(vals == 0)
    return total

def _generate_line_pool_for_points(points, max_coord=20):
    """
    Generate a pool of lines from all pairs of points, plus some random lines.
    """
    pool = set()
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            key = _normalize_line(A, B, C)
            if key != (0, 0, 0):
                pool.add(key)
                
    # Add random lines to diversify
    for _ in range(500):
        A = random.randint(-max_coord, max_coord)
        B = random.randint(-max_coord, max_coord)
        C = random.randint(-max_coord*max_coord, max_coord*max_coord)
        if A != 0 or B != 0:
            pool.add(_normalize_line(A, B, C))
            
    return list(pool)

def _simulate_annealing(points, lines, pool_arr, timeout=580):
    """
    Simulated annealing on joint state of points and lines.
    """
    start_time = time.time()
    current_points = points.copy()
    current_lines = lines.copy()
    current_score = _count_incidences(current_points, current_lines)
    
    best_points = current_points.copy()
    best_lines = current_lines.copy()
    best_score = current_score
    
    # Cooling schedule
    T = 10.0
    alpha = 0.999  # Cooling rate
    min_T = 0.01
    
    n_points = len(current_points)
    n_lines = len(current_lines)
    
    # Precompute for speed
    def get_score(pts, lns):
        return _count_incidences(pts, lns)
    
    while time.time() < start_time + timeout:
        if T < min_T:
            break
            
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.7:
            # Move a point
            i = random.randint(0, n_points - 1)
            old_pt = current_points[i].copy()
            # Random new position
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            new_pt = old_pt + np.array([dx, dy], dtype=np.int64)
            
            # Check if point is already used
            is_used = False
            for j in range(n_points):
                if j != i and np.array_equal(current_points[j], new_pt):
                    is_used = True
                    break
            if is_used:
                continue
                
            # Compute delta
            old_score = get_score(current_points, current_lines)
            current_points[i] = new_pt
            new_score = get_score(current_points, current_lines)
            delta = new_score - old_score
            
            # Accept or reject
            if delta > 0 or random.random() < np.exp(delta / T):
                if new_score > best_score:
                    best_score = new_score
                    best_points = current_points.copy()
                    best_lines = current_lines.copy()
            else:
                current_points[i] = old_pt
                
        else:
            # Swap a line
            i = random.randint(0, n_lines - 1)
            old_line = current_lines[i].copy()
            # Pick a random line from pool
            j = random.randint(0, len(pool_arr) - 1)
            new_line = pool_arr[j].copy()
            
            # Check if line is already in current lines (after normalization)
            is_used = False
            for k in range(n_lines):
                if k != i:
                    # Check if same line
                    l1 = _normalize_line(*old_line)
                    l2 = _normalize_line(*new_line)
                    if l1 == l2:
                        is_used = True
                        break
            if is_used:
                continue
                
            # Compute delta
            old_score = get_score(current_points, current_lines)
            current_lines[i] = new_line
            new_score = get_score(current_points, current_lines)
            delta = new_score - old_score
            
            # Accept or reject
            if delta > 0 or random.random() < np.exp(delta / T):
                if new_score > best_score:
                    best_score = new_score
                    best_points = current_points.copy()
                    best_lines = current_lines.copy()
            else:
                current_lines[i] = old_line
                
        # Cool down
        T *= alpha
        
        # Occasionally re-generate pool to keep it fresh
        if random.random() < 0.01:
            pool = _generate_line_pool_for_points(current_points, max_coord=20)
            pool_arr = np.array(pool, dtype=np.int64)
            
    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    timeout = 590
    
    # 1. Initial points: 9x6 grid
    xs = np.arange(1, 10)
    ys = np.arange(1, 7)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # 2. Generate initial line pool
    pool = _generate_line_pool_for_points(points, max_coord=20)
    pool_arr = np.array(pool, dtype=np.int64)
    
    # 3. Select initial lines: top 27 by incidence
    x = points[:, 0]
    y = points[:, 1]
    A = pool_arr[:, 0]
    B = pool_arr[:, 1]
    C = pool_arr[:, 2]
    
    vals = A[:, None] * x[None, :] + B[:, None] * y[None, :] + C[:, None]
    incidence = (vals == 0).astype(np.int64)
    counts = np.sum(incidence, axis=1)
    
    top_k_indices = np.argsort(counts)[::-1][:27]
    lines = pool_arr[top_k_indices]
    
    # 4. Run simulated annealing
    best_points, best_lines, best_score = _simulate_annealing(points, lines, pool_arr, timeout=timeout - 10)
    
    # 5. Final local search to polish
    improved = True
    while improved and time.time() < start_time + timeout - 5:
        improved = False
        for i in range(len(best_points)):
            old_pt = best_points[i].copy()
            old_score = _count_incidences(best_points, best_lines)
            
            best_cand = None
            best_cand_score = old_score
            
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    cand = old_pt + np.array([dx, dy], dtype=np.int64)
                    is_used = False
                    for j in range(len(best_points)):
                        if j != i and np.array_equal(best_points[j], cand):
                            is_used = True
                            break
                    if is_used:
                        continue
                    
                    new_score = _count_incidences(np.vstack([best_points[:i], cand, best_points[i+1:]]), best_lines)
                    if new_score > best_cand_score:
                        best_cand_score = new_score
                        best_cand = cand
                    
            if best_cand is not None:
                best_points[i] = best_cand
                improved = True
                
        # Also try swapping lines
        for i in range(len(best_lines)):
            old_line = best_lines[i].copy()
            old_score = _count_incidences(best_points, best_lines)
            
            best_cand_line = None
            best_cand_score = old_score
            
            # Try a few random lines from pool
            for _ in range(20):
                j = random.randint(0, len(pool_arr) - 1)
                cand_line = pool_arr[j].copy()
                
                # Check if line is already in current lines
                is_used = False
                for k in range(len(best_lines)):
                    if k != i:
                        l1 = _normalize_line(*old_line)
                        l2 = _normalize_line(*cand_line)
                        if l1 == l2:
                            is_used = True
                            break
                if is_used:
                    continue
                
                new_lines = np.vstack([best_lines[:i], cand_line, best_lines[i+1:]])
                new_score = _count_incidences(best_points, new_lines)
                if new_score > best_cand_score:
                    best_cand_score = new_score
                    best_cand_line = cand_line
                    
            if best_cand_line is not None:
                best_lines[i] = best_cand_line
                improved = True
                
    return best_points, best_lines

# EVOLVE-BLOCK-END
