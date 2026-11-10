import numpy as np
import time
from math import gcd
from itertools import combinations

def _canon(A, B, C):
    A, B, C = int(A), int(B), int(C)
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and (B < 0 or (B == 0 and C < 0))):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_from_two_points(p1, p2):
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    return _canon(A, B, C)

def _compute_all_lines(points):
    """
    Compute all unique lines through pairs of points and their point sets.
    Returns dict: line_key -> set of point indices on that line
    """
    n = len(points)
    line_to_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_from_two_points(points[i], points[j])
            if key not in line_to_points:
                line_to_points[key] = set()
            line_to_points[key].add(i)
            line_to_points[key].add(j)
    return line_to_points

def _objective(points):
    """
    Compute the objective: sum of top-27 line incidences for the given 54 points.
    Returns (score, lines_array).
    """
    n = len(points)
    line_to_points = _compute_all_lines(points)
    
    if not line_to_points:
        return 0, np.zeros((27, 3), dtype=np.int64)
    
    # Count incidences for each line
    counts = []
    for key, pts in line_to_points.items():
        counts.append((len(pts), key))
    
    # Sort by count descending
    counts.sort(key=lambda x: -x[0])
    
    # Take top 27
    top27 = counts[:27]
    score = sum(c for c, _ in top27)
    lines = np.array([k for _, k in top27], dtype=np.int64)
    
    return score, lines

def _objective_fast(points, line_cache=None):
    """
    Faster objective using precomputed line structures where possible.
    """
    n = len(points)
    line_to_points = _compute_all_lines(points)
    
    if not line_to_points:
        return 0, np.zeros((27, 3), dtype=np.int64)
    
    counts = []
    for key, pts in line_to_points.items():
        counts.append((len(pts), key))
    
    counts.sort(key=lambda x: -x[0])
    
    top27 = counts[:27]
    score = sum(c for c, _ in top27)
    lines = np.array([k for _, k in top27], dtype=np.int64)
    
    return score, lines

def _local_search(points, grid_points, rng, max_iters=500, deadline=None):
    """
    Local search: try swapping one point or two points with random grid points.
    """
    current_score, _ = _objective(points)
    P = points.copy()
    G = grid_points
    n_grid = len(G)
    
    # Build a set of current points for fast lookup
    current_set = set(map(tuple, P.tolist()))
    
    iter_count = 0
    no_improve_count = 0
    
    while iter_count < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        
        iter_count += 1
        
        # With 50% probability, do a 2-point swap; otherwise 1-point swap
        if rng.random() < 0.5 and n_grid >= 2:
            # 2-point swap
            idx1 = rng.integers(0, 54)
            idx2 = rng.integers(0, 54)
            if idx1 == idx2:
                continue
            old_pt1 = P[idx1].copy()
            old_pt2 = P[idx2].copy()
            
            best_new_score = current_score
            best_new_pt1 = None
            best_new_pt2 = None
            
            for _ in range(3):
                new_idx1 = rng.integers(0, n_grid)
                new_idx2 = rng.integers(0, n_grid)
                new_pt1 = G[new_idx1].copy()
                new_pt2 = G[new_idx2].copy()
                
                # Check validity
                t1 = tuple(new_pt1.tolist())
                t2 = tuple(new_pt2.tolist())
                t_old1 = tuple(old_pt1.tolist())
                t_old2 = tuple(old_pt2.tolist())
                
                # new_pt1 shouldn't be in current set (unless it's old_pt2)
                if t1 in current_set and t1 != t_old2 and t1 != t_old1:
                    continue
                # new_pt2 shouldn't be in current set (unless it's old_pt1)
                if t2 in current_set and t2 != t_old1 and t2 != t_old2:
                    continue
                # new_pt1 and new_pt2 should be different
                if t1 == t2:
                    continue
                
                P[idx1] = new_pt1
                P[idx2] = new_pt2
                new_score, _ = _objective(P)
                
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_new_pt1 = new_pt1.copy()
                    best_new_pt2 = new_pt2.copy()
                
                P[idx1] = old_pt1
                P[idx2] = old_pt2
            
            if best_new_pt1 is not None:
                current_set.discard(tuple(old_pt1.tolist()))
                current_set.discard(tuple(old_pt2.tolist()))
                current_set.add(tuple(best_new_pt1.tolist()))
                current_set.add(tuple(best_new_pt2.tolist()))
                P[idx1] = best_new_pt1
                P[idx2] = best_new_pt2
                current_score = best_new_score
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # 1-point swap
            idx = rng.integers(0, 54)
            old_pt = P[idx].copy()
            
            best_new_pt = None
            best_new_score = current_score
            
            for _ in range(5):
                new_idx = rng.integers(0, n_grid)
                new_pt = G[new_idx].copy()
                
                t_new = tuple(new_pt.tolist())
                t_old = tuple(old_pt.tolist())
                
                if t_new in current_set and t_new != t_old:
                    continue
                
                P[idx] = new_pt
                new_score, _ = _objective(P)
                
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_new_pt = new_pt.copy()
                
                P[idx] = old_pt
            
            if best_new_pt is not None:
                current_set.discard(tuple(old_pt.tolist()))
                current_set.add(tuple(best_new_pt.tolist()))
                P[idx] = best_new_pt
                current_score = best_new_score
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        # If we've had many iterations without improvement, stop
        if no_improve_count > 100:
            break
    
    return P, current_score

def _structured_points(size_x, size_y, offset_x=0, offset_y=0, scale=1):
    """
    Create a structured grid of points.
    """
    xs = np.arange(size_x) * scale + offset_x
    ys = np.arange(size_y) * scale + offset_y
    px, py = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _is_valid(points, lines):
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    if not np.issubdtype(points.dtype, np.integer):
        return False
    if not np.issubdtype(lines.dtype, np.integer):
        return False
    if len(np.unique(points, axis=0)) != 54:
        return False
    seen = set()
    for row in lines:
        A, B, C = int(row[0]), int(row[1]), int(row[2])
        if A == 0 and B == 0:
            return False
        key = _canon(A, B, C)
        if key in seen:
            return False
        seen.add(key)
    return True

def szemeredi_trotter_incidences():
    start = time.time()
    deadline = start + 580  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Define a grid of candidate points
    size = 15
    xs = np.arange(-size, size + 1)
    ys = np.arange(-size, size + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    grid_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # Try multiple structured grids and random initializations
    initializations = []
    
    # 1. Structured grids with exact 54 points
    for sx in range(4, 10):
        for sy in range(4, 10):
            if sx * sy == 54:
                pts = _structured_points(sx, sy, offset_x=-(sx-1)//2, offset_y=-(sy-1)//2)
                initializations.append(pts)
    
    # 2. Random sets
    for _ in range(15):
        idxs = rng.choice(len(grid_points), size=54, replace=False)
        initializations.append(grid_points[idxs])
    
    # 3. Structured patterns with different scales
    for scale in [1, 2, 3]:
        for offset in range(-5, 6):
            pts = []
            for i in range(54):
                x = (i % 9) * scale + offset
                y = (i // 9) * scale + offset
                pts.append((x, y))
            pts = np.array(pts, dtype=np.int64)
            if len(np.unique(pts, axis=0)) == 54:
                initializations.append(pts)
    
    # 4. Hexagonal-like patterns
    for scale in [1, 2]:
        pts = []
        for i in range(9):
            for j in range(6):
                x = i * scale + (j % 2) * scale // 2
                y = j * scale * 3 // 4
                pts.append((x, y))
        pts = np.array(pts, dtype=np.int64)
        if len(np.unique(pts, axis=0)) == 54:
            initializations.append(pts)
    
    # 5. Concentric rings
    for r in [3, 4, 5]:
        pts = []
        # Center
        pts.append((0, 0))
        # Rings
        for ring in range(1, r + 1):
            for angle in np.linspace(0, 2 * np.pi, ring * 6, endpoint=False):
                x = int(round(ring * np.cos(angle)))
                y = int(round(ring * np.sin(angle)))
                pts.append((x, y))
        pts = np.array(pts[:54], dtype=np.int64)
        if len(np.unique(pts, axis=0)) == 54:
            initializations.append(pts)
    
    # Process each initialization
    time_per_init = 580 / len(initializations)
    
    for init_pts in initializations:
        if time.time() > deadline:
            break
        
        # Ensure 54 distinct points
        if len(init_pts) != 54:
            continue
        if len(np.unique(init_pts, axis=0)) != 54:
            continue
        
        # Local search with time limit
        remaining = deadline - time.time()
        max_iters = int(min(300, max(50, remaining / 0.5)))  # Estimate ~0.5s per iteration
        P, score = _local_search(init_pts, grid_points, rng, max_iters=max_iters, deadline=deadline)
        
        if score > best_score:
            best_score = score
            best_points = P.copy()
            _, best_lines = _objective(P)
    
    # Fallback if something went wrong
    if best_points is None:
        points = np.array([(i, j) for i in range(6) for j in range(9)], dtype=np.int64)
        _, lines = _objective(points)
        best_points = points
        best_lines = lines
    
    if not _is_valid(best_points, best_lines):
        # Fallback to valid grid
        points = np.array([(i, j) for i in range(6) for j in range(9)], dtype=np.int64)
        _, lines = _objective(points)
        best_points = points
        best_lines = lines
    
    return best_points.astype(np.int64), best_lines.astype(np.int64)
