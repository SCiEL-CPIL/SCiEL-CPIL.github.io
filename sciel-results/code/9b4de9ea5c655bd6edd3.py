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

def _incidence_count(points, lines):
    P = points.astype(np.int64)
    L = lines.astype(np.int64)
    # Vectorized check: L[:, 0:1] * P[:, 0:1].T + ...
    vals = L[:, 0:1] * P[:, 0:1].T + L[:, 1:2] * P[:, 1:2].T + L[:, 2:3]
    return int(np.count_nonzero(vals == 0))

def _get_top_lines(points, k=27):
    """
    Given N points, find the k lines that pass through the most pairs of points.
    """
    n = len(points)
    line_counts = {}
    # Iterate over all pairs
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_from_two_points(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    
    # Sort by count
    sorted_lines = sorted(line_counts.items(), key=lambda x: -x[1])
    
    # Pick top k distinct lines
    lines = []
    seen = set()
    for key, count in sorted_lines:
        if key in seen:
            continue
        lines.append(key)
        seen.add(key)
        if len(lines) == k:
            break
            
    # If we don't have enough lines (unlikely for 54 points), pad with arbitrary lines
    # But 54 points generate at least 54 lines (one through each point and origin? No, distinct lines).
    # Actually, just return what we have. The problem guarantees we can form 27 lines.
    return np.array(lines, dtype=np.int64)

def _get_top_points(lines, grid_points, k=54):
    """
    Given M lines, find the k distinct points from grid_points that have the highest incidence count.
    """
    L = lines.astype(np.int64)
    G = grid_points.astype(np.int64)
    
    # Calculate incidence for all grid points
    # G is (N_g, 2), L is (M, 3)
    # vals is (M, N_g)
    vals = L[:, 0:1] * G[:, 0:1].T + L[:, 1:2] * G[:, 1:2].T + L[:, 2:3]
    counts = np.count_nonzero(vals == 0, axis=0)
    
    # Get indices of top k points
    # We need distinct points. Since grid_points are distinct, just pick top k indices.
    # Sort indices by counts descending
    top_indices = np.argsort(counts)[::-1][:k]
    
    return G[top_indices]

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

def _local_refine(points, lines, grid_points, max_iters=200, rng=None):
    """
    Try to improve the current configuration by swapping points.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    current_inc = _incidence_count(points, lines)
    P = points.copy()
    L = lines.copy()
    G = grid_points.astype(np.int64)
    
    # Map grid points to indices for quick lookup?
    # Or just check if a point is in P.
    
    for _ in range(max_iters):
        # Pick a random point to move
        idx = rng.integers(0, 54)
        old_pt = P[idx]
        
        # Try moving it to a random nearby point or a random grid point
        # Let's try a random grid point first
        new_idx = rng.integers(0, len(G))
        new_pt = G[new_idx]
        
        # Check if new_pt is already in P (excluding idx)
        # Simple check
        is_dup = False
        for k in range(54):
            if k == idx:
                continue
            if np.array_equal(P[k], new_pt):
                is_dup = True
                break
        if is_dup:
            continue
            
        P[idx] = new_pt
        # Update lines? No, lines are fixed in this refinement step?
        # The prompt says "returning (points, lines)".
        # If we move points, the optimal lines change.
        # So we should recompute lines.
        new_L = _get_top_lines(P, 27)
        new_inc = _incidence_count(P, new_L)
        
        if new_inc > current_inc:
            current_inc = new_inc
            L = new_L
        else:
            P[idx] = old_pt
            
    return P, L, current_inc

def szemeredi_trotter_incidences():
    start = time.time()
    deadline = start + 600
    rng = np.random.default_rng(42)
    
    # Define a grid of candidate points
    # Size 31x31 seems reasonable. Coordinates -15 to 15.
    # We can expand if needed.
    size = 15
    xs = np.arange(-size, size + 1)
    ys = np.arange(-size, size + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    grid_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    best_points = None
    best_lines = None
    best_inc = -1
    
    n_restarts = 100 # We can do many restarts
    
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        # 1. Initialize points randomly from grid
        idxs = rng.choice(len(grid_points), size=54, replace=False)
        points = grid_points[idxs]
        
        # 2. Get top lines
        lines = _get_top_lines(points, 27)
        
        current_inc = _incidence_count(points, lines)
        
        # 3. Alternating Optimization
        # Loop until convergence or time limit
        prev_lines = lines.copy()
        for _ in range(50): # Max iterations per restart
            if time.time() > deadline - 10:
                break
                
            # Update points
            points = _get_top_points(lines, grid_points, 54)
            
            # Update lines
            lines = _get_top_lines(points, 27)
            
            current_inc = _incidence_count(points, lines)
            
            if current_inc > best_inc:
                best_inc = current_inc
                best_points = points.copy()
                best_lines = lines.copy()
                
            # Check convergence
            if np.array_equal(lines, prev_lines):
                break
            prev_lines = lines.copy()
            
        # 4. Local Refinement
        if time.time() < deadline - 5:
            P_ref, L_ref, inc_ref = _local_refine(best_points, best_lines, grid_points, max_iters=100, rng=rng)
            if inc_ref > best_inc:
                best_inc = inc_ref
                best_points = P_ref
                best_lines = L_ref

    # Fallback if something went wrong
    if best_points is None:
        # Simple grid
        points = np.array([(i, j) for i in range(6) for j in range(9)], dtype=np.int64)
        lines = _get_top_lines(points, 27)
        best_points = points
        best_lines = lines

    if not _is_valid(best_points, best_lines):
        # Fallback to valid grid
        points = np.array([(i, j) for i in range(6) for j in range(9)], dtype=np.int64)
        lines = _get_top_lines(points, 27)
        best_points = points
        best_lines = lines

    return best_points.astype(np.int64), best_lines.astype(np.int64)
