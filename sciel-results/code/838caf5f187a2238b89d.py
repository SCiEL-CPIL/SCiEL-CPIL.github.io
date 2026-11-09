import numpy as np
from itertools import combinations
from math import gcd
import time

def _normalize_line(A, B, C):
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g != 0:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _evaluate_configuration(points_full, line_info, active_mask, total_points):
    """Given a set of active points (via bitmask) and all line info,
       find the top 27 lines by incidence count among active points."""
    candidates = []
    for coeffs, line_mask in line_info:
        count = (active_mask & line_mask).bit_count()
        if count >= 2:
            candidates.append((count, coeffs))
    candidates.sort(key=lambda x: x[0], reverse=True)
    if len(candidates) >= 27:
        top_27 = candidates[:27]
        score = sum(c for c, _ in top_27)
        return score, top_27
    return -1, None

def _build_line_info(points_full):
    """Precompute all lines through pairs of points, with bitmask of points on each line."""
    total_points = len(points_full)
    line_dict = {}
    for i in range(total_points):
        x1, y1 = points_full[i]
        for j in range(i + 1, total_points):
            x2, y2 = points_full[j]
            A = y1 - y2
            B = x2 - x1
            C = -(A * x1 + B * y1)
            norm = _normalize_line(A, B, C)
            if norm is None:
                continue
            if norm not in line_dict:
                line_dict[norm] = 0
            line_dict[norm] |= (1 << i) | (1 << j)
    
    # For each line, compute the full mask of all points on it
    line_info = []
    for coeffs, pair_mask in line_dict.items():
        A, B, C = coeffs
        full_mask = 0
        for idx, (x, y) in enumerate(points_full):
            if A * x + B * y + C == 0:
                full_mask |= (1 << idx)
        if full_mask.bit_count() >= 2:
            line_info.append((coeffs, full_mask))
    return line_info

def _try_removals(points_full, line_info, num_remove, indices, full_mask, deadline):
    """Try removing up to num_remove points and evaluate the best configuration."""
    best_score = -1
    best_active_mask = None
    best_top_27 = None
    
    if num_remove == 0:
        score, top_27 = _evaluate_configuration(points_full, line_info, full_mask, len(points_full))
        if score > best_score:
            best_score = score
            best_active_mask = full_mask
            best_top_27 = top_27
    elif num_remove == 1:
        for r1 in indices:
            if time.time() > deadline - 30:
                break
            active_mask = full_mask & ~(1 << r1)
            score, top_27 = _evaluate_configuration(points_full, line_info, active_mask, len(points_full) - 1)
            if score > best_score:
                best_score = score
                best_active_mask = active_mask
                best_top_27 = top_27
    elif num_remove == 2:
        for r1, r2 in combinations(indices, 2):
            if time.time() > deadline - 30:
                break
            active_mask = full_mask & ~((1 << r1) | (1 << r2))
            score, top_27 = _evaluate_configuration(points_full, line_info, active_mask, len(points_full) - 2)
            if score > best_score:
                best_score = score
                best_active_mask = active_mask
                best_top_27 = top_27
    elif num_remove == 3:
        # Only try if we have time; C(56,3) = 27720
        for r1, r2, r3 in combinations(indices, 3):
            if time.time() > deadline - 30:
                break
            active_mask = full_mask & ~((1 << r1) | (1 << r2) | (1 << r3))
            score, top_27 = _evaluate_configuration(points_full, line_info, active_mask, len(points_full) - 3)
            if score > best_score:
                best_score = score
                best_active_mask = active_mask
                best_top_27 = top_27
    
    return best_score, best_active_mask, best_top_27

def _grid(W, H):
    pts = []
    for y in range(H):
        for x in range(W):
            pts.append((x, y))
    return pts

def _construct_result(points_full, active_mask, top_27):
    pts_out = [points_full[i] for i in range(len(points_full)) if (active_mask >> i) & 1]
    lines_out = [c for _, c in top_27]
    return np.array(pts_out, dtype=np.int64), np.array(lines_out, dtype=np.int64)

def _try_grid(W, H, num_remove, deadline):
    points_full = _grid(W, H)
    total_points = len(points_full)
    if total_points < 54:
        return -1, None, None
    if total_points > 54 + num_remove:
        # We can only remove num_remove points, so we need total_points - num_remove == 54
        # If total_points - num_remove != 54, we need to handle differently
        # For now, only accept if total_points - num_remove == 54
        pass
    
    if total_points - num_remove != 54:
        # Can't use this grid with exactly num_remove removals to get 54 points
        # Try to find a subset of size 54 from this grid
        # This is more complex; for now skip unless total_points == 54 + num_remove
        if total_points != 54 and total_points - num_remove != 54:
            # Need to select 54 points from total_points
            # This is a combinatorial problem; for small grids we can do it
            # But for now, only handle the simple case
            pass
        # If total_points == 54, no removal needed
        # If total_points > 54, we need to remove total_points - 54 points
        # If total_points < 54, can't use
        if total_points < 54:
            return -1, None, None
        # If total_points > 54, we need to remove total_points - 54 points
        # This might be more than num_remove, so we'd need more removals
        # For now, just handle the case where total_points - num_remove == 54
        if total_points - num_remove != 54:
            return -1, None, None
    
    line_info = _build_line_info(points_full)
    full_mask = (1 << total_points) - 1
    indices = list(range(total_points))
    
    best_score, best_active_mask, best_top_27 = _try_removals(
        points_full, line_info, num_remove, indices, full_mask, deadline
    )
    
    if best_score > 0 and best_top_27 is not None:
        return best_score, best_active_mask, best_top_27
    return -1, None, None

def _try_custom_points(deadline):
    """Try some hand-crafted or heuristic point configurations."""
    results = []
    
    # 7x8 grid minus 2 points (best: remove corner points to preserve most lines)
    # A 7x8 grid has 56 points. Remove 2 to get 54.
    # The best removal is likely to remove points that are on the fewest high-incidence lines.
    # Let's try several removal patterns.
    pts_full = _grid(7, 8)
    line_info = _build_line_info(pts_full)
    full_mask = (1 << 56) - 1
    indices = list(range(56))
    
    # Try removing pairs of corner points
    corners = [(0,0), (6,0), (0,7), (6,7)]
    # Convert to indices
    corner_indices = []
    for x, y in corners:
        idx = y * 7 + x
        corner_indices.append(idx)
    
    best_score = -1
    best_mask = None
    best_top27 = None
    
    for i in range(56):
        if time.time() > deadline - 30:
            break
        for j in range(i+1, 56):
            mask = full_mask & ~((1 << i) | (1 << j))
            score, top27 = _evaluate_configuration(pts_full, line_info, mask, 54)
            if score > best_score:
                best_score = score
                best_mask = mask
                best_top27 = top27
    
    if best_score > 0:
        results.append((best_score, best_mask, best_top27, pts_full))
    
    # 6x9 grid exactly
    pts69 = _grid(6, 9)
    li69 = _build_line_info(pts69)
    fm69 = (1 << 54) - 1
    score69, top69 = _evaluate_configuration(pts69, li69, fm69, 54)
    if score69 > 0:
        results.append((score69, fm69, top69, pts69))
    
    # 5x12 grid = 60 points, remove 6
    # This is C(60,6) which is too large. Skip or use heuristic.
    
    return results

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 600
    
    global_best_score = -1
    global_best_points = None
    global_best_lines = None
    
    # Grid configurations to try
    # (W, H, num_remove) where W*H - num_remove should be 54
    configs = [
        (7, 8, 2),   # 56 - 2 = 54
        (6, 9, 0),   # 54
        (5, 11, 1),  # 55 - 1 = 54
        (4, 14, 2),  # 56 - 2 = 54
        (3, 18, 0),  # 54
        (2, 27, 0),  # 54
        (9, 6, 0),   # 54 (transpose of 6x9)
        (8, 7, 2),   # 56 - 2 = 54 (transpose of 7x8)
    ]
    
    for W, H, rem in configs:
        if time.time() > deadline - 60:
            break
        
        score, active_mask, top27 = _try_grid(W, H, rem, deadline)
        if score > global_best_score and top27 is not None:
            global_best_score = score
            pts_full = _grid(W, H)
            pts_out = [pts_full[i] for i in range(len(pts_full)) if (active_mask >> i) & 1]
            global_best_points = np.array(pts_out, dtype=np.int64)
            global_best_lines = np.array([c for _, c in top27], dtype=np.int64)
    
    # Also try the custom points approach for 7x8
    if time.time() < deadline - 120:
        custom_results = _try_custom_points(deadline)
        for score, mask, top27, pts_full in custom_results:
            if score > global_best_score:
                global_best_score = score
                pts_out = [pts_full[i] for i in range(len(pts_full)) if (mask >> i) & 1]
                global_best_points = np.array(pts_out, dtype=np.int64)
                global_best_lines = np.array([c for _, c in top27], dtype=np.int64)
    
    if global_best_points is not None:
        return global_best_points, global_best_lines
    
    # Fallback: 6x9 grid
    pts = _grid(6, 9)
    li = _build_line_info(pts)
    fm = (1 << 54) - 1
    score, top27 = _evaluate_configuration(pts, li, fm, 54)
    if top27 is not None:
        return np.array(pts, dtype=np.int64), np.array([c for _, c in top27], dtype=np.int64)
    
    # Last resort: simple grid
    pts = _grid(9, 6)
    li = _build_line_info(pts)
    fm = (1 << 54) - 1
    score, top27 = _evaluate_configuration(pts, li, fm, 54)
    if top27 is not None:
        return np.array(pts, dtype=np.int64), np.array([c for _, c in top27], dtype=np.int64)
    
    # Absolute fallback
    pts = [(i, 0) for i in range(54)]
    li = _build_line_info(pts)
    fm = (1 << 54) - 1
    score, top27 = _evaluate_configuration(pts, li, fm, 54)
    if top27 is not None:
        return np.array(pts, dtype=np.int64), np.array([c for _, c in top27], dtype=np.int64)
    
    raise RuntimeError("No valid configuration found")

if __name__ == "__main__":
    points, lines = szemeredi_trotter_incidences()
    print(f"Points: {len(points)}, Lines: {len(lines)}")
    # Verify
    count = 0
    for x, y in points:
        for A, B, C in lines:
            if A * x + B * y + C == 0:
                count += 1
    print(f"Incidences: {count}")
