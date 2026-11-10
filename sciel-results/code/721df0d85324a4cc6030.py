import numpy as np
import time
import itertools
from functools import lru_cache

def _is_valid(centers, sides, tol=1e-7):
    """Check all constraints: inside unit square, no overlaps, positive sides."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
        return False
    if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True

def _solve_guillotine(w, h, n, memo={}):
    """
    Recursively solve for the maximum sum of side lengths of n squares
    in a rectangle of width w and height h, using guillotine cuts.
    Returns (max_value, layout_info)
    layout_info is a tuple describing the packing.
    For n=1: ('leaf', s)
    For n>1: ('split', orientation, split_frac, left_info, right_info)
    """
    if n == 1:
        s = min(w, h)
        return s, ('leaf', s)
        
    if (w, h, n) in memo:
        return memo[(w, h, n)]
        
    best_val = 0.0
    best_info = None
    
    # Try vertical splits
    for k in range(1, n):
        # Optimize split position x
        # The function f(x) = V(x, h, k) + V(w-x, h, n-k)
        # We can sample x or use ternary search.
        # Given the concave nature, ternary search is safe.
        lo = 1e-9
        hi = w - 1e-9
        for _ in range(20):
            m1 = lo + (hi - lo) / 3
            m2 = hi - (hi - lo) / 3
            v1, _ = _solve_guillotine(m1, h, k)
            v2, _ = _solve_guillotine(w - m1, h, n - k)
            val1 = v1 + v2
            
            v3, _ = _solve_guillotine(m2, h, k)
            v4, _ = _solve_guillotine(w - m2, h, n - k)
            val2 = v3 + v4
            
            if val1 < val2:
                lo = m1
            else:
                hi = m2
                
        x = (lo + hi) / 2
        val_l, info_l = _solve_guillotine(x, h, k)
        val_r, info_r = _solve_guillotine(w - x, h, n - k)
        total_val = val_l + val_r
        
        if total_val > best_val:
            best_val = total_val
            best_info = ('split', 'v', x, info_l, info_r)
            
    # Try horizontal splits
    for k in range(1, n):
        lo = 1e-9
        hi = h - 1e-9
        for _ in range(20):
            m1 = lo + (hi - lo) / 3
            m2 = hi - (hi - lo) / 3
            v1, _ = _solve_guillotine(w, m1, k)
            v2, _ = _solve_guillotine(w, h - m1, n - k)
            val1 = v1 + v2
            
            v3, _ = _solve_guillotine(w, m2, k)
            v4, _ = _solve_guillotine(w, h - m2, n - k)
            val2 = v3 + v4
            
            if val1 < val2:
                lo = m1
            else:
                hi = m2
                
        y = (lo + hi) / 2
        val_b, info_b = _solve_guillotine(w, y, k)
        val_t, info_t = _solve_guillotine(w, h - y, n - k)
        total_val = val_b + val_t
        
        if total_val > best_val:
            best_val = total_val
            best_info = ('split', 'h', y, info_b, info_t)

    memo[(w, h, n)] = (best_val, best_info)
    return best_val, best_info

def _layout_to_squares(info, x_offset=0.0, y_offset=0.0):
    """
    Convert layout info to a list of (center_x, center_y, side).
    """
    if info[0] == 'leaf':
        s = info[1]
        cx = x_offset + s / 2
        cy = y_offset + s / 2
        return [(cx, cy, s)]
    else:
        _, orient, split, info1, info2 = info
        squares = []
        if orient == 'v':
            # Left part
            squares.extend(_layout_to_squares(info1, x_offset, y_offset))
            # Right part
            squares.extend(_layout_to_squares(info2, x_offset + split, y_offset))
        else:
            # Bottom part
            squares.extend(_layout_to_squares(info1, x_offset, y_offset))
            # Top part
            squares.extend(_layout_to_squares(info2, x_offset, y_offset + split))
        return squares

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares in the unit square to maximize sum of side lengths.
    Uses recursive guillotine optimization.
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 570
    
    # Solve for the optimal guillotine packing
    val, info = _solve_guillotine(1.0, 1.0, n)
    
    # Extract squares
    squares = _layout_to_squares(info)
    
    centers = np.array([(s[0], s[1]) for s in squares])
    sides = np.array([s[2] for s in squares])
    
    # Verify
    if not _is_valid(centers, sides, tol):
        # Fallback to simple grid if something went wrong
        c = np.array([[0.095 + 0.19*i, 0.095 + 0.19*j] for i in range(5) for j in range(2)])
        s = np.full(10, 0.19)
        return c, s
        
    return centers, sides

