"""Circle packing in perimeter-4 rectangle: Grid-based cellular decomposition."""
import numpy as np
import time

N = 21


def construct():
    """
    Constructs a packing of 21 circles in a rectangle of perimeter 4.
    Uses a 3x7 grid (or 7x3) which maximizes the sum of radii for a uniform grid.
    """
    # Optimal grid for N=21 is 3x7 or 7x3.
    # For 3 columns and 7 rows:
    # w + h = 2
    # r = min(w/6, h/14)
    # Max r when w/6 = h/14 => 14w = 6h => h = 7/3 w
    # w + 7/3 w = 2 => 10/3 w = 2 => w = 0.6
    # h = 1.4
    # r = 0.6/6 = 0.1
    # Sum = 21 * 0.1 = 2.1
    
    # Alternative: 7 columns and 3 rows
    # w + h = 2
    # r = min(w/14, h/6)
    # Max r when w/14 = h/6 => 6w = 14h => w = 7/3 h
    # 7/3 h + h = 2 => 10/3 h = 2 => h = 0.6
    # w = 1.4
    # r = 1.4/14 = 0.1
    # Sum = 2.1
    
    # We choose 3x7 (w=0.6, h=1.4)
    w = 0.6
    h = 1.4
    nx = 3
    ny = 7
    
    cw = w / nx
    ch = h / ny
    r = min(cw, ch) / 2.0
    
    x = np.zeros(N)
    y = np.zeros(N)
    r_arr = np.zeros(N)
    
    for i in range(N):
        row = i // nx
        col = i % nx
        x[i] = (col + 0.5) * cw
        y[i] = (row + 0.5) * ch
        r_arr[i] = r
        
    # Verify validity
    # Check boundaries
    if np.any(x < r_arr - 1e-9) or np.any(x > w - r_arr + 1e-9):
        pass # Should not happen
    if np.any(y < r_arr - 1e-9) or np.any(y > h - r_arr + 1e-9):
        pass # Should not happen
        
    # Check overlaps
    # Since they are in disjoint cells, they cannot overlap if r <= inradius.
    # Inradius is min(dist to walls).
    # For centered circle, dist to wall is cw/2 or ch/2.
    # r = min(cw, ch)/2.
    # So it fits perfectly.
    
    return np.column_stack((x, y, r_arr))
