import numpy as np
from math import gcd
import itertools

def normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) to a canonical form."""
    if A == 0 and B == 0:
        return (0, 0, 0) # Should not happen for valid lines from points
        
    # Find GCD
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    
    # Make first non-zero coefficient positive
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def count_incidences(points, lines):
    """Count total incidences between points and lines."""
    # points: (N, 2), lines: (M, 3)
    # Vectorized check: A*x + B*y + C == 0
    # points[:, 0] is x, points[:, 1] is y
    x = points[:, 0]
    y = points[:, 1]
    A = lines[:, 0]
    B = lines[:, 1]
    C = lines[:, 2]
    
    # Compute A*x + B*y + C for all pairs
    # This creates an (M, N) matrix
    vals = A[:, np.newaxis] * x[np.newaxis, :] + B[:, np.newaxis] * y[np.newaxis, :] + C[:, np.newaxis]
    inc = np.sum(vals == 0)
    return inc

def szemeredi_trotter_incidences():
    # Factors of 54
    factors = [1, 2, 3, 6, 9, 18, 27, 54]
    
    best_points = None
    best_lines = None
    best_score = -1
    
    for W in factors:
        H = 54 // W
        # Generate grid points
        # Using 0-based indexing for simplicity, shifts don't change incidence count
        xs = np.arange(W)
        ys = np.arange(H)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        # Generate all unique lines from pairs of points
        lines_set = set()
        n_pts = len(points)
        for i in range(n_pts):
            x1, y1 = points[i]
            for j in range(i + 1, n_pts):
                x2, y2 = points[j]
                # Line through (x1, y1) and (x2, y2)
                # A = y1 - y2
                # B = x2 - x1
                # C = x1*y2 - x2*y1
                A = int(y1 - y2)
                B = int(x2 - x1)
                C = int(x1 * y2 - x2 * y1)
                norm = normalize_line(A, B, C)
                if norm != (0, 0, 0):
                    lines_set.add(norm)
        
        lines_arr = np.array(list(lines_set), dtype=np.int64)
        
        if len(lines_arr) < 27:
            # Pad with dummy lines if necessary? No, we need 27 distinct lines.
            # If a grid has fewer than 27 unique lines, it's invalid or we need to handle it.
            # But any grid with >= 3 non-collinear points has many lines.
            # 1x54 has 54 horizontal? No, just 1 horizontal line if we consider standard lines?
            # Wait, for 1x54 points (x=0, y=0..53), lines are vertical x=0 (1 line) and lines through pairs.
            # Lines through (0, y1) and (0, y2) is x=0. All pairs give x=0.
            # So only 1 unique line? No, that's wrong.
            # If all points are collinear, there is only 1 line.
            # We need 27 distinct lines.
            # So 1x54 is invalid because we can't form 27 distinct lines from 54 collinear points?
            # Actually, the problem says "27 lines". It doesn't say they must be distinct from each other?
            # "two rows describing the same line ... is invalid". So they must be distinct.
            # If we only have 1 unique line, we can't provide 27 distinct lines.
            # So 1x54 and 54x1 are likely invalid or score 0 if we can't fill 27 slots.
            # However, the grid 1x54 consists of points on a line.
            # Any line passing through 2 points is that line.
            # So there is only 1 unique line.
            # We need 27 distinct lines. So this configuration is invalid.
            continue

        # Count incidences for each line
        # We want to pick the top 27 lines with the most incidences.
        # Calculate incidence count for each line
        x = points[:, 0]
        y = points[:, 1]
        
        # Efficiently count incidences for all lines in lines_arr
        # lines_arr shape (M, 3)
        # We can do this in chunks or vectorized.
        # M is at most ~1400. N is 54.
        # 1400 * 54 = 75600 operations. Very fast.
        
        A = lines_arr[:, 0]
        B = lines_arr[:, 1]
        C = lines_arr[:, 2]
        
        # (M, 1) * (1, N) + ...
        vals = A[:, np.newaxis] * x[np.newaxis, :] + B[:, np.newaxis] * y[np.newaxis, :] + C[:, np.newaxis]
        counts = np.sum(vals == 0, axis=1)
        
        # Sort indices by counts descending
        sorted_indices = np.argsort(counts)[::-1]
        
        # Take top 27
        top_27_indices = sorted_indices[:27]
        
        # Check if we have at least 27 lines
        if len(lines_arr) < 27:
            continue
            
        selected_lines = lines_arr[top_27_indices]
        score = np.sum(counts[top_27_indices])
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
            best_lines = selected_lines.copy()
            
    # If no valid grid found (should not happen), fallback to baseline
    if best_points is None:
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        best_lines = np.array(rows, dtype=np.int64)

    return best_points, best_lines
