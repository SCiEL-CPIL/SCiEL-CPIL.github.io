# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
from math import gcd
from collections import defaultdict


def _normalize_line(A, B, C):
    """Normalize a line (A, B, C) so that:
    - gcd(|A|, |B|, |C|) divides all coefficients (divide by gcd)
    - first nonzero entry is positive
    Returns tuple (a, b, c) with small ints.
    """
    A, B, C = int(A), int(B), int(C)
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    # Make first nonzero entry positive
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)


def _count_incidences(points, lines):
    """Count exact incidences using int64 arithmetic."""
    count = 0
    for x, y in points:
        for A, B, C in lines:
            if A * x + B * y + C == 0:
                count += 1
    return count


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization via dual-space hypergraph approach.
    
    We search over structured point sets and line sets to maximize incidences.
    The strategy:
    1. Generate candidate point sets from various geometric structures.
    2. For each candidate point set, find the 27 lines that maximize incidences.
    3. Return the best (points, lines) pair.
    
    Key insight: Instead of fixing the grid structure, we explore:
    - Rectangular grids with varying aspect ratios
    - Points concentrated along few lines (high multiplicity)
    - Hybrid structures combining dense clusters with sparse spread
    
    The baseline (k=3 grid) gives 81 incidences. We try to exceed this.
    """
    
    best_inc = -1
    best_points = None
    best_lines = None
    
    # Strategy 1: Original grid construction with k=3
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    lines = np.array(rows, dtype=np.int64)
    inc = _count_incidences(points, lines)
    if inc > best_inc:
        best_inc = inc
        best_points = points.copy()
        best_lines = lines.copy()
    
    # Strategy 2: Try different grid dimensions
    # Points: X x Y grid where X*Y = 54
    # Lines: try to find 27 lines maximizing incidences
    for X in [1, 2, 3, 6, 9, 18, 27, 54]:
        Y = 54 // X
        if X * Y != 54:
            continue
        
        # Generate grid points
        xs = np.arange(1, X + 1)
        ys = np.arange(1, Y + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        cand_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        # Find best 27 lines for this point set
        # Candidate lines: all lines through pairs of points, plus horizontal/vertical
        # But that's too many. Let's use a smarter approach.
        
        # For each line, count how many points it passes through.
        # We want the top 27.
        
        # Generate candidate lines efficiently:
        # - All horizontal lines y = c for c in range
        # - All vertical lines x = c for c in range
        # - All lines with small slopes passing through grid points
        
        line_counts = defaultdict(int)
        
        # Horizontal lines
        for y_val in range(1, Y + 1):
            line_counts[(-1, 0, y_val)] = X  # -x*0 + 1*y - y_val = 0 -> y = y_val, encoded as (0, 1, -y_val)
        
        # Actually let me be more careful with encoding: A*x + B*y + C = 0
        # Horizontal line y = c: 0*x + 1*y + (-c) = 0 -> (0, 1, -c)
        # Vertical line x = c: 1*x + 0*y + (-c) = 0 -> (1, 0, -c)
        
        line_counts = defaultdict(int)
        
        # Horizontal lines
        for y_val in range(1, Y + 1):
            norm = _normalize_line(0, 1, -y_val)
            line_counts[norm] += X
        
        # Vertical lines
        for x_val in range(1, X + 1):
            norm = _normalize_line(1, 0, -x_val)
            line_counts[norm] += Y
        
        # Diagonal and other lines: for each pair of points, compute the line
        # This is O(n^2) = O(54^2) = 2916 pairs, manageable
        for i in range(len(cand_points)):
            for j in range(i + 1, len(cand_points)):
                x1, y1 = cand_points[i]
                x2, y2 = cand_points[j]
                dx = x2 - x1
                dy = y2 - y1
                # Line through (x1,y1) and (x2,y2): 
                # dy*(x - x1) - dx*(y - y1) = 0
                # dy*x - dx*y + (dx*y1 - dy*x1) = 0
                A = dy
                B = -dx
                C = dx * y1 - dy * x1
                if A == 0 and B == 0:
                    continue
                norm = _normalize_line(A, B, C)
                line_counts[norm] += 1
        
        # Get top 27 lines
        sorted_lines = sorted(line_counts.items(), key=lambda x: -x[1])
        top27 = sorted_lines[:27]
        
        # Count incidences for these 27 lines
        inc = 0
        for norm, cnt in top27:
            inc += cnt
        
        if inc > best_inc:
            best_inc = inc
            best_points = cand_points.copy()
            best_lines = np.array([norm for norm, _ in top27], dtype=np.int64)
    
    # Strategy 3: Concentrate points on few lines
    # Put many points on 1-3 lines, rest scattered
    # E.g., 27 points on line y=0, 27 points on line y=1
    # Then lines y=0 and y=1 each have 27 incidences, plus lines through pairs
    
    # Try: 27 points on y=0 at x=1..27, 27 points on y=1 at x=1..27
    points3 = np.zeros((54, 2), dtype=np.int64)
    for i in range(27):
        points3[i] = [i + 1, 0]
        points3[i + 27] = [i + 1, 1]
    
    # Find best 27 lines
    line_counts3 = defaultdict(int)
    for i in range(len(points3)):
        for j in range(i + 1, len(points3)):
            x1, y1 = points3[i]
            x2, y2 = points3[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            norm = _normalize_line(A, B, C)
            line_counts3[norm] += 1
    
    sorted_lines3 = sorted(line_counts3.items(), key=lambda x: -x[1])
    top27_3 = sorted_lines3[:27]
    inc3 = sum(cnt for _, cnt in top27_3)
    
    if inc3 > best_inc:
        best_inc = inc3
        best_points = points3.copy()
        best_lines = np.array([norm for norm, _ in top27_3], dtype=np.int64)
    
    # Strategy 4: 18 points on y=0, 18 on y=1, 18 on y=2
    points4 = np.zeros((54, 2), dtype=np.int64)
    for i in range(18):
        points4[i] = [i + 1, 0]
        points4[i + 18] = [i + 1, 1]
        points4[i + 36] = [i + 1, 2]
    
    line_counts4 = defaultdict(int)
    for i in range(len(points4)):
        for j in range(i + 1, len(points4)):
            x1, y1 = points4[i]
            x2, y2 = points4[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            norm = _normalize_line(A, B, C)
            line_counts4[norm] += 1
    
    sorted_lines4 = sorted(line_counts4.items(), key=lambda x: -x[1])
    top27_4 = sorted_lines4[:27]
    inc4 = sum(cnt for _, cnt in top27_4)
    
    if inc4 > best_inc:
        best_inc = inc4
        best_points = points4.copy()
        best_lines = np.array([norm for norm, _ in top27_4], dtype=np.int64)
    
    # Strategy 5: 9 points on each of 6 lines
    points5 = np.zeros((54, 2), dtype=np.int64)
    for line_idx in range(6):
        for i in range(9):
            points5[line_idx * 9 + i] = [i + 1, line_idx]
    
    line_counts5 = defaultdict(int)
    for i in range(len(points5)):
        for j in range(i + 1, len(points5)):
            x1, y1 = points5[i]
            x2, y2 = points5[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            norm = _normalize_line(A, B, C)
            line_counts5[norm] += 1
    
    sorted_lines5 = sorted(line_counts5.items(), key=lambda x: -x[1])
    top27_5 = sorted_lines5[:27]
    inc5 = sum(cnt for _, cnt in top27_5)
    
    if inc5 > best_inc:
        best_inc = inc5
        best_points = points5.copy()
        best_lines = np.array([norm for norm, _ in top27_5], dtype=np.int64)
    
    # Strategy 6: Try 3x18 grid with optimized lines
    # Already covered in Strategy 2 with X=3, Y=18
    
    # Strategy 7: 6x9 grid
    # Already covered in Strategy 2 with X=6, Y=9
    
    # Strategy 8: 9x6 grid
    # Already covered in Strategy 2 with X=9, Y=6
    
    # Strategy 9: Try non-uniform distributions
    # Put points at positions that maximize collinearities
    # E.g., use a set where many points are collinear
    
    # Try: points at (i, j) for i in 1..6, j in 1..9 but shifted
    # Or use modular arithmetic patterns
    
    # Strategy 10: 2x27 grid
    # Already covered in Strategy 2
    
    # Strategy 11: 1x54 (all on one line)
    # Already covered in Strategy 2 with X=1, Y=54
    
    # Strategy 12: 54x1
    # Already covered in Strategy 2 with X=54, Y=1
    
    # Let's also try: points on a circle or parabola
    # Parabola: y = x^2 for x in 1..54, but that's 54 distinct points
    # Lines through pairs of parabola points
    
    points12 = np.zeros((54, 2), dtype=np.int64)
    for i in range(54):
        points12[i] = [i + 1, (i + 1) ** 2]
    
    line_counts12 = defaultdict(int)
    for i in range(len(points12)):
        for j in range(i + 1, len(points12)):
            x1, y1 = points12[i]
            x2, y2 = points12[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            norm = _normalize_line(A, B, C)
            line_counts12[norm] += 1
    
    sorted_lines12 = sorted(line_counts12.items(), key=lambda x: -x[1])
    top27_12 = sorted_lines12[:27]
    inc12 = sum(cnt for _, cnt in top27_12)
    
    if inc12 > best_inc:
        best_inc = inc12
        best_points = points12.copy()
        best_lines = np.array([norm for norm, _ in top27_12], dtype=np.int64)
    
    # Strategy 13: Try 3 parallel lines with 18 points each, but offset
    points13 = np.zeros((54, 2), dtype=np.int64)
    for line_idx in range(3):
        for i in range(18):
            points13[line_idx * 18 + i] = [i + 1 + line_idx * 5, line_idx * 10]
    
    line_counts13 = defaultdict(int)
    for i in range(len(points13)):
        for j in range(i + 1, len(points13)):
            x1, y1 = points13[i]
            x2, y2 = points13[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            norm = _normalize_line(A, B, C)
            line_counts13[norm] += 1
    
    sorted_lines13 = sorted(line_counts13.items(), key=lambda x: -x[1])
    top27_13 = sorted_lines13[:27]
    inc13 = sum(cnt for _, cnt in top27_13)
    
    if inc13 > best_inc:
        best_inc = inc13
        best_points = points13.copy()
        best_lines = np.array([norm for norm, _ in top27_13], dtype=np.int64)
    
    # Strategy 14: 2 lines with 27 points each, but interleaved to create more cross-lines
    points14 = np.zeros((54, 2), dtype=np.int64)
    for i in range(27):
        points14[i] = [2 * i + 1, 0]  # odd x on y=0
        points14[i + 27] = [2 * i + 2, 1]  # even x on y=1
    
    line_counts14 = defaultdict(int)
    for i in range(len(points14)):
        for j in range(i + 1, len(points14)):
            x1, y1 = points14[i]
            x2, y2 = points14[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            norm = _normalize_line(A, B, C)
            line_counts14[norm] += 1
    
    sorted_lines14 = sorted(line_counts14.items(), key=lambda x: -x[1])
    top27_14 = sorted_lines14[:27]
    inc14 = sum(cnt for _, cnt in top27_14)
    
    if inc14 > best_inc:
        best_inc = inc14
        best_points = points14.copy()
        best_lines = np.array([norm for norm, _ in top27_14], dtype=np.int64)
    
    # Validate final result
    if best_points is None or best_lines is None:
        # Fallback to original
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
    
    # Ensure correct shapes
    assert best_points.shape == (54, 2), f"Points shape {best_points.shape}"
    assert best_lines.shape == (27, 3), f"Lines shape {best_lines.shape}"
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
