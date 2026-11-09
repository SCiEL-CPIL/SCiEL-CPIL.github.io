# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
from itertools import product


def _incidence_count(points, lines):
    """Compute number of (point, line) incidences. points: (n,2), lines: (m,3)."""
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0:1]  # (n,1)
    py = points[:, 1:2]  # (n,1)
    A = lines[0:1, :]    # (1,m)
    B = lines[1:2, :]    # (1,m)
    C = lines[2:3, :]    # (1,m)
    total = px * A + py * B + C  # (n,m)
    return int(np.sum(total == 0))


def _normalize_line(A, B, C):
    """Normalize (A,B,C) so gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    g = gcd(gcd(abs(int(A)), abs(int(B))), abs(int(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = int(A) // g, int(B) // g, int(C) // g
    # Make first nonzero positive
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)


def _line_key(line):
    """Get normalized key for a line tuple (A,B,C)."""
    return _normalize_line(*line)


def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    # Start with a 6x9 grid (or 9x6)
    # Try multiple grid configurations and pick the best
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # Grid dimensions to try: (width, height) where width*height=54
    grid_dims = [(6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2)]
    
    for w, h in grid_dims:
        if time.time() > deadline - 30:
            break
        
        # Create grid points
        xs = np.arange(1, w + 1)
        ys = np.arange(1, h + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        # Generate candidate lines
        # 1. Horizontal lines: y = c -> (0, 1, -c)
        # 2. Vertical lines: x = c -> (1, 0, -c)
        # 3. Diagonal lines with small slopes
        
        candidates = []
        
        # Horizontal lines
        for c in range(1, h + 1):
            candidates.append((0, 1, -c))
        
        # Vertical lines
        for c in range(1, w + 1):
            candidates.append((1, 0, -c))
        
        # Diagonal lines with slopes 1, -1, 2, -2, 1/2 (encoded as 2x-y=0 -> (2,-1,0))
        # For slope m: y = m*x + b -> m*x - y + b = 0 -> (m, -1, b)
        # For slope 1/2: y = (1/2)x + b -> x - 2y + 2b = 0 -> (1, -2, 2b)
        
        # Slope 1: y = x + b -> (1, -1, b)
        # x in [1,w], y in [1,h], so b = y - x in [1-w, h-1]
        for b in range(1 - w, h):
            candidates.append((1, -1, b))
        
        # Slope -1: y = -x + b -> (1, 1, -b)
        # b = y + x in [2, w+h]
        for b in range(2, w + h + 1):
            candidates.append((1, 1, -b))
        
        # Slope 2: y = 2x + b -> (2, -1, b)
        # b = y - 2x in [1-2w, h-2]
        for b in range(1 - 2 * w, h - 1):
            candidates.append((2, -1, b))
        
        # Slope 1/2: y = (1/2)x + b -> (1, -2, 2b)
        # 2b = 2y - x, so 2y - x in [2-1, 2h-1] = [1, 2h-1]
        # Must be even for integer b... actually let's just use (1, -2, c) where c = 2y - x
        for c in range(1, 2 * h):
            candidates.append((1, -2, c))
        
        # Slope -2: y = -2x + b -> (2, 1, -b)
        # b = y + 2x in [3, h+2w]
        for b in range(3, h + 2 * w + 1):
            candidates.append((2, 1, -b))
        
        # Slope -1/2: y = -(1/2)x + b -> (1, 2, -2b)
        # 2b = x + 2y, so x + 2y in [3, w+2h]
        for c in range(3, w + 2 * h + 1):
            candidates.append((1, 2, -c))
        
        # Remove duplicates and invalid lines
        seen = set()
        unique_candidates = []
        for line in candidates:
            key = _line_key(line)
            if key == (0, 0, 0):
                continue
            if key not in seen:
                seen.add(key)
                unique_candidates.append(key)
        
        # Greedily select 27 lines
        selected = []
        selected_set = set()
        current_points = points.copy()
        
        for _ in range(27):
            if time.time() > deadline - 10:
                break
            best_line = None
            best_inc = -1
            for line in unique_candidates:
                if line in selected_set:
                    continue
                # Compute incidences for this line
                A, B, C = line
                vals = A * points[:, 0] + B * points[:, 1] + C
                inc = int(np.sum(vals == 0))
                if inc > best_inc:
                    best_inc = inc
                    best_line = line
            if best_line is None:
                break
            selected.append(best_line)
            selected_set.add(best_line)
        
        if len(selected) == 27:
            lines = np.array(selected, dtype=np.int64)
            score = _incidence_count(points, lines)
            if score > best_score:
                best_score = score
                best_points = points
                best_lines = lines
    
    # If we didn't find a good solution, fall back to seed
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


# EVOLVE-BLOCK-END
