import numpy as np
from math import gcd

def szemeredi_trotter_incidences():
    # Grid dimensions
    a = 6
    b = 9
    
    # Points: (i, j) for i in 1..a, j in 1..b
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Grid lines
    lines = []
    # Vertical lines x = i
    for i in range(1, a + 1):
        lines.append((1, 0, -i))
    # Horizontal lines y = j
    for j in range(1, b + 1):
        lines.append((0, 1, -j))
        
    # Find best diagonal lines
    # We need m - (a + b) = 27 - 15 = 12 more lines
    num_extra = 27 - (a + b)
    
    # Candidate lines from pairs of points
    # To be efficient, we can just check lines defined by pairs
    # There are 54 points, ~1431 pairs
    candidates = {}
    
    p_list = points.tolist()
    for i in range(len(p_list)):
        for j in range(i + 1, len(p_list)):
            x1, y1 = p_list[i]
            x2, y2 = p_list[j]
            
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            # Normalize
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            else:
                continue # Degenerate, shouldn't happen for distinct points
            
            # Fix sign
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            
            # Skip if it's a grid line
            if (A == 1 and B == 0) or (A == 0 and B == 1):
                continue
                
            if key not in candidates:
                candidates[key] = 0
                
            # Count incidences for this line
            # Optimization: only count if we haven't fully processed?
            # Actually, we can just count all points for unique keys at the end.
            # But storing the line is enough.
            
    # Count points for each candidate
    line_counts = []
    for key in candidates:
        A, B, C = key
        count = 0
        for x, y in p_list:
            if A * x + B * y + C == 0:
                count += 1
        line_counts.append((count, key))
        
    # Sort by count descending
    line_counts.sort(key=lambda x: x[0], reverse=True)
    
    # Add top num_extra lines
    added = 0
    for count, key in line_counts:
        if added >= num_extra:
            break
        lines.append(key)
        added += 1
        
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines
