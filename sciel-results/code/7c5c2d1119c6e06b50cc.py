import numpy as np
import math
import time

def _gcd3(a, b, c):
    """Compute gcd of three integers."""
    g = math.gcd(abs(int(a)), abs(int(b)))
    g = math.gcd(g, abs(int(c)))
    return g if g != 0 else 1

def _normalize_line(A, B, C):
    """Normalize a line so it's unique."""
    A, B, C = int(A), int(B), int(C)
    g = _gcd3(A, B, C)
    A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _generate_grid_points(w, h, n=54):
    """
    Generate a grid of points.
    If w*h > n, we take the first n points in row-major order.
    If w*h < n, we pad with points from a second grid or extend?
    The task requires exactly 54 distinct points.
    We will only use grids where w*h >= 54.
    """
    xs = np.arange(w)
    ys = np.arange(h)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    if len(points) > n:
        # To make it more symmetric, we could remove points from the center or edges.
        # Simple approach: take the first n.
        # Better approach: If w*h is close to n, maybe remove points that break symmetry least?
        # For now, just take the first n.
        points = points[:n]
    elif len(points) < n:
        # This case shouldn't happen if we select w,h carefully.
        # Pad with points from a shifted grid?
        # For simplicity, we will only call this with w*h >= 54.
        pass
        
    return points

def _find_best_lines(points, K, deadline):
    """
    Find the 27 lines with the highest incidence for the given points.
    K is the max absolute value for A and B.
    """
    n_pts = len(points)
    px = points[:, 0]
    py = points[:, 1]
    
    candidates = [] # List of (count, normalized_line_tuple)
    
    # We iterate over A and B
    # Optimization: A and B can be assumed such that gcd(A,B)=1?
    # No, line 2x+2y+C=0 is same as x+y+C/2=0.
    # If C is integer, 2x+2y+C=0 implies x+y = -C/2.
    # If C is even, it's the same line as x+y+C/2=0.
    # If C is odd, it passes through no integer points?
    # Wait, if A=2, B=2, C=1, 2x+2y+1=0 => 2(x+y) = -1, no integer solution.
    # So we can restrict to gcd(A,B)=1 to avoid redundant checks and empty lines.
    
    for A in range(-K, K + 1):
        if time.time() > deadline:
            break
        for B in range(-K, K + 1):
            if A == 0 and B == 0:
                continue
            if math.gcd(abs(A), abs(B)) != 1:
                continue
            
            # Compute values V = A*x + B*y
            V = A * px + B * py
            
            # Get unique values and counts
            # We want to find C such that A*x + B*y + C = 0 => C = -V
            # So for each unique value v in V, the line is A*x + B*y - v = 0 => C = -v
            # The count is the number of points with V == v.
            
            unique_vals, counts = np.unique(V, return_counts=True)
            
            for val, count in zip(unique_vals, counts):
                if count == 0:
                    continue
                C = -val
                line_norm = _normalize_line(A, B, C)
                # Only add if count is significant?
                # We need top 27. Even count=1 might be needed if we run out of better ones.
                # But likely we will have many high-count lines.
                candidates.append((count, line_norm))
                
    if not candidates:
        return None, 0

    # Sort by count descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Select top 27 unique lines
    selected_lines = []
    seen = set()
    
    for count, line_norm in candidates:
        if len(selected_lines) >= 27:
            break
        if line_norm not in seen:
            seen.add(line_norm)
            selected_lines.append(line_norm)
            
    if len(selected_lines) < 27:
        # Fallback: fill with any valid lines
        # This is unlikely given the density
        for A in range(-20, 21):
            for B in range(-20, 21):
                if A == 0 and B == 0: continue
                if math.gcd(abs(A), abs(B)) != 1: continue
                # Just pick a line that hits at least one point
                # We can reuse the unique_vals from a previous A,B? 
                # Simpler: iterate C
                for C in range(-100, 101):
                    norm = _normalize_line(A, B, C)
                    if norm not in seen:
                        # Check if it hits any point?
                        # If not, incidence is 0.
                        # We just need 27 distinct lines.
                        seen.add(norm)
                        selected_lines.append(norm)
                        if len(selected_lines) >= 27:
                            break
                if len(selected_lines) >= 27:
                    break
            if len(selected_lines) >= 27:
                break
                
    if len(selected_lines) < 27:
        return None, 0

    lines_array = np.array(selected_lines, dtype=np.int64)
    
    # Calculate total incidence
    # I = sum over lines of (number of points on line)
    # We already have the counts for the top 27?
    # The `count` in the tuple is the incidence for that specific line.
    # We need to sum them.
    
    # Re-calculate to be safe, or sum from candidates?
    # Summing from candidates is faster.
    total_incidence = 0
    for line in selected_lines:
        A, B, C = line
        # Find the count from candidates list?
        # It's easier to just recompute for the 27 lines.
        val = A * px + B * py + C
        total_incidence += np.sum(val == 0)
        
    return lines_array, total_incidence

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 600.0 - 5.0 # Safety margin
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # Candidate grid dimensions (w, h) such that w*h >= 54
    # We want to explore shapes that might yield high incidences.
    # 6x9, 7x8, 5x11, 4x14, 3x18, 2x27, 1x54
    # Also non-rectangular? Stick to rectangular for now.
    
    grid_configs = [
        (6, 9),
        (9, 6),
        (7, 8),
        (8, 7),
        (5, 11),
        (11, 5),
        (4, 14),
        (14, 4),
        (3, 18),
        (18, 3),
        (2, 27),
        (27, 2),
        (1, 54),
        (54, 1),
    ]
    
    K = 50 # Max slope component
    
    for w, h in grid_configs:
        if time.time() > deadline:
            break
            
        points = _generate_grid_points(w, h, 54)
        
        # Optional: Shift the grid to center at origin?
        # The lines A*x + B*y + C = 0 are translation invariant in terms of incidence structure?
        # No, C changes. But the set of relative incidences is the same.
        # However, the "best" lines might have different C values.
        # Centering might help with symmetry in the search?
        # Let's try centering for 7x8 and 5x11.
        
        lines, score = _find_best_lines(points, K, deadline)
        
        if lines is not None:
            if score > best_score:
                best_score = score
                best_points = points
                best_lines = lines
                
    # If no grid worked, fallback
    if best_points is None:
        points = _generate_grid_points(6, 9, 54)
        lines = np.array([
            [1, 0, -1], [1, 0, -2], [1, 0, -3], [1, 0, -4], [1, 0, -5], [1, 0, -6],
            [0, 1, -1], [0, 1, -2], [0, 1, -3], [0, 1, -4], [0, 1, -5], [0, 1, -6],
            [0, 1, -7], [0, 1, -8], [0, 1, -9],
            [1, -1, 0], [1, -1, 1], [1, -1, 2], [1, -1, 3], [1, -1, 4], [1, -1, 5],
            [1, -1, 6], [1, -1, -1], [1, -1, -2], [1, -1, -3], [1, -1, -4], [1, -1, -5]
        ], dtype=np.int64)
        best_points = points
        best_lines = lines
        
    return best_points, best_lines
