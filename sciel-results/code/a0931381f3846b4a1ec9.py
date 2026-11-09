import numpy as np
import math
import time
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize line coefficients to canonical form."""
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _count_incidence(line, points):
    """Count points on a given line. line is (A, B, C), points is (N, 2) array."""
    A, B, C = line
    # Vectorized check
    vals = A * points[:, 0] + B * points[:, 1] + C
    return int(np.sum(vals == 0))

def szemeredi_trotter_incidences():
    # Deadline guard
    start_time = time.time()
    deadline = 620.0  # 620s safety margin before 630s kill
    
    # 1. Define Points: 6x9 grid
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # 2. Generate Candidate Lines from all pairs of points
    n_points = len(points)
    unique_lines = {}
    
    # Precompute point coords for speed
    p_list = points.tolist()
    
    for i in range(n_points):
        x1, y1 = p_list[i]
        for j in range(i + 1, n_points):
            x2, y2 = p_list[j]
            
            # Line equation: A*x + B*y + C = 0
            A = y2 - y1
            B = x1 - x2
            C = -(A * x1 + B * y1)
            
            # Normalize
            key = _normalize_line(A, B, C)
            if key != (0, 0, 0):
                unique_lines[key] = None
                
    # 3. Count Incidences for Each Unique Line (vectorized)
    line_data = []
    for line in unique_lines.keys():
        count = _count_incidence(line, points)
        line_data.append((count, line))
        
    # 4. Select Top 27 Lines
    line_data.sort(key=lambda item: item[0], reverse=True)
    
    selected_lines = [line_data[i][1] for i in range(min(27, len(line_data)))]
    
    # 5. Local search refinement: try swapping out low-incidence lines
    #    with better candidates not already selected
    selected_set = set(selected_lines)
    total_incidence = sum(_count_incidence(line, points) for line in selected_lines)
    
    # Get candidate lines not in selected set, sorted by individual incidence
    remaining = [(count, line) for count, line in line_data if line not in selected_set]
    remaining.sort(key=lambda x: x[0], reverse=True)
    
    improved = True
    max_iterations = 200  # Safety limit on refinement iterations
    
    while improved and max_iterations > 0:
        improved = False
        max_iterations -= 1
        
        if time.time() > deadline:
            break
            
        # Try to improve by swapping the worst selected line with best remaining
        # Compute per-line incidence for current selection
        line_counts = {}
        for line in selected_lines:
            line_counts[line] = _count_incidence(line, points)
        
        # Find the line with minimum incidence in selection
        worst_idx = min(range(len(selected_lines)), key=lambda i: line_counts[selected_lines[i]])
        worst_line = selected_lines[worst_idx]
        worst_count = line_counts[worst_line]
        
        # Try each remaining candidate to see if swapping improves total
        best_gain = 0
        best_swap_line = None
        
        for count, candidate in remaining[:50]:  # Check top 50 candidates
            if time.time() > deadline:
                break
            # If we swap worst_line with candidate, what's the new total?
            # New total = old_total - worst_count + count(candidate)
            # But we need to be careful: the count is independent since lines are distinct
            gain = count - worst_count
            if gain > best_gain:
                best_gain = gain
                best_swap_line = candidate
        
        if best_swap_line is not None and best_gain > 0:
            # Perform the swap
            selected_lines[worst_idx] = best_swap_line
            selected_set.discard(worst_line)
            selected_set.add(best_swap_line)
            # Update remaining list
            remaining = [(c, l) for c, l in remaining if l != best_swap_line]
            # Add the removed line back to remaining if it has value
            remaining.append((worst_count, worst_line))
            remaining.sort(key=lambda x: x[0], reverse=True)
            improved = True
            
    # 6. Final line array
    lines = np.array(selected_lines, dtype=np.int64)
    
    return points, lines
