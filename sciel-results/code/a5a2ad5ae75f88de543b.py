# EVOLVE-BLOCK-START
import numpy as np
from math import gcd
from itertools import combinations

def szemeredi_trotter_incidences():
    # Define the 54 points as a 6x9 grid
    # x in {0, ..., 5}, y in {0, ..., 8}
    points = []
    for x in range(6):
        for y in range(9):
            points.append((x, y))
    
    # points is a list of 54 tuples
    # We need to find the top 27 lines by incidence count
    
    # Map line coefficients (A, B, C) normalized to the line itself
    # Line: A*x + B*y + C = 0
    # We will collect all unique lines defined by pairs of points
    
    line_counts = {}
    
    # Helper to normalize line (A, B, C)
    def normalize(A, B, C):
        # Find gcd of absolute values
        g = gcd(abs(A), gcd(abs(B), abs(C)))
        if g == 0:
            return (0, 0, 0) # Should not happen for distinct points
        A //= g
        B //= g
        C //= g
        # Make first non-zero positive
        if A < 0:
            A, B, C = -A, -B, -C
        elif A == 0 and B < 0:
            A, B, C = -A, -B, -C
        return (A, B, C)
    
    # Generate all lines from pairs
    # 54 points -> ~1400 pairs
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # A = y2 - y1
            # B = x1 - x2
            # C = x2*y1 - x1*y2
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            key = normalize(A, B, C)
            
            if key not in line_counts:
                line_counts[key] = 0
                
            # We don't count here, we count later for all points
            # Just collect unique lines
            line_counts[key] = 0 

    # Now, for each unique line, count how many of the 54 points lie on it
    final_line_counts = {}
    for key in line_counts:
        A, B, C = key
        count = 0
        for x, y in points:
            if A * x + B * y + C == 0:
                count += 1
        final_line_counts[key] = count
        
    # Sort lines by count descending
    sorted_lines = sorted(final_line_counts.items(), key=lambda item: item[1], reverse=True)
    
    # Take top 27
    top_27_lines = [line for line, count in sorted_lines[:27]]
    
    # Convert to numpy arrays
    points_arr = np.array(points, dtype=np.int64)
    lines_arr = np.array(top_27_lines, dtype=np.int64)
    
    return points_arr, lines_arr

# EVOLVE-BLOCK-END
