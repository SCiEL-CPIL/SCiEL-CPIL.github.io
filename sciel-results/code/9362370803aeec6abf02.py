# EVOLVE-BLOCK-START
import numpy as np
from math import gcd

def szemeredi_trotter_incidences():
    points = [(x, y) for x in range(6) for y in range(9)]
    points_arr = np.array(points, dtype=np.int64)
    
    def normalize(A, B, C):
        g = gcd(abs(A), gcd(abs(B), abs(C)))
        if g == 0:
            return (0, 0, 0)
        A //= g
        B //= g
        C //= g
        if A < 0 or (A == 0 and B < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)
    
    line_counts = {}
    for i in range(54):
        x1, y1 = points[i]
        for j in range(i + 1, 54):
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            key = normalize(A, B, C)
            line_counts[key] = 0
            
    final_line_counts = {}
    for key in line_counts:
        A, B, C = key
        count = 0
        for x, y in points:
            if A * x + B * y + C == 0:
                count += 1
        final_line_counts[key] = count
        
    sorted_lines = sorted(final_line_counts.items(), key=lambda item: item[1], reverse=True)
    top_27_lines = [line for line, count in sorted_lines[:27]]
    lines_arr = np.array(top_27_lines, dtype=np.int64)
    
    return points_arr, lines_arr

# EVOLVE-BLOCK-END
