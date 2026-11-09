import numpy as np
from math import gcd
import time

def szemeredi_trotter_incidences():
    start_time = time.time()
    
    # 1. Generate 7x8 grid points
    xs = np.arange(1, 8)
    ys = np.arange(1, 9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    all_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # 2. Remove 2 points to get 54
    # Remove (1,1) and (7,8) which are at indices 0 and 55
    # Indices: (1,1) is 0. (7,8) is 7*8-1 = 55.
    remove_indices = [0, 55]
    points = np.delete(all_points, remove_indices, axis=0)
    
    # 3. Generate all unique lines defined by pairs of points
    lines_set = set()
    n = len(points)
    
    # Optimization: We can iterate through pairs. 54*53/2 = 1431 pairs.
    # This is fast enough.
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = -(x2 - x1)
            C = (x2 - x1)*y1 - (y2 - y1)*x2
            
            # Normalize
            g = gcd(abs(A), gcd(abs(B), abs(C)))
            if g > 0:
                A, B, C = A//g, B//g, C//g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
                
            lines_set.add((A, B, C))
            
    # 4. Count incidences for each line
    line_list = list(lines_set)
    incidences = []
    for A, B, C in line_list:
        count = 0
        for x, y in points:
            if A*x + B*y + C == 0:
                count += 1
        incidences.append(count)
        
    # 5. Sort lines by incidence count
    line_indices = sorted(range(len(line_list)), key=lambda k: incidences[k], reverse=True)
    
    # 6. Select top 27 lines
    selected_lines = []
    for idx in line_indices[:27]:
        selected_lines.append(line_list[idx])
        
    lines = np.array(selected_lines, dtype=np.int64)
    
    # Safety check
    if time.time() - start_time > 500:
         # Fallback to a simpler static construction if too slow
         # But 1431 pairs is very fast.
         pass
         
    return points, lines
