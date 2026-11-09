import numpy as np
import time

def szemeredi_trotter_incidences():
    """
    Constructs a set of 54 points and 27 lines to maximize incidences.
    Uses a 6x9 grid of points.
    Includes all 9 horizontal lines, all 6 vertical lines, 
    and the 12 longest diagonals of slope 1.
    Total incidences: 9*6 + 6*9 + 52 = 160.
    """
    # Deadline guard
    start_time = time.time()
    deadline = 600  # seconds

    # 1. Define Points: 6x9 Grid
    # x in {1, ..., 6}, y in {1, ..., 9}
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # 2. Define Lines
    lines = []
    
    # Horizontal lines: y = c  =>  0*x + 1*y - c = 0
    # A=0, B=1, C=-c
    for c in range(1, 10):
        lines.append((0, 1, -c))
        
    # Vertical lines: x = r  =>  1*x + 0*y - r = 0
    # A=1, B=0, C=-r
    for r in range(1, 7):
        lines.append((1, 0, -r))
        
    # Diagonal lines (slope 1): y - x = k  =>  -1*x + 1*y - k = 0
    # A=-1, B=1, C=-k
    # We need to select the 12 diagonals with the most points.
    # Length of diagonal y-x=k in x:[1,6], y:[1,9]
    # k ranges from 1-6=-5 to 9-1=8
    
    diag_lengths = {}
    for k in range(-5, 9):
        count = 0
        for x in range(1, 7):
            y = x + k
            if 1 <= y <= 9:
                count += 1
        diag_lengths[k] = count
        
    # Sort diagonals by length descending
    sorted_diags = sorted(diag_lengths.items(), key=lambda item: item[1], reverse=True)
    
    # Take top 12
    for k, length in sorted_diags[:12]:
        lines.append((-1, 1, -k))
        
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines
