import numpy as np
import time
import sys

def szemeredi_trotter_incidences():
    """
    Returns a set of 54 points and 27 lines maximizing incidences.
    Uses a 6x9 grid construction with vertical, horizontal, and diagonal lines.
    """
    # Points: 6 x 9 grid
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    
    # 6 Vertical lines: x = i  -> 1*x + 0*y - i = 0
    for i in range(1, 7):
        lines.append((1, 0, -i))
        
    # 9 Horizontal lines: y = j  -> 0*x + 1*y - j = 0
    for j in range(1, 10):
        lines.append((0, 1, -j))
        
    # 12 Diagonal lines
    # We need 12 lines.
    # In 6x9 grid, slope 1 lines with 6 points: c in {-3, -2, -1, 0, 1, 2, 3}
    # In 6x9 grid, slope -1 lines with 6 points: c in {7, 8, 9, 10, 11, 12, 13} ?
    # Let's verify slope -1 lines y = -x + c -> x + y - c = 0.
    # x in 1..6, y in 1..9.
    # x+y range: 2..15.
    # Len 6 means 6 points.
    # c=7: (1,6)..(6,1) -> 6 pts.
    # c=8: (1,7)..(6,2) -> 6 pts.
    # c=9: (1,8)..(6,3) -> 6 pts.
    # c=10: (1,9)..(5,5) -> 5 pts? (1,9), (2,8), (3,7), (4,6), (5,5). x=6, y=4. Yes 6 pts.
    # Wait. x=1..6. y=9..4. All valid.
    # c=11: (2,9)..(6,5) -> 5 pts.
    # So slope -1 len 6: c in {7, 8, 9, 10}. (4 lines).
    # c=6: (1,5)..(5,1) -> 5 pts.
    # c=12: (3,9)..(6,6) -> 4 pts.
    
    # So we have 7 slope 1 lines of len 6.
    # We have 4 slope -1 lines of len 6.
    # Total 11 lines of len 6.
    # We need 1 more line.
    # Best remaining:
    # Slope 1 len 5: c in {-4, 4}. (2 lines).
    # Slope -1 len 5: c in {6, 11}. (2 lines).
    # So we pick one of these.
    
    # Add 7 slope 1 lines
    for c in [-3, -2, -1, 0, 1, 2, 3]:
        # y = x + c -> x - y + c = 0
        lines.append((1, -1, c))
        
    # Add 4 slope -1 lines
    for c in [7, 8, 9, 10]:
        # y = -x + c -> x + y - c = 0
        lines.append((1, 1, -c))
        
    # Add 1 more line, slope 1, len 5
    c = -4
    lines.append((1, -1, c))
    
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines
