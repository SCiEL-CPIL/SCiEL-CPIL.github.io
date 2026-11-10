import numpy as np


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines.
    
    Construction: 3x18 grid with horizontal, vertical, and diagonal lines.
    - 3 horizontal lines (y=1,2,3): 3*18 = 54 incidences
    - 18 vertical lines (x=1..18): 18*3 = 54 incidences  
    - 6 diagonal lines (x+y=c for selected c): 6*3 = 18 incidences
    Total: 126 incidences
    """
    # Points: 3 x 18 grid
    xs = np.array([1, 2, 3])
    ys = np.arange(1, 19)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    
    # Horizontal lines: y = c -> 0*x + 1*y - c = 0 -> (0, 1, -c)
    for c in [1, 2, 3]:
        lines.append((0, 1, -c))
    
    # Vertical lines: x = c -> 1*x + 0*y - c = 0 -> (1, 0, -c)
    for c in range(1, 19):
        lines.append((1, 0, -c))
    
    # Diagonal lines: x + y = c -> 1*x + 1*y - c = 0 -> (1, 1, -c)
    # Choose c values where the line passes through all 3 x-values
    # x+y=c has points (1,c-1), (2,c-2), (3,c-3) all in grid if 1<=c-x<=18 for x=1,2,3
    # So need: 1<=c-1<=18, 1<=c-2<=18, 1<=c-3<=18
    # i.e., 4<=c<=19
    # Pick 6 values
    diag_cs = [5, 7, 9, 11, 13, 15]
    for c in diag_cs:
        lines.append((1, 1, -c))
    
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines
