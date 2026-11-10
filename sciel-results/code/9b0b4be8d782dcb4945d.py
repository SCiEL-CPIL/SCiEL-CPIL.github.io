# EVOLVE-BLOCK-START
import numpy as np


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization. Construct 54 points and 27 lines in the
    integer lattice to maximize the number of point-line incidences.

    Construction:
      - Points: a 6 x 9 grid, i.e. {0,1,...,5} x {0,1,...,8}  -> 54 points.
      - Lines: 27 lines chosen to maximize incidences:
          * 4 horizontal lines (y = 0,1,2,3)          -> 6 points each  -> 24
          * 9 vertical lines (x = 0,1,...,8)          -> 9 points each  -> 81
          * 12 diagonal lines of slopes ±1, ±2        -> 6+6+6+6+5+5+5+5+4+4+4+4 = 60
        Total incidences I = 24 + 81 + 60 = 165?  Let me recount carefully.
        
        Actually let me re-derive with the observed best score 2.1728395 = 176/81.
        So I = 176. Let me design to achieve 176.

    Design for I = 176:
      Points: 6 x 9 grid = 54 points at (x, y) with x in 0..5, y in 0..8.
      
      Lines (27 total):
        1) 9 vertical lines: x = 0,1,2,...,8? No, x only goes 0..5. 
           Wait, 6 columns. So 6 vertical lines, each with 9 points -> 54 incidences.
        
        Let me reconsider. With 6x9 grid:
        - 6 vertical lines (one per column), each passing through 9 points: 6*9 = 54
        - 9 horizontal lines (one per row), each passing through 6 points: 9*6 = 54
        That's already 108 with 15 lines. We have 12 more lines to add.
        
        Diagonals with slope 1: lines y = x + c. In the 6x9 grid (x:0..5, y:0..8),
        a line y = x + c passes through points where x in [max(0,-c), min(5, 8-c)] and y=x+c in [0,8].
        Length = number of integer x satisfying max(0,-c) <= x <= min(5, 8-c).
        
        For slope 1:
          c=0: x in [0,5], len 6
          c=1: x in [0,5], len 6
          c=2: x in [0,5], len 6
          c=3: x in [0,5], len 6
          c=4: x in [0,4], len 5
          c=5: x in [0,3], len 4
          c=6: x in [0,2], len 3
          c=7: x in [0,1], len 2
          c=8: x in [0,0], len 1
          c=-1: x in [1,5], len 5
          c=-2: x in [2,5], len 4
          c=-3: x in [3,5], len 3
          c=-4: x in [4,5], len 2
          c=-5: x in [5,5], len 1
        
        Slope -1: y = -x + c, i.e., x + y = c.
          c=0: (0,0), len 1
          c=1: (0,1),(1,0), len 2
          c=2: len 3
          c=3: len 4
          c=4: len 5
          c=5: len 6 (x:0..5, y:5..0)
          c=6: len 6 (x:0..5, y:6..1)
          c=7: len 6 (x:0..5, y:7..2)
          c=8: len 6 (x:0..5, y:8..3)
          c=9: len 5 (x:1..5, y:8..4)
          c=10: len 4
          c=11: len 3
          c=12: len 2
          c=13: len 1
        
        Top diagonals by length:
          Slope 1: len 6 (c=0,1,2,3), len 5 (c=4,-1), len 4 (c=5,-2)
          Slope -1: len 6 (c=5,6,7,8), len 5 (c=4,9), len 4 (c=3,10)
          
        Top 12 diagonal lines:
          8 lines of len 6: slope 1 c=0,1,2,3; slope -1 c=5,6,7,8
          4 lines of len 5: slope 1 c=4; slope 1 c=-1; slope -1 c=4; slope -1 c=9
          Total diagonal incidences: 8*6 + 4*5 = 48 + 20 = 68
        
        Total: 54 (vertical) + 54 (horizontal) + 68 (diagonal) = 176.
        
        But that's 6 + 9 + 12 = 27 lines. 
    """
    # Points: 6 x 9 grid
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Lines
    lines = []
    
    # 6 vertical lines: x = 0, 1, 2, 3, 4, 5  => (1, 0, -c)
    for c in range(6):
        lines.append((1, 0, -c))
    
    # 9 horizontal lines: y = 0, 1, 2, ..., 8  => (0, 1, -c)
    for c in range(9):
        lines.append((0, 1, -c))
    
    # 12 diagonal lines
    # Slope 1: y = x + c => -x + y - c = 0 => (-1, 1, -c)
    for c in [0, 1, 2, 3, 4, -1]:
        lines.append((-1, 1, -c))
    
    # Slope -1: y = -x + c => x + y - c = 0 => (1, 1, -c)
    for c in [5, 6, 7, 8, 4, 9]:
        lines.append((1, 1, -c))
    
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines


# EVOLVE-BLOCK-END
