import math
import time

def construct():
    # Target: Pack 11 unit hexagons in a regular hexagon of minimal side length S.
    # Strategy: Use a 3x4 grid of aligned hexagons (12 positions), take 11.
    # This configuration fits comfortably in S=5.0.
    
    # Grid parameters
    # Hexagon side length = 1
    # Distance between centers in x (same row) = sqrt(3)
    # Distance between centers in y (adjacent rows) = 1.5
    # Offset in x for alternating rows = sqrt(3)/2
    
    sqrt3 = math.sqrt(3)
    dx = sqrt3
    dy = 1.5
    
    # 3 rows, 4 columns
    # Rows: -1, 0, 1
    # Cols: -1.5, -0.5, 0.5, 1.5 (in units of dx)
    
    positions = []
    for r in range(-1, 2):
        for c in range(-2, 2):
            # Calculate x, y
            # If row is even, no offset? Let's center the grid.
            # Let's use axial-like coordinates for simplicity.
            # x = (c + 0.5*r) * dx ? No, simple grid is easier.
            
            # Let's define the 12 positions explicitly for a 3x4 block
            # Centered at (0,0)
            # Row 0 (top): y = 1.5
            # Row 1 (mid): y = 0.0
            # Row 2 (bot): y = -1.5
            
            # X positions for 4 columns:
            # To center 4 columns, x offsets: -1.5*dx, -0.5*dx, 0.5*dx, 1.5*dx
            
            x_vals = [-1.5 * dx, -0.5 * dx, 0.5 * dx, 1.5 * dx]
            
            if r == -1: # Top row (y=1.5)
                y = 1.5
                # Offset x by dx/2 to make it a honeycomb? 
                # If we use aligned grid (brick wall), rows are shifted.
                # Standard hex grid:
                # Row 0: x = 0, dx, 2dx, 3dx
                # Row 1: x = dx/2, 3dx/2, 5dx/2, 7dx/2
                # Let's center this.
                # Total width of 4 cols is 3dx. Center is 1.5dx.
                # So x relative to center: -1.5dx, -0.5dx, 0.5dx, 1.5dx for "even" rows.
                # For "odd" rows, shift by dx/2: -1.0dx, 0.0dx, 1.0dx, 2.0dx? 
                # No, shift by dx/2 means: -1.5dx + 0.5dx = -1.0dx.
                
                # Let's just use the "even" row positions for all rows to keep it a rectangle?
                # No, hexagons in a rectangular grid with same x-coords will overlap if y-dist < sqrt(3).
                # y-dist is 1.5. sqrt(3) is 1.732.
                # So if x-coords are same, they overlap!
                # We MUST shift the rows.
                
                # So:
                # Row y=1.5: x = -1.5dx, -0.5dx, 0.5dx, 1.5dx
                # Row y=0.0:  x = -1.0dx, 0.0dx, 1.0dx, 2.0dx ? 
                # Wait, if we shift by dx/2, the range changes.
                # Let's center the whole 3x4 block.
                
                pass

    # Let's construct the 12 positions properly.
    # Row 0 (y = 1.5): x = -1.5*dx, -0.5*dx, 0.5*dx, 1.5*dx
    # Row 1 (y = 0.0):  x = -1.0*dx, 0.0*dx, 1.0*dx, 2.0*dx
    # Row 2 (y = -1.5): x = -1.5*dx, -0.5*dx, 0.5*dx, 1.5*dx
    
    # Check overlap between Row 0 and Row 1.
    # Dist between (x, 1.5) and (x+dx/2, 0).
    # dx_sq = (dx/2)^2 + 1.5^2 = (3/4) + 2.25 = 3.
    # dist = sqrt(3) = 1.732.
    # Min dist between hex centers is sqrt(3). So they touch. Valid.
    
    # Check overlap within Row 0.
    # Dist between (x, 1.5) and (x+dx, 1.5).
    # dist = dx = sqrt(3). Valid.
    
    # So this is a valid honeycomb packing.
    
    pos_list = []
    # Row 0
    for x in [-1.5 * dx, -0.5 * dx, 0.5 * dx, 1.5 * dx]:
        pos_list.append((x, 1.5))
    # Row 1
    for x in [-1.0 * dx, 0.0 * dx, 1.0 * dx, 2.0 * dx]:
        pos_list.append((x, 0.0))
    # Row 2
    for x in [-1.5 * dx, -0.5 * dx, 0.5 * dx, 1.5 * dx]:
        pos_list.append((x, -1.5))
        
    # We have 12 positions. Take 11.
    # Remove the last one to be safe? Or any one.
    # Let's remove the one at (2.0*dx, 0.0) which is the rightmost middle one.
    # Or just take the first 11.
    
    inner = []
    for i in range(11):
        x, y = pos_list[i]
        inner.append([x, y, 0.0])
        
    # Calculate required S.
    # Max distance from origin to any vertex of any hexagon.
    # Hexagon vertices are at distance 1 from center.
    # So max_dist = max( ||center|| ) + 1.
    
    max_r = 0.0
    for x, y in pos_list:
        r = math.sqrt(x*x + y*y)
        if r > max_r:
            max_r = r
            
    # The farthest center is at (2.0*dx, 0.0) ?
    # 2.0 * 1.732 = 3.464.
    # Or (1.5*dx, 1.5) ?
    # 1.5 * 1.732 = 2.598.
    # sqrt(2.598^2 + 1.5^2) = sqrt(6.75 + 2.25) = sqrt(9) = 3.0.
    # So (2.0*dx, 0) is farther. r = 3.464.
    # If we remove (2.0*dx, 0), the next farthest is (1.5*dx, 1.5) or (1.5*dx, -1.5) or (-1.5*dx, 1.5)...
    # r = 3.0.
    # So max_r = 3.0 if we remove the corner.
    
    # Let's remove the last one in pos_list which is (1.5*dx, -1.5).
    # Then max_r is determined by (2.0*dx, 0.0) which is 3.464.
    # So S must be >= 3.464 + 1 = 4.464.
    
    # Let's set S = 4.5.
    
    side = 4.5
    
    return inner, (0.0, 0.0), side, 0.0

