"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Grid-based cellular decomposition approach
    # We partition the rectangle into N cells and place one circle in each cell
    # with radius equal to the inradius of that cell (or slightly less for safety)
    
    # The rectangle has perimeter 4, so if width=w and height=h, then 2(w+h)=4 => w+h=2
    # We want to maximize the sum of radii of N=21 circles
    
    # Strategy: Try different grid configurations (rows x cols) that give at least N cells
    # For a grid of r rows and c cols, we have r*c cells
    # Each cell has dimensions (w/c, h/r)
    # The inradius of each cell is min(w/c, h/r)/2
    # Total sum of radii = N * min(w/c, h/r)/2
    # But we need r*c >= N, and we use exactly N cells
    
    # Actually, for a rectangular grid, all cells have the same inradius
    # So sum = N * min(w/c, h/r) / 2
    # Subject to w + h = 2, w > 0, h > 0
    
    # To maximize min(w/c, h/r):
    # Let a = w/c, b = h/r
    # w = a*c, h = b*r
    # a*c + b*r = 2
    # We want to maximize min(a, b)
    # The maximum of min(a,b) subject to a*c + b*r = 2 occurs when a = b (if possible)
    # Then a*c + a*r = a(c+r) = 2 => a = 2/(c+r)
    # So min(a,b) = 2/(c+r)
    # Sum of radii = N * 2/(c+r) / 2 = N/(c+r)
    
    # So we want to minimize c+r subject to c*r >= N
    
    # Let's enumerate all (r,c) with r*c >= 21
    best_sum = 0.0
    best_config = None
    
    for r in range(1, 30):
        for c in range(1, 30):
            if r * c < N:
                continue
            # We use r*c cells but only place N circles
            # Wait, we need exactly N circles. If r*c > N, we have extra cells.
            # We can either:
            # 1. Use only N of the r*c cells
            # 2. Merge some cells
            
            # For simplicity, let's first consider r*c == N or close to it
            # But r*c must be at least N. If r*c > N, we can still place N circles in N of the cells
            # The sum would be N * inradius (since all cells have same inradius in a uniform grid)
            
            # Actually, for a uniform grid, all cells have the same dimensions
            # So if we use N out of r*c cells, the sum is N * min(w/c, h/r)/2
            # But we can choose which N cells to use - it doesn't matter since all are identical
            
            # So sum = N * min(w/c, h/r) / 2
            # With w+h=2, maximize min(w/c, h/r)
            # As derived, max min(w/c, h/r) = 2/(c+r) when w/c = h/r
            # So sum = N/(c+r)
            
            total_r = N / (r + c)
            if total_r > best_sum:
                best_sum = total_r
                best_config = (r, c)
    
    if best_config is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    r, c = best_config
    # w = 2*c/(r+c), h = 2*r/(r+c)
    w = 2.0 * c / (r + c)
    h = 2.0 * r / (r + c)
    
    # Cell dimensions
    cw = w / c
    ch = h / r
    inradius = min(cw, ch) / 2.0
    
    # Place circles in the first N cells of the r x c grid
    circles = []
    count = 0
    for i in range(r):
        for j in range(c):
            if count >= N:
                break
            x = (j + 0.5) * cw
            y = (i + 0.5) * ch
            circles.append((x, y, inradius - 1e-4))
            count += 1
        if count >= N:
            break
    
    result = np.array(circles)
    
    # Verify constraints
    # Check perimeter
    # Bounding box of circles
    if len(result) > 0:
        min_x = np.min(result[:, 0] - result[:, 2])
        max_x = np.max(result[:, 0] + result[:, 2])
        min_y = np.min(result[:, 1] - result[:, 2])
        max_y = np.max(result[:, 1] + result[:, 2])
        bw = max_x - min_x
        bh = max_y - min_y
        perimeter = 2 * (bw + bh)
        
        # Check disjointness
        valid = True
        for i in range(len(result)):
            for j in range(i+1, len(result)):
                dx = result[i, 0] - result[j, 0]
                dy = result[i, 1] - result[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < result[i, 2] + result[j, 2] - 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid or perimeter > 4.0 + 1e-6:
            # Fallback to simple grid
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
            return np.array(cells[:N])
    
    return result

# EVOLVE-BLOCK-END
