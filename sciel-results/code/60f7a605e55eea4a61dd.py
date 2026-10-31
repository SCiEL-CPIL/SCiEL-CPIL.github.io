"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Pack 21 circles in a perimeter-4 rectangle using force-directed relaxation.
    Tests multiple aspect ratios to find the optimal shape.
    """
    best_value = 0.0
    best_solution = None
    
    # Test several aspect ratios w/h
    # Perimeter 4 => w + h = 2 => h = 2 - w
    # w ranges from 0.8 to 1.2 (symmetric around 1.0)
    aspect_ratios = [1.0, 1.1, 0.9, 1.2, 0.8]
    
    for w in aspect_ratios:
        h = 2.0 - w
        if w <= 0 or h <= 0:
            continue
            
        # Initialize positions in a grid pattern scaled to w, h
        cols = 5
        rows = 5
        positions = np.zeros((N, 2))
        for i in range(N):
            col = i % cols
            row = i // cols
            positions[i, 0] = (col + 0.5) / cols * w
            positions[i, 1] = (row + 0.5) / rows * h
            
        # Initialize radii
        radii = np.full(N, 0.05)
        
        # Simulation parameters
        max_iters = 1000
        dt = 0.01
        growth_rate = 0.0001
        
        for it in range(max_iters):
            forces = np.zeros((N, 2))
            overlaps = False
            
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist_sq = dx * dx + dy * dy
                    dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                    
                    min_dist = radii[i] + radii[j]
                    
                    if dist < min_dist:
                        overlaps = True
                        nx = dx / dist
                        ny = dy / dist
                        overlap = min_dist - dist
                        f_mag = overlap * 10.0
                        
                        forces[i, 0] += nx * f_mag
                        forces[i, 1] += ny * f_mag
                        forces[j, 0] -= nx * f_mag
                        forces[j, 1] -= ny * f_mag
            
            for i in range(N):
                r = radii[i]
                x, y = positions[i]
                
                if x < r:
                    forces[i, 0] += (r - x) * 10.0
                    overlaps = True
                if x > w - r:
                    forces[i, 0] -= (x - (w - r)) * 10.0
                    overlaps = True
                if y < r:
                    forces[i, 1] += (r - y) * 10.0
                    overlaps = True
                if y > h - r:
                    forces[i, 1] -= (y - (h - r)) * 10.0
                    overlaps = True
            
            positions += forces * dt
            
            for i in range(N):
                r = radii[i]
                positions[i, 0] = np.clip(positions[i, 0], r, w - r)
                positions[i, 1] = np.clip(positions[i, 1], r, h - r)
                
            if not overlaps:
                radii += growth_rate
        
        # Check validity
        valid = True
        for i in range(N):
            x, y = positions[i]
            r = radii[i]
            if x < r or x > w - r or y < r or y > h - r:
                valid = False
                break
            for j in range(i + 1, N):
                dx = x - positions[j, 0]
                dy = y - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - 1e-9:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            value = np.sum(radii)
            if value > best_value:
                best_value = value
                # Store a copy
                sol = np.zeros((N, 3))
                sol[:, 0] = positions[:, 0]
                sol[:, 1] = positions[:, 1]
                sol[:, 2] = radii
                best_solution = sol
        else:
            # If invalid, shrink radii to make it valid
            # This is a fallback
            scale = 0.95
            radii *= scale
            # Re-check validity quickly
            # If still invalid, skip this aspect ratio
            # For simplicity, assume scaling fixes it or it's bad
            # Let's just skip if not valid after one shrink attempt
            # Actually, let's just not use it if not valid.
            pass

    if best_solution is None:
        # Fallback: simple grid in 1x1
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.1) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_solution
# EVOLVE-BLOCK-END
