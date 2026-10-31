"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 130  # safety margin before 150s
    
    # Initialize with a grid arrangement
    # 26 circles: try 5x6 grid or 6x5
    # Use 5 columns, 6 rows = 30 cells, take 26
    ncols = 5
    nrows = 6
    positions = []
    for i in range(ncols):
        for j in range(nrows):
            if len(positions) >= N:
                break
            x = (i + 0.5) / ncols
            y = (j + 0.5) / nrows
            positions.append([x, y])
    
    # Initial radii: small
    init_r = 0.5 / max(ncols, nrows) - 0.02
    radii = [init_r] * N
    
    def get_state():
        return np.array(positions), np.array(radii)
    
    def compute_min_radius(pos, rad):
        """For each circle, compute max radius given others' positions."""
        n = len(pos)
        max_r = np.zeros(n)
        for i in range(n):
            # Boundary constraint
            r_bound = min(pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
            r_bound = max(0.001, r_bound)
            # Constraint from other circles
            r_min = r_bound
            for j in range(n):
                if i == j:
                    continue
                dist = np.sqrt((pos[i][0] - pos[j][0])**2 + (pos[i][1] - pos[j][1])**2)
                # Need: dist >= rad[i] + rad[j] => rad[i] <= dist - rad[j]
                r_from_j = dist - rad[j]
                if r_from_j < r_min:
                    r_min = r_from_j
            r_min = max(0.001, r_min)
            max_r[i] = r_min
        return max_r
    
    def total_radius(pos, rad):
        return np.sum(rad)
    
    # Iterative growth: repeatedly grow each circle to its max possible radius
    best_score = -1
    best_result = None
    
    # Run for several passes
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # For each circle, grow to max possible radius
        for i in range(N):
            pos_arr = np.array(positions)
            rad_arr = np.array(radii)
            max_r = compute_min_radius(pos_arr, rad_arr)
            radii[i] = max_r[i]
        
        # After growing all, check for overlaps and resolve by moving circles
        # Use a simple relaxation: if two circles overlap, push them apart
        for relax_iter in range(50):
            if time.time() > deadline:
                break
            max_overlap = 0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[j][0] - positions[i][0]
                    dy = positions[j][1] - positions[i][1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist and dist > 1e-10:
                        overlap = min_dist - dist
                        max_overlap = max(max_overlap, overlap)
                        # Push apart along the line connecting centers
                        nx = dx / dist
                        ny = dy / dist
                        # Move each circle half the overlap
                        positions[i][0] -= nx * overlap * 0.5
                        positions[i][1] -= ny * overlap * 0.5
                        positions[j][0] += nx * overlap * 0.5
                        positions[j][1] += ny * overlap * 0.5
                    elif dist < min_dist and dist <= 1e-10:
                        # Very close, push randomly
                        angle = np.random.uniform(0, 2 * np.pi)
                        positions[i][0] -= np.cos(angle) * 0.001
                        positions[i][1] -= np.sin(angle) * 0.001
                        positions[j][0] += np.cos(angle) * 0.001
                        positions[j][1] += np.sin(angle) * 0.001
            
            # Clamp positions to [0,1]^2
            for i in range(N):
                positions[i][0] = np.clip(positions[i][0], 0, 1)
                positions[i][1] = np.clip(positions[i][1], 0, 1)
            
            if max_overlap < 1e-6:
                break
        
        # After relaxation, recompute radii
        pos_arr = np.array(positions)
        rad_arr = np.array(radii)
        max_r = compute_min_radius(pos_arr, rad_arr)
        radii = list(max_r)
        
        # Check score
        score = total_radius(np.array(positions), np.array(radii))
        if score > best_score:
            best_score = score
            best_result = np.array([[positions[i][0], positions[i][1], radii[i]] for i in range(N)])
    
    if best_result is not None:
        return best_result
    
    # Fallback: return initial grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
