"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Pack 21 circles in a rectangle of perimeter 4, maximizing sum of radii.
    Uses physics-based optimization with smooth penalties.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Initial guess: grid-like arrangement
    # Rectangle with perimeter 4: w + h = 2
    # Start with square-ish: w=1, h=1
    # But we can also vary the aspect ratio
    
    best_result = None
    best_score = 0.0
    
    # Try different initial aspect ratios
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        # w + h = 2, w/h = ar => h = 2/(1+ar), w = 2*ar/(1+ar)
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Initialize circles in a grid pattern
        side = int(np.ceil(np.sqrt(N)))
        positions = []
        radii = []
        
        for i in range(side):
            for j in range(side):
                if len(positions) >= N:
                    break
                x = (i + 0.5) * w / side
                y = (j + 0.5) * h / side
                r = min(w, h) / (2 * side) * 0.9
                positions.append([x, y])
                radii.append(r)
        
        # Pad if needed
        while len(positions) < N:
            positions.append([w/2, h/2])
            radii.append(min(w, h) / (2 * side) * 0.9)
        
        positions = np.array(positions[:N])
        radii = np.array(radii[:N])
        
        # State vector: [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
        state = np.zeros(3 * N)
        for i in range(N):
            state[3*i] = positions[i, 0]
            state[3*i+1] = positions[i, 1]
            state[3*i+2] = radii[i]
        
        # Objective function: maximize sum of radii with penalties
        def objective(s):
            if time.time() > deadline:
                return 0.0  # Return 0 to stop optimization
            
            x = s[0::3]
            y = s[1::3]
            r = s[2::3]
            
            # Sum of radii (we want to maximize, so minimize negative)
            score = -np.sum(r)
            
            # Penalty for circles going outside rectangle
            # x must be in [r, w-r], y must be in [r, h-r]
            pen_x = np.maximum(0, r - x) + np.maximum(0, x - (w - r))
            pen_y = np.maximum(0, r - y) + np.maximum(0, y - (h - r))
            score += 100.0 * (np.sum(pen_x) + np.sum(pen_y))
            
            # Penalty for overlapping circles
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    overlap = r[i] + r[j] - dist
                    if overlap > 0:
                        score += 100.0 * overlap * overlap
            
            # Penalty for negative radii
            score += 100.0 * np.sum(np.maximum(0, -r))
            
            # Penalty for radii too large
            max_r = min(w, h) / 2
            score += 10.0 * np.sum(np.maximum(0, r - max_r))
            
            return score
        
        # Bounds: x in [0, w], y in [0, h], r in [0, min(w,h)/2]
        max_r = min(w, h) / 2
        bounds = []
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-6, max_r))
        
        # Run optimization
        try:
            res = minimize(objective, state, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < 0:
                x = res.x[0::3]
                y = res.x[1::3]
                r = res.x[2::3]
                
                # Validate
                valid = True
                for i in range(N):
                    if x[i] < -1e-6 or x[i] > w + 1e-6 or y[i] < -1e-6 or y[i] > h + 1e-6:
                        valid = False
                        break
                    if r[i] < 1e-8:
                        valid = False
                        break
                
                if valid:
                    # Check overlaps
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                            if dist < r[i] + r[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    current_score = np.sum(r)
                    if current_score > best_score:
                        best_score = current_score
                        best_result = np.column_stack([x, y, r])
                        
        except Exception:
            pass
    
    # If no valid result found, return a simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
