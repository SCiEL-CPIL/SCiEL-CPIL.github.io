import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    best_result = None
    best_score = -1
    
    # Try different aspect ratios
    # w + h = 2, so w in (0, 2), h = 2 - w
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.67, 2.0, 0.5, 1.4, 0.7, 1.3]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = ar * (2.0 / (1.0 + ar))  # w + h = 2, w/h = ar
        h = 2.0 - w
        
        # Grid initialization
        # Choose grid dimensions based on aspect ratio
        if ar >= 1.0:
            cols = int(np.round(np.sqrt(N * ar)))
            rows = int(np.ceil(N / cols))
        else:
            rows = int(np.round(np.sqrt(N / ar)))
            cols = int(np.ceil(N / rows))
        
        # Ensure we have at least N cells
        while cols * rows < N:
            if ar >= 1.0:
                cols += 1
            else:
                rows += 1
        
        # Place N circles in a grid
        positions = []
        radii = []
        for i in range(N):
            row = i // cols
            col = i % cols
            x = (col + 0.5) * w / cols
            y = (row + 0.5) * h / rows
            positions.append([x, y])
        
        positions = np.array(positions)
        
        # Initial radii: limited by spacing and boundaries
        # Spacing-based radius
        dx = w / cols
        dy = h / rows
        r_init = 0.5 * min(dx, dy) * 0.95
        
        # Also limited by boundaries
        for i in range(N):
            r_max = min(positions[i, 0], w - positions[i, 0],
                        positions[i, 1], h - positions[i, 1])
            radii.append(min(r_init, r_max))
        
        radii = np.array(radii)
        
        # Iterative relaxation
        for iteration in range(500):
            if time.time() > deadline:
                break
            
            # Check validity and compute score
            valid = True
            for i in range(N):
                if positions[i, 0] - radii[i] < -1e-9 or positions[i, 0] + radii[i] > w + 1e-9:
                    valid = False
                if positions[i, 1] - radii[i] < -1e-9 or positions[i, 1] + radii[i] > h + 1e-9:
                    valid = False
            
            if valid:
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([positions, radii]).copy()
            
            # Compute forces
            forces = np.zeros_like(positions)
            dradii = np.zeros(N)
            
            # Boundary forces: push circles toward center if too close to boundary
            for i in range(N):
                # Left/Right
                if positions[i, 0] - radii[i] < 0:
                    forces[i, 0] += 10.0
                if positions[i, 0] + radii[i] > w:
                    forces[i, 0] -= 10.0
                # Bottom/Top
                if positions[i, 1] - radii[i] < 0:
                    forces[i, 1] += 10.0
                if positions[i, 1] + radii[i] > h:
                    forces[i, 1] -= 10.0
            
            # Overlap forces: push apart
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = radii[i] + radii[j]
                    
                    if dist < min_dist and dist > 1e-10:
                        overlap = min_dist - dist
                        fx = overlap * dx / dist
                        fy = overlap * dy / dist
                        forces[i, 0] -= fx
                        forces[i, 1] -= fy
                        forces[j, 0] += fx
                        forces[j, 1] += fy
                        
                        # Also reduce radii slightly if overlapping
                        dradii[i] -= overlap * 0.5
                        dradii[j] -= overlap * 0.5
                    elif dist < 1e-10:
                        # Random small push
                        forces[i, 0] += np.random.randn() * 0.01
                        forces[i, 1] += np.random.randn() * 0.01
                        forces[j, 0] -= np.random.randn() * 0.01
                        forces[j, 1] -= np.random.randn() * 0.01
        
            # Try to expand radii toward boundaries
            for i in range(N):
                r_max = min(positions[i, 0], w - positions[i, 0],
                            positions[i, 1], h - positions[i, 1])
                # Also check against neighbors
                for j in range(N):
                    if i != j:
                        dx = positions[j, 0] - positions[i, 0]
                        dy = positions[j, 1] - positions[i, 1]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist > 1e-10:
                            r_max = min(r_max, (dist - radii[j]) / 1.0)
                r_max = max(r_max, 0.001)
                dradii[i] += (r_max - radii[i]) * 0.1
            
            # Apply updates with damping
            learning_rate = 0.05
            positions += forces * learning_rate
            radii += dradii * learning_rate
            
            # Clamp radii to positive
            radii = np.maximum(radii, 0.001)
            
            # Clamp positions to be within rectangle (with margin for radius)
            for i in range(N):
                positions[i, 0] = np.clip(positions[i, 0], radii[i], w - radii[i])
                positions[i, 1] = np.clip(positions[i, 1], radii[i], h - radii[i])
        
        # Final check
        valid = True
        for i in range(N):
            if positions[i, 0] - radii[i] < -1e-8 or positions[i, 0] + radii[i] > w + 1e-8:
                valid = False
            if positions[i, 1] - radii[i] < -1e-8 or positions[i, 1] + radii[i] > h + 1e-8:
                valid = False
            for j in range(i + 1, N):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j] - 1e-8:
                    valid = False
        
        if valid:
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([positions, radii]).copy()
    
    # If no valid result found, fall back to seed
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
