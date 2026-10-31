"""Circle packing in perimeter-4 rectangle using force-directed relaxation."""
N = 21
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 140.0  # safety margin below 150s kill

def _validity_score(x, W, H):
    """Compute penalty: 0 if all circles inside and non-overlapping."""
    n = len(x)
    cx = x[0:n]
    cy = x[n:2*n]
    r = x[2*n:]
    # Inside rectangle
    if np.any(cx < 0) or np.any(cy < 0) or np.any(cx + r > W) or np.any(cy + r > H):
        return -1.0
    # Non-overlapping
    for i in range(n):
        for j in range(i+1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - 1e-9:
                return -1.0
    return np.sum(r)

def _objective(x, W, H):
    """Penalty objective: minimize overlap + boundary violations."""
    n = len(x)
    cx = x[0:n]
    cy = x[n:2*n]
    r = x[2*n:]
    penalty = 0.0
    # Boundary penalties
    penalty += np.sum(np.maximum(0, -cx)**2)
    penalty += np.sum(np.maximum(0, -cy)**2)
    penalty += np.sum(np.maximum(0, cx + r - W)**2)
    penalty += np.sum(np.maximum(0, cy + r - H)**2)
    # Overlap penalties
    for i in range(n):
        for j in range(i+1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx*dx + dy*dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                penalty += overlap**2
    return penalty

def _force_directed_step(cx, cy, r, W, H, dt=0.01, iterations=2000):
    """Apply force-directed relaxation to improve packing."""
    n = len(r)
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    
    for it in range(iterations):
        if time.time() > _DEADLINE:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Repulsion between overlapping circles
        for i in range(n):
            for j in range(i+1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = np.sqrt(dx*dx + dy*dy) + 1e-12
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    force = (min_dist - dist) / dist
                    fx[i] += force * dx
                    fy[i] += force * dy
                    fx[j] -= force * dx
                    fy[j] -= force * dy
        
        # Wall repulsion
        for i in range(n):
            # Left wall
            if cx[i] < r[i]:
                fx[i] += (r[i] - cx[i])
            # Right wall
            if cx[i] + r[i] > W:
                fx[i] -= (cx[i] + r[i] - W)
            # Bottom wall
            if cy[i] < r[i]:
                fy[i] += (r[i] - cy[i])
            # Top wall
            if cy[i] + r[i] > H:
                fy[i] -= (cy[i] + r[i] - H)
        
        # Update positions
        cx += dt * fx
        cy += dt * fy
        
        # Clamp to stay in bounds
        cx = np.clip(cx, 0, W)
        cy = np.clip(cy, 0, H)
    
    return cx, cy, r

def construct():
    start_time = time.time()
    best_score = 0.0
    best_result = None
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.5, 2.0, 0.6, 1.67, 0.7, 1.43, 0.8, 1.25, 1.1, 0.9, 1.3, 0.77, 1.3, 0.85, 1.18, 1.05, 0.95, 1.5, 2/3]
    
    for ar in aspect_ratios:
        if time.time() > _DEADLINE - 5:
            break
        
        # Perimeter = 2*(W+H) = 4, so W+H = 2
        # W/H = ar => W = ar*H => ar*H + H = 2 => H = 2/(1+ar), W = 2*ar/(1+ar)
        H = 2.0 / (1.0 + ar)
        W = 2.0 * ar / (1.0 + ar)
        
        # Initialize circles in a grid pattern
        side = int(np.ceil(np.sqrt(N)))
        cx = np.zeros(N)
        cy = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            col = i % side
            row = i // side
            cx[i] = (col + 0.5) * W / side
            cy[i] = (row + 0.5) * H / side
            r[i] = min(W, H) / (2*side) * 0.95
        
        # Force-directed relaxation
        cx, cy, r = _force_directed_step(cx, cy, r, W, H, dt=0.005, iterations=3000)
        
        # Try to grow radii while maintaining validity
        for grow_iter in range(100):
            if time.time() > _DEADLINE - 5:
                break
            # Check if we can grow all radii slightly
            test_r = r * 1.001
            valid = True
            for i in range(N):
                if cx[i] < test_r[i] or cx[i] + test_r[i] > W:
                    valid = False
                    break
                if cy[i] < test_r[i] or cy[i] + test_r[i] > H:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dx = cx[i] - cx[j]
                        dy = cy[i] - cy[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < test_r[i] + test_r[j] - 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                r = test_r
                # Re-relax positions
                cx, cy, r = _force_directed_step(cx, cy, r, W, H, dt=0.003, iterations=500)
            else:
                break
        
        # Compute score
        score = _validity_score(np.concatenate([cx, cy, r]), W, H)
        if score > 0 and score > best_score:
            best_score = score
            result = np.zeros((N, 3))
            result[:, 0] = cx
            result[:, 1] = cy
            result[:, 2] = r
            best_result = result
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
