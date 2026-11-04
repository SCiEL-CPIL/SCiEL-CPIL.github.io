import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 140  # seconds, safety margin

def construct():
    start_time = time.time()
    
    best_result = None
    best_score = -1
    
    # Try different grid configurations
    configs = [(7, 3), (3, 7), (5, 5), (6, 4), (4, 6)]
    
    for cols, rows in configs:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Aspect ratio matching grid
        W = 2 * cols / (cols + rows)
        H = 2 * rows / (cols + rows)
        
        # Grid positions
        positions = []
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                x = (i + 0.5) * W / cols
                y = (j + 0.5) * H / rows
                positions.append([x, y])
                count += 1
            if count >= N:
                break
        
        positions = np.array(positions)
        
        # Initial radii: slightly less than grid spacing
        r_init = np.full(N, 0.5 * min(W / cols, H / rows) * 0.85)
        
        result = _optimize(positions, r_init, W, H, start_time, DEADLINE)
        
        if result is not None:
            score = np.sum(result[:, 2])
            if score > best_score:
                best_score = score
                best_result = result
    
    if best_result is None:
        # Fallback: 5x5 grid
        W, H = 1.0, 1.0
        positions = []
        for i in range(5):
            for j in range(5):
                positions.append([(i+0.5)*0.2, (j+0.5)*0.2])
        positions = np.array(positions[:N])
        r = np.full(N, 0.099)
        best_result = np.column_stack([positions[:,0], positions[:,1], r])
    
    return best_result

def _optimize(init_pos, init_r, W0, H0, start_time, deadline):
    # Variables: x0,y0,r0, ..., x20,y20,r20, W, H (65 vars)
    
    def objective(v):
        pos = v[:2*N].reshape(N, 2)
        r = v[2*N:3*N]
        W = v[3*N]
        H = v[3*N+1]
        
        penalty = 0.0
        
        # Perimeter: W + H = 2
        penalty += 1000 * (W + H - 2)**2
        
        # Wall constraints (smooth penalty)
        for i in range(N):
            x, y = pos[i]
            ri = r[i]
            # x >= ri
            if x < ri:
                penalty += 1000 * (ri - x)**2
            # x <= W - ri
            if x > W - ri:
                penalty += 1000 * (x - (W - ri))**2
            # y >= ri
            if y < ri:
                penalty += 1000 * (ri - y)**2
            # y <= H - ri
            if y > H - ri:
                penalty += 1000 * (y - (H - ri))**2
            # ri >= 0
            if ri < 0:
                penalty += 1000 * ri**2
        
        # Non-overlap (smooth penalty)
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                min_d = r[i] + r[j]
                if dist < min_d:
                    penalty += 1000 * (min_d - dist)**2
        
        return -np.sum(r) + penalty
    
    v0 = np.concatenate([init_pos.ravel(), init_r, [W0, H0]])
    
    bounds = []
    for _ in range(N):
        bounds.append((0.001, 1.5))  # x
        bounds.append((0.001, 1.5))  # y
        bounds.append((0.001, 0.5))  # r
    bounds.append((0.1, 1.9))  # W
    bounds.append((0.1, 1.9))  # H
    
    try:
        res = minimize(objective, v0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        if time.time() - start_time > deadline - 10:
            return None
        
        v = res.x
        pos = v[:2*N].reshape(N, 2)
        r = v[2*N:3*N]
        W = v[3*N]
        H = v[3*N+1]
        
        # Polish: clip and validate
        r = np.maximum(r, 0.001)
        
        if not _is_valid(pos, r, W, H):
            # Try to repair by shrinking radii
            for scale in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]:
                if _is_valid(pos, r * scale, W, H):
                    r = r * scale
                    break
            else:
                return None
        
        return np.column_stack([pos[:,0], pos[:,1], r])
    
    except Exception:
        return None

def _is_valid(pos, r, W, H, tol=1e-5):
    if abs(W + H - 2) > 0.01:
        return False
    if W < 0.01 or H < 0.01:
        return False
    for i in range(N):
        x, y = pos[i]
        ri = r[i]
        if ri < -tol:
            return False
        if x < ri - tol or x > W - ri + tol:
            return False
        if y < ri - tol or y > H - ri + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i,0] - pos[j,0]
            dy = pos[i,1] - pos[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True
