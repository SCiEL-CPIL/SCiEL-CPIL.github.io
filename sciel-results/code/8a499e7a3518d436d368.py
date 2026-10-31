import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0.0
    best_circles = None
    
    def is_valid(circles, W, H):
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Check boundaries
        if np.any(r > x + 1e-6) or np.any(r > W - x + 1e-6) or np.any(r > y + 1e-6) or np.any(r > H - y + 1e-6):
            return False
        
        # Check overlaps
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        d2 = dx**2 + dy**2
        d = np.sqrt(d2)
        
        req_d = r[:, np.newaxis] + r[np.newaxis, :]
        np.fill_diagonal(req_d, 0)
        np.fill_diagonal(d, np.inf)
        
        # Allow tiny numerical error
        if np.any(req_d > d + 1e-6):
            return False
            
        return True

    def shrink_to_valid(circles, W, H):
        for _ in range(200):
            if is_valid(circles, W, H):
                return circles
            circles = circles.copy()
            circles[:, 2] *= 0.995
            if np.all(circles[:, 2] < 1e-5):
                break
        return circles

    def compute_violations(circles, W, H):
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        d2 = dx**2 + dy**2
        d = np.sqrt(d2)
        
        req_d = r[:, np.newaxis] + r[np.newaxis, :]
        np.fill_diagonal(req_d, 0)
        np.fill_diagonal(d, np.inf)
        
        overlap_viol = np.maximum(0, req_d - d)
        
        v_left = np.maximum(0, r - x)
        v_right = np.maximum(0, r - (W - x))
        v_bottom = np.maximum(0, r - y)
        v_top = np.maximum(0, r - (H - y))
        
        boundary_viol = np.concatenate([v_left, v_right, v_bottom, v_top])
        
        return overlap_viol, boundary_viol

    def relax(circles, W, H, max_iters=2000, tol=1e-9):
        """Physical relaxation: iteratively adjust positions and radii to minimize violations."""
        circles = circles.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            x = circles[:, 0]
            y = circles[:, 1]
            r = circles[:, 2]
            
            # Compute forces from overlaps
            dx = x[:, np.newaxis] - x[np.newaxis, :]
            dy = y[:, np.newaxis] - y[np.newaxis, :]
            d2 = dx**2 + dy**2
            d = np.sqrt(d2)
            
            req_d = r[:, np.newaxis] + r[np.newaxis, :]
            np.fill_diagonal(req_d, 0)
            np.fill_diagonal(d, 1e-10)
            
            # Overlap violation per pair
            viol = np.maximum(0, req_d - d)
            
            # Forces: push apart overlapping circles
            fx = np.zeros(N)
            fy = np.zeros(N)
            fr = np.zeros(N)
            
            for i in range(N):
                for j in range(i+1, N):
                    v = viol[i, j]
                    if v > 0:
                        dist = d[i, j]
                        if dist < 1e-10:
                            dist = 1e-10
                            # Random direction if exactly coincident
                            fx[i] += np.random.randn() * v * 0.5
                            fy[i] += np.random.randn() * v * 0.5
                            fx[j] -= np.random.randn() * v * 0.5
                            fy[j] -= np.random.randn() * v * 0.5
                        else:
                            ux = dx[i, j] / dist
                            uy = dy[i, j] / dist
                            # Each circle moves half the violation
                            fx[i] += ux * v * 0.5
                            fy[i] += uy * v * 0.5
                            fx[j] -= ux * v * 0.5
                            fy[j] -= uy * v * 0.5
                            # Shrink radii slightly to help resolve
                            fr[i] -= v * 0.1
                            fr[j] -= v * 0.1
            
            # Boundary forces
            for i in range(N):
                # Left
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 0.5
                    fr[i] -= (r[i] - x[i]) * 0.1
                # Right
                if x[i] > W - r[i]:
                    fx[i] -= (x[i] - (W - r[i])) * 0.5
                    fr[i] -= (x[i] - (W - r[i])) * 0.1
                # Bottom
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 0.5
                    fr[i] -= (r[i] - y[i]) * 0.1
                # Top
                if y[i] > H - r[i]:
                    fy[i] -= (y[i] - (H - r[i])) * 0.5
                    fr[i] -= (y[i] - (H - r[i])) * 0.1
            
            # Apply updates with damping
            damping = 0.3
            x += damping * fx
            y += damping * fy
            r += damping * fr
            
            # Clamp to bounds
            r = np.maximum(r, 1e-6)
            x = np.clip(x, r, W - r)
            y = np.clip(y, r, H - r)
            
            circles[:, 0] = x
            circles[:, 1] = y
            circles[:, 2] = r
            
            # Check convergence
            max_viol = np.max(viol)
            if max_viol < tol:
                break
        
        return circles

    def expand_radii(circles, W, H, max_iters=500):
        """Try to expand radii while maintaining validity."""
        circles = circles.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            x = circles[:, 0]
            y = circles[:, 1]
            r = circles[:, 2]
            
            # Find the most constrained circle (least room to grow)
            # Try expanding each circle slightly
            best_idx = -1
            best_gain = 0
            
            for i in range(N):
                # Max possible radius given boundaries
                max_r_bound = min(x[i], W - x[i], y[i], H - y[i])
                
                # Check against other circles
                max_r_other = max_r_bound
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx**2 + dy**2)
                    # r_i + r_j <= dist => r_i <= dist - r_j
                    max_r_other = min(max_r_other, dist - r[j])
                
                if max_r_other > r[i] + 1e-8:
                    gain = max_r_other - r[i]
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
            
            if best_idx == -1 or best_gain < 1e-8:
                break
            
            # Expand the most constrained circle
            circles[best_idx, 2] += best_gain * 0.5
            
            # Re-relax to fix any new overlaps
            circles = relax(circles, W, H, max_iters=200)
            
            if not is_valid(circles, W, H):
                circles[best_idx, 2] -= best_gain * 0.5
                break
        
        return circles

    # Hexagonal lattice initialization
    def hex_init(W, H, n_cols, n_rows):
        """Initialize circles in a hexagonal pattern."""
        circles = np.zeros((N, 3))
        
        # Hexagonal spacing
        dx = W / (n_cols + 0.5)
        dy = H / (n_rows + 0.5)
        hex_dy = dy * np.sqrt(3) / 2
        
        idx = 0
        for row in range(n_rows):
            for col in range(n_cols):
                if idx >= N:
                    break
                x = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) * dx
                y = (row + 0.5) * hex_dy
                r = min(dx, hex_dy) * 0.45
                r = min(r, x, W - x, y, H - y)
                circles[idx, 0] = x
                circles[idx, 1] = y
                circles[idx, 2] = max(r, 0.01)
                idx += 1
        
        # Fill remaining circles if needed
        while idx < N:
            # Place in available space
            x = np.random.uniform(0.1 * W, 0.9 * W)
            y = np.random.uniform(0.1 * H, 0.9 * H)
            r = 0.05
            circles[idx, 0] = x
            circles[idx, 1] = y
            circles[idx, 2] = r
            idx += 1
        
        return circles

    # Aspect ratios to try
    aspect_ratios = [1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15, 0.8, 1.2]
    
    for W in aspect_ratios:
        if time.time() > deadline:
            break
        H = 2.0 - W
        
        # Multiple initialization strategies
        for restart in range(6):
            if time.time() > deadline:
                break
            
            np.random.seed(1234 + restart * 7)
            
            if restart < 3:
                # Hexagonal lattice with different densities
                n_cols = 4 + (restart % 2)
                n_rows = 5 - (restart % 2)
                circles = hex_init(W, H, n_cols, n_rows)
            else:
                # Random initialization
                circles = np.zeros((N, 3))
                for i in range(N):
                    circles[i, 0] = np.random.uniform(0.1 * W, 0.9 * W)
                    circles[i, 1] = np.random.uniform(0.1 * H, 0.9 * H)
                    circles[i, 2] = 0.05 + np.random.uniform(0, 0.05)
            
            # Relax to remove overlaps
            circles = relax(circles, W, H, max_iters=3000)
            
            # Expand radii
            circles = expand_radii(circles, W, H, max_iters=300)
            
            # Final relaxation
            circles = relax(circles, W, H, max_iters=1000)
            
            # Validate and update best
            if is_valid(circles, W, H):
                score = np.sum(circles[:, 2])
                if score > best_score:
                    best_score = score
                    best_circles = circles.copy()
            else:
                # Try to shrink to valid
                circles_valid = shrink_to_valid(circles, W, H)
                if is_valid(circles_valid, W, H):
                    score = np.sum(circles_valid[:, 2])
                    if score > best_score:
                        best_score = score
                        best_circles = circles_valid.copy()
    
    # If no valid circles found, return a trivial valid solution
    if best_circles is None:
        # Simple grid solution
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    r = 0.5 / side - 0.02
                    cells.append([
                        (i + 0.5) / side,
                        (j + 0.5) / side,
                        r
                    ])
        best_circles = np.array(cells[:N])
    
    return best_circles
