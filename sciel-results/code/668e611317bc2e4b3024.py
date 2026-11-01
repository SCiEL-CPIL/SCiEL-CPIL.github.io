"""
Circle packing in perimeter-4 rectangle: 21 circles.
Structural change: direct force-directed relaxation with aspect ratio optimization.
Instead of grid-based cellular decomposition, we directly optimize circle positions and radii
using a physics-inspired approach with:
1. Random initialization with multiple restarts
2. Force-directed relaxation to resolve overlaps
3. Iterative radius growth
4. Aspect ratio optimization via 1D search
This allows circles to move freely and potentially find denser packings than grid confinement.
"""
import numpy as np
import time
import random

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def check_validity(x, y, r, w, tol=1e-8):
        h = 2.0 - w
        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        if np.any(r <= 0):
            return False, 0.0
        
        # Boundary checks
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
            
        # Overlap checks
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum * r_sum - tol:
                    return False, 0.0
        return True, float(np.sum(r))
    
    def relax(x, y, r, w, h, max_iter=500, dt=0.01):
        """
        Force-directed relaxation to resolve overlaps and boundary violations.
        Returns updated x, y, r arrays.
        """
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
                
            # Compute forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Pairwise repulsion for overlapping circles
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d2 = dx * dx + dy * dy
                    d = np.sqrt(d2) if d2 > 1e-12 else 1e-12
                    r_sum = r[i] + r[j]
                    
                    if d < r_sum:
                        # Repulsive force proportional to overlap
                        overlap = r_sum - d
                        force = 10.0 * overlap
                        if d > 1e-10:
                            fx[i] += force * dx / d
                            fy[i] += force * dy / d
                            fx[j] -= force * dx / d
                            fy[j] -= force * dy / d
            
            # Boundary forces
            for i in range(N):
                # Left wall
                if x[i] < r[i]:
                    fx[i] += 50.0 * (r[i] - x[i])
                # Right wall
                if x[i] > w - r[i]:
                    fx[i] -= 50.0 * (x[i] - (w - r[i]))
                # Bottom wall
                if y[i] < r[i]:
                    fy[i] += 50.0 * (r[i] - y[i])
                # Top wall
                if y[i] > h - r[i]:
                    fy[i] -= 50.0 * (y[i] - (h - r[i]))
            
            # Update positions
            x += dt * fx
            y += dt * fy
            
            # Project back to valid region
            for i in range(N):
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])
        
        return x, y
    
    def try_grow_radii(x, y, r, w, h, max_steps=100):
        """
        Iteratively try to grow all radii by a small factor.
        Returns updated r if successful, else None.
        """
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
                
            # Try to grow by 1%
            new_r = r * 1.01
            
            # Relax positions with new radii
            x_new, y_new = relax(x.copy(), y.copy(), new_r, w, h, max_iter=100)
            
            # Check validity
            valid, _ = check_validity(x_new, y_new, new_r, w)
            if valid:
                x, y, r = x_new, y_new, new_r
            else:
                break
        
        return x, y, r
    
    def single_trial(w, seed, max_iter=200):
        """
        Single trial: initialize, relax, grow radii.
        Returns (x, y, r, value) or None.
        """
        rng = np.random.RandomState(seed)
        h = 2.0 - w
        
        # Initialize positions in a rough grid pattern with some randomness
        cols = int(np.ceil(np.sqrt(N * w / h))) if h > 0.1 else N
        rows = int(np.ceil(N / cols))
        
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        cell_w = w / cols
        cell_h = h / rows
        
        for i in range(N):
            row, col = divmod(i, cols)
            cx = (col + 0.5) * cell_w + rng.uniform(-0.2, 0.2) * cell_w
            cy = (row + 0.5) * cell_h + rng.uniform(-0.2, 0.2) * cell_h
            x[i] = np.clip(cx, 0, w)
            y[i] = np.clip(cy, 0, h)
            # Initial radius: small fraction of cell size
            r[i] = min(cell_w, cell_h) * 0.25
        
        # Relax to resolve initial overlaps
        x, y = relax(x, y, r, w, h, max_iter=200)
        
        # Check validity
        valid, val = check_validity(x, y, r, w)
        if not valid:
            return None
        
        # Grow radii
        x, y, r = try_grow_radii(x, y, r, w, h)
        
        # Final validity check
        valid, val = check_validity(x, y, r, w)
        if not valid:
            return None
        
        return x, y, r, val
    
    best_value = 0.0
    best_result = None
    
    # Try different aspect ratios
    # Perimeter = 2(w + h) = 4, so w + h = 2
    # w ranges from ~0.3 to ~1.7 (symmetric, so we can restrict to w <= 1.0)
    aspect_ratios = np.linspace(0.3, 1.0, 15)
    
    # Multiple restarts for each aspect ratio
    num_restarts = 3
    
    for w0 in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            
            seed = int(time.time() * 1000) + restart * 100 + int(w0 * 1000)
            result = single_trial(w0, seed)
            
            if result is not None:
                x, y, r, val = result
                if val > best_value:
                    best_value = val
                    best_result = (x, y, r, w0)
    
    # If we have a result, return it
    if best_result is not None:
        x, y, r, w = best_result
        # Shrink slightly to ensure strict inequality
        r_final = r * (1 - 1e-9)
        return np.column_stack((x, y, r_final))
    
    # Fallback: Simple grid in unit square
    cols, rows = 3, 7
    w = 1.0
    h = 1.0
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    for i in range(N):
        row, col = divmod(i, cols)
        cx = (col + 0.5) * (w / cols)
        cy = (row + 0.5) * (h / rows)
        x[i] = cx
        y[i] = cy
        cell_w = w / cols
        cell_h = h / rows
        r[i] = min(cell_w, cell_h) * 0.5  # Inradius of uniform cell
    
    return np.column_stack((x, y, r * (1 - 1e-9)))

if __name__ == "__main__":
    res = construct()
    print(res.shape)
