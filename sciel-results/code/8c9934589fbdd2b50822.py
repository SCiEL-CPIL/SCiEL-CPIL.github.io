"""Circle packing n=32: repulsion forces + simulated annealing + local polish."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _repulsion_force(x, y, r, k=50.0):
    """Compute repulsion forces between overlapping circles and boundary violations."""
    fx = np.zeros(N)
    fy = np.zeros(N)
    fr = np.zeros(N)
    
    # Pairwise repulsion
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            r_sum = r[i] + r[j]
            if d < r_sum and d > 1e-12:
                overlap = r_sum - d
                force = k * overlap
                inv_d = 1.0 / d
                fx[i] += force * dx * inv_d
                fy[i] += force * dy * inv_d
                fx[j] -= force * dx * inv_d
                fy[j] -= force * dy * inv_d
                # Push radii apart
                fr[i] -= force * 0.5
                fr[j] -= force * 0.5
            elif d <= 1e-12:
                # Random small push
                angle = random.uniform(0, 2*np.pi)
                fx[i] += 0.1 * np.cos(angle)
                fy[i] += 0.1 * np.sin(angle)
                fx[j] -= 0.1 * np.cos(angle)
                fy[j] -= 0.1 * np.sin(angle)
    
    # Boundary repulsion
    for i in range(N):
        if x[i] < r[i]:
            ov = r[i] - x[i]
            fx[i] += k * ov
            fr[i] -= k * ov * 0.5
        if x[i] + r[i] > 1.0:
            ov = x[i] + r[i] - 1.0
            fx[i] -= k * ov
            fr[i] -= k * ov * 0.5
        if y[i] < r[i]:
            ov = r[i] - y[i]
            fy[i] += k * ov
            fr[i] -= k * ov * 0.5
        if y[i] + r[i] > 1.0:
            ov = y[i] + r[i] - 1.0
            fy[i] -= k * ov
            fr[i] -= k * ov * 0.5
    
    return fx, fy, fr

def _total_overlap(x, y, r):
    """Compute total overlap (should be 0 for valid config)."""
    total = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            r_sum = r[i] + r[j]
            if d < r_sum:
                total += (r_sum - d)
    for i in range(N):
        if x[i] < r[i]:
            total += (r[i] - x[i])
        if x[i] + r[i] > 1.0:
            total += (x[i] + r[i] - 1.0)
        if y[i] < r[i]:
            total += (r[i] - y[i])
        if y[i] + r[i] > 1.0:
            total += (y[i] + r[i] - 1.0)
    return total

def _is_valid(x, y, r, tol=1e-7):
    """Check validity."""
    for i in range(N):
        if x[i] < r[i] - tol or x[i] + r[i] > 1.0 + tol:
            return False
        if y[i] < r[i] - tol or y[i] + r[i] > 1.0 + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    return True

def _hex_grid_init(n, rng):
    """Generate hexagonal grid points, select n closest to center."""
    points = []
    for q in range(-10, 11):
        for r in range(-10, 11):
            x = 1.5 * q
            y = np.sqrt(3) * (r + 0.5 * (q & 1))
            points.append([x, y])
    points = np.array(points)
    center = np.array([0.5, 0.5])
    points += (center - np.mean(points, axis=0))
    dists = np.sum((points - center)**2, axis=1)
    idx = np.argsort(dists)
    selected = points[idx[:n]]
    # Estimate radii from local spacing
    radii = np.full(n, 0.08)
    return selected, radii

def _random_init(n, rng):
    """Random initialization with small radii."""
    points = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = rng.uniform(0.04, 0.08, size=n)
    return points, radii

def _anneal_and_repel(x0, y0, r0, start_time, deadline, seed=0):
    """Simulated annealing with repulsion forces to find valid packing."""
    rng = np.random.RandomState(seed)
    x = x0.copy().astype(float)
    y = y0.copy().astype(float)
    r = r0.copy().astype(float)
    
    best_sum = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    T = 0.05  # Initial temperature
    T_min = 0.001
    cooling = 0.995
    steps_per_temp = 50
    
    total_steps = 0
    max_steps = 50000
    
    while total_steps < max_steps and time.time() < deadline:
        for _ in range(steps_per_temp):
            if time.time() > deadline:
                break
            
            # Pick random circle
            i = rng.randint(0, N)
            
            # Propose perturbation
            dx = rng.normal(0, 0.02)
            dy = rng.normal(0, 0.02)
            dr = rng.normal(0, 0.01)
            
            new_xi = x[i] + dx
            new_yi = y[i] + dy
            new_ri = max(0.01, r[i] + dr)
            
            # Compute energy change (total overlap)
            old_overlap = _total_overlap(x, y, r)
            
            x[i], y[i], r[i] = new_xi, new_yi, new_ri
            new_overlap = _total_overlap(x, y, r)
            x[i], y[i], r[i] = x[i] - dx, y[i] - dy, r[i] - dr
            
            delta = new_overlap - old_overlap
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                x[i], y[i], r[i] = new_xi, new_yi, new_ri
                
                # Apply repulsion forces
                fx, fy, fr = _repulsion_force(x, y, r, k=10.0)
                x += fx * 0.001
                y += fy * 0.001
                r += fr * 0.001
                
                # Clamp
                x = np.clip(x, 0.01, 0.99)
                y = np.clip(y, 0.01, 0.99)
                r = np.maximum(r, 0.01)
                r = np.minimum(r, 0.5)
                
                # Track best
                current_sum = np.sum(r)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
            
            total_steps += 1
        
        T *= cooling
        if T < T_min:
            break
    
    return best_x, best_y, best_r, best_sum

def _local_polish(x, y, r, start_time, deadline):
    """Local search to maximize sum of radii while maintaining validity."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    best_sum = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    # Try to increase radii and adjust positions
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            # Try increasing radius slightly
            for dr in [0.005, 0.002, 0.001]:
                new_r = r[i] + dr
                if new_r > 0.5:
                    continue
                
                # Check if we can adjust position to maintain validity
                # Simple: try to push away from neighbors
                valid = True
                for j in range(N):
                    if j == i:
                        continue
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    if d < r[i] + r[j] + dr:
                        valid = False
                        break
                if not valid:
                    # Try to move circle i away
                    moved = False
                    for _ in range(20):
                        angle = random.uniform(0, 2*np.pi)
                        step = 0.005
                        nx = x[i] + step * np.cos(angle)
                        ny = y[i] + step * np.sin(angle)
                        
                        # Check boundaries
                        if nx < new_r or nx + new_r > 1.0:
                            continue
                        if ny < new_r or ny + new_r > 1.0:
                            continue
                        
                        # Check all pairs
                        ok = True
                        for j in range(N):
                            if j == i:
                                continue
                            d = np.sqrt((nx-x[j])**2 + (ny-y[j])**2)
                            if d < new_r + r[j]:
                                ok = False
                                break
                        if ok:
                            x[i] = nx
                            y[i] = ny
                            r[i] = new_r
                            best_sum = np.sum(r)
                            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
                            improved = True
                            moved = True
                            break
                    if moved:
                        break
        
        if not improved:
            break
    
    return best_x, best_y, best_r, best_sum

def _grid_init(n):
    """Simple grid initialization."""
    side = int(np.ceil(np.sqrt(n)))
    points = []
    radii = []
    for i in range(side):
        for j in range(side):
            if len(points) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            points.append([x, y])
            radii.append(0.5 / side - 0.01)
    points = np.array(points[:n])
    radii = np.array(radii[:n])
    return points, radii

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_x = None
    best_y = None
    best_r = None
    best_sum = 0.0
    
    rng = np.random.RandomState(42)
    
    # Multiple restarts with different initializations
    num_restarts = 12
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            points, radii = _hex_grid_init(N, rng)
        elif restart == 1:
            points, radii = _grid_init(N)
        elif restart < 5:
            points, radii = _random_init(N, rng)
        else:
            # Hex with jitter
            points, radii = _hex_grid_init(N, rng)
            points += rng.normal(0, 0.05, size=points.shape)
            points = np.clip(points, 0.05, 0.95)
        
        # Run annealing + repulsion
        x, y, r, s = _anneal_and_repel(points[:, 0], points[:, 1], radii, 
                                        start_time, deadline, seed=restart*7+1)
        
        if s > best_sum:
            best_sum = s
            best_x, best_y, best_r = x, y, r
    
    if best_x is None:
        # Fallback
        points, radii = _grid_init(N)
        return np.column_stack((points[:, 0], points[:, 1], radii))
    
    # Local polish
    if time.time() < deadline - 5:
        best_x, best_y, best_r, best_sum = _local_polish(best_x, best_y, best_r, 
                                                          start_time, deadline)
    
    # Final validation
    if not _is_valid(best_x, best_y, best_r, tol=1e-6):
        # Try to fix by reducing radii
        scale = 0.9
        for _ in range(10):
            r_test = best_r * scale
            if _is_valid(best_x, best_y, r_test, tol=1e-6):
                best_r = r_test
                break
            scale *= 0.95
        else:
            # Fallback
            points, radii = _grid_init(N)
            return np.column_stack((points[:, 0], points[:, 1], radii))
    
    result = np.column_stack((best_x, best_y, best_r))
    return result

# EVOLVE-BLOCK-END
