"""Circle packing in unit square n=32.
Uses phyllotaxis initialization + coordinate descent with radius adaptation.
"""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time

def _phyllotaxis_init(n, scale=0.95):
    """Generate n points in [0,1]^2 using phyllotaxis (sunflower) pattern."""
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(n)
    r = np.sqrt((indices + 0.5) / n) * scale
    theta = golden_angle * indices
    x = 0.5 + r * np.cos(theta)
    y = 0.5 + r * np.sin(theta)
    # Clip to [0, 1]
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    # Initial small radii
    radii = np.full(n, 0.01)
    return x, y, radii


def _grid_init(n):
    """Grid initialization as fallback."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
    x = np.array([c[0] for c in cells])
    y = np.array([c[1] for c in cells])
    radii = np.array([c[2] for c in cells])
    return x, y, radii


def _validity_check(x, y, r):
    """Check if circles are valid (inside unit square, disjoint interiors)."""
    n = len(r)
    # Check bounds
    if np.any(x < -1e-9) or np.any(x > 1 + 1e-9):
        return False
    if np.any(y < -1e-9) or np.any(y > 1 + 1e-9):
        return False
    if np.any(r < -1e-9):
        return False
    # Check each circle is inside the square
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
        return False
    if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return False
    # Check pairwise disjointness
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return False
    return True


def _sum_radii(r):
    return float(np.sum(r))


def _max_feasible_radius(x, y, n):
    """For given positions, compute max radius for each circle such that all constraints hold."""
    r = np.zeros(n)
    for i in range(n):
        # Distance to walls
        wall_dist = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
        r[i] = max(wall_dist, 0.0)
    # Iteratively enforce pairwise constraints
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                # Constraint: r[i] + r[j] <= dist
                # Maximize r[i] given r[j]
                max_ri = dist - r[j]
                if max_ri < r[i] - 1e-12:
                    r[i] = max(max_ri, 0.0)
                    changed = True
                max_rj = dist - r[i]
                if max_rj < r[j] - 1e-12:
                    r[j] = max(max_rj, 0.0)
                    changed = True
        if not changed:
            break
    return r


def _coordinate_descent(x, y, r, max_iters=50000, tol=1e-10, time_limit=110.0):
    """Coordinate descent to maximize sum of radii.
    
    Strategy: iterate over circles, for each circle try to increase its radius
    by moving it optimally, or adjust position to allow larger radius.
    """
    n = len(r)
    best_sum = _sum_radii(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        for i in range(n):
            if time.time() - start_time > time_limit:
                    break
            
            # Try to increase radius of circle i
            # First, find max radius at current position
            max_r_at_pos = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            for j in range(n):
                if j == i:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r_at_pos = min(max_r_at_pos, dist - r[j])
            max_r_at_pos = max(max_r_at_pos, 0.0)
            
            if max_r_at_pos <= r[i] + 1e-12:
                # Can't increase at current position, try to move
                # Gradient-like: move towards center of available space
                # Compute repulsive forces from other circles
                fx = 0.0
                fy = 0.0
                for j in range(n):
                    if j == i:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy) + 1e-15
                    if dist < r[i] + r[j] + 0.05:
                        # Push away
                        force = (r[i] + r[j] + 0.05 - dist) / dist
                        fx += force * dx
                        fy += force * dy
                
                # Also push away from walls
                if x[i] < r[i] + 0.05:
                    fx += (r[i] + 0.05 - x[i])
                if x[i] > 1.0 - r[i] - 0.05:
                    fx -= (x[i] - (1.0 - r[i] - 0.05))
                if y[i] < r[i] + 0.05:
                    fy += (r[i] + 0.05 - y[i])
                if y[i] > 1.0 - r[i] - 0.05:
                    fy -= (y[i] - (1.0 - r[i] - 0.05))
                
                # Move a small step
                step = 0.02
                new_xi = x[i] + step * fx
                new_yi = y[i] + step * fy
                
                # Clamp to valid region
                new_xi = np.clip(new_xi, r[i], 1.0 - r[i])
                new_yi = np.clip(new_yi, r[i], 1.0 - r[i])
                
                # Check if this move allows a larger radius
                old_xi, old_yi = x[i], y[i]
                x[i], y[i] = new_xi, new_yi
                
                # Recompute max radius
                new_max_r = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
                for j in range(n):
                    if j == i:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    new_max_r = min(new_max_r, dist - r[j])
                new_max_r = max(new_max_r, 0.0)
                
                if new_max_r > r[i] + 1e-10:
                    r[i] = new_max_r
                    improved = True
                else:
                    # Revert
                    x[i], y[i] = old_xi, old_yi
            
            # Try increasing radius directly if feasible
            new_r = r[i] + 0.005
            # Check if new radius is feasible at current position
            feasible = True
            if x[i] - new_r < 0 or x[i] + new_r > 1 or y[i] - new_r < 0 or y[i] + new_r > 1:
                feasible = False
            if feasible:
                for j in range(n):
                    if j == i:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < new_r + r[j] - 1e-10:
                        feasible = False
                        break
            if feasible:
                r[i] = new_r
                improved = True
            else:
                # Try to find the exact max radius
                max_r = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
                for j in range(n):
                    if j == i:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    max_r = min(max_r, dist - r[j])
                max_r = max(max_r, 0.0)
                if max_r > r[i] + 1e-10:
                    r[i] = max_r
                    improved = True
        
        current_sum = _sum_radii(r)
        if current_sum > best_sum:
            best_sum = current_sum
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        if not improved and it > 100:
            # Try a different strategy: random perturbation
            if np.random.random() < 0.1:
                idx = np.random.randint(n)
                x[idx] += np.random.randn() * 0.01
                y[idx] += np.random.randn() * 0.01
                x[idx] = np.clip(x[idx], 0.0, 1.0)
                y[idx] = np.clip(y[idx], 0.0, 1.0)
                # Recompute radius
                max_r = min(x[idx], 1.0 - x[idx], y[idx], 1.0 - y[idx])
                for j in range(n):
                    if j == idx:
                        continue
                    dx = x[idx] - x[j]
                    dy = y[idx] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    max_r = min(max_r, dist - r[j])
                r[idx] = max(max_r, 0.0)
    
    return best_x, best_y, best_r, best_sum


def _local_search(x, y, r, time_limit=100.0):
    """Multi-start local search with coordinate descent."""
    n = len(r)
    best_sum = -np.inf
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    start_time = time.time()
    
    # Multiple restarts with different initializations
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        if restart == 0:
            # Use provided initial positions
            xi, yi, ri = x.copy(), y.copy(), r.copy()
        elif restart == 1:
            # Phyllotaxis
            xi, yi, ri = _phyllotaxis_init(n)
        elif restart == 2:
            # Grid
            xi, yi, ri = _grid_init(n)
        elif restart == 3:
            # Phyllotaxis with different scale
            xi, yi, ri = _phyllotaxis_init(n, scale=0.85)
        else:
            # Random
            xi = np.random.uniform(0.1, 0.9, n)
            yi = np.random.uniform(0.1, 0.9, n)
            ri = np.full(n, 0.01)
        
        # Run coordinate descent
        time_left = time_limit - (time.time() - start_time)
        if time_left < 5:
            break
        
        cx, cy, cr, csum = _coordinate_descent(xi, yi, ri, max_iters=20000, time_limit=time_left - 2)
        
        if csum > best_sum:
            best_sum = csum
            best_x, best_y, best_r = cx, cy, cr
    
    return best_x, best_y, best_r, best_sum


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Initial phyllotaxis
    x, y, r = _phyllotaxis_init(N)
    
    # Run local search
    x, y, r, total = _local_search(x, y, r, time_limit=110.0)
    
    # Final validity check and fix
    if time.time() < deadline - 2:
        # Ensure all circles are valid
        for i in range(N):
            # Clamp radius to fit in square
            max_r = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            r[i] = min(r[i], max_r)
        
        # Enforce pairwise constraints
        for _ in range(50):
            changed = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < r[i] + r[j]:
                        # Reduce the larger radius
                        excess = (r[i] + r[j] - dist) / 2.0
                        if r[i] >= r[j]:
                            r[i] = max(r[i] - excess, 0.0)
                        else:
                            r[j] = max(r[j] - excess, 0.0)
                        changed = True
            if not changed:
                break
    
    result = np.column_stack([x, y, r])
    return result
# EVOLVE-BLOCK-END
