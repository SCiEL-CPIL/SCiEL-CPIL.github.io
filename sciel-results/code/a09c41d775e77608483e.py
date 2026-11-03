import numpy as np
import time

N = 26

def _max_radius_for_circle(i, centers, radii):
    """Compute the maximum radius circle i can have given other circles' positions."""
    x, y = centers[i]
    max_r = min(x, 1 - x, y, 1 - y)
    for j in range(N):
        if j == i:
            continue
        dx = x - centers[j][0]
        dy = y - centers[j][1]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist > 0:
            max_r = min(max_r, dist - radii[j])
    return max(0.0, max_r)

def _smooth_objective(centers, radii, beta=50.0):
    """
    Smooth surrogate: maximize sum(radii) - penalty for violations.
    Uses soft-min via log-sum-exp for the nonsmooth min in radius constraints.
    """
    n = len(centers)
    # Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    # For each circle, the effective max radius from boundaries:
    bound_r = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                         np.minimum(centers[:, 1], 1 - centers[:, 1]))
    
    # Pairwise constraints: r_i + r_j <= dist(i,j)
    # We want to penalize when r_i + r_j > dist(i,j)
    total_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            violation = radii[i] + radii[j] - dist
            # Soft penalty
            if violation > 0:
                total_penalty += violation * violation
            else:
                # Smooth transition
                total_penalty += np.logaddexp(0, beta * violation) / beta - 1/beta
    
    # Boundary penalty
    for i in range(n):
        for bound_val in [centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1]]:
            violation = radii[i] - bound_val
            if violation > 0:
                total_penalty += violation * violation
            else:
                total_penalty += np.logaddexp(0, beta * violation) / beta - 1/beta
    
    return np.sum(radii) - 10.0 * total_penalty

def _gradient_polish(centers, radii, max_iters=200, step_size=0.005, deadline=None):
    """
    Gradient-based polish using finite differences on the smooth objective.
    """
    centers = centers.copy()
    radii = radii.copy()
    current_score = _smooth_objective(centers, radii)
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            # Try small perturbations in x, y, and radius
            best_delta = 0.0
            best_dx, best_dy, best_dr = 0.0, 0.0, 0.0
            
            # Sample perturbations
            for dx in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                for dy in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                    for dr in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                        new_x = centers[i, 0] + dx
                        new_y = centers[i, 1] + dy
                        new_r = radii[i] + dr
                        
                        if new_x < 0 or new_x > 1 or new_y < 0 or new_y > 1 or new_r < 0:
                            continue
                        
                        centers[i, 0] = new_x
                        centers[i, 1] = new_y
                        radii[i] = new_r
                        new_score = _smooth_objective(centers, radii)
                        
                        if new_score > current_score + best_delta + 1e-12:
                            best_delta = new_score - current_score
                            best_dx, best_dy, best_dr = dx, dy, dr
            
            # Restore
            centers[i, 0] -= best_dx
            centers[i, 1] -= best_dy
            radii[i] -= best_dr
            
            if best_delta > 1e-12:
                centers[i, 0] += best_dx
                centers[i, 1] += best_dy
                radii[i] += best_dr
                current_score += best_delta
                improved = True
        
        if not improved:
            break
    
    return centers, radii, np.sum(radii)

def _high_precision_polish(centers, radii, deadline=None):
    """
    Final high-precision polish: iteratively adjust each circle to its optimal
    position/radius given others fixed, using a fine grid around current position.
    """
    centers = centers.copy()
    radii = radii.copy()
    current_score = np.sum(radii)
    
    for iteration in range(50):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            best_pos = (centers[i, 0], centers[i, 1])
            best_r = radii[i]
            
            # Fine grid around current position
            base_r = radii[i]
            for dx in np.linspace(-0.02, 0.02, 9):
                for dy in np.linspace(-0.02, 0.02, 9):
                    cx = centers[i, 0] + dx
                    cy = centers[i, 1] + dy
                    if cx < 0 or cx > 1 or cy < 0 or cy > 1:
                        continue
                    
                    max_r = min(cx, 1 - cx, cy, 1 - cy)
                    for j in range(N):
                        if j == i:
                            continue
                        ddx = cx - centers[j, 0]
                        ddy = cy - centers[j, 1]
                        dist = np.sqrt(ddx * ddx + ddy * ddy)
                        if dist > 0:
                            max_r = min(max_r, dist - radii[j])
                    max_r = max(0.0, max_r)
                    
                    if max_r > best_r + 1e-12:
                        best_r = max_r
                        best_pos = (cx, cy)
            
            if best_pos[0] != centers[i, 0] or best_pos[1] != centers[i, 1] or best_r > radii[i] + 1e-12:
                centers[i, 0] = best_pos[0]
                centers[i, 1] = best_pos[1]
                radii[i] = best_r
                improved = True
        
        if not improved:
            break
    
    return centers, radii, np.sum(radii)

def _eval_packing(centers, radii):
    """Evaluate a packing: return (sum_of_radii, is_valid)."""
    sum_r = np.sum(radii)
    
    # Check boundaries
    for i in range(N):
        if centers[i][0] < -1e-9 or centers[i][0] > 1 + 1e-9:
            return sum_r, False
        if centers[i][1] < -1e-9 or centers[i][1] > 1 + 1e-9:
            return sum_r, False
        if radii[i] < -1e-9:
            return sum_r, False
        if radii[i] > centers[i][0] + 1e-9:
            return sum_r, False
        if radii[i] > 1 - centers[i][0] + 1e-9:
            return sum_r, False
        if radii[i] > centers[i][1] + 1e-9:
            return sum_r, False
        if radii[i] > 1 - centers[i][1] + 1e-9:
            return sum_r, False
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - 1e-9:
                return sum_r, False
    
    return sum_r, True

def _best_fit_local_search(centers, radii, max_iters=2000, deadline=None):
    """Iteratively move each circle to maximize its radius given others fixed.
    Uses two-scale approach: coarse random search + fine grid refinement."""
    centers = centers.copy()
    radii = radii.copy()
    current_score = np.sum(radii)
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            # Try to maximize radius of circle i by moving it
            best_pos = (centers[i][0], centers[i][1])
            best_r = radii[i]
            
            # Coarse: random perturbations
            for _ in range(15):
                angle = np.random.uniform(0, 2 * np.pi)
                dist = np.random.uniform(0, 0.08)
                nx = centers[i][0] + dist * np.cos(angle)
                ny = centers[i][1] + dist * np.sin(angle)
                if 0 <= nx <= 1 and 0 <= ny <= 1:
                    max_r = min(nx, 1 - nx, ny, 1 - ny)
                    for j in range(N):
                        if j == i:
                            continue
                        dx = nx - centers[j][0]
                        dy = ny - centers[j][1]
                        d = np.sqrt(dx * dx + dy * dy)
                        if d > 0:
                            max_r = min(max_r, d - radii[j])
                    max_r = max(0.0, max_r)
                    
                    if max_r > best_r + 1e-10:
                        best_r = max_r
                        best_pos = (nx, ny)
                        improved = True
            
            # Fine: grid refinement around best position found
            if improved:
                fine_best_r = best_r
                fine_best_pos = best_pos
                for fdx in np.linspace(-0.015, 0.015, 7):
                    for fdy in np.linspace(-0.015, 0.015, 7):
                        fx = best_pos[0] + fdx
                        fy = best_pos[1] + fdy
                        if fx < 0 or fx > 1 or fy < 0 or fy > 1:
                            continue
                        max_r = min(fx, 1 - fx, fy, 1 - fy)
                        for j in range(N):
                            if j == i:
                                continue
                            dx = fx - centers[j][0]
                            dy = fy - centers[j][1]
                            d = np.sqrt(dx * dx + dy * dy)
                            if d > 0:
                                max_r = min(max_r, d - radii[j])
                        max_r = max(0.0, max_r)
                        
                        if max_r > fine_best_r + 1e-12:
                            fine_best_r = max_r
                            fine_best_pos = (fx, fy)
                
                best_r = fine_best_r
                best_pos = fine_best_pos
            
            # Update if improved
            if best_r > radii[i] + 1e-10 or best_pos[0] != centers[i][0] or best_pos[1] != centers[i][1]:
                centers[i][0] = best_pos[0]
                centers[i][1] = best_pos[1]
                radii[i] = best_r
    
    return centers, radii, np.sum(radii)

def _grid_init():
    """Create initial grid packing."""
    # Use a 5x5 grid + 1 extra
    centers = []
    radii = []
    for i in range(5):
        for j in range(5):
            centers.append(((i + 0.5) / 5.0, (j + 0.5) / 5.0))
            radii.append(0.09)
    # Add one more
    centers.append((0.5, 0.5))
    radii.append(0.09)
    
    return np.array(centers), np.array(radii)

def _random_init(seed=42):
    """Create random initial packing."""
    np.random.seed(seed)
    centers = np.random.rand(N, 2)
    radii = np.full(N, 0.05)
    return centers, radii

def _hex_init():
    """Create hexagonal-like initial packing."""
    centers = []
    radii = []
    # Hexagonal packing
    r0 = 0.09
    dy = r0 * 2 * 0.866  # sqrt(3)/2
    for row in range(6):
        x_offset = 0 if row % 2 == 0 else r0
        for col in range(5):
            x = r0 + col * 2 * r0 + x_offset
            y = r0 + row * dy
            if x < 1 and y < 1:
                centers.append((x, y))
                radii.append(r0)
            if len(centers) >= N:
                break
        if len(centers) >= N:
            break
    
    # If we don't have enough, add random
    while len(centers) < N:
        centers.append((np.random.rand(), np.random.rand()))
        radii.append(0.05)
    
    return np.array(centers[:N]), np.array(radii[:N])

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = -np.inf
    best_centers = None
    best_radii = None
    
    # Generate multiple initial configurations
    inits = []
    
    # Grid init
    c, r = _grid_init()
    inits.append((c, r))
    
    # Random inits
    for seed in [42, 123, 456, 789, 1011, 2020]:
        c, r = _random_init(seed)
        inits.append((c, r))
    
    # Hex init
    c, r = _hex_init()
    inits.append((c, r))
    
    # Run best-fit local search on each
    for centers, radii in inits:
        if time.time() > deadline:
            break
        
        try:
            # First, compute proper radii for initial positions
            for i in range(N):
                radii[i] = _max_radius_for_circle(i, centers, radii)
            
            # Run local search
            c_opt, r_opt, score = _best_fit_local_search(centers, radii, 
                                                         max_iters=400, 
                                                         deadline=deadline)
            
            # Gradient polish
            c_polish, r_polish, score_polish = _gradient_polish(c_opt, r_opt, 
                                                                 max_iters=100, 
                                                                 step_size=0.003,
                                                                 deadline=deadline)
            
            # High-precision polish
            c_hp, r_hp, score_hp = _high_precision_polish(c_polish, r_polish, 
                                                           deadline=deadline)
            
            # Validate
            valid_score, is_valid = _eval_packing(c_hp, r_hp)
            if is_valid and valid_score > best_score:
                best_score = valid_score
                best_centers = c_hp.copy()
                best_radii = r_hp.copy()
        except Exception:
            pass
    
    # If no valid solution found, use a simple grid fallback
    if best_centers is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells)
    
    return np.column_stack((best_centers, best_radii))
