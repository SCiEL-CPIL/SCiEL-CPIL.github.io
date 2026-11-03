"""Pack 50 equal circles in a unit square, maximize common radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers."""
    wall_r = np.min(0.5 - np.abs(centers))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _project_to_square(centers):
    """Clip centers to be strictly inside the square."""
    return np.clip(centers, -0.499, 0.499)

def _expand_radius(centers, target_r, max_iter=1000, tol=1e-10):
    """
    Hard-disk radius expansion: project points to satisfy separation >= 2*target_r
    and wall distance >= target_r.
    Returns (success, refined_centers).
    """
    centers = centers.copy()
    n = len(centers)
    for _ in range(max_iter):
        moved = False
        # Wall projection
        for i in range(n):
            for k in range(2):
                if centers[i, k] + target_r > 0.5:
                    centers[i, k] = 0.5 - target_r
                    moved = True
                elif centers[i, k] - target_r < -0.5:
                    centers[i, k] = -0.5 + target_r
                    moved = True
        # Pairwise separation projection
        for i in range(n):
            for j in range(i + 1, n):
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                min_dist = 2.0 * target_r
                if dist < min_dist - tol:
                    if dist < 1e-12:
                        # Coincident: push apart randomly
                        d = np.random.randn(2)
                        dist = np.linalg.norm(d)
                    # Push both apart by half the violation
                    shift = (min_dist - dist) / (2.0 * dist) * d
                    centers[i] += shift
                    centers[j] -= shift
                    moved = True
        if not moved:
            return True, centers
    # Check if converged
    for i in range(n):
        if np.any(np.abs(centers[i]) + target_r > 0.5 + 1e-8):
            return False, centers
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(centers[i] - centers[j]) < 2.0 * target_r - 1e-8:
                return False, centers
    return True, centers

def _slsqp_polish(centers, time_limit=10.0):
    """
    SLSQP epigraph polish: maximize r subject to
    r <= 0.5 - |c_ik| for all i,k
    r <= |c_i - c_j| / 2 for all i<j
    """
    x0 = np.concatenate([centers.ravel(), [_radius(centers)]])
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    def objective(x):
        return -x[-1]  # maximize r
    
    # Constraints: each returns >= 0
    cons = []
    # Wall constraints: 0.5 - |c_ik| - r >= 0
    for i in range(N):
        for k in range(2):
            def wall_con(x, i=i, k=k):
                return 0.5 - abs(x[2*i + k]) - x[-1]
            cons.append({'type': 'ineq', 'fun': wall_con})
    # Pairwise constraints: |c_i - c_j|/2 - r >= 0
    for i in range(N):
        for j in range(i + 1, N):
            def pair_con(x, i=i, j=j):
                ci = x[2*i:2*i+2]
                cj = x[2*j:2*j+2]
                return 0.5 * np.linalg.norm(ci - cj) - x[-1]
            cons.append({'type': 'ineq', 'fun': pair_con})
    
    start_time = time.time()
    def callback(xk):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
        return False
    
    try:
        res = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-12, 'disp': False},
            callback=callback
        )
        if res.x is not None:
            return res.x[:2*N].reshape(N, 2)
    except RuntimeError:
        pass
    except Exception:
        pass
    return centers

def _make_grid_rect(n_x, n_y):
    ticks_x = (np.arange(n_x) + 0.5) / n_x - 0.5
    ticks_y = (np.arange(n_y) + 0.5) / n_y - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    return grid[:N].copy()

def _make_hexagonal_grid(n_x, n_y, spacing, offset_x=0, offset_y=0):
    points = []
    dy = spacing * np.sqrt(3) / 2
    for j in range(n_y):
        for i in range(n_x):
            x = i * spacing + offset_x
            y = j * dy + offset_y
            if j % 2 != 0:
                x += spacing / 2
            points.append((x, y))
    points = np.array(points)
    return points[:N]

def construct():
    start_time = time.time()
    best_centers = None
    best_r = 0.0
    
    # Generate initial configurations
    inits = []
    
    # 1. Rectangular Grid 7x8
    inits.append(_make_grid_rect(7, 8))
    # 2. Rectangular Grid 8x7
    inits.append(_make_grid_rect(8, 7))
    # 3. Hexagonal Grid
    for sp in [0.14, 0.15, 0.16]:
        c_hex = _make_hexagonal_grid(10, 10, sp, offset_x=-0.5, offset_y=-0.5)
        c_hex = c_hex - np.mean(c_hex, axis=0)
        max_ext = np.max(np.abs(c_hex))
        if max_ext > 0.45:
            c_hex = c_hex * (0.45 / max_ext)
        inits.append(c_hex)
    # 4. Random
    np.random.seed(42)
    inits.append(np.random.uniform(-0.45, 0.45, size=(N, 2)))
    # 5. Jittered Grid
    inits.append(_make_grid_rect(7, 8) + np.random.uniform(-0.02, 0.02, size=(N, 2)))
    
    # Phase 1: Radius expansion from each initial configuration
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        time_left = DEADLINE - 30 - (time.time() - start_time)
        alloc_time = min(8.0, time_left / max(1, len(inits) - i))
        
        # Start with a small radius and expand
        r_start = 0.02
        r_step = 0.002
        r = r_start
        success = False
        while time.time() - start_time < start_time + alloc_time:
            ok, refined = _expand_radius(init, r, max_iter=200, tol=1e-10)
            if ok:
                r += r_step
            else:
                # Back off and try finer steps
                r -= r_step
                r_step /= 2
                if r_step < 1e-5:
                    break
                ok, refined = _expand_radius(init, r, max_iter=200, tol=1e-10)
                if ok:
                    success = True
                    break
                else:
                    break
        
        if success or r > 0.03:
            r_actual = _radius(refined)
            if r_actual > best_r:
                best_r = r_actual
                best_centers = refined.copy()
    
    # Phase 2: SLSQP epigraph polish on the best candidate
    if best_centers is not None and time.time() - start_time < DEADLINE - 15:
        time_left = DEADLINE - 15 - (time.time() - start_time)
        polished = _slsqp_polish(best_centers, time_limit=time_left * 0.8)
        r_polish = _radius(polished)
        if r_polish > best_r:
            best_r = r_polish
            best_centers = polished.copy()
    
    # Phase 3: If time remains, try more random restarts with expansion + SLSQP
    while time.time() - start_time < DEADLINE - 8:
        seed = int(time.time() * 1000) % 100000
        np.random.seed(seed)
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        time_left = DEADLINE - 8 - (time.time() - start_time)
        if time_left < 5:
            break
        
        # Quick expansion
        r = 0.03
        r_step = 0.003
        refined = c_rand.copy()
        for _ in range(50):
            if time.time() - start_time > DEADLINE - 10:
                break
            ok, refined = _expand_radius(refined, r, max_iter=100, tol=1e-9)
            if ok:
                r += r_step
            else:
                break
        
        r_actual = _radius(refined)
        if r_actual > best_r:
            best_r = r_actual
            best_centers = refined.copy()
        
        # SLSQP polish if time permits
        if time.time() - start_time < DEADLINE - 5:
            time_left = DEADLINE - 5 - (time.time() - start_time)
            polished = _slsqp_polish(best_centers, time_limit=time_left * 0.7)
            r_polish = _radius(polished)
            if r_polish > best_r:
                best_r = r_polish
                best_centers = polished.copy()
    
    return best_centers
