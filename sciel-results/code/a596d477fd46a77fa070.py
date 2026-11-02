"""Heilbronn triangles: place 11 points to maximize the minimum triangle area."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools

N = 11
S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0
DEADLINE = 140.0

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)

def _elapsed(start_time):
    return time.time() - start_time

def _cartesian_to_barycentric(pts):
    """Convert (x,y) points to (a,b,c) barycentric coords with a+b+c=1."""
    n = len(pts)
    abc = np.zeros((n, 3))
    c = pts[:, 1] / S3H
    b = pts[:, 0] - 0.5 * c
    a = 1.0 - b - c
    abc[:, 0] = a
    abc[:, 1] = b
    abc[:, 2] = c
    return abc

def _barycentric_to_cartesian(abc):
    """Convert (a,b,c) barycentric coords to (x,y) cartesian points."""
    n = len(abc)
    pts = np.zeros((n, 2))
    pts[:, 0] = abc[:, 1] + 0.5 * abc[:, 2]
    pts[:, 1] = abc[:, 2] * S3H
    return pts

def _all_triangle_areas(pts):
    """Compute all triangle areas (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return areas

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points (vectorized)."""
    return np.min(_all_triangle_areas(pts))

def _min_triangle_area_with_ijk(pts):
    """Compute min area and identify the argmin triangle (i,j,k)."""
    areas = _all_triangle_areas(pts)
    min_idx = np.argmin(areas)
    min_area = areas[min_idx]
    i, j, k = _TRIPLETS[min_idx]
    return min_area, (i, j, k)

def _soft_min_area(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    areas = _all_triangle_areas(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -100, 100)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min

def _project_to_triangle(pts):
    """Project points into the equilateral triangle."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / np.sqrt(3)
        x_max = 1 - y / np.sqrt(3)
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts

def _push_apart_min_triangle(pts, strength=0.01, max_iter=3000, start_time=0, deadline=140):
    """
    Push-apart local search: identify the minimum area triangle and push its
    vertices apart to increase that area.
    """
    current = pts.copy()
    current_min, current_ijk = _min_triangle_area_with_ijk(current)
    best = current.copy()
    best_min = current_min
    
    for it in range(max_iter):
        if _elapsed(start_time) > deadline - 5:
            break
        
        # Find the minimum area triangle
        min_area, (i, j, k) = _min_triangle_area_with_ijk(current)
        
        # Push each vertex away from the midpoint of the other two
        p_i = current[i]
        p_j = current[j]
        p_k = current[k]
        
        mid_jk = (p_j + p_k) / 2.0
        dir_i = p_i - mid_jk
        norm_i = np.linalg.norm(dir_i)
        if norm_i > 1e-10:
            dir_i /= norm_i
        else:
            dir_i = np.array([1.0, 0.0])
        
        mid_ik = (p_i + p_k) / 2.0
        dir_j = p_j - mid_ik
        norm_j = np.linalg.norm(dir_j)
        if norm_j > 1e-10:
            dir_j /= norm_j
        else:
            dir_j = np.array([0.0, 1.0])
        
        mid_ij = (p_i + p_j) / 2.0
        dir_k = p_k - mid_ij
        norm_k = np.linalg.norm(dir_k)
        if norm_k > 1e-10:
            dir_k /= norm_k
        else:
            dir_k = np.array([-0.5, np.sqrt(3)/2])
        
        # Apply push with adaptive step size
        step = strength * (1.0 + 0.3 * it / max_iter)
        new_pts = current.copy()
        new_pts[i] += step * dir_i
        new_pts[j] += step * dir_j
        new_pts[k] += step * dir_k
        
        # Project to triangle
        new_pts = _project_to_triangle(new_pts)
        
        new_min, _ = _min_triangle_area_with_ijk(new_pts)
        
        if new_min >= current_min - 1e-12:
            current = new_pts
            current_min = new_min
            if current_min > best_min:
                best_min = current_min
                best = current.copy()
        else:
            # Try smaller step
            step2 = step * 0.1
            new_pts2 = current.copy()
            new_pts2[i] += step2 * dir_i
            new_pts2[j] += step2 * dir_j
            new_pts2[k] += step2 * dir_k
            new_pts2 = _project_to_triangle(new_pts2)
            new_min2, _ = _min_triangle_area_with_ijk(new_pts2)
            
            if new_min2 >= current_min - 1e-12:
                current = new_pts2
                current_min = new_min2
                if current_min > best_min:
                    best_min = current_min
                    best = current.copy()
            else:
                # If even small step fails, try a different direction
                # Push all three vertices away from centroid
                centroid = (p_i + p_j + p_k) / 3.0
                new_pts3 = current.copy()
                for idx, p in [(i, p_i), (j, p_j), (k, p_k)]:
                    d = p - centroid
                    nrm = np.linalg.norm(d)
                    if nrm > 1e-10:
                        d /= nrm
                    else:
                        d = np.array([1.0, 0.0])
                    new_pts3[idx] += step2 * d
                new_pts3 = _project_to_triangle(new_pts3)
                new_min3, _ = _min_triangle_area_with_ijk(new_pts3)
                
                if new_min3 >= current_min - 1e-12:
                    current = new_pts3
                    current_min = new_min3
                    if current_min > best_min:
                        best_min = current_min
                        best = current.copy()
    
    return best, best_min / TRI_AREA

def _slp_iteration(points, start_time, deadline=140, n_iters=200):
    """
    SLP (Sequential Linear Programming) style iteration:
    At each step, identify the minimum area triangle, push its vertices apart.
    """
    current = points.copy()
    current_min, _ = _min_triangle_area_with_ijk(current)
    best = current.copy()
    best_min = current_min
    
    for it in range(n_iters):
        if _elapsed(start_time) > deadline - 5:
            break
        
        # Find the k smallest area triangles
        areas = _all_triangle_areas(current)
        sorted_idx = np.argsort(areas)
        k_small_idx = sorted_idx[:5]
        
        improved = False
        for sidx in k_small_idx:
            if _elapsed(start_time) > deadline - 5:
                break
            
            i, j, k = _TRIPLETS[sidx]
            
            p_i = current[i]
            p_j = current[j]
            p_k = current[k]
            
            # Push each vertex away from the other two
            mid_jk = (p_j + p_k) / 2.0
            dir_i = p_i - mid_jk
            norm_i = np.linalg.norm(dir_i)
            if norm_i > 1e-10:
                dir_i /= norm_i
            
            mid_ik = (p_i + p_k) / 2.0
            dir_j = p_j - mid_ik
            norm_j = np.linalg.norm(dir_j)
            if norm_j > 1e-10:
                dir_j /= norm_j
            
            mid_ij = (p_i + p_j) / 2.0
            dir_k = p_k - mid_ij
            norm_k = np.linalg.norm(dir_k)
            if norm_k > 1e-10:
                dir_k /= norm_k
            
            step = 0.005
            new_pts = current.copy()
            new_pts[i] += step * dir_i
            new_pts[j] += step * dir_j
            new_pts[k] += step * dir_k
            new_pts = _project_to_triangle(new_pts)
            
            new_min, _ = _min_triangle_area_with_ijk(new_pts)
            
            if new_min > current_min + 1e-10:
                current = new_pts
                current_min = new_min
                improved = True
                if current_min > best_min:
                    best_min = current_min
                    best = current.copy()
        
        if not improved:
            break
    
    return best, best_min / TRI_AREA

def _nelder_mead_optimize(points, start_time, deadline=140):
    """Run Nelder-Mead optimization on the points."""
    def objective(flat):
        pts = flat.reshape(N, 2)
        bary = _cartesian_to_barycentric(pts)
        penalty = 1000.0 * (np.sum(np.maximum(0, -bary[:, 0])**2) +
                           np.sum(np.maximum(0, -bary[:, 1])**2) +
                           np.sum(np.maximum(0, -bary[:, 2])**2))
        min_a = _min_triangle_area(pts)
        return -min_a / TRI_AREA + penalty
    
    try:
        res = minimize(
            objective,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        polished = res.x.reshape(N, 2)
        polished = _project_to_triangle(polished)
        score = _min_triangle_area(polished) / TRI_AREA
        return polished, score
    except Exception:
        return points, _min_triangle_area(points) / TRI_AREA

def _random_in_triangle(rng, n):
    """Generate n random points in the triangle."""
    pts = []
    for _ in range(n):
        while True:
            u, v = rng.random(), rng.random()
            if u + v < 1.0:
                break
        x = v + 0.5 * (1 - u - v)
        y = S3H * (1 - u - v)
        pts.append([x, y])
    return np.array(pts)

def _layered_initialization(rng):
    """
    Initialize points in a 3-4-4 layer structure relative to the triangle.
    """
    pts = []
    
    # Layer 1: 3 points near the base
    for i in range(3):
        x = 0.15 + 0.35 * i + rng.normal(0, 0.02)
        y = rng.uniform(0.02, 0.12)
        pts.append([x, y])
    
    # Layer 2: 4 points in the middle
    for i in range(4):
        x = 0.2 + 0.25 * i + rng.normal(0, 0.02)
        y = rng.uniform(0.25, 0.45) * S3H
        max_x_at_y = 0.5 + 0.5 * (1 - y/S3H)
        min_x_at_y = 0.5 - 0.5 * (1 - y/S3H)
        x = np.clip(x, min_x_at_y + 0.02, max_x_at_y - 0.02)
        pts.append([x, y])
    
    # Layer 3: 4 points near the top
    for i in range(4):
        x = 0.3 + 0.15 * i + rng.normal(0, 0.02)
        y = rng.uniform(0.55, 0.85) * S3H
        max_x_at_y = 0.5 + 0.5 * (1 - y/S3H)
        min_x_at_y = 0.5 - 0.5 * (1 - y/S3H)
        x = np.clip(x, min_x_at_y + 0.02, max_x_at_y - 0.02)
        pts.append([x, y])
    
    return np.array(pts)

def _symmetric_init():
    """Symmetric configuration (known good starting point)."""
    return np.array([
        [0.5, 0.05],
        [0.2, 0.1],
        [0.8, 0.1],
        [0.35, 0.3],
        [0.65, 0.3],
        [0.5, 0.4],
        [0.25, 0.5],
        [0.75, 0.5],
        [0.4, 0.65],
        [0.6, 0.65],
        [0.5, 0.75],
    ])

def construct():
    start_time = time.time()
    
    best_points = None
    best_score = -float('inf')
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Symmetric configuration
    initial_configs.append(_symmetric_init())
    
    # 2. Layered initialization
    initial_configs.append(_layered_initialization(rng))
    
    # 3. Random points
    for _ in range(3):
        initial_configs.append(_random_in_triangle(rng, N))
    
    # 4. Low-discrepancy based
    def van_der_corput(n, base):
        f = 1.0
        r = 0.0
        while n > 0:
            f /= base
            r += f * (n % base)
            n //= base
        return r
    
    sobol_pts = []
    for i in range(1, N + 1):
        x1 = van_der_corput(i, 2)
        x2 = van_der_corput(i, 3)
        sobol_pts.append([x1, x2])
    sobol_pts = np.array(sobol_pts)
    cart_pts = np.column_stack([
        sobol_pts[:, 1] + 0.5 * (1 - sobol_pts[:, 0] - sobol_pts[:, 1]),
        S3H * (1 - sobol_pts[:, 0] - sobol_pts[:, 1])
    ])
    initial_configs.append(cart_pts)
    
    # Phase 1: Differential Evolution in barycentric space
    if _elapsed(start_time) < DEADLINE - 60:
        try:
            def _de_obj_2d(flat_bc):
                """DE objective using 2D barycentric (b, c) pairs."""
                bc = flat_bc.reshape(-1, 2)
                b = bc[:, 0]
                c = bc[:, 1]
                a = 1.0 - b - c
                abc = np.column_stack([a, b, c])
                # Penalty for constraint violation
                pen = 0.0
                for i in range(N):
                    if a[i] < 0:
                        pen += a[i] ** 2 * 500
                    if b[i] < 0:
                        pen += b[i] ** 2 * 500
                    if c[i] < 0:
                        pen += c[i] ** 2 * 500
                pts = _barycentric_to_cartesian(abc)
                ma = _min_triangle_area(pts)
                return -ma + pen
            
            bounds_de = [(0, 1), (0, 1)] * N
            de_result = differential_evolution(
                _de_obj_2d,
                bounds_de,
                maxiter=30,
                popsize=12,
                seed=42,
                tol=1e-4,
                polish=False,
                init='latinhypercube'
            )
            
            if de_result.success or de_result.fun < 0:
                de_pts = _barycentric_to_cartesian(np.column_stack([
                    1.0 - de_result.x[0::2] - de_result.x[1::2],
                    de_result.x[0::2],
                    de_result.x[1::2]
                ]))
                de_pts = _project_to_triangle(de_pts)
                de_score = _min_triangle_area(de_pts) / TRI_AREA
                
                if de_score > best_score:
                    best_score = de_score
                    best_points = de_pts.copy()
                    
        except Exception:
            pass
    
    # Phase 2: Push-apart + Nelder-Mead from each initial config
    for idx, init_pts in enumerate(initial_configs):
        if _elapsed(start_time) > DEADLINE - 25:
            break
        
        # Push-apart local search
        pushed, push_score = _push_apart_min_triangle(
            init_pts, strength=0.005, max_iter=2000, 
            start_time=start_time, deadline=DEADLINE
        )
        
        if push_score > best_score:
            best_score = push_score
            best_points = pushed.copy()
        
        # Nelder-Mead polish
        if _elapsed(start_time) > DEADLINE - 15:
            break
        
        polished, nm_score = _nelder_mead_optimize(pushed, start_time, DEADLINE)
        
        if nm_score > best_score:
            best_score = nm_score
            best_points = polished.copy()
    
    # Phase 3: SLP iterations from best result
    if best_points is not None and _elapsed(start_time) < DEADLINE - 10:
        slp_pts, slp_score = _slp_iteration(best_points, start_time, DEADLINE, n_iters=150)
        
        if slp_score > best_score:
            best_score = slp_score
            best_points = slp_pts.copy()
        
        # Final Nelder-Mead polish
        if _elapsed(start_time) < DEADLINE - 5:
            polished, nm_score = _nelder_mead_optimize(best_points, start_time, DEADLINE)
            if nm_score > best_score:
                best_score = nm_score
                best_points = polished.copy()
    
    # Phase 4: Additional multi-start if time permits
    if best_points is not None and _elapsed(start_time) < DEADLINE - 15:
        for _ in range(5):
            if _elapsed(start_time) > DEADLINE - 8:
                break
            
            # Perturb best and re-optimize
            perturbed = best_points + rng.normal(0, 0.03, size=(N, 2))
            perturbed = _project_to_triangle(perturbed)
            
            pushed, push_score = _push_apart_min_triangle(
                perturbed, strength=0.003, max_iter=1000, 
                start_time=start_time, deadline=DEADLINE
            )
            
            if push_score > best_score:
                best_score = push_score
                best_points = pushed.copy()
    
    # Final polish
    if best_points is not None and _elapsed(start_time) < DEADLINE - 3:
        polished, nm_score = _nelder_mead_optimize(best_points, start_time, DEADLINE)
        if nm_score > best_score:
            best_score = nm_score
            best_points = polished.copy()
    
    # Fallback
    if best_points is None:
        best_points = _random_in_triangle(rng, N)
    
    return best_points
