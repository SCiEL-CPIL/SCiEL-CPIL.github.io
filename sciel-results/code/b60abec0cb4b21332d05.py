"""Optimized packing of 12 unit hexagons in a minimal outer hexagon."""
import time
import math
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = 145.0  # safety margin before the 150s kill

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _hex_vertices(theta):
    """Regular hexagon, side 1, circumradius 1, rotated by theta, centered at origin."""
    return np.array([[math.cos(theta + k * math.pi / 3.0),
                      math.sin(theta + k * math.pi / 3.0)] for k in range(6)])

def _hex_axes(theta):
    """6 edge-normal directions (unit vectors) of a hexagon oriented by theta."""
    return np.array([[math.cos(theta + math.pi / 6.0 + k * math.pi / 3.0),
                      math.sin(theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])

def _project_polygon(verts, axes):
    """Project polygon vertices onto given axes. Returns array of shape (len(axes), 2) = (min, max)."""
    P = verts @ axes.T
    return np.vstack([P.min(axis=0), P.max(axis=0)]).T  # shape (n_axes, 2)

def _hex_axes_from_verts(verts):
    """Compute edge normals from polygon vertices."""
    axes = []
    for k in range(6):
        p1 = verts[k]
        p2 = verts[(k + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        nl = np.linalg.norm(normal)
        if nl > 1e-9:
            axes.append(normal / nl)
    return np.array(axes)

def _sat_overlap(A, B):
    """SAT overlap test between two hexagons. Returns True if overlapping."""
    EPS = 1e-9
    axesA = _hex_axes_from_verts(A)
    axesB = _hex_axes_from_verts(B)
    all_axes = np.vstack([axesA, axesB])
    
    projA = _project_polygon(A, all_axes)
    projB = _project_polygon(B, all_axes)
    
    for i in range(len(all_axes)):
        if projA[i, 1] < projB[i, 0] - EPS or projB[i, 1] < projA[i, 0] - EPS:
            return False
    return True

def _fast_sat_overlap(verts_i, verts_j, axes_i, axes_j):
    """Faster SAT using precomputed axes."""
    EPS = 1e-9
    all_axes = np.vstack([axes_i, axes_j])
    projA = _project_polygon(verts_i, all_axes)
    projB = _project_polygon(verts_j, all_axes)
    
    for i in range(len(all_axes)):
        if projA[i, 1] < projB[i, 0] - EPS or projB[i, 1] < projA[i, 0] - EPS:
            return False
    return True

def _outer_hex_side(all_pts, theta):
    """Compute side length of smallest regular hexagon (orientation theta, optimal center) containing points."""
    axes = _hex_axes(theta)
    P = all_pts @ axes.T
    c = (P.min(axis=0) + P.max(axis=0)) / 2.0
    max_dev = np.max(P.max(axis=0) - c)
    return 2.0 * max_dev / math.sqrt(3.0)

def _compute_best_outer(all_pts):
    """Find best outer hexagon orientation and center for given set of points."""
    best_side = float('inf')
    best_center = np.array([0.0, 0.0])
    best_angle = 0.0
    
    for ang in np.linspace(0.0, math.pi / 3.0, 361):
        axes = _hex_axes(ang)
        P = all_pts @ axes.T
        c = (P.min(axis=0) + P.max(axis=0)) / 2.0
        side = 2.0 * np.max(P.max(axis=0) - c) / math.sqrt(3.0)
        if side < best_side - 1e-12:
            best_side = side
            best_center = axes.T @ c
            best_angle = float(ang)
    
    return best_side, best_center, best_angle

def _objective(centers, angles, outer_theta, outer_center):
    """Compute objective: outer side length + overlap penalty + containment penalty."""
    verts = [_hex_vertices(angles[k]) + centers[k] for k in range(N)]
    all_pts = np.vstack(verts)
    
    axes = _hex_axes(outer_theta)
    P = all_pts @ axes.T
    c = np.array(outer_center) @ axes.T
    max_dev = np.max(P.max(axis=0) - c)
    side = 2.0 * max_dev / math.sqrt(3.0)
    
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            if _sat_overlap(verts[i], verts[j]):
                axesA = _hex_axes_from_verts(verts[i])
                axesB = _hex_axes_from_verts(verts[j])
                all_axes = np.vstack([axesA, axesB])
                projA = _project_polygon(verts[i], all_axes)
                projB = _project_polygon(verts[j], all_axes)
                min_sep = float('inf')
                for k in range(len(all_axes)):
                    sep = min(projA[k, 1] - projB[k, 0], projB[k, 1] - projA[k, 0])
                    min_sep = min(min_sep, sep)
                if min_sep < 0:
                    overlap_pen += 10.0 * min_sep * min_sep
    
    contain_pen = 0.0
    for k in range(N):
        Pk = verts[k] @ axes.T
        for face in range(6):
            exceed = (Pk[:, face] - c[face]).max() - outer_side * math.sqrt(3.0) / 2.0
            if exceed > 0:
                contain_pen += 5.0 * exceed * exceed
    
    return side + overlap_pen + contain_pen

def _objective_fast(centers, angles, outer_theta, outer_center, outer_side):
    """Faster objective computation."""
    verts = [_hex_vertices(angles[k]) + centers[k] for k in range(N)]
    all_pts = np.vstack(verts)
    
    axes = _hex_axes(outer_theta)
    P = all_pts @ axes.T
    c = np.array(outer_center) @ axes.T
    max_dev = np.max(P.max(axis=0) - c)
    side = 2.0 * max_dev / math.sqrt(3.0)
    
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            if _sat_overlap(verts[i], verts[j]):
                overlap_pen += 1.0
    
    contain_pen = 0.0
    for k in range(N):
        Pk = verts[k] @ axes.T
        for face in range(6):
            exceed = (Pk[:, face] - c[face]).max() - outer_side * math.sqrt(3.0) / 2.0
            if exceed > 0:
                contain_pen += 5.0 * exceed * exceed
    
    return side + overlap_pen + contain_pen

# ---------------------------------------------------------------------------
# Initial configuration generators
# ---------------------------------------------------------------------------
def _construct_initial_config(seed_type=0):
    """Generate initial configuration based on seed type."""
    if seed_type == 0:
        # Hexagonal cluster
        d = 2.0 * math.sqrt(3.0) + 0.001
        a1 = np.array([d, 0.0])
        a2 = np.array([d / 2.0, d * math.sqrt(3.0) / 2.0])
        coords = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
                  (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2)]
        pts = []
        for (i, j) in coords:
            pts.append(i * a1 + j * a2)
        pts = np.array(pts)
        c = pts.mean(axis=0)
        pts = pts - c
        angles = np.zeros(N)
        return pts, angles
    
    elif seed_type == 1:
        # Triangular lattice
        d = 2.0 * math.sqrt(3.0) + 0.001
        a1 = np.array([d, 0.0])
        a2 = np.array([d / 2.0, d * math.sqrt(3.0) / 2.0])
        pts = []
        for i in range(-1, 3):
            for j in range(-1, 2):
                pts.append(i * a1 + j * a2)
        pts = np.array(pts)
        c = pts.mean(axis=0)
        pts = pts - c
        angles = np.zeros(N)
        return pts, angles
    
    elif seed_type == 2:
        # Random in disk
        pts = []
        for _ in range(N):
            r = random.uniform(0, 4.0)
            theta = random.uniform(0, 2 * math.pi)
            pts.append([r * math.cos(theta), r * math.sin(theta)])
        pts = np.array(pts)
        c = pts.mean(axis=0)
        pts = pts - c
        angles = np.array([random.uniform(-math.pi/6, math.pi/6) for _ in range(N)])
        return pts, angles
    
    else:
        # Ring + center
        pts = [np.array([0.0, 0.0])]
        for k in range(11):
            ang = 2 * math.pi * k / 11
            r = 2.5 * math.sqrt(3.0)
            pts.append([r * math.cos(ang), r * math.sin(ang)])
        pts = np.array(pts)
        c = pts.mean(axis=0)
        pts = pts - c
        angles = np.zeros(N)
        return pts, angles

# ---------------------------------------------------------------------------
# Nelder-Mead style local optimization
# ---------------------------------------------------------------------------
def _nelder_mead(centers, angles, outer_theta, outer_center, max_iters=500, step_size=0.5):
    """Nelder-Mead optimization on positions and angles."""
    centers = centers.copy()
    angles = angles.copy()
    
    best_obj = _objective(centers, angles, outer_theta, outer_center)
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        improved = False
        for i in range(N):
            for dx in [-step_size, step_size]:
                for dy in [-step_size, step_size]:
                    new_centers = centers.copy()
                    new_centers[i] += np.array([dx, dy])
                    new_obj = _objective(new_centers, angles, outer_theta, outer_center)
                    if new_obj < best_obj - 1e-10:
                        centers = new_centers
                        best_obj = new_obj
                        improved = True
            
            for dangle in [-step_size * 0.3, step_size * 0.3]:
                new_angles = angles.copy()
                new_angles[i] += dangle
                new_obj = _objective(centers, new_angles, outer_theta, outer_center)
                if new_obj < best_obj - 1e-10:
                    angles = new_angles
                    best_obj = new_obj
                    improved = True
        
        if not improved:
            step_size *= 0.8
            if step_size < 0.001:
                break
    
    return centers, angles, best_obj

# ---------------------------------------------------------------------------
# Simulated annealing
# ---------------------------------------------------------------------------
def _simulated_annealing(centers, angles, outer_theta, outer_center, max_iters=5000):
    """Simulated annealing to find better packings."""
    best_centers = centers.copy()
    best_angles = angles.copy()
    best_obj = _objective(centers, angles, outer_theta, outer_center)
    
    current_obj = best_obj
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        i = random.randint(0, N - 1)
        move_type = random.random()
        
        if move_type < 0.7:
            delta = np.array([random.gauss(0, T), random.gauss(0, T)])
            new_centers = centers.copy()
            new_centers[i] += delta
            new_angles = angles.copy()
        else:
            new_centers = centers.copy()
            new_angles = angles.copy()
            new_angles[i] += random.gauss(0, T * 0.5)
        
        new_obj = _objective(new_centers, new_angles, outer_theta, outer_center)
        delta_E = new_obj - current_obj
        
        if delta_E < 0 or random.random() < math.exp(-delta_E / max(T, 1e-10)):
            centers = new_centers
            angles = new_angles
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_centers = centers.copy()
                best_angles = angles.copy()
        
        T = max(T * cooling, T_min)
    
    return best_centers, best_angles, best_obj

# ---------------------------------------------------------------------------
# SciPy-based optimization
# ---------------------------------------------------------------------------
def _scipy_optimize(centers, angles, outer_theta, outer_center, max_time=30):
    """Use scipy.optimize.minimize with L-BFGS-B or COBYLA."""
    start = time.time()
    
    # Flatten parameters: [x0, y0, a0, x1, y1, a1, ...]
    x0 = np.concatenate([centers.ravel(), angles])
    
    def obj_flat(x):
        if time.time() - start > max_time:
            return float('inf')
        c = x[:2*N].reshape(N, 2)
        a = x[2*N:]
        return _objective(c, a, outer_theta, outer_center)
    
    # Try multiple methods
    best_result = None
    best_fun = float('inf')
    
    for method in ['L-BFGS-B', 'COBYLA']:
        if time.time() > DEADLINE - 5:
            break
        try:
            res = minimize(obj_flat, x0, method=method, 
                          options={'maxiter': 200, 'maxfev': 500})
            if res.fun < best_fun:
                best_fun = res.fun
                best_result = res.x
        except:
            pass
    
    if best_result is None:
        return centers.copy(), angles.copy(), float('inf')
    
    c = best_result[:2*N].reshape(N, 2)
    a = best_result[2*N:]
    return c, a, best_fun

# ---------------------------------------------------------------------------
# Differential evolution (global search)
# ---------------------------------------------------------------------------
def _differential_evolution_global(outer_theta, outer_center, max_time=60):
    """Global search using differential evolution."""
    start = time.time()
    
    # Bounds: positions in [-5, 5], angles in [-pi/6, pi/6]
    bounds = [(-5.0, 5.0)] * (2 * N) + [(-math.pi/6, math.pi/6)] * N
    
    def obj_de(x):
        if time.time() - start > max_time:
            return float('inf')
        c = x[:2*N].reshape(N, 2)
        a = x[2*N:]
        return _objective(c, a, outer_theta, outer_center)
    
    try:
        res = differential_evolution(obj_de, bounds, maxiter=100, popsize=15,
                                     tol=1e-6, seed=42, polish=True)
        if res.success or res.fun < float('inf'):
            c = res.x[:2*N].reshape(N, 2)
            a = res.x[2*N:]
            return c, a, res.fun
    except:
        pass
    
    return None, None, float('inf')

# ---------------------------------------------------------------------------
# Main construct function
# ---------------------------------------------------------------------------
def construct():
    start = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_side = float('inf')
    best_centers = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    def _consider(centers, angles, side, center, angle):
        nonlocal best_side, best_centers, best_angles, best_center, best_angle
        if side < best_side:
            best_side = side
            best_centers = centers.copy()
            best_angles = angles.copy()
            best_center = center
            best_angle = angle
    
    # Phase 1: Quick local search from multiple seeds
    seed_types = [0, 1, 2, 3]
    outer_angles = [0.0, math.pi / 6.0, math.pi / 3.0 * 0.5]
    
    for seed_type in seed_types:
        for oa in outer_angles:
            if time.time() > DEADLINE - 30:
                break
            
            centers, angles = _construct_initial_config(seed_type)
            
            # Quick Nelder-Mead
            bc, ba, bobj = _nelder_mead(centers, angles, oa, (0.0, 0.0), max_iters=100, step_size=0.5)
            
            # Compute best outer hexagon
            all_pts = np.vstack([_hex_vertices(ba[k]) + bc[k] for k in range(N)])
            side, center, angle = _compute_best_outer(all_pts)
            
            _consider(bc, ba, side, center, angle)
    
    # Phase 2: Refine best configuration with scipy
    if best_centers is not None and time.time() < DEADLINE - 20:
        all_pts = np.vstack([_hex_vertices(best_angles[k]) + best_centers[k] for k in range(N)])
        side, center, angle = _compute_best_outer(all_pts)
        
        # SciPy optimization
        c_scipy, a_scipy, obj_scipy = _scipy_optimize(best_centers, best_angles, angle, center, max_time=20)
        
        if obj_scipy < float('inf'):
            all_pts = np.vstack([_hex_vertices(a_scipy[k]) + c_scipy[k] for k in range(N)])
            side2, center2, angle2 = _compute_best_outer(all_pts)
            _consider(c_scipy, a_scipy, side2, center2, angle2)
    
    # Phase 3: Global search with differential evolution if time permits
    if time.time() < DEADLINE - 40:
        c_de, a_de, obj_de = _differential_evolution_global(0.0, (0.0, 0.0), max_time=30)
        
        if c_de is not None and obj_de < float('inf'):
            all_pts = np.vstack([_hex_vertices(a_de[k]) + c_de[k] for k in range(N)])
            side_de, center_de, angle_de = _compute_best_outer(all_pts)
            _consider(c_de, a_de, side_de, center_de, angle_de)
    
    # Phase 4: Final polish on best configuration
    if best_centers is not None and time.time() < DEADLINE - 10:
        all_pts = np.vstack([_hex_vertices(best_angles[k]) + best_centers[k] for k in range(N)])
        side, center, angle = _compute_best_outer(all_pts)
        
        # Fine Nelder-Mead polish
        bc_final, ba_final, _ = _nelder_mead(best_centers, best_angles, angle, center, max_iters=200, step_size=0.1)
        all_pts = np.vstack([_hex_vertices(ba_final[k]) + bc_final[k] for k in range(N)])
        side_final, center_final, angle_final = _compute_best_outer(all_pts)
        
        _consider(bc_final, ba_final, side_final, center_final, angle_final)
    
    # Fallback
    if best_centers is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Format output
    inner = []
    for k in range(N):
        inner.append([float(best_centers[k][0]), float(best_centers[k][1]), float(math.degrees(best_angles[k]))])
    
    return inner, best_center, best_side, math.degrees(best_angle)
