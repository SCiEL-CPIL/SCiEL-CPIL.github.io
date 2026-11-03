"""Simulated annealing packing of 12 unit hexagons in a regular hexagon."""
import time
import math
import random
import numpy as np

N = 12
DEADLINE = 145.0  # safety margin before the 150s kill

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _hex_vertices(theta):
    # Regular hexagon, side 1, circumradius 1, rotated by theta
    return np.array([[math.cos(theta + k * math.pi / 3.0),
                      math.sin(theta + k * math.pi / 3.0)] for k in range(6)])

def _norms(theta):
    # 6 edge-normal directions of a regular hexagon oriented by theta
    return np.array([[math.cos(theta + math.pi / 6.0 + k * math.pi / 3.0),
                      math.sin(theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])

def _bounding_side_fast(pts, theta):
    # side length of smallest regular hexagon (orientation theta, centered at origin)
    # containing the set of points pts.
    nrm = _norms(theta)
    P = pts @ nrm.T
    req = P.max(axis=0)
    return 2.0 * float(np.max(req)) / math.sqrt(3.0)

def _compute_center(pts, theta):
    # Compute optimal center for bounding hexagon with orientation theta
    nrm = _norms(theta)
    P = pts @ nrm.T
    c = (P.max(axis=0) + P.min(axis=0)) / 2.0
    c_world = nrm.T @ c
    return c_world

def _fast_sat_overlap(A, B):
    """Fast SAT overlap test between two hexagons. Returns True if overlapping."""
    EPS = 1e-9
    for poly in (A, B):
        for k in range(6):
            p1 = poly[k]
            p2 = poly[(k + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            projA = A @ normal
            projB = B @ normal
            if projA.max() < projB.min() - EPS or projB.max() < projA.min() - EPS:
                return False
    return True

def _min_separation(A, B):
    """Find minimum separation axis and depth for two hexagons.
    Returns (min_depth, best_normal) where min_depth < 0 means overlap."""
    min_depth = float('inf')
    best_normal = None
    for poly in (A, B):
        for k in range(6):
            p1 = poly[k]
            p2 = poly[(k + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            nl = math.hypot(normal[0], normal[1])
            if nl < 1e-9:
                continue
            normal = normal / nl
            projA = A @ normal
            projB = B @ normal
            sep = min(projA.max() - projB.min(), projB.max() - projA.min())
            if sep < min_depth:
                min_depth = sep
                best_normal = normal
    return min_depth, best_normal

def _is_valid(centers, angles, outer_side, outer_theta, outer_center):
    """Check if all hexagons are non-overlapping and contained in outer hexagon."""
    EPS = 1e-9
    verts = [_hex_vertices(angles[k]) + centers[k] for k in range(N)]
    
    # Check pairwise overlaps via SAT
    for i in range(N):
        for j in range(i + 1, N):
            if _fast_sat_overlap(verts[i], verts[j]):
                return False
    
    # Check containment in outer hexagon
    nrm = _norms(outer_theta)
    c = np.array(outer_center) @ nrm.T
    for k in range(N):
        P = verts[k] @ nrm.T
        for face in range(6):
            if (P[:, face] - c[face]).max() > outer_side * math.sqrt(3.0) / 2.0 + EPS:
                return False
    
    return True

def _smooth_cost(centers, angles, outer_theta, outer_center, beta=100.0):
    """Compute a smooth cost that penalizes overlaps and boundary violations.
    Uses soft-max approximation for the bounding hexagon side length."""
    verts = np.array([_hex_vertices(angles[k]) + centers[k] for k in range(N)])
    
    # Bounding hexagon side with soft-max
    nrm = _norms(outer_theta)
    c = np.array(outer_center) @ nrm.T
    all_pts = verts.reshape(-1, 2)
    P = all_pts @ nrm.T
    
    # Soft-max for each face
    eps = 1e-10
    smax = np.zeros(6)
    for face in range(6):
        vals = P[:, face] - c[face]
        max_val = vals.max()
        smax[face] = max_val + math.log(sum(math.exp(v - max_val) for v in vals)) / beta
    
    base_side = 2.0 * float(np.max(smax)) / math.sqrt(3.0)
    
    # Overlap penalty (soft)
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            min_depth, _ = _min_separation(verts[i], verts[j])
            if min_depth < 0:
                # Soft penalty: smooth function of overlap depth
                overlap_penalty += math.exp(-beta * abs(min_depth))
    
    # Containment penalty (soft)
    contain_penalty = 0.0
    for k in range(N):
        Pk = verts[k] @ nrm.T
        for face in range(6):
            exceed = (Pk[:, face] - c[face]).max() - outer_side * math.sqrt(3.0) / 2.0
            if exceed > 0:
                contain_penalty += math.exp(-beta * exceed)
    
    return base_side + 5.0 * overlap_penalty + 5.0 * contain_penalty

# ---------------------------------------------------------------------------
# Seed generators
# ---------------------------------------------------------------------------
def _seed_lattice():
    """Triangular lattice seed, centered at origin."""
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
    return pts

def _seed_hex_cluster():
    """Hexagonal cluster: center + 6 neighbors + 5 more."""
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
    return pts

def _seed_random():
    """Random positions in a disk of radius 5."""
    pts = []
    for _ in range(N):
        r = random.uniform(0, 4.0)
        theta = random.uniform(0, 2 * math.pi)
        pts.append([r * math.cos(theta), r * math.sin(theta)])
    pts = np.array(pts)
    c = pts.mean(axis=0)
    pts = pts - c
    return pts

# ---------------------------------------------------------------------------
# Physics relaxation
# ---------------------------------------------------------------------------
def _physics_relax(centers, angles, outer_side, outer_theta, outer_center, steps=500):
    """Simple physics-based relaxation to resolve overlaps."""
    centers = centers.copy()
    angles = angles.copy()
    vel = np.zeros((N, 2))
    
    for s in range(steps):
        if time.time() > DEADLINE:
            break
        verts = [_hex_vertices(angles[k]) + centers[k] for k in range(N)]
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                A = verts[i]
                B = verts[j]
                min_sep, best_axis = _min_separation(A, B)
                if min_sep < -1e-9 and best_axis is not None:
                    f = 0.3 * (-min_sep)
                    vel[i] += f * best_axis
                    vel[j] -= f * best_axis
        
        # Containment force
        nrm = _norms(outer_theta)
        c = np.array(outer_center) @ nrm.T
        for i in range(N):
            P = verts[i] @ nrm.T
            for face in range(6):
                exceed = (P[:, face] - c[face]).max() - outer_side * math.sqrt(3.0) / 2.0
                if exceed > 0:
                    vel[i] -= 0.5 * exceed * nrm[face]
        
        # Mild attraction to center
        for i in range(N):
            vel[i] += -0.01 * centers[i]
        
        # Integrate
        vel *= 0.9
        centers += vel * 0.05
    
    return centers, angles

# ---------------------------------------------------------------------------
# Simulated annealing with smooth cost
# ---------------------------------------------------------------------------
def _simulated_annealing(centers, angles, outer_theta, max_iters=20000, use_smooth=True):
    """Simulated annealing to minimize bounding hexagon side length.
    Uses smooth cost to avoid hard rejections."""
    best_side = float('inf')
    best_centers = centers.copy()
    best_angles = angles.copy()
    
    # Initial evaluation
    all_pts = np.vstack([_hex_vertices(angles[k]) + centers[k] for k in range(N)])
    side = _bounding_side_fast(all_pts, outer_theta)
    if side < best_side:
        best_side = side
        best_centers = centers.copy()
        best_angles = angles.copy()
    
    T = 0.5  # initial temperature
    T_min = 0.001
    cooling = 0.9997
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        # Propose move
        i = random.randint(0, N - 1)
        move_type = random.random()
        
        if move_type < 0.7:
            # Move center
            delta = np.array([random.gauss(0, T), random.gauss(0, T)])
            new_centers = centers.copy()
            new_centers[i] += delta
            new_angles = angles.copy()
        else:
            # Rotate
            new_centers = centers.copy()
            new_angles = angles.copy()
            new_angles[i] += random.gauss(0, T * 0.5)
        
        # Evaluate using smooth cost
        if use_smooth:
            new_cost = _smooth_cost(new_centers, new_angles, outer_theta, (0.0, 0.0), beta=50.0)
            # Also compute exact side for tracking
            all_pts = np.vstack([_hex_vertices(new_angles[k]) + new_centers[k] for k in range(N)])
            new_side_exact = _bounding_side_fast(all_pts, outer_theta)
        else:
            all_pts = np.vstack([_hex_vertices(new_angles[k]) + new_centers[k] for k in range(N)])
            new_side_exact = _bounding_side_fast(all_pts, outer_theta)
            new_cost = new_side_exact
        
        # Check validity (non-overlapping) - only for exact tracking
        if not _is_valid(new_centers, new_angles, new_side_exact, outer_theta, (0.0, 0.0)):
            # Still accept with probability based on cost difference
            if new_cost < side:
                side = new_cost
                centers = new_centers
                angles = new_angles
                if new_side_exact < best_side:
                    best_side = new_side_exact
                    best_centers = centers.copy()
                    best_angles = angles.copy()
            T = max(T * cooling, T_min)
            continue
        
        # Accept/reject based on smooth cost
        if new_cost < side:
            side = new_cost
            centers = new_centers
            angles = new_angles
            if new_side_exact < best_side:
                best_side = new_side_exact
                best_centers = centers.copy()
                best_angles = angles.copy()
        else:
            delta_E = new_cost - side
            if random.random() < math.exp(-delta_E / max(T, 1e-10)):
                side = new_cost
                centers = new_centers
                angles = new_angles
                if new_side_exact < best_side:
                    best_side = new_side_exact
                    best_centers = centers.copy()
                    best_angles = angles.copy()
        
        T = max(T * cooling, T_min)
    
    return best_centers, best_angles, best_side

# ---------------------------------------------------------------------------
# Outer angle/center refinement
# ---------------------------------------------------------------------------
def _refine_outer(centers, angles):
    """Sweep outer angle and compute optimal center; return best (side, center, angle)."""
    all_pts = np.vstack([_hex_vertices(angles[k]) + centers[k] for k in range(N)])
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for ang in np.linspace(0.0, math.pi / 3.0, 361):
        nrm = _norms(ang)
        P = all_pts @ nrm.T
        # Optimal center in normal space
        c_normal = (P.max(axis=0) + P.min(axis=0)) / 2.0
        c_world = nrm.T @ c_normal
        side = 2.0 * float(np.max(P.max(axis=0) - c_normal)) / math.sqrt(3.0)
        if side < best_side - 1e-12:
            best_side = side
            best_center = (float(c_world[0]), float(c_world[1]))
            best_angle = float(ang)
    
    return best_side, best_center, best_angle

# ---------------------------------------------------------------------------
# High-precision polish
# ---------------------------------------------------------------------------
def _polish(centers, angles, outer_theta, outer_center, outer_side, iterations=2000):
    """Local polish: small perturbations with acceptance only if valid and improves."""
    best_side = outer_side
    best_centers = centers.copy()
    best_angles = angles.copy()
    
    for it in range(iterations):
        if time.time() > DEADLINE:
            break
        
        i = random.randint(0, N - 1)
        move_type = random.random()
        step = 0.01 * (1.0 + 0.5 * random.random())
        
        if move_type < 0.8:
            delta = np.array([random.gauss(0, step), random.gauss(0, step)])
            new_centers = centers.copy()
            new_centers[i] += delta
            new_angles = angles.copy()
        else:
            new_centers = centers.copy()
            new_angles = angles.copy()
            new_angles[i] += random.gauss(0, step * 0.5)
        
        # Check validity
        if not _is_valid(new_centers, new_angles, outer_side, outer_theta, outer_center):
            continue
        
        # Compute new bounding side
        all_pts = np.vstack([_hex_vertices(new_angles[k]) + new_centers[k] for k in range(N)])
        new_side = _bounding_side_fast(all_pts, outer_theta)
        
        if new_side < best_side - 1e-10:
            best_side = new_side
            best_centers = new_centers
            best_angles = new_angles
            centers = new_centers
            angles = new_angles
    
    return best_centers, best_angles, best_side

# ---------------------------------------------------------------------------
# Entry point
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

    # Try multiple seeds and outer angles
    seed_funcs = [_seed_lattice, _seed_hex_cluster, _seed_random]
    outer_angles = [0.0, math.pi / 6.0, math.pi / 3.0 * 0.5, math.pi / 3.0 * 0.25]

    for seed_func in seed_funcs:
        for oa in outer_angles:
            if time.time() > DEADLINE - 10:
                break
            
            centers = seed_func()
            angles = np.zeros(N)
            
            # Physics relaxation to resolve initial overlaps
            centers, angles = _physics_relax(centers, angles, 8.0, oa, (0.0, 0.0), steps=200)
            
            # Simulated annealing
            bc, ba, bs = _simulated_annealing(centers, angles, oa, max_iters=10000, use_smooth=True)
            
            # Refine outer
            side, center, angle = _refine_outer(bc, ba)
            
            # Polish
            bc2, ba2, side2 = _polish(bc, ba, angle, center, side, iterations=500)
            
            _consider(bc2, ba2, side2, center, angle)
    
    # Final fine sweep on best configuration
    if best_centers is not None:
        side, center, angle = _refine_outer(best_centers, best_angles)
        # Final polish with the refined outer
        bc2, ba2, side2 = _polish(best_centers, best_angles, angle, center, side, iterations=1000)
        _consider(bc2, ba2, side2, center, angle)

    # Fallback
    if best_centers is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Format output
    inner = []
    for k in range(N):
        inner.append([float(best_centers[k][0]), float(best_centers[k][1]), float(math.degrees(best_angles[k]))])
    
    return inner, best_center, best_side, math.degrees(best_angle)
