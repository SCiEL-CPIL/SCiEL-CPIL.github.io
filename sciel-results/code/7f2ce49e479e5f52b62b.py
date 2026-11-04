import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
S3 = np.sqrt(3.0)

# 6 unit-hexagon edge normals (angles 0,60,...300)
NORM = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])


def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon with given center, angle, side."""
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    R = np.array([[ca, -sa], [sa, ca]])
    base = np.array([
        [1.0, 0.0],
        [0.5, S3 / 2.0],
        [-0.5, S3 / 2.0],
        [-1.0, 0.0],
        [-0.5, -S3 / 2.0],
        [0.5, -S3 / 2.0],
    ]) * side
    verts = base @ R.T + np.array([cx, cy])
    return verts


def outer_side_for_positions(C, A):
    """Smallest outer hexagon (centered at origin, rot 0) side length enclosing all hexagons."""
    m = 0.0
    for k in range(6):
        n = NORM[k]
        proj = C @ n + 1.0
        m = max(m, float(proj.max()))
    return 2.0 * m / S3


def check_non_overlap(C, A, eps=1e-9):
    """Exact SAT check for non-overlap of all pairs."""
    for i in range(N):
        for j in range(i + 1, N):
            rel = (A[i] - A[j]) * np.pi / 180.0
            c, s = np.cos(rel), np.sin(rel)
            R = np.array([[c, -s], [s, c]])
            n_j = NORM @ R
            axes = np.vstack([NORM, n_j])
            d = C[i] - C[j]
            sep = np.abs(d @ axes.T).max()
            if sep < 2.0 - eps:
                return False
    return True


def inside_outer(C, A, S, eps=1e-9):
    """All hexagons inside outer hexagon of side S centered at origin rot 0."""
    apothem = S * S3 / 2.0
    for k in range(6):
        n = NORM[k]
        proj = (C @ n) + 1.0
        if proj.max() > apothem + eps:
            return False
    return True


def valid_config(C, A, S):
    return check_non_overlap(C, A) and inside_outer(C, A, S)


def lattice_positions(lattice_coords):
    """Convert hexagonal lattice coords to Cartesian.
    Hex lattice: (q, r) -> (S3*(q + r/2), 1.5*r)
    Adjacent hexagons are distance sqrt(3) apart in x or 1.5 in y.
    """
    C = np.zeros((len(lattice_coords), 2))
    for i, (q, r) in enumerate(lattice_coords):
        C[i, 0] = S3 * (q + r / 2.0)
        C[i, 1] = 1.5 * r
    return C


# Define several candidate lattice arrangements for 11 hexagons
# Hex lattice adjacency: neighbors of (q,r) are (q+1,r),(q-1,r),(q,r+1),(q,r-1),(q+1,r-1),(q-1,r+1)

def arrangement_3431():
    """3-4-3-1 staggered rows"""
    coords = []
    # Row 0: 3 hexagons
    for i in range(3):
        coords.append((i - 1, 0))
    # Row 1: 4 hexagons
    for i in range(4):
        coords.append((i - 1.5, 1))  # offset by half
    # Row 2: 3 hexagons
    for i in range(3):
        coords.append((i - 1, 2))
    # Row 3: 1 hexagon
    coords.append((0, 3))
    return coords


def arrangement_compact():
    """Compact cluster: center + 6 ring + 4 more"""
    coords = []
    coords.append((0, 0))  # center
    # 6 neighbors
    for dq, dr in [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]:
        coords.append((dq, dr))
    # 4 more at distance 2
    for dq, dr in [(2,0),(-2,0),(0,2),(0,-2)]:
        coords.append((dq, dr))
    return coords[:11]


def arrangement_hex_cluster():
    """Hexagonal number arrangement: 1+6+... but we only need 11"""
    # Center (1) + first ring (6) + 4 from second ring
    coords = [(0, 0)]
    ring1 = [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]
    coords.extend(ring1)
    # Pick 4 from second ring that form a compact shape
    second_ring = [(2,0),(-2,0),(0,2),(0,-2),(2,-1),(-2,1),(1,1),(-1,-1)]
    # Choose 4 that minimize bounding box
    coords.extend(second_ring[:4])
    return coords


def arrangement_row_stagger():
    """Two rows: 6 and 5 staggered"""
    coords = []
    for i in range(6):
        coords.append((i - 2.5, 0))
    for i in range(5):
        coords.append((i - 2.0, 1))
    return coords


def arrangement_cross():
    """Cross/plus shape extended"""
    coords = []
    coords.append((0, 0))
    coords.append((1, 0))
    coords.append((-1, 0))
    coords.append((0, 1))
    coords.append((0, -1))
    coords.append((1, -1))
    coords.append((-1, 1))
    coords.append((2, 0))
    coords.append((-2, 0))
    coords.append((0, 2))
    coords.append((0, -2))
    return coords


def optimize_rotation(C_base, A_base, deadline=None):
    """Optimize global rotation to minimize outer hexagon side length."""
    def obj(theta_deg):
        theta = np.deg2rad(theta_deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        C_rot = C_base @ R.T
        A_rot = A_base + theta_deg
        side = outer_side_for_positions(C_rot, A_rot)
        if not check_non_overlap(C_rot, A_rot):
            return side + 100.0
        return side
    
    # Try multiple starting points with deadline guard
    best_side = float('inf')
    best_theta = 0.0
    starts = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
    for start in starts:
        if deadline is not None and time.time() > deadline:
            break
        try:
            res = minimize(obj, [start], method='Nelder-Mead',
                           options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
            if res.fun < best_side:
                best_side = res.fun
                best_theta = float(res.x[0])
        except Exception:
            continue
    
    return best_theta, best_side


def optimize_free(C0, A0, deadline, maxiter=5000):
    """Free Nelder-Mead optimization as fallback."""
    def objective(flat):
        C = np.column_stack([flat[0::3], flat[2::3]])
        A = flat[1::3]
        side = outer_side_for_positions(C, A)
        # Separation penalty
        viol = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = C[i] - C[j]
                rel = (A[i] - A[j]) * np.pi / 180.0
                c, s = np.cos(rel), np.sin(rel)
                R = np.array([[c, -s], [s, c]])
                axes = np.vstack([NORM, NORM @ R])
                sep = np.abs(d @ axes.T).max()
                if sep < 2.0:
                    viol += (2.0 - sep) ** 2
        return side + 50.0 * viol
    
    flat0 = np.empty(3 * N)
    flat0[0::3] = C0[:, 0]
    flat0[1::3] = A0
    flat0[2::3] = C0[:, 1]
    
    try:
        res = minimize(objective, flat0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-6,
                                'adaptive': True})
        f = res.x
        Cn = np.column_stack([f[0::3], f[2::3]])
        An = f[1::3]
        return Cn, An
    except Exception:
        return C0, A0


def construct():
    start = time.time()
    deadline = start + 140.0
    
    best = None
    best_side = float('inf')
    
    # Strategy 1: Lattice-based arrangements with global rotation optimization
    arrangements = [
        arrangement_3431(),
        arrangement_compact(),
        arrangement_hex_cluster(),
        arrangement_row_stagger(),
        arrangement_cross(),
    ]
    
    for arr in arrangements:
        if time.time() > deadline - 20:
            break
        
        C_base = lattice_positions(arr)
        A_base = np.zeros(N)
        
        # Center the configuration
        centroid = C_base.mean(axis=0)
        C_base = C_base - centroid
        
        # Optimize rotation with deadline guard
        theta_opt, side_opt = optimize_rotation(C_base, A_base, deadline=deadline - 5)
        
        if time.time() > deadline - 10:
            break
        
        # Apply optimal rotation
        theta = np.deg2rad(theta_opt)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        C_rot = C_base @ R.T
        A_rot = A_base + theta_opt
        
        Sn = outer_side_for_positions(C_rot, A_rot)
        
        if valid_config(C_rot, A_rot, Sn):
            if Sn < best_side:
                best_side = Sn
                best = (C_rot, A_rot, Sn)
        
        # Also try free optimization from this starting point
        if time.time() > deadline - 15:
            break
        
        Cn, An = optimize_free(C_rot, A_rot, deadline, maxiter=1500)
        Sn2 = outer_side_for_positions(Cn, An)
        
        if valid_config(Cn, An, Sn2):
            if Sn2 < best_side:
                best_side = Sn2
                best = (Cn, An, Sn2)
    
    # Strategy 2: Random restarts with free optimization
    if time.time() < deadline - 30:
        rng = np.random.RandomState(42)
        for trial in range(3):
            if time.time() > deadline - 20:
                break
            
            # Random compact cluster
            C0 = rng.uniform(-2.0, 2.0, (N, 2))
            A0 = rng.uniform(-30, 30, N)
            
            Cn, An = optimize_free(C0, A0, deadline, maxiter=1000)
            Sn = outer_side_for_positions(Cn, An)
            
            if valid_config(Cn, An, Sn):
                if Sn < best_side:
                    best_side = Sn
                    best = (Cn, An, Sn)
    
    # Fallback: if nothing valid found, use a widely spaced valid config
    if best is None:
        Cn = np.zeros((N, 2))
        for i in range(N):
            Cn[i] = [8.0 * i, 0.0]
        An = np.zeros(N)
        Sn = outer_side_for_positions(Cn, An)
        best = (Cn, An, Sn)
    
    Cn, An, Sn = best
    inner = [[float(Cn[i, 0]), float(Cn[i, 1]), float(An[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(Sn), 0.0
