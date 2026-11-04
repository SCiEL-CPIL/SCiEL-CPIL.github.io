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

# 12 separation directions (6 normals + 6 rotated by 30 deg)
# Used for a faster, approximate separation check in the objective
N12 = np.vstack([NORM, NORM @ np.array([[0.5, -S3 / 2.0], [S3 / 2.0, 0.5]])])

# unit hexagon vertices
UNIT = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])


def outer_side(C):
    """Smallest outer hexagon (centered at origin, rot 0) side length enclosing all hexagons."""
    m = 0.0
    for k in range(6):
        n = NORM[k]
        proj = C @ n + 1.0
        m = max(m, float(proj.max()))
    return 2.0 * m / S3


def separation_violation_fast(C, A):
    """
    Fast approximate separation check.
    Uses 12 fixed directions. This is an approximation of the true Minkowski sum
    separation which depends on relative angles. However, for local search,
    this gradient-like signal is often sufficient to push apart overlapping hexagons.
    """
    viol = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = C[i] - C[j]
            sep = np.abs(d @ N12.T).max()
            if sep < 2.0:
                viol += (2.0 - sep) ** 2
    return viol


def objective_flat(flat):
    C = flat[0::3].reshape(N, 2)
    A = flat[1::3]
    side = outer_side(C)
    viol = separation_violation_fast(C, A)
    return side + 100.0 * viol


def exact_nonoverlap(C, A, eps=1e-9):
    """Exact check: two convex polygons non-overlapping iff a separating axis exists.
    Use SAT on the 12 edge normals (union of both orientations)."""
    for i in range(N):
        for j in range(i + 1, N):
            # relative angle
            rel = (A[i] - A[j]) * np.pi / 180.0
            c, s = np.cos(rel), np.sin(rel)
            R = np.array([[c, -s], [s, c]])
            # normals of hexagon j in hexagon i's frame = NORM @ R
            n_j = NORM @ R
            axes = np.vstack([NORM, n_j])
            d = C[i] - C[j]
            sep = np.abs(d @ axes.T).max()
            if sep < 2.0 - eps:
                return False
    return True


def inside_outer(C, S, eps=1e-9):
    """All hexagons inside outer hexagon of side S centered at origin rot 0."""
    apothem = S * S3 / 2.0
    for k in range(6):
        n = NORM[k]
        proj = (C @ n) + 1.0
        if proj.max() > apothem + eps:
            return False
    return True


def valid(C, A, S):
    return exact_nonoverlap(C, A) and inside_outer(C, A, S)


def init_flower():
    C = [[0.0, 0.0]]
    for k in range(6):
        a = k * np.pi / 3
        C.append([S3 * np.cos(a), S3 * np.sin(a)])
    # 4 more in a ring at radius ~2*sqrt(3)
    for k in range(4):
        a = k * np.pi / 2 + np.pi / 6
        C.append([2.0 * S3 * np.cos(a), 2.0 * S3 * np.sin(a)])
    C = np.array(C[:N])
    A = np.zeros(N)
    return C, A


def init_ring():
    C = [[0.0, 0.0]]
    for k in range(10):
        a = k * 2 * np.pi / 10
        C.append([2.2 * S3 * np.cos(a), 2.2 * S3 * np.sin(a)])
    C = np.array(C[:N])
    A = np.zeros(N)
    return C, A


def init_lattice():
    # 3-4-3-1 staggered lattice rows
    C = []
    rows = [
        (3, 0.0),
        (4, 1.5),
        (3, 3.0),
        (1, 4.5),
    ]
    for count, y in rows:
        for i in range(count):
            x = (i - (count - 1) / 2.0) * S3
            C.append([x, y])
    C = np.array(C[:N])
    A = np.zeros(N)
    return C, A


def init_staggered():
    C = []
    # two staggered rows of 6 and 5
    for i in range(6):
        C.append([(i - 2.5) * S3, 0.0])
    for i in range(5):
        C.append([(i - 2.0) * S3 + S3 / 2, 1.5])
    C = np.array(C[:N])
    A = np.zeros(N)
    return C, A


def init_random_cluster():
    rng = np.random.RandomState(42)
    C = rng.uniform(-1.5 * S3, 1.5 * S3, (N, 2))
    A = rng.uniform(-15, 15, N)
    return C, A


def optimize_nm(C, A, deadline, maxiter=20000):
    """Local polish with Nelder-Mead."""
    flat0 = np.empty(3 * N)
    flat0[0::3] = C[:, 0]
    flat0[1::3] = A
    flat0[2::3] = C[:, 1]
    try:
        res = minimize(objective_flat, flat0, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-6,
                                "adaptive": True})
    except Exception:
        return C, A
    f = res.x
    Cn = np.column_stack([f[0::3], f[2::3]])
    An = f[1::3]
    return Cn, An


def rigid_objective(params, base_C, base_A):
    """
    Objective for rigid body optimization.
    params: [theta, tx, ty]
    base_C, base_A: The local coordinates and angles of the cluster.
    We rotate the whole cluster by theta and translate by tx, ty.
    """
    theta, tx, ty = params
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    
    # Rotate centers
    C_rot = base_C @ R.T
    C_new = C_rot + np.array([tx, ty])
    
    # Angles are rotated by theta as well
    A_new = base_A + np.degrees(theta)
    
    side = outer_side(C_new)
    # We can skip violation check here if we assume the base cluster was valid,
    # as rigid transformation preserves internal distances.
    # However, due to floating point or if base was slightly invalid, we add a small penalty.
    viol = separation_violation_fast(C_new, A_new)
    return side + 10.0 * viol


def optimize_rigid(C, A, deadline, maxiter=5000):
    """
    Optimize rotation and translation of a rigid cluster.
    This is a 3D optimization, much faster and less prone to local minima 
    regarding the global alignment.
    """
    # Center the cluster at origin for better optimization behavior
    centroid = C.mean(axis=0)
    base_C = C - centroid
    base_A = A.copy()
    
    best_params = [0.0, 0.0, 0.0]
    best_val = rigid_objective(best_params, base_C, base_A)
    
    # Try multiple initial angles to escape local minima
    # The hexagon has 6-fold symmetry, so 0 to 60 deg is sufficient, but we sample more.
    angles_to_try = [0, np.pi/12, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]
    
    for theta0 in angles_to_try:
        if time.time() > deadline - 5:
            break
        params0 = [theta0, 0.0, 0.0]
        try:
            res = minimize(rigid_objective, params0, args=(base_C, base_A),
                          method='Nelder-Mead',
                          options={'maxiter': maxiter, 'xatol': 1e-7, 'fatol': 1e-7,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    theta, tx, ty = best_params
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    C_final = base_C @ R.T + np.array([tx, ty])
    A_final = base_A + np.degrees(theta)
    
    return C_final, A_final


def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin

    inits = [
        init_flower(), 
        init_ring(), 
        init_lattice(), 
        init_staggered(), 
        init_random_cluster()
    ]

    best_C = None
    best_A = None
    best_side = float('inf')

    # Phase 1: Local Search on diverse inits
    for C, A in inits:
        if time.time() > deadline - 30:
            break
        
        # Polish with NM
        Cn, An = optimize_nm(C, A, deadline, maxiter=10000)
        Sn = outer_side(Cn, An)
        
        # Check validity
        if valid(Cn, An, Sn):
            if Sn < best_side:
                best_side = Sn
                best_C = Cn
                best_A = An
        
        # Also check the initial state
        Si = outer_side(C, A)
        if valid(C, A, Si):
            if Si < best_side:
                best_side = Si
                best_C = C
                best_A = A

    # Phase 2: Rigid Body Optimization
    # This is the structural improvement. It treats the best-found cluster as a single object
    # and optimizes its global placement.
    if best_C is not None and time.time() < deadline - 10:
        C_rig, A_rig = optimize_rigid(best_C, best_A, deadline, maxiter=10000)
        S_rig = outer_side(C_rig, A_rig)
        if valid(C_rig, A_rig, S_rig):
            if S_rig < best_side:
                best_side = S_rig
                best_C = C_rig
                best_A = A_rig

    # Fallback if nothing valid found
    if best_C is None:
        # Use the most robust init and force it valid by spreading out if needed
        C, A = init_flower()
        Cn, An = optimize_nm(C, A, deadline, maxiter=30000)
        Sn = outer_side(Cn, An)
        if not valid(Cn, An, Sn):
            # Fallback: spread out widely (valid but large)
            Cn = np.zeros((N, 2))
            for i in range(N):
                Cn[i] = [6.0 * i, 0.0]
            An = np.zeros(N)
            Sn = outer_side(Cn, An)
        best_C = Cn
        best_A = An
        best_side = Sn

    inner = [[float(best_C[i, 0]), float(best_C[i, 1]), float(best_A[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(best_side), 0.0
