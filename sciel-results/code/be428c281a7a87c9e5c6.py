import numpy as np
import time
import scipy.optimize

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

# unit hexagon vertices
UNIT = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])


def outer_side_and_angle(C, A, outer_angle_deg=0.0):
    """Smallest outer hexagon (centered at origin, given rotation) side length enclosing all hexagons."""
    # For a regular hexagon centered at origin with side S and rotation phi:
    # facet normals at angles phi + k*60, apothem = S * sqrt(3)/2
    # A hexagon with center c and rotation a has vertices c + R(a)*UNIT
    # Max projection of any vertex onto normal n: dot(c, n) + max_dot_unit(n rotated by -a)
    # max_dot_unit for a unit hexagon in direction n is the support function
    # For a regular hexagon with side 1, support in direction n is 1 (the apothem is sqrt(3)/2 * ... )
    # Actually: for unit hexagon (side 1), the apothem is sqrt(3)/2.
    # The support function h(n) = max over vertices of dot(v, n).
    # For a regular hexagon with vertices at distance 1 from center (circumradius 1),
    # the support in direction at angle theta relative to a vertex direction is:
    # h(theta) = cos(theta mod 60) ... no, let me think more carefully.
    #
    # Unit hexagon: vertices at angles 0,60,120,180,240,300 degrees, radius 1.
    # Support function h(phi) = max_k cos(phi - k*pi/3) for phi in [0, 2pi)
    # This equals cos(phi mod pi/3) where phi is reduced to [0, pi/3].
    # Actually h(phi) = cos(min_k |phi - k*pi/3|) but that's not right either.
    # Let's just compute: for direction n at angle alpha, the max dot with unit hex vertices:
    # vertices at (cos(k*pi/3), sin(k*pi/3)) for k=0..5
    # dot = cos(alpha - k*pi/3), max over k
    # This is cos(delta) where delta = min_k |alpha - k*pi/3| (mod pi/3, take min)
    # Actually it's just: the hexagon's support function is h(alpha) = cos(alpha mod pi/3)
    # when alpha mod pi/3 is in [0, pi/3]. But cos is decreasing on [0, pi/3],
    # so h(alpha) = cos(alpha mod pi/3). Wait, that gives h(0)=1, h(pi/3)=0.5, h(pi/6)=cos(pi/6)=sqrt(3)/2.
    # But the apothem is sqrt(3)/2, which is h(pi/6). Yes that's correct.
    #
    # So for hexagon i with center C[i] and rotation A[i] (in degrees):
    # For outer normal at angle phi + k*pi/3:
    # support = dot(C[i], n_k) + h((phi + k*pi/3 - A[i]*pi/180) mod pi/3)
    # where h(theta) = cos(theta) for theta in [0, pi/3]
    
    phi = np.deg2rad(outer_angle_deg)
    m = 0.0
    for k in range(6):
        n_angle = phi + k * np.pi / 3
        n = np.array([np.cos(n_angle), np.sin(n_angle)])
        # For each inner hexagon, compute support in direction n
        # The inner hexagon i has rotation A[i] degrees
        # The relative angle between n and the hexagon's "vertex direction" is:
        # delta_i = (n_angle - A[i]*pi/180) mod (pi/3), reduced to [0, pi/3]
        deltas = (n_angle - np.deg2rad(A)) % (np.pi / 3)
        # Ensure in [0, pi/3]
        deltas = np.minimum(deltas, np.pi / 3 - deltas)  # this is wrong, let me think
        # Actually: (x mod pi/3) gives [0, pi/3). But we want the minimum angular distance to a multiple of pi/3.
        # delta = (n_angle - A_i_rad) mod (pi/3)
        # Then h = cos(delta) if delta <= pi/3, which it always is after mod.
        # But wait: if the angle difference is, say, 150 degrees, mod 60 gives 30 degrees. cos(30) = sqrt(3)/2. Correct.
        # If difference is 90 degrees, mod 60 = 30 degrees. cos(30) = sqrt(3)/2. But actual support at 90 degrees from a vertex:
        # The hexagon has edges between vertices. At 90 degrees from a vertex (which is 30 degrees from an edge normal),
        # the support is the apothem = sqrt(3)/2. Yes, cos(30 deg) = sqrt(3)/2. Correct.
        # If difference is 60 degrees, mod 60 = 0. cos(0) = 1. But 60 degrees from a vertex is another vertex. Correct.
        # If difference is 30 degrees, mod 60 = 30. cos(30) = sqrt(3)/2. Correct (apothem direction).
        # So h = cos(delta) where delta = (n_angle - A_i_rad) mod (pi/3).
        # But we need to be careful: mod gives [0, pi/3), so delta is in [0, pi/3).
        # cos is decreasing on [0, pi/3], so this is fine.
        
        support = C @ n + np.cos(deltas)
        m = max(m, float(support.max()))
    return 2.0 * m / S3, outer_angle_deg


def find_optimal_outer_angle(C, A):
    """Find the outer hexagon angle that minimizes the side length."""
    def side_at_angle(angle_deg):
        s, _ = outer_side_and_angle(C, A, angle_deg)
        return s
    
    # Coarse search
    angles = np.linspace(0, 60, 601, endpoint=False)
    sides = np.array([side_at_angle(a) for a in angles])
    best_idx = np.argmin(sides)
    best_angle = angles[best_idx]
    best_side = sides[best_idx]
    
    # Fine search
    fine_angles = np.linspace(best_angle - 0.5, best_angle + 0.5, 501)
    fine_sides = np.array([side_at_angle(a) for a in fine_angles])
    best_idx_fine = np.argmin(fine_sides)
    best_angle = fine_angles[best_idx_fine]
    best_side = fine_sides[best_idx_fine]
    
    # Local optimization
    try:
        res = scipy.optimize.minimize(
            side_at_angle, np.array([best_angle]), method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        if res.fun < best_side:
            best_angle = float(res.x[0])
            best_side = float(res.fun)
    except Exception:
        pass
    
    # Normalize to [0, 60)
    best_angle = best_angle % 60.0
    return best_side, best_angle


def separation_violation(C, A):
    """For each pair, the max separation along the 12 directions minus 2. Negative => overlap."""
    # Use 12 separation directions (6 normals + 6 rotated by 30 deg)
    N12 = np.vstack([NORM, NORM @ np.array([[0.5, -S3 / 2.0], [S3 / 2.0, 0.5]])])
    viol = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = C[i] - C[j]
            sep = np.abs(d @ N12.T).max()
            if sep < 2.0:
                viol += (2.0 - sep) ** 2
    return viol


def objective_inner(flat):
    """Objective for inner configuration only: outer side length (with optimal outer angle) + penalty."""
    C = flat[0::3].reshape(N, 2)
    A = flat[1::3]
    # Use a quick estimate of outer side (fix outer angle to 0 for speed in optimization)
    # The final answer will use the optimal outer angle
    m = 0.0
    for k in range(6):
        n = NORM[k]
        proj = C @ n + 1.0
        m = max(m, float(proj.max()))
    side = 2.0 * m / S3
    viol = separation_violation(C, A)
    return side + 100.0 * viol


def exact_nonoverlap(C, A, eps=1e-9):
    """Exact check: two convex polygons non-overlapping iff a separating axis exists."""
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


def inside_outer(C, A, S, outer_angle_deg, eps=1e-9):
    """All hexagons inside outer hexagon of side S centered at origin with given rotation."""
    phi = np.deg2rad(outer_angle_deg)
    apothem = S * S3 / 2.0
    for k in range(6):
        n_angle = phi + k * np.pi / 3
        n = np.array([np.cos(n_angle), np.sin(n_angle)])
        deltas = (n_angle - np.deg2rad(A)) % (np.pi / 3)
        support = C @ n + np.cos(deltas)
        if support.max() > apothem + eps:
            return False
    return True


def valid(C, A, S, outer_angle_deg):
    return exact_nonoverlap(C, A) and inside_outer(C, A, S, outer_angle_deg)


def init_rigid_cluster():
    """Tight triangular-lattice cluster with slight expansion for strict disjointness."""
    d = S3
    u = np.array([1.5, d / 2])
    v = np.array([0.0, d])
    lattice_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, -1), (2, -1), (-2, 1)
    ]
    C = np.array([i * u + j * v for i, j in lattice_coords])
    # Scale up slightly to ensure strict disjointness
    C *= 1.005
    A = np.zeros(N)
    return C, A


def init_flower():
    C = [[0.0, 0.0]]
    for k in range(6):
        a = k * np.pi / 3
        C.append([S3 * np.cos(a), S3 * np.sin(a)])
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
    for i in range(6):
        C.append([(i - 2.5) * S3, 0.0])
    for i in range(5):
        C.append([(i - 2.0) * S3 + S3 / 2, 1.5])
    C = np.array(C[:N])
    A = np.zeros(N)
    return C, A


def init_random_cluster():
    rng = np.random.RandomState(0)
    C = rng.uniform(-1.5 * S3, 1.5 * S3, (N, 2))
    A = rng.uniform(-15, 15, N)
    return C, A


def optimize_nm(C, A, deadline, maxiter=5000):
    """Local polish with Nelder-Mead on inner configuration."""
    flat0 = np.empty(3 * N)
    flat0[0::3] = C[:, 0]
    flat0[1::3] = A
    flat0[2::3] = C[:, 1]
    try:
        res = scipy.optimize.minimize(objective_inner, flat0, method="Nelder-Mead",
                                      options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-6,
                                               "adaptive": True})
    except Exception:
        return C, A
    f = res.x
    Cn = np.column_stack([f[0::3], f[2::3]])
    An = f[1::3]
    return Cn, An


def construct():
    start = time.time()
    deadline = start + 140.0

    inits = [
        init_rigid_cluster(),
        init_flower(),
        init_ring(),
        init_lattice(),
        init_staggered(),
        init_random_cluster(),
    ]

    best = None
    best_side = float('inf')

    # For each initialization: polish with NM, then find optimal outer angle
    for C, A in inits:
        if time.time() > deadline - 30:
            break
        
        # Polish inner configuration
        Cn, An = optimize_nm(C, A, deadline, maxiter=3000)
        
        # Find optimal outer angle for this inner configuration
        Sn, outer_angle = find_optimal_outer_angle(Cn, An)
        
        if valid(Cn, An, Sn, outer_angle):
            if Sn < best_side:
                best_side = Sn
                best = (Cn.copy(), An.copy(), Sn, outer_angle)
        
        # Also check the unpolished initial
        Si, oi = find_optimal_outer_angle(C, A)
        if valid(C, A, Si, oi) and Si < best_side:
            best_side = Si
            best = (C.copy(), A.copy(), Si, oi)

    # If nothing valid, fallback to a wide spread
    if best is None:
        Cn = np.zeros((N, 2))
        for i in range(N):
            Cn[i] = [6.0 * i, 0.0]
        An = np.zeros(N)
        Sn, outer_angle = find_optimal_outer_angle(Cn, An)
        if not valid(Cn, An, Sn, outer_angle):
            # Even wider
            for i in range(N):
                Cn[i] = [10.0 * i, 0.0]
            Sn, outer_angle = find_optimal_outer_angle(Cn, An)
        best = (Cn, An, Sn, outer_angle)

    Cn, An, Sn, outer_angle = best
    inner = [[float(Cn[i, 0]), float(Cn[i, 1]), float(An[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(Sn), float(outer_angle)
