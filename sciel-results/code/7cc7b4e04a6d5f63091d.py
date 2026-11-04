import numpy as np
import time
from scipy.optimize import minimize

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
R30 = np.array([[0.5, -S3 / 2.0], [S3 / 2.0, 0.5]])
N12 = np.vstack([NORM, NORM @ R30])


def outer_side(C):
    """Smallest outer hexagon (centered at origin, rot 0) side length enclosing all hexagons."""
    m = 0.0
    for k in range(6):
        n = NORM[k]
        proj = C @ n + 1.0
        m = max(m, float(proj.max()))
    return 2.0 * m / S3


def separation_violation(C, A):
    """For each pair, the max separation along the 12 directions minus 2. Negative => overlap."""
    viol = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            rel = (A[i] - A[j]) * np.pi / 180.0
            c, s = np.cos(rel), np.sin(rel)
            R = np.array([[c, -s], [s, c]])
            n_j = NORM @ R
            axes = np.vstack([NORM, n_j])
            d = C[i] - C[j]
            sep = np.abs(d @ axes.T).max()
            if sep < 2.0:
                viol += (2.0 - sep) ** 2
    return viol


def objective_flat(flat):
    C = flat[0::3].reshape(N, 2)
    A = flat[1::3]
    side = outer_side(C)
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


def init_random_cluster(seed=0):
    rng = np.random.RandomState(seed)
    C = rng.uniform(-1.5 * S3, 1.5 * S3, (N, 2))
    A = rng.uniform(-15, 15, N)
    return C, A


def optimize(C, A, deadline, maxiter=20000):
    flat0 = np.empty(3 * N)
    flat0[0::3] = C[:, 0]
    flat0[1::3] = A
    flat0[2::3] = C[:, 1]
    try:
        res = minimize(objective_flat, flat0, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-8, "fatol": 1e-8,
                                "adaptive": True})
    except Exception:
        return C, A
    f = res.x
    Cn = np.column_stack([f[0::3], f[2::3]])
    An = f[1::3]
    return Cn, An


def coordinate_descent(C, A, deadline, n_sweeps=50):
    """Fine-grained coordinate descent with golden-section-like steps."""
    step = 0.5
    for sweep in range(n_sweeps):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(N):
            # Optimize x[i]
            best_x = C[i, 0]
            best_val = objective_flat(np.concatenate([C[:, 0], A, C[:, 1]]))
            for dx in [-step, step]:
                C[i, 0] += dx
                val = objective_flat(np.concatenate([C[:, 0], A, C[:, 1]]))
                if val < best_val:
                    best_val = val
                    best_x = C[i, 0]
                C[i, 0] -= dx
            C[i, 0] = best_x

            # Optimize y[i]
            best_y = C[i, 1]
            best_val = objective_flat(np.concatenate([C[:, 0], A, C[:, 1]]))
            for dy in [-step, step]:
                C[i, 1] += dy
                val = objective_flat(np.concatenate([C[:, 0], A, C[:, 1]]))
                if val < best_val:
                    best_val = val
                    best_y = C[i, 1]
                C[i, 1] -= dy
            C[i, 1] = best_y

            # Optimize angle[i]
            best_a = A[i]
            best_val = objective_flat(np.concatenate([C[:, 0], A, C[:, 1]]))
            for da in [-step, step]:
                A[i] += da
                val = objective_flat(np.concatenate([C[:, 0], A, C[:, 1]]))
                if val < best_val:
                    best_val = val
                    best_a = A[i]
                A[i] -= da
            A[i] = best_a

            if C[i, 0] != best_x or C[i, 1] != best_y or A[i] != best_a:
                improved = True

        if not improved:
            break
        step *= 0.5
    return C, A


def construct():
    start = time.time()
    deadline = start + 140.0

    inits = [
        init_flower(),
        init_ring(),
        init_lattice(),
        init_staggered(),
        init_random_cluster(0),
        init_random_cluster(1),
        init_random_cluster(2),
    ]

    best = None
    best_side = float('inf')

    # First pass: local optimization from multiple starts
    for C, A in inits:
        if time.time() > deadline - 30:
            break
        Cn, An = optimize(C, A, deadline, maxiter=10000)
        Sn = outer_side(Cn)
        if valid(Cn, An, Sn):
            if Sn < best_side:
                best_side = Sn
                best = (Cn.copy(), An.copy(), Sn)
        Si = outer_side(C)
        if valid(C, A, Si) and Si < best_side:
            best_side = Si
            best = (C.copy(), A.copy(), Si)

    # Second pass: coordinate descent polish on best solution
    if best is not None and time.time() < deadline - 10:
        Cn, An, Sn = best
        Cn2, An2 = coordinate_descent(Cn.copy(), An.copy(), deadline)
        Sn2 = outer_side(Cn2)
        if valid(Cn2, An2, Sn2) and Sn2 < best_side:
            best_side = Sn2
            best = (Cn2, An2, Sn2)

    # Third pass: refine with more Nelder-Mead iterations
    if best is not None and time.time() < deadline - 5:
        Cn, An, Sn = best
        Cn3, An3 = optimize(Cn, An, deadline, maxiter=20000)
        Sn3 = outer_side(Cn3)
        if valid(Cn3, An3, Sn3) and Sn3 < best_side:
            best_side = Sn3
            best = (Cn3, An3, Sn3)

    # Fallback
    if best is None:
        C, A = init_flower()
        Cn, An = optimize(C, A, deadline, maxiter=30000)
        Sn = outer_side(Cn)
        if not valid(Cn, An, Sn):
            Cn = np.zeros((N, 2))
            for i in range(N):
                Cn[i] = [6.0 * i, 0.0]
            An = np.zeros(N)
            Sn = outer_side(Cn)
        best = (Cn, An, Sn)

    Cn, An, Sn = best
    inner = [[float(Cn[i, 0]), float(Cn[i, 1]), float(An[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(Sn), 0.0
