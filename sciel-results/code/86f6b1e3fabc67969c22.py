"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 11
# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3.0) / 2.0])
VERTS = np.array([V0, V1, V2])
TRI_AREA = 0.5 * np.sqrt(3.0)  # area of the equilateral triangle with side 1


def _min_triangle_area(points):
    """Minimum area among all C(n,3) triangles formed by the points."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                a = 0.5 * abs((pj[0] - pi[0]) * (pk[1] - pi[1]) - (pk[0] - pi[0]) * (pj[1] - pi[1]))
                if a < min_a:
                    min_a = a
    return min_a


def _barycentric_to_cartesian(b):
    """Convert barycentric coords (n,3) to cartesian (n,2)."""
    return b[:, 0] * V0 + b[:, 1] * V1 + b[:, 2] * V2


def _cartesian_to_barycentric(pts):
    """Convert cartesian (n,2) to barycentric (n,3)."""
    # Solve: p = b0*V0 + b1*V1 + b2*V2, b0+b1+b2=1
    # Using V0 as origin: p = b1*(V1-V0) + b2*(V2-V0)
    # Matrix A = [V1-V0, V2-V0] as columns
    A = np.array([[V1[0] - V0[0], V2[0] - V0[0]],
                  [V1[1] - V0[1], V2[1] - V0[1]]])
    Ainv = np.linalg.inv(A)
    rel = pts - V0
    b12 = rel @ Ainv.T  # (n,2): [b1, b2]
    b = np.zeros((len(pts), 3))
    b[:, 1] = b12[:, 0]
    b[:, 2] = b12[:, 1]
    b[:, 0] = 1.0 - b12[:, 0] - b12[:, 1]
    return b


def _project_barycentric(b):
    """Project barycentric coords to the simplex (each row sums to 1, all >= 0)."""
    b = b.copy()
    for i in range(len(b)):
        row = b[i]
        # Simple projection: clip negatives and renormalize
        neg = row < 0
        if neg.any():
            row[neg] = 0
            s = row.sum()
            if s > 0:
                row = row / s
            else:
                row = np.array([1.0, 0.0, 0.0])
        b[i] = row
    return b


def _fps_init(n_points, rng, n_init=300):
    """Farthest point sampling to get diverse initial points inside triangle."""
    # Start with a random point
    def rand_point():
        u = rng.random()
        v = rng.random()
        if u + v > 1.0:
            u, v = 1.0 - u, 1.0 - v
        return np.array([u + 0.5 * v, (np.sqrt(3.0) / 2.0) * v])

    pts = [rand_point()]
    for _ in range(n_points - 1):
        # Find farthest candidate
        best_d = -1
        best_p = None
        for _ in range(50):
            cand = rand_point()
            d = min(np.linalg.norm(cand - p) for p in pts)
            if d > best_d:
                best_d = d
                best_p = cand
        pts.append(best_p)
    return np.array(pts)


def _optimize_from_start(start_pts, deadline, max_iter=500):
    """Run SLSQP optimization from a starting configuration. Returns (points, min_area)."""
    b = _cartesian_to_barycentric(start_pts)

    def objective(b_flat):
        bb = b_flat.reshape(-1, 3)
        pts = _barycentric_to_cartesian(bb)
        return -_min_triangle_area(pts)

    # Constraints: each point's barycentric coords sum to 1
    cons = []
    for i in range(N):
        cons.append({
            'type': 'eq',
            'fun': lambda b_flat, i=i: b_flat[3 * i] + b_flat[3 * i + 1] + b_flat[3 * i + 2] - 1.0,
            'jac': lambda b_flat, i=i: np.concatenate([np.zeros(3 * i),
                                                       np.ones(3),
                                                       np.zeros(3 * (N - i - 1))])
        })

    # Bounds: each barycentric coord in [0, 1]
    bounds = [(0.0, 1.0)] * (3 * N)

    t0 = time.time()
    result = minimize(
        objective,
        b.flatten(),
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': max_iter, 'ftol': 1e-10}
    )

    if time.time() > deadline:
        return start_pts, _min_triangle_area(start_pts)

    if result.success:
        pts = _barycentric_to_cartesian(result.x.reshape(-1, 3))
        # Project to ensure inside triangle
        pts = _project_cartesian(pts)
        return pts, _min_triangle_area(pts)
    else:
        return start_pts, _min_triangle_area(start_pts)


def _project_cartesian(pts):
    """Project cartesian points to inside the triangle."""
    b = _cartesian_to_barycentric(pts)
    b = _project_barycentric(b)
    return _barycentric_to_cartesian(b)


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    rng = np.random.RandomState(42)

    best_pts = None
    best_area = 0.0

    # Initial FPS-based configuration
    fps_pts = _fps_init(N, rng, n_init=N)
    best_pts = fps_pts.copy()
    best_area = _min_triangle_area(fps_pts)

    # Grid-based initial configurations for diversity
    # Create several diverse starting points
    start_configs = [fps_pts]

    # Regular grid in barycentric space
    for grid_n in [3, 4, 5]:
        grid_pts = []
        for i in range(grid_n):
            for j in range(grid_n):
                for k in range(grid_n):
                    if i + j + k == grid_n - 1:
                        b = np.array([i / (grid_n - 1), j / (grid_n - 1), k / (grid_n - 1)])
                        p = b[0] * V0 + b[1] * V1 + b[2] * V2
                        grid_pts.append(p)
        if len(grid_pts) >= N:
            # Randomly select N points
            idx = rng.choice(len(grid_pts), N, replace=False)
            start_configs.append(np.array([grid_pts[i] for i in idx]))

    # Random configurations
    for _ in range(5):
        rand_pts = np.array([
            np.array([u + 0.5 * v, (np.sqrt(3.0) / 2.0) * v])
            for u, v in [(rng.random(), rng.random()) for _ in range(N)]
            if u + v <= 1.0
        ])
        if len(rand_pts) == N:
            start_configs.append(rand_pts)

    # Optimize from each start configuration
    for config in start_configs:
        if time.time() > deadline - 10.0:
            break
        pts, area = _optimize_from_start(config, deadline, max_iter=300)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # If we still have time, do a few more random restarts
    if time.time() < deadline - 20.0:
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            rand_pts = np.array([
                np.array([u + 0.5 * v, (np.sqrt(3.0) / 2.0) * v])
                for u, v in [(rng.random(), rng.random()) for _ in range(N)]
                if u + v <= 1.0
            ])
            if len(rand_pts) != N:
                continue
            pts, area = _optimize_from_start(rand_pts, deadline, max_iter=200)
            if area > best_area:
                best_area = area
                best_pts = pts.copy()

    # Final refinement with more iterations if time permits
    if time.time() < deadline - 10.0 and best_pts is not None:
        pts, area = _optimize_from_start(best_pts, deadline, max_iter=500)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # Ensure all points are strictly inside (with small margin)
    if best_pts is not None:
        b = _cartesian_to_barycentric(best_pts)
        # Shrink slightly toward centroid to ensure interior
        centroid_b = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        b = 0.999 * b + 0.001 * centroid_b
        best_pts = _barycentric_to_cartesian(b)

    return np.array(best_pts)
