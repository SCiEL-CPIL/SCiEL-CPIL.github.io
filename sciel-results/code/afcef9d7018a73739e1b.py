"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

# Vertices of the equilateral triangle with area 1: (0,0), (1,0), (0.5, sqrt(3)/2)
_V1 = np.array([0.0, 0.0])
_V2 = np.array([1.0, 0.0])
_V3 = np.array([0.5, np.sqrt(3) / 2.0])
_AREA = np.sqrt(3) / 4.0  # area of the equilateral triangle

def _barycentric_to_cartesian(u, v):
    """Convert barycentric coords (u, v) to Cartesian. w = 1-u-v.
    Point = u*V1 + v*V2 + w*V3.
    Valid when u>=0, v>=0, u+v<=1."""
    w = 1.0 - u - v
    x = v * _V2[0] + w * _V3[0]
    y = v * _V2[1] + w * _V3[1]
    return x, y

def _min_tri_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points.
    points: array of shape (n, 2) in Cartesian coords."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(pj-pi, pk-pi)|
                dx1 = points[j][0] - points[i][0]
                dy1 = points[j][1] - points[i][1]
                dx2 = points[k][0] - points[i][0]
                dy2 = points[k][1] - points[i][1]
                area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                if area < min_area:
                    min_area = area
    return min_area

def _objective(barycentric_params):
    """Objective for differential_evolution: minimize negative min area.
    barycentric_params: flat array of 2*N values (u, v) for each point."""
    points = np.zeros((N, 2))
    for i in range(N):
        u = barycentric_params[2 * i]
        v = barycentric_params[2 * i + 1]
        x, y = _barycentric_to_cartesian(u, v)
        points[i] = [x, y]
    area = _min_tri_area(points)
    return -area

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # Bounds in barycentric space: u in [0,1], v in [0, 1-u]
    # But differential_evolution needs rectangular bounds.
    # We'll use u in [0, 1], v in [0, 1] and handle constraint via penalty or clipping.
    # Better: use a transformation. Let u = a, v = b*(1-a) where a,b in [0,1].
    # Then u+v = a + b*(1-a) = a + b - ab. For this to be <= 1, we need a + b - ab <= 1.
    # a + b - ab = 1 - (1-a)(1-b) <= 1 always. So u+v <= 1 automatically.
    # u = a, v = b*(1-a). Both in [0,1].
    
    def _transform(params):
        """Convert from (a, b) in [0,1]^2 to (u, v) in triangle."""
        u = np.zeros(N)
        v = np.zeros(N)
        for i in range(N):
            a = params[2 * i]
            b = params[2 * i + 1]
            u[i] = a
            v[i] = b * (1.0 - a)
        return u, v

    def _obj_transformed(params):
        u, v = _transform(params)
        points = np.zeros((N, 2))
        for i in range(N):
            x = v[i] * _V2[0] + (1.0 - u[i] - v[i]) * _V3[0]
            y = v[i] * _V2[1] + (1.0 - u[i] - v[i]) * _V3[1]
            points[i] = [x, y]
        area = _min_tri_area(points)
        return -area

    bounds = [(0.0, 1.0)] * (2 * N)

    # Run differential evolution
    best_result = None
    try:
        result = differential_evolution(
            _obj_transformed,
            bounds=bounds,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=True,
            callback=None
        )
        if result.success or result.fun < 0:
            best_result = result.x
    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        # Return a reasonable default
        if best_result is not None:
            u, v = _transform(best_result)
            points = np.zeros((N, 2))
            for i in range(N):
                x = v[i] * _V2[0] + (1.0 - u[i] - v[i]) * _V3[0]
                y = v[i] * _V2[1] + (1.0 - u[i] - v[i]) * _V3[1]
                points[i] = [x, y]
            return points
        # Fallback: random points
        rng = np.random.RandomState(7)
        points = []
        while len(points) < N:
            a = rng.random()
            b = rng.random()
            u = a
            v = b * (1.0 - a)
            x = v * _V2[0] + (1.0 - u - v) * _V3[0]
            y = v * _V2[1] + (1.0 - u - v) * _V3[1]
            points.append([x, y])
        return np.array(points)

    if best_result is not None:
        u, v = _transform(best_result)
        points = np.zeros((N, 2))
        for i in range(N):
            x = v[i] * _V2[0] + (1.0 - u[i] - v[i]) * _V3[0]
            y = v[i] * _V2[1] + (1.0 - u[i] - v[i]) * _V3[1]
            points[i] = [x, y]
        return points

    # Fallback
    rng = np.random.RandomState(7)
    points = []
    while len(points) < N:
        a = rng.random()
        b = rng.random()
        u = a
        v = b * (1.0 - a)
        x = v * _V2[0] + (1.0 - u - v) * _V3[0]
        y = v * _V2[1] + (1.0 - u - v) * _V3[1]
        points.append([x, y])
    return np.array(points)
# EVOLVE-BLOCK-END
