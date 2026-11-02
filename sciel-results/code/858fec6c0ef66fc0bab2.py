"""Circle packing in unit square: 26 disjoint circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
EPS = 1e-8
DEADLINE = 145.0  # safety margin before 150s kill


def _bounds():
    lo = np.zeros(3 * N)
    hi = np.zeros(3 * N)
    for i in range(N):
        lo[3*i] = 0.0
        hi[3*i] = 1.0
        lo[3*i+1] = 0.0
        hi[3*i+1] = 1.0
        lo[3*i+2] = 1e-6
        hi[3*i+2] = 0.5
    return (lo, hi)


def _enforce_validity(params):
    """Enforce all constraints: walls and pairwise non-overlap by shrinking radii."""
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    # Iterative overlap resolution
    for _ in range(100):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        
        # Shrink the larger radius in overlapping pairs, or split if equal
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            # Simple heuristic: shrink the one with larger radius, or both if close
            if p[i, 2] >= p[j, 2]:
                shrink_i = min(ov * 0.5, p[i, 2] - 1e-6)
                shrink_j = min(ov - shrink_i, p[j, 2] - 1e-6)
                p[i, 2] -= shrink_i
                p[j, 2] -= shrink_j
            else:
                shrink_j = min(ov * 0.5, p[j, 2] - 1e-6)
                shrink_i = min(ov - shrink_j, p[i, 2] - 1e-6)
                p[j, 2] -= shrink_j
                p[i, 2] -= shrink_i
        
        # Re-check walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    return p


def _score_valid(p):
    """Compute sum of radii for a valid configuration."""
    return np.sum(p[:, 2])


def _hex_initial(seed):
    """Hexagonal grid initialization."""
    rng = np.random.RandomState(seed)
    points = []
    # Generate a 7x7 hex grid
    for i in range(7):
        for j in range(7):
            x = i
            y = j * 1.5 + (0.5 if i % 2 else 0)
            points.append((x, y))
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    max_coord = np.max(np.abs(points))
    scale = 0.45 / max_coord
    points *= scale
    points += 0.5
    
    # Select N points closest to center
    dists = np.sum((points - 0.5)**2, axis=1)
    indices = np.argsort(dists)[:N]
    
    p = points[indices]
    radii = np.full(N, 0.05)
    
    params = np.empty(3 * N)
    params[0::3] = p[:, 0]
    params[1::3] = p[:, 1]
    params[2::3] = radii
    
    params += rng.uniform(-1e-4, 1e-4, size=params.size)
    return params


def _grid_initial(seed):
    """Grid of equal circles with small jitter."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4))
    cells = cells[:N]
    arr = np.array(cells, dtype=float).reshape(-1)
    arr += rng.uniform(-1e-5, 1e-5, size=arr.size)
    return arr

def _random_initial(seed):
    """Random initialization with corner bias."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.1, 0.9, size=(N, 2))
    # Bias some towards corners
    corners = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
    for i in range(4):
        points[i] = corners[i] + rng.uniform(-0.05, 0.05, size=2)
    
    radii = np.full(N, 0.05)
    params = np.empty(3 * N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii
    return params


def _optimize_contact_graph(params, deadline):
    """
    Iteratively optimize using SLSQP on the Delaunay contact graph.
    """
    p = params.reshape(N, 3).copy()
    p = _enforce_validity(p)
    best_p = p.copy()
    best_score = _score_valid(p)
    
    lo, hi = _bounds()
    bounds_list = list(zip(lo, hi))
    
    max_iterations = 50
    for it in range(max_iterations):
        if time.time() > deadline - 5:
            break
            
        centers = p[:, :2]
        
        # Build Delaunay triangulation to find contact edges
        try:
            tri = Delaunay(centers)
            edge_set = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = simplex[i], simplex[j]
                        if a > b:
                            a, b = b, a
                        edge_set.add((a, b))
        except Exception:
            edge_set = set()
            
        if not edge_set:
            break
            
        edge_list = list(edge_set)
        eu = np.array([e[0] for e in edge_list])
        ev = np.array([e[1] for e in edge_list])
        m = len(eu)
        
        def obj(x):
            return -np.sum(x[2::3])
        
        def obj_grad(x):
            g = np.zeros_like(x)
            g[2::3] = -1.0
            return g
            
        def contact_func(x):
            xi = x[eu * 3]
            yi = x[eu * 3 + 1]
            ri = np.maximum(x[eu * 3 + 2], 1e-6)
            xj = x[ev * 3]
            yj = x[ev * 3 + 1]
            rj = np.maximum(x[ev * 3 + 2], 1e-6)
            dx = xi - xj
            dy = yi - yj
            dist = np.sqrt(dx * dx + dy * dy)
            return dist - (ri + rj)
            
        def contact_grad(x):
            jacobian = np.zeros((m, 3 * N))
            xi = x[eu * 3]
            yi = x[eu * 3 + 1]
            xj = x[ev * 3]
            yj = x[ev * 3 + 1]
            dx = xi - xj
            dy = yi - yj
            dist = np.sqrt(dx * dx + dy * dy)
            dist_safe = np.maximum(dist, 1e-10)
            ux = dx / dist_safe
            uy = dy / dist_safe
            
            for k in range(m):
                i = eu[k]
                j = ev[k]
                jacobian[k, 3*i] = ux[k]
                jacobian[k, 3*i+1] = uy[k]
                jacobian[k, 3*i+2] = -1.0
                jacobian[k, 3*j] = -ux[k]
                jacobian[k, 3*j+1] = -uy[k]
                jacobian[k, 3*j+2] = -1.0
            return jacobian
            
        def wall_func(x):
            xi = x[0::3]
            yi = x[1::3]
            ri = np.maximum(x[2::3], 1e-6)
            return np.minimum(np.minimum(xi - ri, 1.0 - xi - ri),
                              np.minimum(yi - ri, 1.0 - yi - ri))
            
        def wall_grad(x):
            jacobian = np.zeros((N, 3 * N))
            xi = x[0::3]
            yi = x[1::3]
            ri = np.maximum(x[2::3], 1e-6)
            
            for i in range(N):
                wx_left = xi[i] - ri[i]
                wx_right = 1.0 - xi[i] - ri[i]
                wy_bottom = yi[i] - ri[i]
                wy_top = 1.0 - yi[i] - ri[i]
                wmin = min(wx_left, wx_right, wy_bottom, wy_top)
                
                if wx_left <= wmin + 1e-12:
                    jacobian[i, 3*i] += 1.0
                    jacobian[i, 3*i+2] -= 1.0
                if wx_right <= wmin + 1e-12:
                    jacobian[i, 3*i] -= 1.0
                    jacobian[i, 3*i+2] -= 1.0
                if wy_bottom <= wmin + 1e-12:
                    jacobian[i, 3*i+1] += 1.0
                    jacobian[i, 3*i+2] -= 1.0
                if wy_top <= wmin + 1e-12:
                    jacobian[i, 3*i+1] -= 1.0
                    jacobian[i, 3*i+2] -= 1.0
            return jacobian
            
        constraints = [
            {'type': 'ineq', 'fun': contact_func, 'jac': contact_grad},
            {'type': 'ineq', 'fun': wall_func, 'jac': wall_grad}
        ]
        
        try:
            res = minimize(obj, p.reshape(-1), jac=obj_grad, method='SLSQP',
                           bounds=bounds_list, constraints=constraints,
                           options={'maxiter': 100, 'ftol': 1e-8})
            
            if hasattr(res, 'x'):
                new_p = _enforce_validity(res.x)
                new_score = _score_valid(new_p)
                
                if new_score > best_score:
                    best_score = new_score
                    best_p = new_p
                    p = new_p
                else:
                    # If no improvement, break to avoid cycling
                    break
        except Exception:
            break
            
    return best_p


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_params = None
    best_score = -np.inf

    initializers = [
        lambda s: _hex_initial(s),
        lambda s: _grid_initial(s),
        lambda s: _random_initial(s),
    ]

    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678,
             11, 22, 33, 44, 55, 66, 77, 88, 99, 100]
    
    for seed in seeds:
        if time.time() > deadline - 15:
            break
            
        for init_fn in initializers:
            if time.time() > deadline - 10:
                break
            
            params = init_fn(seed)
            params_valid = _enforce_validity(params)
            
            # Run the contact graph optimizer
            refined = _optimize_contact_graph(params_valid, deadline)
            refined_valid = _enforce_validity(refined)
            score = _score_valid(refined_valid)
            
            if score > best_score:
                best_score = score
                best_params = refined_valid.copy()

    if best_params is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_params
