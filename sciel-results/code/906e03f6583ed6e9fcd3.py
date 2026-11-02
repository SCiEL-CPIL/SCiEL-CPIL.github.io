"""Circle packing in unit square: 26 circles, maximize sum of radii.
Strategy: Delaunay contact-graph iterative solver.
- Maintain a Delaunay triangulation of circle centers.
- Solve a small QP (SLSQP) on only the contact-graph edges + boundary constraints.
- Iterate: solve QP -> update positions/radii -> re-triangulate -> repeat.
- Multiple diverse initializations + final polish.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
EPS = 1e-12
DEADLINE = 140.0  # Safety margin for 150s limit


def _max_radius(pos, r):
    """Maximal radius for each circle given current positions and radii."""
    n = len(r)
    mr = np.empty(n)
    for i in range(n):
        dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        for j in range(n):
            if i != j:
                d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                dmin = min(dmin, d - r[j])
        mr[i] = max(0.0, dmin)
    return mr


def _delaunay_edges(pos):
    """Return set of unique edges (i,j) with i<j from Delaunay triangulation."""
    try:
        tri = Delaunay(pos)
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = simplex[a], simplex[b]
                    if i > j:
                        i, j = j, i
                    edges.add((i, j))
        return edges
    except Exception:
        # Fallback: connect each point to its 3 nearest neighbors
        edges = set()
        n = len(pos)
        for i in range(n):
            dists = np.array([np.hypot(pos[i,0]-pos[j,0], pos[i,1]-pos[j,1]) for j in range(n) if j != i])
            idx = np.argsort(dists)[:4]
            for j in idx:
                j = int(j)
                if j > i:
                    edges.add((i, j))
                else:
                    edges.add((j, i))
        return edges


def _solve_contact_qp(pos, rad, edges):
    """
    Solve: maximize sum(r) s.t.
      ||p_i - p_j|| >= r_i + r_j  for (i,j) in edges
      p_i in [r_i, 1-r_i] x [r_i, 1-r_i]
      r_i >= 0
    Using SLSQP on the small variable set.
    Returns updated (pos, rad) or None on failure.
    """
    n = N
    # Variables: [x0, y0, x1, y1, ..., x_{n-1}, y_{n-1}, r0, r1, ..., r_{n-1}]
    x0 = np.concatenate([pos.ravel(), rad])
    
    def obj(x):
        return -np.sum(x[2*n:])
    
    def obj_grad(x):
        g = np.zeros(3*n)
        g[2*n:] = -1.0
        return g
    
    # Inequality constraints: g(x) >= 0
    cons = []
    
    # Pair constraints for contact edges
    for (i, j) in edges:
        def pair_con(x, i=i, j=j):
            xi, yi = x[2*i], x[2*i+1]
            xj, yj = x[2*j], x[2*j+1]
            ri, rj = x[2*n+i], x[2*n+j]
            d = np.hypot(xi - xj, yi - yj)
            return d - ri - rj
        cons.append({'type': 'ineq', 'fun': pair_con})
    
    # Boundary constraints
    for i in range(n):
        def bound_con(x, i=i, axis=0, side=0):
            xi = x[2*i + axis]
            ri = x[2*n + i]
            if side == 0:  # xi - ri >= 0
                return xi - ri
            else:  # 1 - xi - ri >= 0
                return 1.0 - xi - ri
        cons.append({'type': 'ineq', 'fun': lambda x, i=i, a=0, s=0: x[2*i+a] - x[2*n+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i, a=0, s=1: 1.0 - x[2*i+a] - x[2*n+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i, a=1, s=0: x[2*i+a] - x[2*n+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i, a=1, s=1: 1.0 - x[2*i+a] - x[2*n+i]})
    
    bounds = [(0, 1)] * (2*n) + [(0, 0.5)] * n
    
    try:
        res = minimize(
            obj, x0, jac=obj_grad,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        if res.success or res.fun < -np.sum(rad):
            new_pos = res.x[:2*n].reshape(n, 2)
            new_rad = res.x[2*n:]
            # Clip to valid range
            new_rad = np.clip(new_rad, 0, 0.5)
            new_pos = np.clip(new_pos, 0, 1)
            return new_pos, new_rad
    except Exception:
        pass
    return None


def _polish(pos, rad, max_iter=50):
    """Final polish: fix positions, maximize radii exactly via iterative update."""
    for _ in range(max_iter):
        if time.time() > DEADLINE - 3:
            break
        rad_new = _max_radius(pos, rad)
        if np.allclose(rad_new, rad, atol=1e-12):
            rad = rad_new
            break
        rad = rad_new
    return pos, rad


def _validate(pos, rad):
    """Check all circles are in square and non-overlapping."""
    for i in range(N):
        if rad[i] <= 0:
            return False
        if pos[i,0] < rad[i] - 1e-9 or pos[i,0] > 1 - rad[i] + 1e-9:
            return False
        if pos[i,1] < rad[i] - 1e-9 or pos[i,1] > 1 - rad[i] + 1e-9:
            return False
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(pos[i,0]-pos[j,0], pos[i,1]-pos[j,1])
            if d < rad[i] + rad[j] - 1e-9:
                return False
    return True


def _score(pos, rad):
    """Sum of radii if valid, else -inf."""
    if not _validate(pos, rad):
        return -np.inf
    return float(np.sum(rad))


def _initial_grid(side, rng=None):
    """Grid initialization."""
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side))
    cells = cells[:N]
    pos = np.array(cells, dtype=float)
    if rng is not None:
        pos = pos + 0.03 * (rng.random((N, 2)) * 2 - 1)
        pos = np.clip(pos, 0.05, 0.95)
    rad = np.full(N, 0.5 / side - 1e-3)
    rad = _max_radius(pos, rad)
    return pos, rad


def _initial_hex(rng=None):
    """Hexagonal-like initialization."""
    side = int(np.ceil(np.sqrt(N)))
    pos_list = []
    for i in range(side):
        for j in range(side):
            if len(pos_list) >= N:
                break
            x = (j + 0.5 * (i % 2)) / side
            y = (i + 0.5) / side
            pos_list.append((x, y))
    pos = np.array(pos_list[:N], dtype=float)
    if rng is not None:
        pos = pos + 0.02 * (rng.random((N, 2)) * 2 - 1)
        pos = np.clip(pos, 0.05, 0.95)
    rad = np.full(N, 0.05)
    rad = _max_radius(pos, rad)
    return pos, rad


def _initial_random(rng):
    """Random initialization."""
    pos = rng.random((N, 2)) * 0.8 + 0.1
    rad = np.full(N, 0.03)
    rad = _max_radius(pos, rad)
    return pos, rad


def _initial_corner_weighted(rng):
    """Place some circles near corners/edges for better boundary utilization."""
    pos = np.zeros((N, 2))
    # 4 corners
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + 0.03*(rng.random()-0.5), cy + 0.03*(rng.random()-0.5)]
    # Edge midpoints
    edges = [(0.5, 0.08), (0.5, 0.92), (0.08, 0.5), (0.92, 0.5)]
    for i, (cx, cy) in enumerate(edges):
        pos[4+i] = [cx + 0.03*(rng.random()-0.5), cy + 0.03*(rng.random()-0.5)]
    # Remaining: random interior
    for i in range(8, N):
        pos[i] = rng.random(2) * 0.7 + 0.15
    pos = np.clip(pos, 0.05, 0.95)
    rad = np.full(N, 0.04)
    rad = _max_radius(pos, rad)
    return pos, rad


def _delaunay_optimize(pos, rad, max_rounds=20, rng=None):
    """Iterative Delaunay contact-graph QP solver."""
    best_pos, best_rad = pos.copy(), rad.copy()
    best_score = _score(pos, rad)
    
    for rnd in range(max_rounds):
        if time.time() > DEADLINE - 5:
            break
        
        # Get current contact graph
        edges = _delaunay_edges(pos)
        
        # Solve QP on contact graph
        result = _solve_contact_qp(pos, rad, edges)
        if result is None:
            # Try with slightly perturbed positions to escape local min
            if rng is not None and rnd % 3 == 2:
                pert = pos + 0.01 * (rng.random((N, 2)) * 2 - 1)
                pert = np.clip(pert, 0.02, 0.98)
                rad_p = _max_radius(pert, rad)
                result = _solve_contact_qp(pert, rad_p, _delaunay_edges(pert))
                if result is not None:
                    pos, rad = result
                continue
            else:
                break
        
        new_pos, new_rad = result
        
        # Update
        pos, rad = new_pos, new_rad
        
        # Score
        s = _score(pos, rad)
        if s > best_score:
            best_score = s
            best_pos, best_rad = pos.copy(), rad.copy()
    
    return best_pos, best_rad


def construct():
    t_start = time.time()
    rng = np.random.default_rng(12345)
    
    best_pos = None
    best_rad = None
    best_score = -1.0
    
    def consider(pos, rad):
        nonlocal best_pos, best_rad, best_score
        s = _score(pos, rad)
        if s > best_score:
            best_score = s
            best_pos = pos.copy()
            best_rad = rad.copy()
    
    # Generate diverse initializations
    inits = []
    
    # Grids
    for side in [4, 5, 6, 7]:
        pos, rad = _initial_grid(side, rng=None)
        inits.append((pos, rad))
        # Perturbed grids
        for k in range(3):
            if time.time() > DEADLINE - 30:
                break
            pos_p, rad_p = _initial_grid(side, rng=rng)
            inits.append((pos_p, rad_p))
    
    # Hex
    pos, rad = _initial_hex(rng=None)
    inits.append((pos, rad))
    for k in range(3):
        if time.time() > DEADLINE - 30:
            break
        pos_p, rad_p = _initial_hex(rng=rng)
        inits.append((pos_p, rad_p))
    
    # Random
    for k in range(5):
        if time.time() > DEADLINE - 30:
            break
        pos, rad = _initial_random(rng)
        inits.append((pos, rad))
    
    # Corner-weighted
    for k in range(3):
        if time.time() > DEADLINE - 30:
            break
        pos, rad = _initial_corner_weighted(rng)
        inits.append((pos, rad))
    
    # Run Delaunay optimization on each initialization
    for pos0, rad0 in inits:
        if time.time() > DEADLINE - 10:
            break
        
        # First do a quick polish to get valid starting point
        pos0, rad0 = _polish(pos0, rad0, max_iter=10)
        
        # Run Delaunay iterative solver
        pos_opt, rad_opt = _delaunay_optimize(pos0, rad0, max_rounds=15, rng=rng)
        
        # Final polish
        pos_opt, rad_opt = _polish(pos_opt, rad_opt, max_iter=30)
        
        consider(pos_opt, rad_opt)
    
    # If we have a best, do additional refinement rounds
    if best_pos is not None and time.time() < DEADLINE - 15:
        # Multiple refinement passes on the best solution
        for _ in range(3):
            if time.time() > DEADLINE - 5:
                break
            pos_ref, rad_ref = _delaunay_optimize(best_pos, best_rad, max_rounds=10, rng=rng)
            pos_ref, rad_ref = _polish(pos_ref, rad_ref, max_iter=30)
            consider(pos_ref, rad_ref)
    
    # Final polish on best
    if best_pos is not None:
        best_pos, best_rad = _polish(best_pos, best_rad, max_iter=50)
        s = _score(best_pos, best_rad)
        if s > best_score:
            best_score = s
    
    # Build output
    if best_pos is not None:
        out = np.column_stack([best_pos, best_rad])
        return out
    
    # Fallback
    pos, rad = _initial_grid(6, rng=None)
    pos, rad = _polish(pos, rad, max_iter=20)
    return np.column_stack([pos, rad])


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
