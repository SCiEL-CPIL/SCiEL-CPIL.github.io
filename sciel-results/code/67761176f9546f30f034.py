"""Packing 50 circles: contact-graph assembly + soft-min SA + L-BFGS polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 50
HALF = 0.5
DEADLINE = 142.0  # safety margin before 150s kill


def _compute_r(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # boundary: min distance from each point to the square edge
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    # pairwise using cKDTree for efficiency
    tree = cKDTree(pts)
    pairs = tree.query_pairs(2.0 * r + 0.1)  # only check nearby pairs
    if pairs:
        min_pair = np.inf
        for i, j in pairs:
            dist = np.linalg.norm(pts[i] - pts[j])
            if dist < min_pair:
                min_pair = dist
        r = min(r, min_pair / 2.0)
    else:
        # If no pairs found within 2r+0.1, the min pair distance is > 2r+0.1
        # But we need the actual min pair distance to compute r correctly
        # This is a shortcut - let's just compute all pairs for safety
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        r = min(r, float(np.min(dist)) / 2.0)
    return r


def _compute_r_full(pts):
    """Compute radius using full pairwise distances (for accuracy)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r = min(r, float(np.min(dist)) / 2.0)
    return r


def _violation(pts, r_target):
    """Total violation: sum of pairwise overlaps + wall penetrations."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_viol = np.maximum(0.0, 2.0 * r_target - dist)
    iu = np.triu_indices(N, k=1)
    pair_viol = pair_viol[iu]
    
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    wall_viol = np.maximum(0.0, r_target - d)
    
    return float(np.sum(pair_viol) + np.sum(wall_viol))


def _soft_min_obj(pts, r_target, beta=500.0):
    """Smooth surrogate for the packing problem."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, k=1)
    pair_dist = dist[iu]
    pair_viol = 2.0 * r_target - pair_dist
    pair_viol_clipped = np.clip(pair_viol, -50, 50)
    soft_pair = np.log1p(np.exp(beta * pair_viol_clipped)) / beta
    
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    
    wall_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    wall_viol = r_target - wall_dist
    wall_viol_clipped = np.clip(wall_viol, -50, 50)
    soft_wall = np.log1p(np.exp(beta * wall_viol_clipped)) / beta
    
    return float(np.sum(soft_pair) + np.sum(soft_wall))


def _soft_min_grad(pts, r_target, beta=500.0):
    """Gradient of the soft-min objective with respect to pts."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    dist_safe = np.where(dist > 1e-12, dist, 1e-12)
    
    pair_viol = 2.0 * r_target - dist
    pair_viol_clipped = np.clip(pair_viol, -50, 50)
    sig = 1.0 / (1.0 + np.exp(-beta * pair_viol_clipped))
    
    grad = np.zeros_like(pts)
    for i in range(N):
        for j in range(i + 1, N):
            d = dist[i, j]
            if d < 1e-12:
                continue
            s = sig[i, j]
            diff_ij = (pts[i] - pts[j]) / d
            grad[i] -= s * diff_ij
            grad[j] += s * diff_ij
    
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    
    wall_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    wall_viol = r_target - wall_dist
    wall_viol_clipped = np.clip(wall_viol, -50, 50)
    sig_w = 1.0 / (1.0 + np.exp(-beta * wall_viol_clipped))
    
    for i in range(N):
        dl = d_left[i]
        dr = d_right[i]
        db = d_bottom[i]
        dt = d_top[i]
        wmin = wall_dist[i]
        s = sig_w[i]
        if abs(dl - wmin) < 1e-10:
            grad[i, 0] -= s
        if abs(dr - wmin) < 1e-10:
            grad[i, 0] += s
        if abs(db - wmin) < 1e-10:
            grad[i, 1] -= s
        if abs(dt - wmin) < 1e-10:
            grad[i, 1] += s
    
    return grad


def _contact_graph(pts, r_target, k=6):
    """Build a sparse contact graph: for each circle, find its k nearest neighbors."""
    tree = cKDTree(pts)
    graph = {}
    for i in range(N):
        _, indices = tree.query(pts[i], k=k + 1)  # +1 for self
        indices = indices[1:]  # exclude self
        graph[i] = set(indices.tolist())
    # Make symmetric
    for i in range(N):
        for j in graph[i]:
            if i not in graph[j]:
                graph[j].add(i)
    return graph


def _solve_contact_constraints(pts, r_target, graph, max_iter=100, tol=1e-10):
    """
    Iteratively adjust positions to satisfy distance constraints on the contact graph.
    For each edge (i,j) in the graph, if dist < 2*r_target, push apart.
    Also enforce wall constraints.
    """
    pts = pts.copy()
    for iteration in range(max_iter):
        moved = False
        # Pairwise constraints from contact graph
        edges = set()
        for i in range(N):
            for j in graph[i]:
                if i < j:
                    edges.add((i, j))
        
        for i, j in edges:
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 2.0 * r_target and dist > 1e-12:
                # Push apart
                correction = (2.0 * r_target - dist) / (2.0 * dist)
                pts[i] += correction * diff
                pts[j] -= correction * diff
                moved = True
        
        # Wall constraints
        for i in range(N):
            x, y = pts[i]
            # Left wall
            if x < -HALF + r_target:
                pts[i, 0] = -HALF + r_target
                moved = True
            # Right wall
            if x > HALF - r_target:
                pts[i, 0] = HALF - r_target
                moved = True
            # Bottom wall
            if y < -HALF + r_target:
                pts[i, 1] = -HALF + r_target
                moved = True
            # Top wall
            if y > HALF - r_target:
                pts[i, 1] = HALF - r_target
                moved = True
        
        if not moved:
            break
    
    # Final clip
    pts = np.clip(pts, -HALF, HALF)
    return pts


def _hex_init(rng, jitter=0.0):
    """Hexagonal close-packed lattice initialization."""
    pts = []
    a = 0.12
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        while len(pts) < N and x < HALF - a + 1e-9:
            y = -HALF + a + row * dy
            if y < HALF - a + 1e-9:
                pts.append([x, y])
            x += dx
        row += 1
    
    pts = np.array(pts[:N])
    if jitter > 0:
        pts += rng.normal(0, jitter, size=pts.shape)
        pts = np.clip(pts, -HALF, HALF)
    return pts


def _grid_init(rng, jitter=0.0):
    """Grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:N]
    if jitter > 0:
        pts = pts.copy()
        pts += rng.normal(0, jitter, size=pts.shape)
        pts = np.clip(pts, -HALF, HALF)
    return pts


def _simulated_annealing(pts0, r_target, deadline, rng, n_steps=5000):
    """Simulated annealing to minimize violation for target radius r_target."""
    pts = pts0.copy()
    current_viol = _violation(pts, r_target)
    best_pts = pts.copy()
    best_viol = current_viol
    
    t_start = 0.04
    t_end = 0.00005
    cooling = (t_end / t_start) ** (1.0 / n_steps)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        temp = t_start * (cooling ** step)
        
        # Random move: pick 1-3 circles and perturb
        n_move = rng.integers(1, 4)
        indices = rng.choice(N, size=n_move, replace=False)
        deltas = rng.normal(0, temp, size=(n_move, 2))
        
        new_pts = pts.copy()
        new_pts[indices] += deltas
        new_pts = np.clip(new_pts, -HALF, HALF)
        
        new_viol = _violation(new_pts, r_target)
        
        delta = new_viol - current_viol
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            pts = new_pts
            current_viol = new_viol
            if current_viol < best_viol:
                best_viol = current_viol
                best_pts = pts.copy()
    
    return best_pts, best_viol


def _lbfgs_polish(pts0, r_target, deadline, beta=500.0, maxiter=300):
    """Use L-BFGS-B to minimize the soft-min objective for target radius."""
    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return _soft_min_obj(p, r_target, beta)
    
    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        g = _soft_min_grad(p, r_target, beta)
        return g.ravel()
    
    x0 = pts0.ravel()
    try:
        res = minimize(obj, x0, method='L-BFGS-B',
                       jac=grad,
                       bounds=[(-HALF, HALF)] * (2 * N),
                       options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-10})
        pts = res.x.reshape(N, 2)
    except Exception:
        pts = pts0.copy()
    pts = np.clip(pts, -HALF, HALF)
    return pts


def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # Multi-start initializations
    inits = [
        _hex_init(rng, jitter=0.0),
        _hex_init(rng, jitter=0.02),
        _grid_init(rng, jitter=0.0),
        _grid_init(rng, jitter=0.02),
    ]
    
    # Evaluate initial radii
    init_radii = [_compute_r_full(pts) for pts in inits]
    
    best_pts = inits[0].copy()
    best_r = init_radii[0]
    for i, (pts, r) in enumerate(zip(inits, init_radii)):
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    # Binary search on radius with SA + contact graph + L-BFGS refinement
    r_lo = best_r
    r_hi = 0.075  # upper bound slightly above known best
    
    # First, try to improve from best initial
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 20:
        # SA to improve from best init
        sa_pts, sa_viol = _simulated_annealing(best_pts, best_r, DEADLINE - 30, rng, n_steps=3000)
        sa_r = _compute_r_full(sa_pts)
        if sa_r > best_r:
            best_r = sa_r
            best_pts = sa_pts.copy()
    
    # Binary search for maximum feasible radius
    for _ in range(8):
        if time.time() > DEADLINE - 15:
            break
        
        r_mid = (r_lo + r_hi) / 2.0
        
        # Try to find a configuration with radius >= r_mid
        # Start from best_pts, run SA targeting r_mid
        deadline_inner = DEADLINE - 10
        sa_pts, sa_viol = _simulated_annealing(best_pts, r_mid, deadline_inner, rng, n_steps=2000)
        
        # Contact graph refinement
        if time.time() < deadline_inner - 5:
            graph = _contact_graph(sa_pts, r_mid, k=6)
            cg_pts = _solve_contact_constraints(sa_pts, r_mid, graph, max_iter=50)
            
            # L-BFGS polish
            if time.time() < deadline_inner - 3:
                lbfgs_pts = _lbfgs_polish(cg_pts, r_mid, deadline_inner - 2, beta=500.0, maxiter=100)
                final_pts = lbfgs_pts
            else:
                final_pts = cg_pts
        else:
            final_pts = sa_pts
        
        final_r = _compute_r_full(final_pts)
        
        if final_r >= r_mid - 1e-10:
            # Feasible, try higher
            r_lo = r_mid
            if final_r > best_r:
                best_r = final_r
                best_pts = final_pts.copy()
        else:
            # Not feasible, try lower
            r_hi = r_mid
    
    # Final polish with best known radius
    if time.time() < DEADLINE - 5:
        graph = _contact_graph(best_pts, best_r, k=6)
        cg_pts = _solve_contact_constraints(best_pts, best_r, graph, max_iter=100)
        lbfgs_pts = _lbfgs_polish(cg_pts, best_r, DEADLINE - 2, beta=800.0, maxiter=200)
        final_r = _compute_r_full(lbfgs_pts)
        if final_r > best_r:
            best_r = final_r
            best_pts = lbfgs_pts.copy()
    
    # Ensure validity
    best_pts = np.clip(best_pts, -HALF, HALF)
    if not np.all(np.isfinite(best_pts)):
        # Fallback to hex init
        best_pts = _hex_init(rng, jitter=0.0)
    
    return best_pts
