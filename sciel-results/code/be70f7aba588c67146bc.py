"""Circle packing in unit square n=32.

Maximize sum of radii of 32 disjoint circles inside [0,1]^2.
Strategy: hexagonal close-packing initialization + L-BFGS-B on a smooth
penalty objective (jointly optimizing positions and radii), multi-start,
then a tight final polish.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 32

# EVOLVE-BLOCK-START

def _pairwise(pos):
    """Vectorized pairwise squared distances and coordinate diffs.

    pos: (n, 2). Returns dist (n,n) and the diff arrays.
    """
    diff = pos[:, None, :] - pos[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 0.0)
    return d2, diff

def _objective_penalty(x, beta, gamma):
    """x = [x0,y0,x1,y1,...,xn-1,yn-1, r0,...,rn-1]  (2n + n = 3n).

    Objective = -sum(r)  +  beta * sum_{i<j} max(0, r_i+r_j - d_ij)^2
                +  gamma * boundary_penalties.
    We minimize this.
    """
    n = N
    pos = x[:2*n].reshape(n, 2)
    r = x[2*n:]
    d2, _ = _pairwise(pos)
    d = np.sqrt(d2)
    min_d = r[:, None] + r[None, :]
    # only off-diagonal
    overlap = np.maximum(0.0, min_d - d)
    # zero the diagonal (it's 0 anyway)
    np.fill_diagonal(overlap, 0.0)
    pen = beta * np.sum(overlap * overlap)
    # boundary penalties: circle must satisfy r <= x <= 1-r
    bx = np.maximum(0.0, r - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - r))
    by = np.maximum(0.0, r - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - r))
    bpen = gamma * (np.sum(bx * bx) + np.sum(by * by))
    return -np.sum(r) + pen + bpen

def _objective_grad(x, beta, gamma):
    n = N
    pos = x[:2*n].reshape(n, 2)
    r = x[2*n:]
    d2, diff = _pairwise(pos)
    d = np.sqrt(d2)
    min_d = r[:, None] + r[None, :]
    overlap = np.maximum(0.0, min_d - d)
    np.fill_diagonal(overlap, 0.0)
    # grad of penalty w.r.t. positions
    # dpen/dpos_i = 2*beta * sum_j overlap_ij * (pos_i - pos_j)/d_ij
    inv_d = np.zeros_like(d)
    mask = d > 1e-12
    inv_d[mask] = 1.0 / d[mask]
    # weight matrix W_ij = 2*beta*overlap_ij*inv_d_ij
    W = 2.0 * beta * overlap * inv_d
    W = W * 0.5  # symmetric; we'll sum over j for each i
    grad_pos = np.zeros((n, 2))
    # grad_pos_i = sum_j W_ij * (pos_i - pos_j)
    # = (W @ 1)_i * pos_i - sum_j W_ij * pos_j
    rowsum = W.sum(axis=1)
    grad_pos = rowsum[:, None] * pos - (W @ pos)
    # boundary gradient
    bx = np.maximum(0.0, r - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - r))
    by = np.maximum(0.0, r - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - r))
    # d/dpos
    dbx = np.zeros_like(pos[:, 0])
    dby = np.zeros_like(pos[:, 1])
    m1 = (pos[:, 0] < r)
    m2 = (pos[:, 0] > 1.0 - r)
    dbx[m1] = -2.0 * gamma * (r[m1] - pos[m1, 0])
    dbx[m2] = 2.0 * gamma * (pos[m2, 0] - (1.0 - r[m2]))
    m1 = (pos[:, 1] < r)
    m2 = (pos[:, 1] > 1.0 - r)
    dby[m1] = -2.0 * gamma * (r[m1] - pos[m1, 1])
    dby[m2] = 2.0 * gamma * (pos[m2, 1] - (1.0 - r[m2]))
    # gradient w.r.t. radii
    # from -sum(r): -1
    # from penalty: 2*beta * sum_j overlap_ij (d overlap_ij / dr_i = 1 when overlap>0)
    grad_r = -np.ones(n) + 2.0 * beta * overlap.sum(axis=1)
    # boundary: r contributes to (r - x) and (1 - r - ...) 
    # d/d r of max(0, r - x)^2 = 2(r-x) if r>x
    # d/d r of max(0, x - (1-r))^2 = 2(x-(1-r)) * d(1-r)/dr * (-1)?? careful
    # term A = max(0, r - x)^2 ; dA/dr = 2(r-x) if r>x
    # term B = max(0, x - (1-r))^2 = max(0, x+ r -1)^2 ; dB/dr = 2(x+r-1) if x+r>1
    dbx_r = np.zeros_like(r)
    m1 = (r > pos[:, 0])
    m2 = (pos[:, 0] + r > 1.0)
    dbx_r[m1] += 2.0 * gamma * (r[m1] - pos[m1, 0])
    dbx_r[m2] += 2.0 * gamma * (pos[m2, 0] + r[m2] - 1.0)
    dby_r = np.zeros_like(r)
    m1 = (r > pos[:, 1])
    m2 = (pos[:, 1] + r > 1.0)
    dby_r[m1] += 2.0 * gamma * (r[m1] - pos[m1, 1])
    dby_r[m2] += 2.0 * gamma * (pos[m2, 1] + r[m2] - 1.0)
    grad_r += dbx_r + dby_r

    g = np.zeros(3 * n)
    g[:2*n] = grad_pos.reshape(-1)
    g[2*n:] = grad_r
    return g

def _extract(x):
    n = N
    pos = x[:2*n].reshape(n, 2)
    r = x[2*n:]
    return pos, r

def _is_valid(pos, r, tol=1e-8):
    # boundary
    if np.any(r < -tol):
        return False
    if np.any(pos[:, 0] < r - tol) or np.any(pos[:, 0] > 1.0 - r + tol):
        return False
    if np.any(pos[:, 1] < r - tol) or np.any(pos[:, 1] > 1.0 - r + tol):
        return False
    d2, _ = _pairwise(pos)
    d = np.sqrt(d2)
    min_d = r[:, None] + r[None, :]
    np.fill_diagonal(min_d, 0.0)
    if np.any(d < min_d - tol):
        return False
    return True

def _clean(pos, r, tol=1e-10):
    """Clamp to be exactly valid (shrink radii / nudge positions minimally)."""
    pos = pos.copy()
    r = r.copy()
    r = np.maximum(r, 0.0)
    # clamp positions
    for i in range(N):
        pos[i, 0] = np.clip(pos[i, 0], r[i], 1.0 - r[i])
        pos[i, 1] = np.clip(pos[i, 1], r[i], 1.0 - r[i])
    # resolve overlaps by shrinking radii slightly if needed
    for _ in range(200):
        d2, _ = _pairwise(pos)
        d = np.sqrt(d2)
        min_d = r[:, None] + r[None, :]
        np.fill_diagonal(min_d, 0.0)
        worst = np.max(np.where(np.eye(N, dtype=bool), 0.0, min_d - d))
        if worst <= tol:
            break
        # shrink the radii that are involved in worst overlap
        idx = np.unravel_index(np.argmax(np.where(np.eye(N, dtype=bool), 0.0, min_d - d)), (N, N))
        i, j = idx
        # reduce both radii by half the overlap
        dec = max(worst, 0.0) / 2.0 + 1e-9
        r[i] = max(0.0, r[i] - dec)
        r[j] = max(0.0, r[j] - dec)
        pos[i, 0] = np.clip(pos[i, 0], r[i], 1.0 - r[i])
        pos[i, 1] = np.clip(pos[i, 1], r[i], 1.0 - r[i])
        pos[j, 0] = np.clip(pos[j, 0], r[j], 1.0 - r[j])
        pos[j, 1] = np.clip(pos[j, 1], r[j], 1.0 - r[j])
    return pos, r

def _hex_initialization():
    """Hexagonal close packing of equal circles, then let radii grow.

    Build a hex lattice of N circles that fits in the square, equal radii,
    then return a feasible starting point.
    """
    # Hex lattice: rows offset. Choose spacing so it fits.
    # We'll place circles on a triangular lattice and scale to fit [0,1]^2.
    # Generate a triangular lattice of points, keep N, scale to fit.
    pts = []
    # triangular lattice with unit spacing in x, sqrt(3)/2 in y, offset rows
    L = 8
    for row in range(L):
        for col in range(L):
            x = col + (0.5 if row % 2 else 0.0)
            y = row * np.sqrt(3.0) / 2.0
            pts.append([x, y])
    pts = np.array(pts)
    # take first N (or a centered subset). We'll just take N.
    pts = pts[:N]
    # scale to fit in [0,1]^2 with margin
    # find the spacing: nearest neighbor distance in this lattice is 1.0
    # We want radius r such that 2r <= min spacing and circles fit.
    # Scale positions to fill the square.
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    w = xmax - xmin
    h = ymax - ymin
    scale = min(1.0 / max(w, 1e-9), 1.0 / max(h, 1e-9))
    # But we want to use the full square: scale so the bounding box fits [0,1]
    pts = (pts - np.array([xmin, ymin])) * scale
    # center in the square
    cx = (pts[:, 0].max() - pts[:, 0].min()) / 2.0
    cy = (pts[:, 1].max() - pts[:, 1].min()) / 2.0
    pts[:, 0] += 0.5 - cx
    pts[:, 1] += 0.5 - cy
    # radius: half of nearest neighbor distance
    d2, _ = _pairwise(pts)
    np.fill_diagonal(d2, np.inf)
    dmin = np.sqrt(d2.min())
    r = 0.5 * dmin
    # ensure fits in boundary
    r = min(r, pts[:, 0].min(), pts[:, 0].max() and (1.0 - pts[:, 0].max()),
            pts[:, 1].min(), 1.0 - pts[:, 1].max())
    r = max(r, 1e-4)
    pos = pts.copy()
    r = np.full(N, r)
    return pos, r

def _square_grid_initialization():
    """Square grid of equal circles (fallback / diversity start)."""
    side = int(np.ceil(np.sqrt(N)))
    pos = np.zeros((N, 2))
    k = 0
    for i in range(side):
        for j in range(side):
            if k >= N:
                break
            pos[k, 0] = (i + 0.5) / side
            pos[k, 1] = (j + 0.5) / side
            k += 1
    spacing = 1.0 / side
    r = 0.5 * spacing
    r = np.full(N, r)
    return pos, r

def _staggered_grid_initialization():
    """Staggered (brick) grid: better than square for equal circles."""
    # Use rows with offset
    n_rows = 6
    per_row = [6, 5, 6, 5, 6, 4]
    pos = np.zeros((N, 2))
    k = 0
    row_h = 1.0 / n_rows
    for ri, cnt in enumerate(per_row):
        if k >= N:
            break
        y = (ri + 0.5) * row_h
        offset = (0.5 * row_h) if ri % 2 else 0.0
        # horizontal spacing
        spacing = 1.0 / cnt
        for ci in range(cnt):
            if k >= N:
                break
            x = (ci + 0.5) * spacing
            pos[k, 0] = x
            pos[k, 1] = y
            k += 1
    # radius: limited by both horizontal and vertical/offset spacing
    # compute min pairwise distance
    d2, _ = _pairwise(pos)
    np.fill_diagonal(d2, np.inf)
    dmin = np.sqrt(d2.min())
    r = 0.5 * dmin
    r = min(r, pos[:, 0].min(), 1.0 - pos[:, 0].max(), pos[:, 1].min(), 1.0 - pos[:, 1].max())
    r = max(r, 1e-4)
    r = np.full(N, r)
    return pos, r

def _make_x(pos, r):
    n = N
    x = np.zeros(3 * n)
    x[:2*n] = pos.reshape(-1)
    x[2*n:] = r
    return x

def _optimize_start(pos, r, deadline, beta0=10.0, gamma0=50.0):
    """Run L-BFGS-B from a start, annealing beta down, returning best (pos,r,score)."""
    n = N
    x = _make_x(pos, r)
    # bounds: positions in [0,1], radii in [0, 0.5]
    lb = np.zeros(3 * n)
    ub = np.ones(3 * n)
    ub[2*n:] = 0.5
    # also bound positions by radius is implicit via penalty; keep simple bounds
    best_score = np.sum(r)
    best_x = x.copy()
    best_pos, best_r = pos.copy(), r.copy()
    if not _is_valid(best_pos, best_r):
        best_pos, best_r = _clean(best_pos, best_r)
        best_score = np.sum(best_r)

    beta = beta0
    gamma = gamma0
    # anneal beta down over a few phases
    phases = [
        (beta0, gamma0, 40),
        (beta0 * 0.5, gamma0, 60),
        (beta0 * 0.2, gamma0, 80),
        (beta0 * 0.1, gamma0 * 2, 100),
        (beta0 * 0.05, gamma0 * 5, 120),
    ]
    for b, g, iters in phases:
        if time.time() > deadline - 5.0:
            break
        def obj(xx, b=b, g=g):
            return _objective_penalty(xx, b, g)
        def grad(xx, b=b, g=g):
            return _objective_grad(xx, b, g)
        try:
            res = minimize(obj, x, jac=grad, method='L-BFGS-B',
                           bounds=list(zip(lb, ub)),
                           options={'maxiter': iters, 'maxfun': iters * 3,
                                    'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            x = res.x
        except Exception:
            pass
        # evaluate
        p, rr = _extract(x)
        if _is_valid(p, rr):
            sc = np.sum(rr)
            if sc > best_score:
                best_score = sc
                best_x = x.copy()
                best_pos, best_r = p.copy(), rr.copy()
        else:
            p2, rr2 = _clean(p, rr)
            sc = np.sum(rr2)
            if sc > best_score:
                best_score = sc
                best_pos, best_r = p2, rr2
                # re-optimize from cleaned
                x = _make_x(p2, rr2)
    # final tight polish
    if time.time() < deadline - 4.0:
        b, g, iters = (beta0 * 0.02, gamma0 * 10, 200)
        def obj(xx):
            return _objective_penalty(xx, b, g)
        def grad(xx):
            return _objective_grad(xx, b, g)
        try:
            res = minimize(obj, best_x, jac=grad, method='L-BFGS-B',
                           bounds=list(zip(lb, ub)),
                           options={'maxiter': iters, 'maxfun': iters * 3,
                                    'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
            x = res.x
        except Exception:
            x = best_x
        p, rr = _extract(x)
        if _is_valid(p, rr):
            sc = np.sum(rr)
            if sc > best_score:
                best_score = sc
                best_pos, best_r = p.copy(), rr.copy()
        else:
            p2, rr2 = _clean(p, rr)
            sc = np.sum(rr2)
            if sc > best_score:
                best_score = sc
                best_pos, best_r = p2, rr2
    return best_pos, best_r, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    inits = [
        _hex_initialization,
        _staggered_grid_initialization,
        _square_grid_initialization,
    ]

    best_pos, best_r, best_score = None, None, -1.0

    # Multi-start with a couple of seeds each
    rng = np.random.default_rng(12345)
    for init_fn in inits:
        for trial in range(2):
            if time.time() > deadline - 20.0:
                break
            pos, r = init_fn()
            # add small random jitter for diversity on 2nd trial
            if trial > 0:
                pos = pos + rng.normal(0, 0.01, size=pos.shape)
                pos, r = _clean(pos, r)
            p, rr, sc = _optimize_start(pos, r, deadline)
            if sc > best_score:
                best_score = sc
                best_pos, best_r = p, rr

    # If still nothing valid (shouldn't happen), fallback to a guaranteed grid
    if best_pos is None or not _is_valid(best_pos, best_r):
        pos, r = _square_grid_initialization()
        pos, r = _clean(pos, r)
        best_pos, best_r = pos, r
        best_score = np.sum(r)

    # Final global polish from the best solution with very tight settings
    if time.time() < deadline - 6.0:
        p, rr, sc = _optimize_start(best_pos, best_r, deadline, beta0=5.0, gamma0=100.0)
        if sc > best_score:
            best_score = sc
            best_pos, best_r = p, rr

    # Ensure final validity
    best_pos, best_r = _clean(best_pos, best_r)

    result = np.zeros((N, 3))
    result[:, 0] = best_pos[:, 0]
    result[:, 1] = best_pos[:, 1]
    result[:, 2] = best_r
    return result

# EVOLVE-BLOCK-END
