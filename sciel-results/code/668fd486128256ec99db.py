"""Circle packing in perimeter-4 rectangle: contact-graph rigid-body enumeration."""
import numpy as np
import time
import itertools
import random
from scipy.optimize import minimize

N = 21
DEADLINE = 150.0


def _rigid_solve(graph, W, H, tol=1e-9, max_iter=80):
    """Solve the rigid-body system for a given contact graph.
    Returns (cx, cy, r) arrays or None on failure."""
    n = N
    # Unknowns: edge lengths for circle-circle and circle-corner contacts
    cc_edges = [(i, j) for i, j in graph['cc']]
    cn_edges = [(i, c) for i, c in graph['cn']]
    m = len(cc_edges) + len(cn_edges)
    if m < n:
        return None

    # Initial guess for edge lengths
    r_guess = 0.08
    L = np.empty(m)
    for k, (i, j) in enumerate(cc_edges):
        L[k] = 2.0 * r_guess
    for k, (i, c) in enumerate(cn_edges):
        L[k] = r_guess

    def residual(L):
        L = np.clip(L, 1e-4, None)
        # Build node positions: n circles + 4 corners
        # We'll do iterative relaxation of positions given fixed edge lengths
        # Corners fixed
        corners = np.array([[0.0, 0.0], [W, 0.0], [0.0, H], [W, H]])
        # Circle positions: start from grid
        cx = np.linspace(0.15, W - 0.15, n)
        cy = np.linspace(0.15, H - 0.15, n)
        pos = np.column_stack([cx, cy])

        # Build adjacency: node index 0..n-1 are circles, n..n+3 are corners
        # For each edge, compute constraint
        # We'll use a simple iterative relaxation
        for it in range(max_iter):
            # For each edge, compute current distance vs target
            for k, (i, j) in enumerate(cc_edges):
                d = pos[i] - pos[j]
                dist = np.sqrt(d[0] * d[0] + d[1] * d[1]) + 1e-12
                err = (dist - L[k]) / dist
                pos[i] -= 0.5 * err * d
                pos[j] += 0.5 * err * d
            for k, (i, c) in enumerate(cn_edges):
                node = n + c
                d = pos[i] - corners[c]
                dist = np.sqrt(d[0] * d[0] + d[1] * d[1]) + 1e-12
                err = (dist - L[k]) / dist
                pos[i] -= err * d  # corner fixed

        # Compute radii from circle-corner edges
        r = np.zeros(n)
        for k, (i, c) in enumerate(cn_edges):
            r[i] = L[k]

        # Check consistency: circle-circle distances should equal r[i]+r[j]
        max_err = 0.0
        for k, (i, j) in enumerate(cc_edges):
            d = pos[i] - pos[j]
            dist = np.sqrt(d[0] * d[0] + d[1] * d[1])
            max_err = max(max_err, abs(dist - (r[i] + r[j])))

        # Residual vector: mismatch between edge lengths and computed
        res = np.zeros(m)
        for k, (i, j) in enumerate(cc_edges):
            d = pos[i] - pos[j]
            dist = np.sqrt(d[0] * d[0] + d[1] * d[1])
            res[k] = dist - L[k]
        for k, (i, c) in enumerate(cn_edges):
            d = pos[i] - corners[c]
            dist = np.sqrt(d[0] * d[0] + d[1] * d[1])
            res[k] = dist - L[k]

        return res, pos, r

    # Use scipy to minimize sum of squared residuals
    def obj(L):
        L = np.clip(L, 1e-4, None)
        res, pos, r = residual(L)
        return np.sum(res * res)

    res_opt = minimize(obj, L, method='L-BFGS-B',
                       options={'maxiter': 50, 'ftol': 1e-12})
    L_opt = np.clip(res_opt.x, 1e-4, None)
    res_val, pos, r = residual(L_opt)
    obj_val = np.sum(res_val * res_val)
    if obj_val > 1e-8:
        return None

    # Validate: all circles inside rectangle, positive radii
    cx = pos[:, 0]
    cy = pos[:, 1]
    if np.any(r < 1e-4):
        return None
    if np.any(cx < r - tol) or np.any(cx > W - r + tol):
        return None
    if np.any(cy < r - tol) or np.any(cy > H + tol):
        return None
    # Check all pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2)
            if d < r[i] + r[j] - 1e-6:
                return None
    return cx, cy, r


def _generate_graphs(W, H, n_graphs=200, seed=42):
    """Generate random plausible contact graphs."""
    rng = random.Random(seed)
    graphs = []

    # Build a grid-based initial layout
    side = int(np.ceil(np.sqrt(N)))
    cx = np.array([(c + 0.5) / side * W for i in range(N) for c in range(side)][:N])
    cy = np.array([(r + 0.5) / side * H for i in range(side) for r in range(side)][:N])

    # Compute Delaunay-like neighbors (use simple k-nearest)
    from scipy.spatial import Delaunay
    try:
        pts = np.column_stack([cx, cy])
        tri = Delaunay(pts)
        cc_edges = set()
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    cc_edges.add((min(simplex[i], simplex[j]), max(simplex[i], simplex[j])))
    except Exception:
        # Fallback: connect each circle to 4 nearest neighbors
        cc_edges = set()
        for i in range(N):
            d = np.array([(cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2 for j in range(N)])
            d[i] = np.inf
            nearest = np.argsort(d)[:4]
            for j in nearest:
                cc_edges.add((min(i, j), max(i, j)))

    cc_list = sorted(cc_edges)

    # Assign each circle to a corner based on proximity
    corners = np.array([[0, 0], [W, 0], [0, H], [W, H]])
    cn_edges = set()
    for i in range(N):
        d = np.array([(cx[i] - c[0]) ** 2 + (cy[i] - c[1]) ** 2 for c in corners])
        c = int(np.argmin(d))
        cn_edges.add((i, c))

    # Ensure each corner has at least one circle
    for c in range(4):
        if not any(i_c[1] == c for i_c in cn_edges):
            # Find circle closest to corner c
            d = np.array([(cx[i] - corners[c][0]) ** 2 + (cy[i] - corners[c][1]) ** 2 for i in range(N)])
            i = int(np.argmin(d))
            cn_edges.add((i, c))

    cn_list = sorted(cn_edges)

    # Generate variations by randomly adding/removing edges
    for g in range(n_graphs):
        cc = list(cc_list)
        cn = list(cn_list)

        # Randomly add some edges
        for _ in range(rng.randint(0, 5)):
            i, j = rng.randint(0, N - 1), rng.randint(0, N - 1)
            if i != j:
                cc.append((min(i, j), max(i, j)))

        # Randomly remove some edges
        if len(cc) > N + 3:
            for _ in range(rng.randint(0, 3)):
                cc.pop(rng.randint(0, len(cc) - 1))

        # Randomly reassign some circles to corners
        for _ in range(rng.randint(0, 3)):
            i = rng.randint(0, N - 1)
            c = rng.randint(0, 3)
            cn = [(x, y) for x, y in cn if x != i]
            cn.append((i, c))

        # Ensure each corner has at least one circle
        for c in range(4):
            if not any(x[1] == c for x in cn):
                i = rng.randint(0, N - 1)
                cn = [(x, y) for x, y in cn if x != i]
                cn.append((i, c))

        graphs.append({'cc': cc, 'cn': cn})

    return graphs


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin

    best_value = 0.0
    best_result = None

    # Try different aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.85, 1.15, 0.8, 1.2, 0.75, 1.25, 0.7, 1.3, 0.65, 1.35, 0.6, 1.4, 0.55, 1.45, 0.5]

    for w in aspect_ratios:
        if time.time() > deadline - 5:
            break
        h = 2.0 - w
        if h < 0.3 or w < 0.3:
            continue

        # Generate graphs for this aspect ratio
        graphs = _generate_graphs(w, h, n_graphs=80, seed=int(w * 1000))

        for graph in graphs:
            if time.time() > deadline - 5:
                break
            result = _rigid_solve(graph, w, h)
            if result is not None:
                cx, cy, r = result
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    best_result = (cx.copy(), cy.copy(), r.copy(), w)

    # Fallback: grid packing
    if best_result is None or best_value < 0.5:
        side = int(np.ceil(np.sqrt(N)))
        W, H = 1.0, 1.0
        cx = np.array([(c + 0.5) / side * W for i in range(N) for c in range(side)][:N])
        cy = np.array([(r + 0.5) / side * H for i in range(side) for r in range(side)][:N])
        r = np.full(N, 0.5 / side - 1e-3)
        if best_result is None:
            best_result = (cx, cy, r, W)

    cx, cy, r, w = best_result
    # Safety shrink
    r = r * (1 - 1e-6)
    return np.column_stack((cx, cy, r))
