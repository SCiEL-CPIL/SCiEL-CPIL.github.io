"""Per-graph QAOA angle design via analytic-gradient ascent with a deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 150.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _expected_cut(n, edge_set, gammas, betas):
    """Exact expected cut value for a given QAOA parameter schedule.

    edge_set: list of (i,j) undirected edges (i<j).
    Returns (value, state_amplitude_vector) so gradients can be computed cheaply.
    """
    m = len(edge_set)
    # Precompute edge bitmasks
    emask = [(1 << i) | (1 << j) for (i, j) in edge_set]

    # Start state |+>^n -> uniform amplitudes
    # We work in the computational basis. Amplitudes all 1/sqrt(2^n).
    # Represent state as array of length 2^n of complex amplitudes.
    size = 1 << n
    # Use real/imag arrays; start all equal real.
    amp = np.full(size, 1.0 / np.sqrt(size), dtype=np.float64)

    def apply_x_all(phase):
        # exp(-i beta * sum X): acts as rotation on each qubit.
        # For each bit position, swap pairs and apply phase.
        # X on qubit k maps |b> -> |b xor (1<<k)>.
        # exp(-i beta sum_k X_k) = product_k exp(-i beta X_k)
        # exp(-i beta X) = cos(beta) I - i sin(beta) X
        # So on pairs differing in bit k: [a, b] -> [cos a - i sin b, -i sin a + cos b]
        c = np.cos(phase)
        s = np.sin(phase)
        nonlocal amp
        for k in range(n):
            bit = 1 << k
            # indices where bit is 0
            # iterate over blocks
            # For each index idx with bit k = 0, partner = idx ^ bit
            # new[idx] = c*old[idx] - i*s*old[partner]
            # new[partner] = -i*s*old[idx] + c*old[partner]
            # We need to do this in place carefully; do it with a copy of affected.
            # Gather all pairs
            # Efficient: use reshape. For bit k, the array can be viewed with stride.
            # Simpler: loop but vectorize over the non-k bits.
            # Build index arrays
            # We'll do it via np operations on a reshaped view.
            # Reshape so that the k-th bit is the last axis of size 2.
            # size = 2^n; bit k has stride 2^k.
            # Create view with shape: (2^k, 2^(n-k-1), 2)
            shape = [2] * n
            # We want axis for bit k last.
            # Current index order: index = sum b_j 2^j, j=0..n-1, so bit j is axis j in a
            # C-order reshape with shape (2,2,...,2) where axis j corresponds to bit j.
            # Move axis k to the end.
            v = amp.reshape([2] * n)
            v = np.moveaxis(v, k, -1)
            flat = v.reshape(-1, 2)
            a0 = flat[:, 0].copy()
            a1 = flat[:, 1].copy()
            # new0 = c*a0 - i*s*a1 ; new1 = -i*s*a0 + c*a1
            new0 = c * a0 - 1j * s * a1
            new1 = -1j * s * a0 + c * a1
            flat[:, 0] = new0
            flat[:, 1] = new1
            amp = v.reshape(size)

    def apply_C(phase):
        # exp(-i gamma C), C = sum_edges (1 - Z_i Z_j)/2 = m/2 - 1/2 sum Z_i Z_j
        # exp(-i gamma C) = exp(-i gamma m/2) * prod_edges exp(i gamma/2 Z_i Z_j)
        # exp(i gamma/2 ZZ) on a pair: if Z_i Z_j = +1 -> phase e^{i gamma/2}; if -1 -> e^{-i gamma/2}
        # So it multiplies amplitude of basis state |b> by exp(i gamma/2 * (product of bits-as-z))
        # where z_i = +1 if bit 0 else -1. product = (-1)^(parity of edge in state).
        # Actually Z_i Z_j eigenvalue = (-1)^(b_i + b_j). So phase = exp(i gamma/2 * (-1)^(b_i+b_j)).
        # = exp(i gamma/2) if b_i==b_j, else exp(-i gamma/2).
        # Equivalent to: multiply by exp(i gamma/2) * exp(-i gamma * (b_i+b_j mod 2))
        # Let's just compute per-state phase.
        # For efficiency, accumulate phase per state from edges.
        # phase_state = (m/2 - 1/2 * sum_e zprod_e) * gamma, where zprod_e = (-1)^(b_i+b_j)
        # exp(-i gamma C) = exp(-i gamma (m/2 - 1/2 sum zprod)) = exp(-i gamma m/2) exp(i gamma/2 sum zprod)
        # sum zprod over edges. We can compute parity per edge.
        nonlocal amp
        # Compute for each state the sum of zprod = sum_e (-1)^(b_i+b_j)
        # We'll accumulate an integer parity count: for each edge, add +1 if b_i==b_j else -1.
        # Do it vectorized: for each edge, compute parity mask.
        # Build array of zprod sum.
        zsum = np.zeros(size, dtype=np.float64)
        for (i, j) in edge_set:
            bi = (np.arange(size) >> i) & 1
            bj = (np.arange(size) >> j) & 1
            same = (bi == bj)
            zsum += np.where(same, 1.0, -1.0)
        phase = np.exp(1j * (phase * (m / 2.0 - 0.5 * zsum)))
        amp = amp * phase

    for (g, b) in zip(gammas, betas):
        apply_C(g)
        apply_X_all(b)

    # Expected cut: for each edge, prob(b_i != b_j)
    # cut_val = sum_e P(b_i xor b_j = 1)
    probs = np.abs(amp) ** 2
    cut = 0.0
    for (i, j) in edge_set:
        bi = (np.arange(size) >> i) & 1
        bj = (np.arange(size) >> j) & 1
        cut += np.sum(probs[bi != bj])
    return cut, amp


def _objective(params, n, edge_set):
    """params: (p,2) array. Returns expected cut value."""
    gammas = params[:, 0]
    betas = params[:, 1]
    val, _ = _expected_cut(n, edge_set, gammas, betas)
    return val


def _exact_optimum(n, edge_set):
    """Exhaustive MaxCut optimum."""
    best = 0
    size = 1 << n
    for s in range(size):
        c = 0
        for (i, j) in edge_set:
            if ((s >> i) & 1) != ((s >> j) & 1):
                c += 1
        if c > best:
            best = c
    return best


def _gradient(params, n, edge_set, eps=1e-4):
    """Numerical gradient of objective w.r.t. params."""
    grad = np.zeros_like(params)
    base = _objective(params, n, edge_set)
    for idx in range(params.size):
        p2 = params.copy()
        flat = p2.reshape(-1)
        flat[idx] += eps
        p2 = flat.reshape(params.shape)
        up = _objective(p2, n, edge_set)
        p3 = params.copy()
        flat3 = p3.reshape(-1)
        flat3[idx] -= eps
        p3 = flat3.reshape(params.shape)
        down = _objective(p3, n, edge_set)
        grad.reshape(-1)[idx] = (up - down) / (2 * eps)
    return grad


def _optimize_graph(n, edge_set, init, max_iters=40, deadline_check=None):
    """Gradient ascent with line search on the 4 params."""
    params = init.copy()
    best_params = params.copy()
    best_val = _objective(params, n, edge_set)
    for it in range(max_iters):
        if deadline_check is not None and deadline_check() < 30:
            break
        g = _gradient(params, n, edge_set)
        # line search
        lr = 0.05
        improved = False
        for _ in range(30):
            cand = params + lr * g
            cand = np.clip(cand, -np.pi, np.pi)
            v = _objective(cand, n, edge_set)
            if v > best_val + 1e-9:
                params = cand
                best_params = cand.copy()
                best_val = v
                improved = True
                break
            lr *= 0.5
        if not improved:
            # try smaller steps in a grid around
            found = False
            for step in np.linspace(-0.3, 0.3, 13):
                for axis in range(4):
                    cand = params.copy()
                    cand.reshape(-1)[axis] += step
                    cand = np.clip(cand, -np.pi, np.pi)
                    v = _objective(cand, n, edge_set)
                    if v > best_val + 1e-9:
                        params = cand
                        best_params = cand.copy()
                        best_val = v
                        found = True
                        break
                if found:
                    break
            if not found:
                break
    return best_params, best_val


def _graph_features(n, edge_set):
    """Simple features for a graph to guide initial guess."""
    m = len(edge_set)
    deg = np.zeros(n)
    for (i, j) in edge_set:
        deg[i] += 1
        deg[j] += 1
    return {
        "n": n,
        "m": m,
        "density": m / (n * (n - 1) / 2.0) if n > 1 else 0.0,
        "avg_deg": deg.mean() if n else 0.0,
        "max_deg": deg.max() if n else 0.0,
    }


def _initial_guess(feats):
    """Heuristic initial 4-parameter guess based on graph features."""
    n = feats["n"]
    density = feats["density"]
    # Typical good QAOA angles: gamma ~ 0.4-0.6, beta ~ 0.3-0.5 for small graphs.
    gamma = 0.5 - 0.1 * density
    beta = 0.4 + 0.05 * (feats["avg_deg"] / max(n, 1))
    return np.array([[gamma, beta], [gamma * 0.8, beta * 1.1]])


def _generate_dev_graphs(seed=113, n_list=(6, 8, 10), densities=(0.25, 0.55)):
    """Generate development graphs similar to evaluation structure."""
    rng = np.random.RandomState(seed)
    graphs = []
    for n in n_list:
        for d in densities:
            m_max = n * (n - 1) // 2
            m = int(round(d * m_max))
            m = max(1, min(m, m_max))
            edges = []
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            idx = rng.choice(len(pairs), size=m, replace=False)
            for k in idx:
                edges.append(pairs[k])
            graphs.append((n, edges))
    return graphs


def _build_angle_model():
    """Learn a mapping from graph features to good angles on dev graphs.

    Returns a function feats -> params (4x1).
    """
    if _remaining() < 60:
        return None
    dev = _generate_dev_graphs()
    # Optimize each dev graph, collect (features, best_params)
    data = []
    for (n, edges) in dev:
        if _remaining() < 45:
            break
        feats = _graph_features(n, edges)
        init = _initial_guess(feats)
        best_params, best_val = _optimize_graph(n, edges, init, max_iters=25)
        data.append((feats, best_params))
    if not data:
        return None
    # Fit: for each of the 4 parameters, regress on features (n, density, avg_deg).
    X = np.array([[f["n"], f["density"], f["avg_deg"]] for f, _ in data])
    X = np.column_stack([X, np.ones(len(X))])
    model = {}
    for axis in range(4):
        y = np.array([p[axis] for _, p in data])
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            coef = np.zeros(X.shape[1])
        model[axis] = coef
    return model


def _apply_model(model, feats):
    if model is None:
        return _initial_guess(feats)
    x = np.array([feats["n"], feats["density"], feats["avg_deg"], 1.0])
    params = np.zeros(4)
    for axis in range(4):
        params[axis] = float(np.dot(x, model[axis]))
    return params.reshape(2, 2)


_MODEL = None
_MODEL_BUILT = False


def angles(n, edges, p):
    global _MODEL, _MODEL_BUILT
    # Normalize edges
    edge_set = []
    for e in edges:
        i, j = int(e[0]), int(e[1])
        if i > j:
            i, j = j, i
        if i != j:
            edge_set.append((i, j))
    edge_set = sorted(set(edge_set))

    # Build model once (with deadline guard)
    if not _MODEL_BUILT:
        _MODEL = _build_angle_model()
        _MODEL_BUILT = True

    feats = _graph_features(n, edge_set)
    init = _apply_model(_MODEL, feats)

    # Refine on this specific graph if time permits
    if _remaining() > 40:
        best_params, best_val = _optimize_graph(n, edge_set, init, max_iters=20)
    else:
        best_params = init

    return best_params[:p, :]
# EVOLVE-BLOCK-END
