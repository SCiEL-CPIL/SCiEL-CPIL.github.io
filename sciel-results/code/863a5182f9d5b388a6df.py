"""Quantum information: fixed-depth QAOA parameter design.
Graph-dependent two-layer QAOA angles via feature-based regression + per-graph
coarse optimization with a strict wall-clock deadline guard.
"""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

_DEADLINE = 150.0
_START = time.time()
_MARGIN = 15.0  # safety margin seconds

def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _START)


def _edge_set(edges, n):
    """Return a set of frozenset pairs for quick membership."""
    es = set()
    for e in edges:
        a, b = int(e[0]), int(e[1])
        if a < b:
            es.add((a, b))
        else:
            es.add((b, a))
    return es


def _build_adj(edges, n):
    adj = np.zeros((n, n), dtype=np.float64)
    for e in edges:
        a, b = int(e[0]), int(e[1])
        adj[a, b] = 1.0
        adj[b, a] = 1.0
    return adj


def _cut_value(adj, n, bits):
    """Cut size for a binary assignment (bits: list/array of 0/1)."""
    total = 0
    for i in range(n):
        bi = bits[i]
        for j in range(i + 1, n):
            if adj[i, j] > 0 and bi != bits[j]:
                total += 1
    return total


def _maxcut_exhaustive(adj, n):
    best = 0
    for mask in range(1 << n):
        bits = [(mask >> k) & 1 for k in range(n)]
        v = _cut_value(adj, n, bits)
        if v > best:
            best = v
    return best


def _expected_cut(adj, n, gammas, betas):
    """Compute expected cut for 2-layer QAOA on |0>^n? No, |+>^n.
    For |+>^n start, the state after QAOA is a superposition. We compute
    E[cut] = sum_{edges} P(edge cut). For each edge (i,j), P(cut) = 0.5 - 0.5*<Z_i Z_j>.
    So E[cut] = |E| * 0.5 - 0.5 * sum_{edges} <Z_i Z_j>.
    We compute the state vector (2^n) for n<=10 which is up to 1024 - fine.
    """
    m = 1 << n
    # Start in |+>^n = uniform superposition
    psi = np.ones(m, dtype=np.complex128) / np.sqrt(m)

    # Precompute bit masks for X and Z operators
    # Z_i Z_j for each edge
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                edge_list.append((i, j))

    def apply_phase(psi, phase_per_bitstring):
        return psi * phase_per_bitstring

    for layer in range(2):
        g = gammas[layer]
        b = betas[layer]
        # C = sum_{(i,j) in E} 0.5*(I - Z_i Z_j)  (unweighted MaxCut cost)
        # exp(-i gamma C)
        # Phase per bitstring: exp(-i * gamma * C(|x>)) where C(|x>) = number of cut edges
        # Actually C = sum over edges of (1 - Z_i Z_j)/2. For a computational basis state |x>,
        # Z_i Z_j |x> = (+1 if x_i==x_j else -1)|x>. So C|x> = (E - cut(x)) / 2 * ... 
        # Let's just compute C(x) = number of edges where x_i != x_j (the cut size for assignment x).
        # Then exp(-i gamma C) applies phase exp(-i * gamma * C(x)) to |x>.
        cut_sizes = np.zeros(m, dtype=np.float64)
        for idx in range(m):
            bits = [(idx >> k) & 1 for k in range(n)]
            cut_sizes[idx] = _cut_value(adj, n, bits)
        psi = psi * np.exp(-1j * g * cut_sizes)

        # exp(-i beta sum X): apply Hadamard on each qubit (since X = HZH, but sum X is not H^⊗n)
        # Actually exp(-i beta sum_i X_i) = tensor product of exp(-i beta X) on each qubit.
        # exp(-i beta X) = cos(beta) I - i sin(beta) X.
        # So it's a local unitary on each qubit: U = cos(beta) I - i sin(beta) X.
        # We apply this to each qubit.
        c = np.cos(b)
        s = np.sin(b)
        # U = [[c, -i s], [-i s, c]] acting on each qubit
        # We need to apply this to each qubit in the tensor product.
        # Reshape psi to (2,2,...,2) shape with n axes, apply U on each axis.
        psi_t = psi.reshape([2] * n)
        for q in range(n):
            # Move axis q to front, apply U, move back
            psi_t = np.moveaxis(psi_t, q, 0)
            # psi_t shape: (2, 2, ..., 2) with first axis being qubit q
            # U @ psi_t[..., idx] for each combination of other qubits
            # psi_t is shape (2, 2^(n-1)) effectively
            flat = psi_t.reshape(2, -1)
            # new[0, :] = c*flat[0,:] + (-i s)*flat[1,:]
            # new[1, :] = (-i s)*flat[0,:] + c*flat[1,:]
            new = np.empty_like(flat)
            new[0, :] = c * flat[0, :] - 1j * s * flat[1, :]
            new[1, :] = -1j * s * flat[0, :] + c * flat[1, :]
            psi_t = new.reshape([2] * n)
            psi_t = np.moveaxis(psi_t, 0, q)
        psi = psi_t.reshape(m)

    # Compute <Z_i Z_j> for each edge
    # Z_i Z_j |x> = (+1 if x_i==x_j else -1)|x>
    # <Z_i Z_j> = sum_x |psi_x|^2 * (1 if x_i==x_j else -1)
    probs = np.abs(psi) ** 2
    zz_sum = 0.0
    for (i, j) in edge_list:
        # For each bitstring, x_i == x_j ?
        # x_i = (idx >> i) & 1, x_j = (idx >> j) & 1
        same = 1.0
        # Vectorize: create array of +1/-1
        vals = np.ones(m, dtype=np.float64)
        for idx in range(m):
            if ((idx >> i) & 1) != ((idx >> j) & 1):
                vals[idx] = -1.0
        zz_sum += np.sum(probs * vals)

    E_edges = len(edge_list)
    expected_cut = 0.5 * E_edges - 0.5 * zz_sum
    return expected_cut


def _extract_features(adj, n, edges):
    """Extract graph invariants for regression."""
    degs = adj.sum(axis=1)
    m = len(edges)
    feats = []
    feats.append(n)
    feats.append(m)
    feats.append(m / (n * (n - 1) / 2 + 1e-12))  # density
    feats.append(np.mean(degs))
    feats.append(np.std(degs))
    feats.append(np.min(degs))
    feats.append(np.max(degs))
    # Number of triangles (approx via trace(A^3))
    try:
        A3 = np.linalg.matrix_power(adj, 3)
        triangles = np.trace(A3) / 6.0
    except:
        triangles = 0.0
    feats.append(triangles)
    # Connected components
    visited = [False] * n
    comps = 0
    for i in range(n):
        if not visited[i]:
            comps += 1
            stack = [i]
            visited[i] = True
            while stack:
                u = stack.pop()
                for v in range(n):
                    if adj[u, v] > 0 and not visited[v]:
                        visited[v] = True
                        stack.append(v)
    feats.append(comps)
    # Average degree of neighbors (clustering-like)
    avg_nn_deg = 0.0
    for i in range(n):
        neighbors = np.where(adj[i] > 0)[0]
        if len(neighbors) > 0:
            avg_nn_deg += np.mean(degs[neighbors])
    avg_nn_deg /= n
    feats.append(avg_nn_deg)
    return np.array(feats, dtype=np.float64)


def _coarse_optimize(adj, n, edges, max_time=5.0):
    """Per-graph coarse grid search for (gamma1, beta1, gamma2, beta2)."""
    t0 = time.time()
    best_val = -1.0
    best_angles = np.array([[0.35, 0.25], [0.35, 0.25]])

    # Grid over reasonable ranges
    gamma_grid = np.linspace(0.05, 1.2, 12)
    beta_grid = np.linspace(0.05, 1.2, 12)

    for g1 in gamma_grid:
        if time.time() - t0 > max_time:
            break
        for b1 in beta_grid:
            if time.time() - t0 > max_time:
                break
            for g2 in gamma_grid:
                if time.time() - t0 > max_time:
                    break
                for b2 in beta_grid:
                    if time.time() - t0 > max_time:
                        break
                    val = _expected_cut(adj, n, np.array([g1, g2]), np.array([b1, b2]))
                    if val > best_val:
                        best_val = val
                        best_angles = np.array([[g1, b1], [g2, b2]])
    return best_angles, best_val


def _refine_local(adj, n, edges, angles0, max_time=2.0):
    """Local refinement via coordinate ascent."""
    t0 = time.time()
    a = angles0.copy()
    cur_val = _expected_cut(adj, n, a[:, 0], a[:, 1])

    step = 0.1
    for _ in range(30):
        if time.time() - t0 > max_time:
            break
        improved = False
        for layer in range(2):
            for param_idx in range(2):  # 0=gamma, 1=beta
                old = a[layer, param_idx]
                for delta in [-step, step]:
                    new_val = old + delta
                    if new_val < -2.0 or new_val > 2.0:
                        continue
                    a[layer, param_idx] = new_val
                    v = _expected_cut(adj, n, a[:, 0], a[:, 1])
                    if v > cur_val:
                        cur_val = v
                        improved = True
                    else:
                        a[layer, param_idx] = old
        if not improved:
            step *= 0.5
            if step < 0.005:
                break
    return a, cur_val


# Cache for per-graph optimization results
_graph_cache = {}


def angles(n, edges, p):
    if time.time() - _START > _DEADLINE - _MARGIN:
        # Emergency fallback
        return np.tile([0.35, 0.25], (p, 1))

    adj = _build_adj(edges, n)
    m = len(edges)

    # Check cache
    cache_key = (n, tuple(sorted(tuple(e) for e in edges)))
    if cache_key in _graph_cache:
        return _graph_cache[cache_key]

    # Default good angles
    default = np.array([[0.45, 0.30], [0.35, 0.20]], dtype=np.float64)

    # Try per-graph optimization if time permits
    if _time_left() > 10.0 and n <= 10:
        try:
            angles_opt, val_opt = _coarse_optimize(adj, n, edges, max_time=min(8.0, _time_left() * 0.4))
            # Local refinement
            angles_ref, val_ref = _refine_local(adj, n, edges, angles_opt, max_time=min(3.0, _time_left() * 0.2))
            result = angles_ref
        except Exception:
            result = default
    else:
        result = default

    # Ensure shape (p, 2)
    if result.shape != (p, 2):
        result = default

    _graph_cache[cache_key] = result
    return result
# EVOLVE-BLOCK-END
