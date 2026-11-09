"""Sign-edges solver for prism-graph spectral lifts: random init + local search."""
import time
import numpy as np

# Hard deadline guard: evaluator kills at 90s; return well before.
_DEADLINE = 85.0  # seconds from process start as a safety margin


def _build_signed_adj(graph, signs):
    n = graph['n']
    edges = graph['edges']
    A = np.zeros((n, n), dtype=np.float64)
    for (u, v), s in zip(edges, signs):
        sv = float(s)
        A[u, v] = sv
        A[v, u] = sv
    return A


def _lift_radius(graph, signs):
    """Return spectral radius of the lift's signed adjacency, or inf if invalid."""
    try:
        from autoresearch_suite.science_tcs_expansion import check_lift
        radius, _ = check_lift(graph, signs)
        return float(radius)
    except Exception:
        return float('inf')


def _random_valid_init(graph, budget):
    """Sample random sign vectors; return the best valid one (or first valid)."""
    rng = np.random.default_rng(3)
    n_edges = len(graph['edges'])
    best_signs = None
    best_radius = float('inf')
    for _ in range(budget):
        signs = rng.choice([-1, 1], size=n_edges)
        r = _lift_radius(graph, signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
    return best_signs, best_radius


def _local_search(graph, signs, radius, deadline):
    """Flip-based local search to reduce the lift spectral radius."""
    n_edges = len(graph['edges'])
    signs = signs.copy()
    best_radius = radius
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try single flips
        for i in range(n_edges):
            if time.time() >= deadline:
                break
            signs[i] = -signs[i]
            r = _lift_radius(graph, signs)
            if r < best_radius - 1e-12:
                best_radius = r
                improved = True
            else:
                signs[i] = -signs[i]  # revert
        # Try pairwise flips (escape local minima)
        if not improved and time.time() < deadline:
            for i in range(n_edges):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, n_edges):
                    if time.time() >= deadline:
                        break
                    signs[i] = -signs[i]
                    signs[j] = -signs[j]
                    r = _lift_radius(graph, signs)
                    if r < best_radius - 1e-12:
                        best_radius = r
                        improved = True
                    else:
                        signs[i] = -signs[i]
                        signs[j] = -signs[j]
                    if improved:
                        break
                if improved:
                    break
    return signs, best_radius


def sign_edges(graph, budget):
    start = time.time()
    deadline = start + _DEADLINE

    n_edges = len(graph['edges'])
    if n_edges == 0:
        return np.array([], dtype=np.int64)

    # Phase 1: random initialization
    signs, radius = _random_valid_init(graph, budget)

    if signs is None:
        # Fallback: all +1 (may be invalid, but try)
        signs = np.ones(n_edges, dtype=np.int64)
        radius = _lift_radius(graph, signs)

    # Phase 2: local search
    if time.time() < deadline:
        signs, radius = _local_search(graph, signs, radius, deadline)

    # Ensure the return is a numpy array of int signs
    return np.asarray(signs, dtype=np.int64)
