"""Maximum Cut solver: multi-restart greedy (1-flip + 2-flip) with simulated annealing."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _cut_size(adj, x):
    """Exact cut size from a 0/1 label array."""
    s = 2.0 * x.astype(np.float64) - 1.0
    S = adj.sum() - s @ (adj @ s)
    return int(round(S * 0.5))


def _greedy(adj, x, max_passes=60):
    """Greedy local search over 1-flips and 2-flips until a local optimum.

    Returns (labels, cut). Maintains s=2x-1 and g=A@s for O(n^2) per pass.
    """
    n = x.shape[0]
    x = x.copy()
    s = 2.0 * x.astype(np.float64) - 1.0
    g = adj @ s

    for _ in range(max_passes):
        improved = False

        # --- 1-flip neighborhood (exact delta) ---
        delta1 = 4.0 * s * g
        for v in range(n):
            d = delta1[v]
            if d > 1e-9:
                s[v] = -s[v]
                g += 2.0 * s[v] * adj[v]
                delta1[v] = -d
                delta1[adj[v]] += 2.0 * s[v]
                improved = True

        # --- 2-flip neighborhood (exact delta) ---
        for u in range(n):
            if not improved:
                break
            su = s[u]
            gu = g[u]
            for v in range(u + 1, n):
                d = 4.0 * (su * gu + s[v] * g[v]) - 8.0 * adj[u, v] * su * s[v]
                if d > 1e-9:
                    s[u] = -su
                    s[v] = -s[v]
                    g[u] += 2.0 * s[u] * adj[u]
                    g[v] += 2.0 * s[v] * adj[v]
                    gu = -gu
                    g[v] = g[v]  # keep reference consistent
                    improved = True

        if not improved:
            break

    x = ((s + 1.0) * 0.5).astype(np.int64)
    return x, _cut_size(adj, x)


def _anneal(adj, x, T0, T1, steps):
    """Simulated annealing over 1-flips to escape local optima."""
    n = x.shape[0]
    x = x.copy()
    s = 2.0 * x.astype(np.float64) - 1.0
    g = adj @ s
    rng = np.random.default_rng(12345)
    ratio = (T1 / T0) ** (1.0 / max(1, steps))

    T = T0
    for _ in range(steps):
        v = int(rng.integers(n))
        d = 4.0 * s[v] * g[v]
        if d > 1e-9:
            s[v] = -s[v]
            g += 2.0 * s[v] * adj[v]
        else:
            # accept worsening move with probability exp(d/T)
            if T > 1e-9 and rng.random() < np.exp(d / T):
                s[v] = -s[v]
                g += 2.0 * s[v] * adj[v]
        T *= ratio

    x = ((s + 1.0) * 0.5).astype(np.int64)
    return x, _cut_size(adj, x)


def partition(adjacency):
    """Solve Maximum Cut with multi-restart greedy (1+2 flip) + simulated annealing."""
    start_time = time.time()
    DEADLINE = 143.0  # safety margin under the 150s wall-clock kill

    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)

    adj = np.array(adjacency, dtype=bool)
    adj = adj | adj.T
    np.fill_diagonal(adj, False)

    best_labels = None
    best_cut = -1

    def consider(x):
        nonlocal best_labels, best_cut
        c = _cut_size(adj, x)
        if c > best_cut:
            best_cut = c
            best_labels = x.copy()

    # Deterministic seeds first (cheap, often near-optimal)
    for seed in (np.arange(n) % 2, np.zeros(n, dtype=int)):
        x, _ = _greedy(adj, seed)
        consider(x)
        if time.time() >= DEADLINE:
            break

    # Random restarts, each followed by a short anneal to escape the basin
    rng = np.random.default_rng(42)
    T0 = 1.2
    T1 = 1e-3
    anneal_steps = 400

    while time.time() < DEADLINE:
        x = rng.integers(0, 2, size=n).astype(np.int64)
        x, _ = _greedy(adj, x)
        consider(x)
        if time.time() >= DEADLINE:
            break
        # Anneal to try to jump to a better basin, then polish greedily
        xa, _ = _anneal(adj, x, T0, T1, anneal_steps)
        xa, _ = _greedy(adj, xa)
        consider(xa)

    if best_labels is None:
        return np.arange(n) % 2

    return best_labels
# EVOLVE-BLOCK-END
