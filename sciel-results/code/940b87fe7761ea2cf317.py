"""Optimized solver for spectral graph lifts on prism graphs.

Model: a two-lift with per-edge signs s_e. The lift's spectrum equals the
spectrum of the base signed adjacency matrix A_s (2m x 2m). We minimize the
spectral radius max|eig| of A_s subject to the lift being connected and
3-regular (3-regularity is automatic for a prism lift; connectivity is the
real constraint).

Search: random restarts + 1-flip local search on the exact spectral radius,
with a strict wall-clock deadline guard. We validate candidates with
check_lift to be safe.
"""

import time
import numpy as np

try:
    from autoresearch_suite.science_tcs_expansion import check_lift
    _HAVE_CHECK = True
except Exception:
    _HAVE_CHECK = False

_DEADLINE = 78.0  # evaluator kills at 90s; keep a comfortable margin
_START = None


def _now():
    return time.monotonic() - _START


def _deadline():
    return _START is not None and _now() > _DEADLINE


def _build_signed_adj(m, top_s, bot_s, match_s):
    """Base signed adjacency (2m x 2m) for the prism.

    top_s, bot_s, match_s: arrays of length m (per-edge signs).
    Edge layout: [0..m-1] top cycle, [m..2m-1] bottom cycle, [2m..3m-1] matching.
    """
    A = np.zeros((2 * m, 2 * m))
    for i in range(m):
        j = (i + 1) % m
        A[i, j] += top_s[i]
        A[j, i] += top_s[i]
        A[m + i, m + j] += bot_s[i]
        A[m + j, m + i] += bot_s[i]
        A[i, m + i] += match_s[i]
        A[m + i, i] += match_s[i]
    return A


def _spectral_radius(m, top_s, bot_s, match_s):
    A = _build_signed_adj(m, top_s, bot_s, match_s)
    ev = np.linalg.eigvalsh(A)
    return float(np.max(np.abs(ev)))


def _lift_connected(m, top_s, bot_s, match_s):
    """Check connectivity of the two-lift via BFS on the 4m lift vertices."""
    # Lift vertex (v, side). Edge (u,v) with sign s: (u,0)->(v, s==1?0:1) etc.
    # Build adjacency of lift.
    n_lift = 4 * m
    adj = [[] for _ in range(n_lift)]

    def add(u, v, s):
        # (u,0) -> (v, 0) if s==1 else (v,1); (u,1) -> (v,1) if s==1 else (v,0)
        adj[u * 2 + 0].append(v * 2 + (0 if s == 1 else 1))
        adj[v * 2 + (0 if s == 1 else 1)].append(u * 2 + 0)
        adj[u * 2 + 1].append(v * 2 + (1 if s == 1 else 0))
        adj[v * 2 + (1 if s == 1 else 0)].append(u * 2 + 1)

    for i in range(m):
        j = (i + 1) % m
        add(i, j, int(top_s[i]))
        add(m + i, m + j, int(bot_s[i]))
        add(i, m + i, int(match_s[i]))

    seen = [False] * n_lift
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                cnt += 1
                stack.append(v)
    return cnt == n_lift


def sign_edges(graph, budget):
    global _START
    if _START is None:
        _START = time.monotonic()

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    assert n_edges == 3 * m, f"expected {3*m} edges, got {n_edges}"

    rng = np.random.default_rng(3)

    def rand_pattern():
        top_s = rng.choice([-1, 1], size=m)
        bot_s = rng.choice([-1, 1], size=m)
        match_s = rng.choice([-1, 1], size=m)
        return top_s, bot_s, match_s

    def evaluate(top_s, bot_s, match_s):
        if not _lift_connected(m, top_s, bot_s, match_s):
            return float('inf')
        return _spectral_radius(m, top_s, bot_s, match_s)

    def local_search(top_s, bot_s, match_s, iters):
        cur = evaluate(top_s, bot_s, match_s)
        for _ in range(iters):
            if _deadline():
                break
            grp = int(rng.integers(0, 3))
            i = int(rng.integers(0, m))
            if grp == 0:
                top_s[i] = -top_s[i]
                arr, idx = top_s, i
            elif grp == 1:
                bot_s[i] = -bot_s[i]
                arr, idx = bot_s, i
            else:
                match_s[i] = -match_s[i]
                arr, idx = match_s, i
            val = evaluate(top_s, bot_s, match_s)
            if val < cur - 1e-12:
                cur = val
            else:
                arr[idx] = -arr[idx]  # revert
        return top_s, bot_s, match_s, cur

    best = None
    best_r = float('inf')

    # Initial feasible seed: make matching alternating to guarantee connectivity.
    def feasible_seed():
        top_s = np.ones(m, dtype=int)
        bot_s = np.ones(m, dtype=int)
        match_s = np.ones(m, dtype=int)
        match_s[1::2] = -1  # alternating matching -> connected
        return top_s, bot_s, match_s

    top_s, bot_s, match_s = feasible_seed()
    best_r = evaluate(top_s, bot_s, match_s)
    best = (top_s.copy(), bot_s.copy(), match_s.copy())

    restarts = 0
    while not _deadline():
        # Occasionally restart from scratch, otherwise continue local search
        if restarts % 50 == 0:
            top_s, bot_s, match_s = rand_pattern()
            if not _lift_connected(m, top_s, bot_s, match_s):
                # fix connectivity by forcing alternating matching
                match_s = np.ones(m, dtype=int)
                match_s[1::2] = -1
        top_s, bot_s, match_s, r = local_search(top_s, bot_s, match_s, iters=60)
        if r < best_r - 1e-12:
            best_r = r
            best = (top_s.copy(), bot_s.copy(), match_s.copy())
        restarts += 1

    top_s, bot_s, match_s = best

    # Final validation with check_lift if available; if invalid, fall back to
    # the guaranteed-feasible alternating-matching pattern.
    signs = np.empty(n_edges, dtype=int)
    signs[0:m] = top_s
    signs[m:2 * m] = bot_s
    signs[2 * m:3 * m] = match_s

    if _HAVE_CHECK:
        try:
            check_lift(graph, signs)
        except Exception:
            # fallback to a known-connected pattern
            top_s, bot_s, match_s = feasible_seed()
            signs[0:m] = top_s
            signs[m:2 * m] = bot_s
            signs[2 * m:3 * m] = match_s
            try:
                check_lift(graph, signs)
            except Exception:
                pass

    return signs
