"""Construct a large (5,5)-free graph: no K5, no independent 5-set.
Goal: maximize n (score = n / 37). Known optimum: Exoo's 42-vertex graph.
Strategy: fast verification + randomized incremental construction with restarts,
deadline-guarded, returning the best valid graph found.
"""
import time
import random
import numpy as np

_DEADLINE = 300.0  # seconds; evaluator kills at 330s
_MARGIN = 25.0     # safety margin


def _elapsed():
    return time.time() - _START


def _has_k5_or_ind5(adj):
    """Return True if graph has a K5 or an independent set of size 5.
    adj: n x n boolean numpy array (symmetric, zero diag)."""
    n = adj.shape[0]
    # Check clique number >= 5 and independence number >= 5.
    # Independence number of G = clique number of complement.
    def max_clique_size(adj_m, limit):
        # Branch and bound: find a clique of size >= limit.
        # Returns True if a clique of size `limit` exists.
        adj_m = adj_m.astype(bool)
        n = adj_m.shape[0]

        # Greedy upper bound via coloring is complex; use simple branch-and-bound.
        # We only need to detect size-5 cliques, so keep it small.
        found = [False]

        def expand(cand, size):
            if found[0]:
                return
            if size >= limit:
                found[0] = True
                return
            if not cand:
                return
            # Upper bound: size + len(cand) < limit -> prune
            if size + len(cand) < limit:
                return
            # Try vertices in cand
            for idx in range(len(cand) - 1, -1, -1):
                v = cand[idx]
                new_cand = [u for u in cand[:idx] if adj_m[v, u]]
                expand(new_cand, size + 1)
                if found[0]:
                    return

        expand(list(range(n)), 0)
        return found[0]

    if max_clique_size(adj, 5):
        return True
    # complement
    comp = (~adj)
    np.fill_diagonal(comp, False)
    if max_clique_size(comp, 5):
        return True
    return False


def _try_build(n_target, rng, adj_seed=None):
    """Randomized incremental construction of a (5,5)-free graph on n_target vertices.
    Adds vertices one at a time, choosing random adjacency consistent with (5,5)-freedom.
    Returns (n, edges) if successful, else None."""
    n = n_target
    if adj_seed is not None:
        adj = adj_seed.copy()
        cur = adj.shape[0]
    else:
        adj = np.zeros((n, n), dtype=bool)
        cur = 0

    # We'll build up to n vertices.
    # Use a simple randomized greedy: for each new vertex, try random subsets of
    # existing vertices to connect to, accept if (5,5)-free maintained.
    # To keep verification fast, only check the new vertex's local structure.
    # Full verification at the end.

    for v in range(cur, n):
        # Try several random adjacency patterns
        placed = False
        for attempt in range(200):
            # Random subset of existing vertices
            if v == 0:
                adj_new = np.zeros((n, n), dtype=bool)
            else:
                # Start from current adj, set row/col v
                adj_new = adj.copy()
                mask = rng.random(v) < 0.5
                adj_new[v, :v] = mask
                adj_new[:v, v] = mask
            # Quick local check: does adding v create a K5 or ind5?
            # A K5 containing v: v + 4 mutually adjacent neighbors.
            # An ind5 containing v: v + 4 mutually non-adjacent neighbors.
            # Check neighbors of v.
            neigh = np.where(adj_new[v, :v])[0]
            nonneigh = np.where(~adj_new[v, :v])[0]
            nonneigh = nonneigh[nonneigh < v]
            # Check if neigh contains a K4
            if len(neigh) >= 4:
                # check clique of size 4 in induced subgraph on neigh
                sub = adj_new[np.ix_(neigh, neigh)]
                if _has_clique_size(sub, 4):
                    continue
            # Check if nonneigh contains a K4 in complement (i.e., independent set of size 4 in G)
            if len(nonneigh) >= 4:
                sub = adj_new[np.ix_(nonneigh, nonneigh)]
                comp = ~sub
                np.fill_diagonal(comp, False)
                if _has_clique_size(comp, 4):
                    continue
            # Also need to ensure existing graph remains (5,5)-free (it should by construction)
            # Accept
            adj = adj_new
            placed = True
            break
        if not placed:
            # Try with different randomization
            for attempt in range(200):
                if v == 0:
                    adj_new = np.zeros((n, n), dtype=bool)
                else:
                    adj_new = adj.copy()
                    p = rng.random()
                    mask = rng.random(v) < p
                    adj_new[v, :v] = mask
                    adj_new[:v, v] = mask
                neigh = np.where(adj_new[v, :v])[0]
                nonneigh = np.where(~adj_new[v, :v])[0]
                nonneigh = nonneigh[nonneigh < v]
                if len(neigh) >= 4:
                    sub = adj_new[np.ix_(neigh, neigh)]
                    if _has_clique_size(sub, 4):
                        continue
                if len(nonneigh) >= 4:
                    sub = adj_new[np.ix_(nonneigh, nonneigh)]
                    comp = ~sub
                    np.fill_diagonal(comp, False)
                    if _has_clique_size(comp, 4):
                        continue
                adj = adj_new
                placed = True
                break
        if not placed:
            return None

    # Full verification
    if _has_k5_or_ind5(adj):
        return None
    edges = [(int(i), int(j)) for i in range(n) for j in range(i + 1, n) if adj[i, j]]
    return n, edges


def _has_clique_size(adj_m, k):
    """Check if adj_m (m x m bool) has a clique of size k. m is small."""
    m = adj_m.shape[0]
    if m < k:
        return False
    found = [False]

    def expand(cand, size):
        if found[0]:
            return
        if size >= k:
            found[0] = True
            return
        if not cand:
            return
        if size + len(cand) < k:
            return
        for idx in range(len(cand) - 1, -1, -1):
            v = cand[idx]
            new_cand = [u for u in cand[:idx] if adj_m[v, u]]
            expand(new_cand, size + 1)
            if found[0]:
                return

    expand(list(range(m)), 0)
    return found[0]


# Precompute Paley graph on 37 as a seed/fallback
def _paley37():
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = np.zeros((q, q), dtype=bool)
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i, j] = True
                adj[j, i] = True
    return adj


_START = time.time()


def construct():
    global _START
    _START = time.time()
    deadline = _START + _DEADLINE - _MARGIN

    best_n = 0
    best_edges = None

    # Fallback: Paley 37
    paley = _paley37()
    if not _has_k5_or_ind5(paley):
        best_n = 37
        best_edges = [(int(i), int(j)) for i in range(37) for j in range(i + 1, 37) if paley[i, j]]

    rng = random.Random(12345)

    # Try to build larger graphs with randomized construction
    # Try targets from 42 down, with multiple restarts
    for n_target in [42, 41, 40, 39, 38]:
        if _elapsed() > deadline - 30:
            break
        # Multiple restarts
        for restart in range(50):
            if _elapsed() > deadline:
                break
            try:
                res = _try_build(n_target, rng)
            except Exception:
                res = None
            if res is not None:
                n, edges = res
                if n > best_n:
                    best_n = n
                    best_edges = edges
                    if n >= 42:
                        return n, edges
            # Also try seeded from Paley (extend)
            if restart < 10:
                try:
                    # Try to extend Paley 37 to n_target
                    adj_seed = np.zeros((n_target, n_target), dtype=bool)
                    adj_seed[:37, :37] = paley
                    res = _try_build(n_target, rng, adj_seed=adj_seed)
                    if res is not None:
                        n, edges = res
                        if n > best_n:
                            best_n = n
                            best_edges = edges
                            if n >= 42:
                                return n, edges
                except Exception:
                    pass

    return best_n, best_edges
