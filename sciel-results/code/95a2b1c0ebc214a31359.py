"""Construct a large (5,5)-free graph: no K5, no independent 5-set.
Goal: maximize n (score = n / 37). Known optimum: Exoo's 42-vertex graph.
Strategy: Use known constructions + efficient verification + local search.
"""
import time
import random
import numpy as np

_DEADLINE = 300.0  # seconds; evaluator kills at 330s
_MARGIN = 25.0     # safety margin


def _elapsed():
    return time.time() - _START


def _has_k5(adj, n):
    """Check if graph has a K5 using efficient bitset-based approach."""
    if n < 5:
        return False
    # Convert adjacency to bitsets for fast intersection
    adj_bits = [0] * n
    for i in range(n):
        bits = 0
        for j in range(n):
            if i != j and adj[i, j]:
                bits |= (1 << j)
        adj_bits[i] = bits
    
    # Check all 5-tuples
    from itertools import combinations
    for combo in combinations(range(n), 5):
        a, b, c, d, e = combo
        if (adj_bits[a] & (1 << b)) and (adj_bits[a] & (1 << c)) and \
           (adj_bits[a] & (1 << d)) and (adj_bits[a] & (1 << e)) and \
           (adj_bits[b] & (1 << c)) and (adj_bits[b] & (1 << d)) and \
           (adj_bits[b] & (1 << e)) and \
           (adj_bits[c] & (1 << d)) and (adj_bits[c] & (1 << e)) and \
           (adj_bits[d] & (1 << e)):
            return True
    return False


def _has_ind5(adj, n):
    """Check if graph has an independent set of size 5."""
    if n < 5:
        return False
    # Independent set in G = clique in complement
    comp = ~adj
    np.fill_diagonal(comp, False)
    return _has_k5(comp, n)


def _has_k5_or_ind5(adj):
    """Return True if graph has a K5 or an independent set of size 5."""
    n = adj.shape[0]
    if _has_k5(adj, n):
        return True
    if _has_ind5(adj, n):
        return True
    return False


def _try_build(n_target, rng, adj_seed=None):
    """Try to construct a (5,5)-free graph on n_target vertices.
    Uses randomized local search starting from seed or random initialization."""
    n = n_target
    if adj_seed is not None:
        adj = adj_seed.copy()
    else:
        # Start with random graph
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    adj[i, j] = True
                    adj[j, i] = True

    # Local search: try to fix violations by flipping edges
    for iteration in range(1000):
        if _elapsed() > _DEADLINE - _MARGIN:
            break
        
        # Check if valid
        if not _has_k5_or_ind5(adj):
            edges = [(int(i), int(j)) for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            return n, edges
        
        # Find a violation and try to fix it
        # For simplicity, try random edge flips
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        # Flip edge (i, j)
        adj[i, j] = not adj[i, j]
        adj[j, i] = adj[i, j]
        
        # If this creates a violation, revert
        if _has_k5_or_ind5(adj):
            adj[i, j] = not adj[i, j]
            adj[j, i] = adj[i, j]

    return None


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

    # Try to build larger graphs
    # Strategy 1: Try to extend Paley 37 to larger sizes
    for n_target in [42, 41, 40, 39, 38]:
        if _elapsed() > deadline - 10:
            break
        # Try multiple times with different random seeds
        for restart in range(20):
            if _elapsed() > deadline:
                break
            try:
                # Create seed with Paley 37 + random edges for new vertices
                adj_seed = np.zeros((n_target, n_target), dtype=bool)
                adj_seed[:37, :37] = paley
                # Add random edges between new vertices and old ones
                for i in range(37, n_target):
                    for j in range(i):
                        if rng.random() < 0.5:
                            adj_seed[i, j] = True
                            adj_seed[j, i] = True
                
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

    # Strategy 2: Try random construction from scratch
    for n_target in [42, 41, 40, 39, 38]:
        if _elapsed() > deadline - 5:
            break
        for restart in range(10):
            if _elapsed() > deadline:
                break
            try:
                res = _try_build(n_target, rng)
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
