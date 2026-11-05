"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 40
_M = 401
_DEADLINE = 120.0
_MARGIN = 8.0


def _books_from_adj(adj):
    """Return the book size for every edge: for edge (u,v), count of w with adj[u,w] and adj[v,w]."""
    n = adj.shape[0]
    books = np.zeros((n, n), dtype=np.int32)
    # For each edge (u,v), book = (A @ A)[u,v] - (adj[u,v] ? 1 : 0) but A^2[u,v] counts common neighbors
    # A^2[u,v] = sum_w A[u,w]*A[v,w] = number of common neighbors of u and v
    A2 = adj @ adj
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                books[u, v] = A2[u, v]
                books[v, u] = A2[u, v]
    return books


def _max_book(books):
    return int(books.max())


def _edge_list_from_adj(adj):
    n = adj.shape[0]
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                edges.append([u, v])
    return edges


def _make_random_regular_like(rng, n, m):
    """Build a graph with m edges, trying to keep degrees balanced (regular-like)."""
    adj = np.zeros((n, n), dtype=np.int8)
    # start with a random regular-ish base via a random permutation pairing
    # We'll build edges greedily preferring low-degree vertices to keep degrees balanced.
    deg = np.zeros(n, dtype=np.int32)
    remaining = m
    # Pre-generate candidate pairs
    all_pairs = []
    for u in range(n):
        for v in range(u + 1, n):
            all_pairs.append((u, v))
    rng.shuffle(all_pairs)
    # Greedy: repeatedly pick a pair that minimizes max(deg[u], deg[v]) among available
    # To keep it fast, bucket by current degree sum.
    # Simpler: iterate shuffled pairs, but bias toward low-degree.
    # We'll do a weighted pick: probability proportional to (1/(deg[u]+1))*(1/(deg[v]+1))
    # But that's O(n^2) per pick; with m=401 and n=40 it's fine.
    chosen = set()
    for _ in range(m):
        # sample a few candidates and pick the best (lowest degree sum)
        best = None
        best_key = None
        for _ in range(30):
            u, v = rng.choice(all_pairs)
            if (u, v) in chosen:
                continue
            key = deg[u] + deg[v]
            if best_key is None or key < best_key:
                best_key = key
                best = (u, v)
        if best is None:
            # fallback: scan
            for u, v in all_pairs:
                if (u, v) not in chosen:
                    best = (u, v)
                    break
        if best is None:
            break
        u, v = best
        chosen.add((u, v))
        adj[u, v] = 1
        adj[v, u] = 1
        deg[u] += 1
        deg[v] += 1
    return adj


def _swap_move(adj, books, rng, n):
    """Pick an edge to remove and a non-edge to add (preserving edge count)."""
    # remove a random edge
    edges = []
    nonedges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                edges.append((u, v))
            else:
                nonedges.append((u, v))
    if not edges or not nonedges:
        return None
    ru, rv = rng.choice(edges)
    su, sv = rng.choice(nonedges)
    return (ru, rv), (su, sv)


def _apply_swap(adj, books, remove, add, n):
    """Apply a swap, updating adj and books incrementally. Returns delta max book info."""
    ru, rv = remove
    su, sv = add
    # We'll recompute books for affected vertices only, but for simplicity and correctness
    # with a small graph we recompute affected rows/cols.
    # Affected: neighbors of ru, rv, su, sv and the vertices themselves.
    # For safety and speed on n=40, recompute books for a subset.
    # Actually books depends on A^2 which is global. Recomputing full A^2 is 40x40 matmul ~ fast.
    # But we want incremental. Let's just recompute full A^2 for the affected check; n=40 is tiny.
    adj[ru, rv] = 0
    adj[rv, ru] = 0
    adj[su, sv] = 1
    adj[sv, su] = 1
    A2 = adj @ adj
    new_books = A2.copy()
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u, v]:
                new_books[u, v] = 0
                new_books[v, u] = 0
    return new_books


def construct():
    t0 = time.time()
    limit = t0 + _DEADLINE - _MARGIN
    rng = random.Random(12345)
    np_rng = np.random.default_rng(12345)

    best_adj = None
    best_books = None
    best_score = None

    # Initial random regular-like graph
    adj = _make_random_regular_like(rng, _N, _M)
    books = _books_from_adj(adj)
    score = _max_book(books)
    best_adj = adj.copy()
    best_books = books.copy()
    best_score = score

    # Simulated annealing
    T = 10.0
    T_min = 0.05
    cooling = 0.995
    step = 0
    while time.time() < limit:
        step += 1
        move = _swap_move(adj, books, rng, _N)
        if move is None:
            break
        remove, add = move
        # Compute new books
        new_books = _apply_swap(adj, books, remove, add, _N)
        new_score = _max_book(new_books)
        delta = new_score - score
        if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
            # accept
            score = new_score
            books = new_books
            if score < best_score:
                best_score = score
                best_adj = adj.copy()
                best_books = books.copy()
        else:
            # revert
            adj[remove[0], remove[1]] = 1
            adj[remove[1], remove[0]] = 1
            adj[add[0], add[1]] = 0
            adj[add[1], add[0]] = 0
            books = _books_from_adj(adj)  # recompute to be safe
        T *= cooling
        if T < T_min:
            T = T_min
            # restart temperature occasionally to escape local minima
            if rng.random() < 0.1:
                T = 5.0

    # Final: return best
    return _edge_list_from_adj(best_adj)
# EVOLVE-BLOCK-END
