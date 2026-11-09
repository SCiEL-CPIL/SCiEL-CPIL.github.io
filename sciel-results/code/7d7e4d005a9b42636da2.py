import random
import time

_START = time.time()
_DEADLINE = _START + 110.0  # Safety margin before 120s limit

_RNG = random.Random(20240607)

def _time_left():
    return _DEADLINE - time.time()

def _paley17():
    squares = {x * x % 17 for x in range(1, 17)}
    n = 17
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % 17 in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return n, adj

def _find_k4(adj, n):
    """Return a 4-tuple of vertices forming a K4, or None if K4-free."""
    for u in range(n):
        au = adj[u]
        # neighbours of u that are greater than u
        vv = au & (~((1 << (u + 1)) - 1))
        while vv:
            v = (vv & -vv).bit_length() - 1
            vv &= vv - 1
            # common neighbours of u and v, greater than v
            c = au & adj[v] & (~((1 << (v + 1)) - 1))
            if not c:
                continue
            cc = c
            while cc:
                x = (cc & -cc).bit_length() - 1
                cc &= cc - 1
                # a neighbour of x inside c gives the 4th vertex
                ymask = adj[x] & c
                if ymask:
                    y = (ymask & -ymask).bit_length() - 1
                    return [u, v, x, y]
    return None

def _find_clique(adj, n, k):
    """Return a k-clique (list of k vertices) in the graph (adj,n), or None."""
    full = (1 << n) - 1
    found = [None]

    def rec(cand, start, chosen):
        if found[0] is not None:
            return
        if len(chosen) == k:
            found[0] = list(chosen)
            return
        need = k - len(chosen)
        usable = cand & (~((1 << start) - 1))
        if usable.bit_count() < need:
            return
        c = usable
        while c:
            v = (c & -c).bit_length() - 1
            c &= c - 1
            chosen.append(v)
            rec(cand & adj[v], v + 1, chosen)
            chosen.pop()
            if found[0] is not None:
                return

    rec(full, 0, [])
    return found[0]

def _find_indep6(adj, n):
    """Return an independent set of 6 vertices in (adj,n), or None if alpha < 6."""
    full = (1 << n) - 1
    comp = [full & ~adj[i] & ~(1 << i) for i in range(n)]
    return _find_clique(comp, n, 6)

def _is_valid(adj, n):
    return _find_k4(adj, n) is None and _find_indep6(adj, n) is None

def _repair(adj, n, max_iters=50000):
    """Flip random edges to eliminate K4s and independent 6-sets."""
    for _ in range(max_iters):
        if _time_left() <= 2.0:
            break
        k4 = _find_k4(adj, n)
        if k4 is not None:
            a, b = _RNG.sample(k4, 2)
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            continue
        i6 = _find_indep6(adj, n)
        if i6 is not None:
            a, b = _RNG.sample(i6, 2)
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
            continue
        return adj, n, True
    return adj, n, _is_valid(adj, n)

def _try_add_vertex(adj, n, p):
    """Try to add one vertex with random neighbourhood (prob p) and re-repair."""
    new_n = n + 1
    new_adj = adj + [0]
    for i in range(n):
        if _RNG.random() < p:
            new_adj[i] |= (1 << new_n)
            new_adj[new_n] |= (1 << i)
    new_adj, new_n, ok = _repair(new_adj, new_n)
    return new_adj, new_n, ok

def _to_dict(n, adj):
    edges = []
    for i in range(n):
        m = adj[i]
        for j in range(i + 1, n):
            if (m >> j) & 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}

def construct():
    # Start with Paley 17
    n, adj = _paley17()
    best_n = n
    best_adj = adj[:]
    
    # Try to grow the graph
    # We try different connection probabilities for the new vertex
    # and multiple attempts for each probability
    # If an addition succeeds, we continue growing
    # If it fails, we try a different probability or restart growth from best
    
    while _time_left() > 5.0:
        grown = False
        # Try a range of probabilities
        # p=0.5 is a good start, but we can explore around it
        for p in [0.5, 0.45, 0.55, 0.4, 0.6]:
            if _time_left() <= 5.0:
                break
            # Try multiple random neighborhoods for this p
            for _ in range(10):
                if _time_left() <= 5.0:
                    break
                new_adj, new_n, ok = _try_add_vertex(adj, n, p)
                if ok and new_n > n:
                    adj, n = new_adj, new_n
                    if n > best_n:
                        best_n = n
                        best_adj = adj[:]
                    grown = True
                    break # Restart growth loop with new n
        if not grown:
            # If we couldn't add a vertex with any of these probabilities,
            # maybe try a slightly different approach or break
            # For now, break to avoid infinite loop of failures
            # We could also try to repair the current graph more aggressively
            # but if it's already valid, we just can't grow it easily
            break

    # Final verification
    if _is_valid(best_adj, best_n):
        return _to_dict(best_n, best_adj)
    else:
        # Fallback to Paley 17
        return _to_dict(17, _paley17()[1])

# EVOLVE-BLOCK-START
def construct():
    return construct()
# EVOLVE-BLOCK-END
