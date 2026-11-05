import time
import random

def _clebsch_edges(offset=0):
    """Generate edges of Clebsch graph (16 vertices, degree 5, C4-free, alpha=5)."""
    edges = []
    verts = [i for i in range(32) if bin(i).count('1') % 2 == 0]
    for i in range(16):
        for j in range(i + 1, 16):
            diff = verts[i] ^ verts[j]
            if bin(diff).count('1') == 2:
                edges.append([offset + i, offset + j])
    return edges

def _has_c4(adj, n):
    """Check if any two vertices share 2+ common neighbors."""
    for u in range(n):
        for v in adj[u]:
            if v > u:
                if len(adj[u] & adj[v]) >= 2:
                    return True
    return False

def _has_indep_11(adj, n, deadline):
    """Check if an independent set of size 11 exists. Returns True if found."""
    def backtrack(remaining, current):
        if time.time() > deadline:
            return False
        if len(current) >= 11:
            return True
        if not remaining:
            return False
        if len(current) + len(remaining) < 11:
            return False
        v = min(remaining, key=lambda x: len(adj[x] & remaining))
        remaining.discard(v)
        if backtrack(remaining - adj[v], current + [v]):
            return True
        if backtrack(remaining, current):
            return True
        remaining.add(v)
        return False
    return backtrack(set(range(n)), [])

def construct():
    start_time = time.time()
    deadline = start_time + 130

    # Base: two disjoint Clebsch graphs (n=32, alpha=10, C4-free)
    edges = _clebsch_edges(0) + _clebsch_edges(16)
    n = 32
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    best_n = n
    best_edges = edges[:]

    # Validate base
    if _has_c4(adj, n) or _has_indep_11(adj, n, deadline):
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

    # Try to extend by adding vertices
    random.seed(42)
    while time.time() < deadline - 10 and n < 43:
        improved = False
        for _ in range(300):
            if time.time() > deadline - 10:
                break
            new_n = n + 1
            new_adj = [set(s) for s in adj]
            new_adj.append(set())

            order = list(range(n))
            random.shuffle(order)
            for v in order:
                ok = True
                for u in new_adj[new_n - 1]:
                    if len(adj[u] & adj[v]) >= 2:
                        ok = False
                        break
                if ok:
                    new_adj[new_n - 1].add(v)
                    new_adj[v].add(new_n - 1)

            if not new_adj[new_n - 1]:
                continue
            if _has_c4(new_adj, new_n):
                continue
            if _has_indep_11(new_adj, new_n, deadline):
                continue

            for v in new_adj[new_n - 1]:
                edges.append([new_n - 1, v])
            adj = new_adj
            n = new_n
            best_n = n
            best_edges = edges[:]
            improved = True
            break

        if not improved:
            break

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
