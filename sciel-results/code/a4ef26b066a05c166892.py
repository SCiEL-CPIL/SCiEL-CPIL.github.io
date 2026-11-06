import time
import random
import sys

N = 40
DEADLINE = 110.0  # safety margin below 120s

def _has_c6(adj):
    # adj: list of N ints, adj[i] bitmask of neighbors of i
    for u in range(N):
        bu = adj[u]
        # iterate neighbors of u
        v = (bu & -bu).bit_length() - 1 if bu else -1
        # use a loop over set bits
        mask = bu
        while mask:
            lb = mask & -mask
            v = lb.bit_length() - 1
            mask ^= lb
            # common neighbors of N(u)\{v} and N(v)\{u}
            cu = bu & ~lb          # N(u) without v
            cv = adj[v] & ~((1 << u) | (1 << v))  # N(v) without u and v
            if (cu & cv).bit_count() >= 2:
                return True
    return False

def _greedy(order, adj, edges):
    for (u, v) in order:
        if adj[u] >> v & 1:
            continue
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edges.append((u, v))
        if _has_c6(adj):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            edges.pop()

def _local_search(adj, edges, rng, deadline):
    edge_set = set(edges)
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # 1-for-1 swaps
        for (u, v) in edges:
            if time.time() >= deadline:
                break
            if not (adj[u] >> v & 1):
                continue
            # try removing (u,v)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            edge_set.discard((u, v))
            # try adding some random non-edge
            for _ in range(8):
                a = rng.randrange(N)
                b = rng.randrange(N)
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                if (a, b) in edge_set:
                    continue
                if a == u and b == v:
                    continue
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                if not _has_c6(adj):
                    edge_set.add((a, b))
                    edges.append((a, b))
                    improved = True
                    break
                else:
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
            else:
                # restore (u,v)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
                continue
            # (u,v) stays removed; (a,b) added
            # rebuild edge list consistency
            # edges already updated: we appended (a,b); need to ensure (u,v) not in edges
            # edges is a list; remove (u,v) if present
            if (u, v) in edges:
                edges.remove((u, v))
        # 1-for-2 swaps: remove one edge, try adding two
        for _ in range(200):
            if time.time() >= deadline:
                break
            if not edges:
                break
            idx = rng.randrange(len(edges))
            (u, v) = edges[idx]
            if not (adj[u] >> v & 1):
                continue
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            edge_set.discard((u, v))
            added = 0
            added_edges = []
            for _ in range(30):
                a = rng.randrange(N)
                b = rng.randrange(N)
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                if (a, b) in edge_set:
                    continue
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                if not _has_c6(adj):
                    edge_set.add((a, b))
                    added_edges.append((a, b))
                    added += 1
                    if added == 2:
                        break
                else:
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
            if added == 2:
                edges.extend(added_edges)
                if (u, v) in edges:
                    edges.remove((u, v))
                improved = True
            else:
                # revert
                for (a, b) in added_edges:
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    edge_set.discard((a, b))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
    return edges

def construct():
    start = time.time()
    deadline = start + DEADLINE
    best_edges = None
    best_count = -1
    rng = random.Random(12345)

    while time.time() < deadline - 1.0:
        adj = [0] * N
        edges = []
        order = [(u, v) for u in range(N) for v in range(u + 1, N)]
        rng.shuffle(order)
        _greedy(order, adj, edges)
        cur = len(edges)
        if cur > best_count:
            best_count = cur
            best_edges = list(edges)
        # local search
        edges2 = _local_search(adj, list(edges), rng, deadline - 1.0)
        cur2 = len(edges2)
        if cur2 > best_count:
            best_count = cur2
            best_edges = list(edges2)

    # final safety
    if best_edges is None:
        # fallback: 8 copies of K5
        best_edges = []
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    best_edges.append([5 * k + a, 5 * k + b])
    # ensure list of lists
    return [[u, v] for (u, v) in best_edges]
