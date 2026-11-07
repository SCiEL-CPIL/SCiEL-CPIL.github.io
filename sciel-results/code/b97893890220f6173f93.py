"""Evolved program: dense 15-vertex graph avoiding equal-degree 3-path endpoints.

Strategy:
- The constraint "no simple 3-edge path s-u-v-t has deg(s)==deg(t)" is equivalent to:
  for every edge (u,v), the union of N(u) and N(v) contains at most ONE vertex of
  each degree value (since s in N(u)\{v}, t in N(v)\{u}, s!=t, and the path is simple).
- We search over graphs, ordering vertices by degree so that high-degree vertices
  carry the "burden" of constraints. A greedy construction plus local search
  (edge add/remove) finds dense valid graphs quickly.
"""
import random
import time

N = 15


def _is_valid(adj, deg):
    """Check the constraint efficiently using the N(u)∪N(v) degree-uniqueness rule."""
    for u in range(N):
        nu = adj[u]
        for v in nu:
            if v <= u:
                continue
            # union of neighbors of u and v
            seen = set()
            ok = True
            for w in nu:
                d = deg[w]
                if d in seen:
                    ok = False
                    break
                seen.add(d)
            if ok:
                for w in adj[v]:
                    d = deg[w]
                    if d in seen:
                        ok = False
                        break
                    seen.add(d)
            if not ok:
                return False
    return True


def _deg_from_adj(adj):
    return [len(a) for a in adj]


def _try_add(adj, deg, u, v, deadline):
    """Try to add edge (u,v) if it keeps validity. Returns True on success."""
    if u == v or v in adj[u]:
        return False
    # degrees of u and v change: old->old+1
    du, dv = deg[u], deg[v]
    # tentatively update
    adj[u].add(v)
    adj[v].add(u)
    deg[u] = du + 1
    deg[v] = dv + 1
    if _is_valid(adj, deg) and time.time() < deadline:
        return True
    # revert
    adj[u].discard(v)
    adj[v].discard(u)
    deg[u] = du
    deg[v] = dv
    return False


def _try_remove(adj, deg, u, v, deadline):
    if u == v or v not in adj[u]:
        return False
    du, dv = deg[u], deg[v]
    adj[u].discard(v)
    adj[v].discard(u)
    deg[u] = du - 1
    deg[v] = dv - 1
    if _is_valid(adj, deg) and time.time() < deadline:
        return True
    adj[u].add(v)
    adj[v].add(u)
    deg[u] = du
    deg[v] = dv
    return False


def _greedy_build(rng, deadline):
    """Build a valid graph by greedily adding edges in a randomized order."""
    adj = [set() for _ in range(N)]
    deg = [0] * N
    edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    rng.shuffle(edges)
    for (u, v) in edges:
        if time.time() > deadline:
            break
        _try_add(adj, deg, u, v, deadline)
    return adj, deg


def _local_search(adj, deg, rng, deadline):
    """Improve edge count via random add/remove moves."""
    best_edges = sum(len(a) for a in adj) // 2
    best_adj = [set(a) for a in adj]
    best_deg = list(deg)
    cur_edges = best_edges
    stall = 0
    while time.time() < deadline:
        # pick a random move
        if rng.random() < 0.6:
            # try to add a random missing edge
            u = rng.randrange(N)
            v = rng.randrange(N)
            if u != v and v not in adj[u]:
                if _try_add(adj, deg, u, v, deadline):
                    cur_edges += 1
                    stall = 0
                    if cur_edges > best_edges:
                        best_edges = cur_edges
                        best_adj = [set(a) for a in adj]
                        best_deg = list(deg)
                else:
                    stall += 1
        else:
            # try to remove a random edge (to enable future adds)
            u = rng.randrange(N)
            if not adj[u]:
                continue
            v = rng.choice(list(adj[u]))
            if _try_remove(adj, deg, u, v, deadline):
                cur_edges -= 1
                stall = 0
        if stall > 200:
            break
    return best_adj, best_deg, best_edges


def construct():
    deadline = time.time() + 70.0  # safety margin before 90s kill
    rng = random.Random(12345)

    best_adj = None
    best_deg = None
    best_edges = 0

    # Multiple restarts with different seeds
    for seed in range(200):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        adj, deg = _greedy_build(rng, deadline)
        cur_edges = sum(len(a) for a in adj) // 2
        if cur_edges > best_edges:
            best_edges = cur_edges
            best_adj = [set(a) for a in adj]
            best_deg = list(deg)

        # local search from this start
        if time.time() < deadline:
            la, ld, le = _local_search(adj, deg, rng, deadline)
            if le > best_edges:
                best_edges = le
                best_adj = la
                best_deg = ld

    if best_adj is None:
        # fallback: empty graph (always valid)
        return []

    # build edge list with strict int types
    edges = []
    for u in range(N):
        for v in best_adj[u]:
            if u < v:
                edges.append([int(u), int(v)])
    return edges


# Quick self-test (disabled in production to save time)
if __name__ == "__main__":
    t0 = time.time()
    e = construct()
    print("edges:", len(e), "time:", time.time() - t0)
