import random
import time

N = 20
C = 20
DEADLINE = time.time() + 140.0  # safety margin under 150s


def _common_neighbours(adj, u, v):
    """List of w such that (u,w) and (v,w) are both present."""
    au = adj[u]
    av = adj[v]
    # iterate the smaller set
    if len(au) > len(av):
        au, av = av, au
    out = []
    for w in au:
        if w in av:
            out.append(w)
    return out


def _edge_ok(adj, colour_of, u, v, c):
    """Check whether adding edge (u,v) with colour c keeps every 4-cycle rainbow."""
    for w in _common_neighbours(adj, u, v):
        cuw = colour_of[u][w]
        cvw = colour_of[v][w]
        # the new 4-cycle edges are (u,v)=c, (u,w)=cuw, (v,w)=cvw, and the fourth
        # edge of the cycle is (w,?) ... but a 4-cycle through u,v,w needs a 4th vertex.
        pass
    # Correct approach: a 4-cycle containing edge (u,v) is u-v-x-y-u.
    # x must be a neighbour of v, y a neighbour of u, and (x,y) present, x!=y, x!=u, y!=v.
    # The four edges: (u,v)=c, (v,x)=cvx, (x,y)=cxy, (y,u)=cyu.
    # All four colours must be distinct.
    for x in adj[v]:
        if x == u:
            continue
        for y in adj[u]:
            if y == v or y == x:
                continue
            if x in adj[y]:
                cvx = colour_of[v][x]
                cxy = colour_of[x][y]
                cyu = colour_of[y][u]
                if c == cvx or c == cxy or c == cyu:
                    return False
                if cvx == cxy or cvx == cyu or cxy == cyu:
                    # pre-existing cycle not rainbow -> shouldn't happen if we maintained it,
                    # but guard anyway
                    return False
    return True


def _try_build(seed, edge_order):
    random.Random(seed).shuffle(edge_order)
    adj = [set() for _ in range(N)]
    colour_of = [[-1] * N for _ in range(N)]
    placed = []
    for (u, v) in edge_order:
        if u in adj[v]:
            continue
        # try colours in a randomised order, pick first valid
        cols = list(range(C))
        random.Random(seed + u * 31 + v).shuffle(cols)
        for c in cols:
            if _edge_ok(adj, colour_of, u, v, c):
                adj[u].add(v)
                adj[v].add(u)
                colour_of[u][v] = c
                colour_of[v][u] = c
                placed.append([u, v, c])
                break
    return placed


def construct():
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    best = []
    seed = 12345
    while time.time() < DEADLINE:
        cand = _try_build(seed, list(all_edges))
        if len(cand) > len(best):
            best = cand
        seed += 1
    return best
