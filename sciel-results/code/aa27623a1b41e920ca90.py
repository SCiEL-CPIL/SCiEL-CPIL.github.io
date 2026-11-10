import time
import random
import itertools

N = 16
ALL_PAIRS = list(itertools.combinations(range(N), 2))

# Precompute, for each vertex, the list of vertices at distance 2 (via a middle).
# dist2[u] = set of v such that there exists w with u-w and w-v, u!=v, u!=w, v!=w.
DIST2 = [set() for _ in range(N)]
for u in range(N):
    for w in range(N):
        if u == w:
            continue
        for v in range(N):
            if v == u or v == w:
                continue
            DIST2[u].add(v)

def _creates_cube(edge, adj):
    """Return True if adding edge (u,v) to adjacency `adj` (list of int bitsets)
    creates a 3-cube (Q3) as a subgraph.
    A cube contains the edge (u,v); the other 6 edges form a 6-cycle alternating
    u,v. So we need x,y,z with:
      u-x, x-y, y-v, v-z, z-w, w-u  -> here w = x (6-cycle back to u)
    i.e. a 6-cycle u - x - y - v - z - x? No, that's not a 6-cycle (x repeats).
    Correct: cube with edge uv: the 6 other edges are a 6-cycle
      u - x - y - v - z - w - u
    with x,y,z,w distinct from u,v and from each other.
    """
    u, v = edge
    # u-x and v-z and w-u and y-v  => x in N(u), z in N(v), w in N(u), y in N(v)
    # and x-y, v-z, z-w, w-u edges.
    # The cycle edges: (u,x),(x,y),(y,v),(v,z),(z,w),(w,u).
    # So x,w in N(u); y,z in N(v); and x-y, v-z, z-w, w-u present.
    Nu = adj[u]
    Nv = adj[v]
    # candidate x in N(u), candidate y in N(v) with x-y edge
    # candidate z in N(v), candidate w in N(u) with z-w edge
    # and x,y,z,w all distinct and not u,v.
    # Iterate over pairs (x,w) in N(u), (y,z) in N(v).
    # 6-cycle: u-x-y-v-z-w-u. Edges needed: x-y, v-z, z-w, w-u(=u-w).
    # u-w and v-z are within N(u)/N(v) already? w in N(u) means u-w yes.
    # z in N(v) means v-z yes. So we need x-y and z-w.
    # So: choose x,w in N(u), y,z in N(v), with x-y edge and z-w edge, all distinct.
    # Equivalently: there exist x in N(u), w in N(u), y in N(v), z in N(v),
    #   x-y in adj, z-w in adj, {x,w,y,z} distinct and not u,v.
    Nu_list = [a for a in range(N) if a != u and a != v and (Nu >> a) & 1]
    Nv_list = [a for a in range(N) if a != u and a != v and (Nv >> a) & 1]
    # Precompute edge presence for the new graph: adj plus (u,v).
    # We'll check x-y and z-w using adj (these don't involve u or v edges being added,
    # since x,y in N(v) and N(u) respectively... x-y: x in N(u), y in N(v); not u/v.
    # z-w: z in N(v), w in N(u); not u/v.)
    for x in Nu_list:
        for y in Nv_list:
            if y == x:
                continue
            if not ((adj[x] >> y) & 1):
                continue
            # now need z in N(v), w in N(u), z != x,y, w != x,y, z-w edge.
            for z in Nv_list:
                if z == x or z == y:
                    continue
                for w in Nu_list:
                    if w == x or w == y or w == z:
                        continue
                    if ((adj[z] >> w) & 1):
                        return True
    return False


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    best_edges = None
    best_count = -1

    def time_left():
        return deadline - time.time()

    # Greedy with randomized edge order, many trials.
    rng = random.Random(12345)

    trial = 0
    while time_left() > 2.0:
        trial += 1
        # Random edge ordering (shuffle a copy)
        order = ALL_PAIRS[:]
        rng.shuffle(order)

        adj = [0] * N
        edges = []
        count = 0
        # Optional: seed with a random base to diversify
        for e in order:
            if time_left() < 0.5:
                break
            u, v = e
            if (adj[u] >> v) & 1:
                continue
            if not _creates_cube(e, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
                count += 1

        if count > best_count:
            best_count = count
            best_edges = edges

    if best_edges is None:
        # fallback: K3,13
        best_edges = [[a, b] for a in range(3) for b in range(3, 16)]
    return best_edges
