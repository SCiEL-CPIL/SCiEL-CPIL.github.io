import time
import random
from itertools import combinations

N = 20
C = 20
DEADLINE = 120.0
START = time.time()

def elapsed():
    return time.time() - START

def make_adj(col_edges):
    # adj[u][v] = colour or -1
    adj = [[-1]*N for _ in range(N)]
    for u, v, c in col_edges:
        adj[u][v] = c
        adj[v][u] = c
    return adj

def all_four_cycles(adj):
    # yield each 4-cycle as tuple of 4 edges (u,v,c)
    # A 4-cycle: vertices a,b,c,d distinct with edges ab,bc,cd,da
    # To avoid duplicates, enumerate unordered sets of 4 vertices and the 3 cycle structures
    n = N
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    # three ways to form a 4-cycle on {a,b,c,d}
                    for (x,y,z,w) in [(a,b,c,d), (a,b,d,c), (a,c,b,d)]:
                        if adj[x][y] >= 0 and adj[y][z] >= 0 and adj[z][w] >= 0 and adj[w][x] >= 0:
                            yield (x,y,adj[x][y]), (y,z,adj[y][z]), (z,w,adj[z][w]), (w,x,adj[w][x])

def is_rainbow(col_edges):
    adj = make_adj(col_edges)
    for cyc in all_four_cycles(adj):
        cols = set(e[2] for e in cyc)
        if len(cols) != 4:
            return False
    return True

def count_edges(col_edges):
    return len(col_edges)

def _seed_star():
    return [[0, v, 0] for v in range(1, N)]

def _try_add(col_edges, adj, u, v, c):
    if u == v: return False
    if adj[u][v] >= 0: return False
    # check: adding edge (u,v,c) must not create any 4-cycle with repeated colour
    # New 4-cycles involving this edge: for each w such that there's a path u-?-?-v of length 3
    # i.e., find x,y with adj[u][x]>=0, adj[x][y]>=0, adj[y][v]>=0, and x!=v,y!=u, x!=y
    # The cycle u-x-y-v-u would have colours c, adj[u][x], adj[x][y], adj[y][v]
    # Need all 4 distinct.
    for x in range(N):
        if x == v or adj[u][x] < 0: continue
        for y in range(N):
            if y == u or y == x or adj[x][y] < 0 or adj[y][v] < 0: continue
            cols = {c, adj[u][x], adj[x][y], adj[y][v]}
            if len(cols) != 4:
                return False
    return True

def _add_edge(col_edges, adj, u, v, c):
    col_edges.append([u, v, c])
    adj[u][v] = c
    adj[v][u] = c

def construct():
    random.seed(42)
    best = _seed_star()
    best_count = count_edges(best)
    # greedy: try to add all possible edges with some colour
    col_edges = list(best)
    adj = make_adj(col_edges)
    # order edges randomly
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(all_pairs)
    for u, v in all_pairs:
        if elapsed() > DEADLINE - 10:
            break
        if adj[u][v] >= 0:
            continue
        # try each colour
        colours = list(range(C))
        random.shuffle(colours)
        added = False
        for c in colours:
            if _try_add(col_edges, adj, u, v, c):
                _add_edge(col_edges, adj, u, v, c)
                added = True
                break
    if count_edges(col_edges) > best_count:
        best = col_edges
        best_count = count_edges(best)
    # local search: try to add more edges (maybe with re-colouring not done; just try remaining)
    # second pass with different random order
    col_edges2 = list(best)
    adj2 = make_adj(col_edges2)
    random.shuffle(all_pairs)
    for u, v in all_pairs:
        if elapsed() > DEADLINE - 5:
            break
        if adj2[u][v] >= 0:
            continue
        colours = list(range(C))
        random.shuffle(colours)
        for c in colours:
            if _try_add(col_edges2, adj2, u, v, c):
                _add_edge(col_edges2, adj2, u, v, c)
                break
    if count_edges(col_edges2) > best_count:
        best = col_edges2
        best_count = count_edges(col_edges2)
    # final verification
    if not is_rainbow(best):
        # fallback to seed
        return _seed_star()
    return best
