"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time, itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

_BASE = {tuple(e) for e in P['edges']}
_N = P['n']
# All candidate bipartite (parity-different) non-base edges
_CANDS = []
for u in range(_N):
    for v in range(u + 1, _N):
        if (u - v) % 2 and (u, v) not in _BASE:
            _CANDS.append((u, v))

def _bipartite_ok(edges):
    # Triangle-free is guaranteed if graph stays bipartite.
    # Base path is bipartite (even/odd). Adding only even-odd edges keeps it bipartite.
    return True

def _diam(edges):
    # Build adjacency and BFS from each vertex; return max shortest distance.
    adj = [[] for _ in range(_N)]
    for u, v in _BASE:
        adj[u].append(v); adj[v].append(u)
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    INF = 10**9
    best = 0
    for s in range(_N):
        dist = [-1] * _N
        dist[s] = 0
        q = [s]
        head = 0
        while head < len(q):
            x = q[head]; head += 1
            for y in adj[x]:
                if dist[y] < 0:
                    dist[y] = dist[x] + 1
                    q.append(y)
        for d in dist:
            if d < 0:
                return INF  # disconnected
            if d > best:
                best = d
    return best

def _valid(edges):
    if not _bipartite_ok(edges):
        return False
    if _diam(edges) <= 4:
        return True
    return False

# Precomputed candidate "structural" families to search (small, fast).
# Family A: pick k even-hubs and k odd-hubs, connect hubs to a spread of vertices.
# We do a bounded exhaustive search over small edge-counts first.

def _search():
    deadline = time.time() + 80.0
    best = None
    # Try edge counts from small upward.
    # To keep it fast, restrict candidate set to "long" edges (large span) which help diameter most.
    # But to be safe for correctness, include all bipartite candidates; limit combinations.
    # Use a greedy+local-search as the main structural method.
    
    # Greedy: start empty, repeatedly add the edge that most reduces current diameter.
    def _greedy():
        cur = []
        cur_diam = _diam(cur)
        while cur_diam > 4 and time.time() < deadline:
            best_edge = None
            best_d = cur_diam
            for e in _CANDS:
                if e in cur:
                    continue
                # quick: skip if same parity (already filtered) 
                d = _diam(cur + [e])
                if d < best_d:
                    best_d = d
                    best_edge = e
                    if d <= 4:
                        break
            if best_edge is None:
                break
            cur.append(best_edge)
            cur_diam = best_d
        return cur, cur_diam
    
    g, gd = _greedy()
    if gd <= 4:
        best = g
    else:
        # fallback: a known-good larger construction
        best = None
    
    # Local search: try to remove edges, and swap edges to reduce count.
    if best is not None:
        cur = list(best)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # removal pass
            for i in range(len(cur) - 1, -1, -1):
                if time.time() > deadline:
                    break
                cand = cur[:i] + cur[i+1:]
                if _valid(cand):
                    cur = cand
                    improved = True
                    break
            if time.time() > deadline:
                break
            # swap pass: remove one, add another different one
            for i in range(len(cur)):
                if time.time() > deadline:
                    break
                for e in _CANDS:
                    if e in cur:
                        continue
                    cand = cur[:i] + cur[i+1:] + [e]
                    if _valid(cand):
                        cur = cand
                        improved = True
                        break
                if improved:
                    break
        best = cur
    
    # Final verification
    if best is not None and _valid(best):
        return [[int(u), int(v)] for u, v in best]
    return None

def construct():
    res = _search()
    if res is None:
        # Guaranteed fallback: connect 0 and 23 to a central bipartite hub pair.
        # Use edges (0,11),(0,13),(1,12),(1,14),(22,11),(22,13),(23,12),(23,14)
        # All even-odd? 0 even,11 odd ok; 0 even,13 odd ok; 1 odd,12 even ok; 1 odd,14 even ok;
        # 22 even,11 odd ok; 22 even,13 odd ok; 23 odd,12 even ok; 23 odd,14 even ok.
        fb = [[0,11],[0,13],[1,12],[1,14],[22,11],[22,13],[23,12],[23,14]]
        if _valid(fb):
            return [[int(u), int(v)] for u, v in fb]
        # last resort: all bipartite edges
        return [[int(u), int(v)] for u, v in _CANDS]
    return res
# EVOLVE-BLOCK-END
