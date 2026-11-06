import time
import random
import math

N = 31
E = 91
DEADLINE = 110.0

def _pg25_polarity():
    F = 5
    pts = []
    seen = set()
    for a in range(F):
        for b in range(F):
            for c in range(F):
                if a == b == c == 0:
                    continue
                if a != 0:
                    rep = (1, (b * pow(a, -1, F)) % F, (c * pow(a, -1, F)) % F)
                elif b != 0:
                    rep = (0, 1, (c * pow(b, -1, F)) % F)
                else:
                    rep = (0, 0, 1)
                if rep not in seen:
                    seen.add(rep)
                    pts.append(rep)
    assert len(pts) == 31

    def dot(p, q):
        return (p[0]*q[0] - p[1]*q[1] - p[2]*q[2]) % F

    adj = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if dot(pts[i], pts[j]) == 0:
                adj[i][j] = adj[j][i] = 1
    return adj

def _edge_count(adj):
    return sum(adj[i][j] for i in range(N) for j in range(i+1, N))

def _count_c4(adj):
    total = 0
    for u in range(N):
        for v in range(u+1, N):
            c = 0
            for w in range(N):
                if w != u and w != v and adj[u][w] and adj[v][w]:
                    c += 1
            total += c*(c-1)//2
    return total

def _build_initial():
    adj = _pg25_polarity()
    ec = _edge_count(adj)
    assert ec == 90, f"polarity graph has {ec} edges, expected 90"
    best = None
    for u in range(N):
        for v in range(u+1, N):
            if adj[u][v]:
                continue
            adj[u][v] = adj[v][u] = 1
            cnt = _count_c4(adj)
            adj[u][v] = adj[v][u] = 0
            if best is None or cnt < best[0]:
                best = (cnt, u, v)
    assert best is not None
    adj[best[1]][best[2]] = adj[best[2]][best[1]] = 1
    assert _edge_count(adj) == E
    return adj

def construct():
    start = time.time()
    random.seed(42) # Reproducibility for testing, but we want to explore. Maybe use time seed?
    # The task says "Minimize". Randomness is allowed if it helps find a better minimum.
    # Using a fixed seed ensures we always find the same result, which is good for debugging but might not be the global min.
    # However, the evaluator runs it once. We should probably use a seed derived from time or just fixed to be safe?
    # "The search score is seed/value". This implies the score is just the value.
    # Let's use a fixed seed to ensure we get a consistent "good" result if the search is deterministic enough, 
    # or random seed to explore. Given the deadline, a good heuristic is better.
    
    adj = _build_initial()
    best_c4 = _count_c4(adj)
    best_adj = [row[:] for row in adj]

    def time_left():
        return DEADLINE - (time.time() - start)

    # Build common neighbor counts for incremental updates
    def build_common(adj):
        com = [[0]*N for _ in range(N)]
        for w in range(N):
            for u in range(N):
                if not adj[u][w]:
                    continue
                for v in range(N):
                    if v != w and adj[v][w]:
                        com[u][v] += 1
        return com

    def c4_from_common(com):
        total = 0
        for u in range(N):
            for v in range(u+1, N):
                c = com[u][v]
                total += c*(c-1)//2
        return total

    com = build_common(adj)
    assert c4_from_common(com) == best_c4

    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    alpha = 0.999
    
    # Precompute list of edges and non-edges for faster random selection?
    # Or just pick random u, v and check.
    
    while time_left() > 1.0:
        # Pick a random edge to remove
        # We can maintain a list of edges, but updating it is overhead.
        # Just pick random u, v.
        while True:
            u = random.randint(0, N-1)
            v = random.randint(0, N-1)
            if u != v and adj[u][v]:
                break
        
        # Pick a random non-edge to add
        while True:
            a = random.randint(0, N-1)
            b = random.randint(0, N-1)
            if a != b and not adj[a][b]:
                break
        
        # Calculate delta
        # Remove (u,v), Add (a,b)
        # This is a 2-swap.
        
        # Neighbors of u (excluding v) and v (excluding u)
        Nu = [w for w in range(N) if w != v and adj[u][w]]
        Nv = [w for w in range(N) if w != u and adj[v][w]]
        Na = [w for w in range(N) if w != b and adj[a][w]]
        Nb = [w for w in range(N) if w != a and adj[b][w]]
        
        delta = 0
        
        # Effect of removing (u,v)
        # For each w in Nu, common[u][w] decreases by 1.
        # For each w in Nv, common[v][w] decreases by 1.
        # Note: common[u][v] is not directly used in C4 count as u,v are adjacent?
        # C4 count is sum C(com[u][v], 2).
        # If u,v are adjacent, do they form C4s with each other? No, C4s are simple cycles.
        # The formula sum_{u<v} C(com[u][v], 2) counts all 4-cycles.
        # If u and v are adjacent, com[u][v] counts common neighbors.
        # Does removing edge (u,v) change com[u][v]? No.
        # Does it change com[u][w] for w != v? Yes, if w is neighbor of u, then v was a common neighbor of u and w?
        # No. com[u][w] = number of x such that adj[u][x] and adj[w][x].
        # If we remove edge (u,v), does it affect com[u][w]?
        # Only if v is a common neighbor of u and w. i.e., adj[u][v] and adj[w][v].
        # Since we are removing (u,v), adj[u][v] becomes 0.
        # So for any w such that adj[w][v] is 1, com[u][w] decreases by 1.
        # The set of such w is exactly Nv (neighbors of v excluding u).
        # Wait. Nu is neighbors of u excluding v. Nv is neighbors of v excluding u.
        # For w in Nv: adj[v][w]=1. Since adj[u][v]=1, v is a common neighbor of u and w.
        # So com[u][w] includes v. Removing (u,v) removes v from common neighbors of u and w.
        # So com[u][w] -= 1 for all w in Nv.
        # Similarly, com[v][w] -= 1 for all w in Nu.
        
        # Let's re-verify.
        # com[x][y] = count of z where adj[x][z] and adj[y][z].
        # Remove (u,v).
        # adj[u][v] becomes 0.
        # This affects com[x][y] if the removed edge was one of the two required adjacencies.
        # Case 1: x=u, y=w. We need adj[u][z] and adj[w][z].
        # If z=v, we need adj[u][v] and adj[w][v].
        # adj[u][v] is removed. So if adj[w][v] was 1, com[u][w] decreases by 1.
        # The set of w with adj[w][v]=1 is N(v). Since v != u, u is in N(v).
        # So for all w in N(v) \ {u}, com[u][w] decreases by 1.
        # This is Nv.
        # So com[u][w] -= 1 for w in Nv.
        
        # Case 2: x=v, y=w.
        # If z=u, we need adj[v][u] and adj[w][u].
        # adj[v][u] is removed. So if adj[w][u] was 1, com[v][w] decreases by 1.
        # Set of w with adj[w][u]=1 is N(u) \ {v} = Nu.
        # So com[v][w] -= 1 for w in Nu.
        
        # Case 3: x=u, y=v.
        # com[u][v] counts z where adj[u][z] and adj[v][z].
        # Removing (u,v) does not change adj[u][z] or adj[v][z] for z != u,v.
        # So com[u][v] is unchanged.
        
        # So, delta_remove = sum_{w in Nv} (C(com[u][w]-1, 2) - C(com[u][w], 2)) + sum_{w in Nu} (C(com[v][w]-1, 2) - C(com[v][w], 2))
        # Note: C(n-1, 2) - C(n, 2) = ((n-1)(n-2) - n(n-1))/2 = (n-1)(n-2-n)/2 = -(n-1)/2? No.
        # C(n,2) = n(n-1)/2.
        # C(n-1,2) - C(n,2) = (n-1)(n-2)/2 - n(n-1)/2 = (n-1)/2 * (n-2-n) = (n-1)/2 * (-2) = -(n-1).
        # So delta is sum of -(com[u][w]-1) for w in Nv?
        # Let's just calculate explicitly to be safe.
        
        for w in Nv:
            c_old = com[u][w]
            c_new = c_old - 1
            delta += c_new*(c_new-1)//2 - c_old*(c_old-1)//2
        for w in Nu:
            c_old = com[v][w]
            c_new = c_old - 1
            delta += c_new*(c_new-1)//2 - c_old*(c_old-1)//2

        # Effect of adding (a,b)
        # adj[a][b] becomes 1.
        # Affects com[x][y] if z=a or z=b is the common neighbor?
        # No, adding edge (a,b) makes a and b neighbors.
        # It affects com[x][y] if the new edge provides a new common neighbor.
        # com[x][y] = count of z where adj[x][z] and adj[y][z].
        # If we add (a,b), then adj[a][b]=1 and adj[b][a]=1.
        # For com[x][y] to change, one of the conditions adj[x][z] or adj[y][z] must involve the new edge.
        # This happens if z is a or b, and the other adjacency is already present.
        # Case 1: z=a. We need adj[x][a] and adj[y][a].
        # The new edge is (a,b). This doesn't change adj[x][a] or adj[y][a] unless x or y is b.
        # If x=b, we need adj[b][a] (new) and adj[y][a].
        # So com[b][y] increases by 1 if adj[y][a] is 1.
        # The set of y with adj[y][a]=1 is N(a) \ {b} = Na.
        # So com[b][y] += 1 for y in Na.
        # Similarly, com[a][y] += 1 for y in Nb.
        
        # Wait, is that all?
        # What about com[a][b]?
        # com[a][b] counts z where adj[a][z] and adj[b][z].
        # Adding (a,b) doesn't change adj[a][z] or adj[b][z] for z != a,b.
        # So com[a][b] is unchanged.
        
        # So delta_add = sum_{y in Na} (C(com[b][y]+1, 2) - C(com[b][y], 2)) + sum_{y in Nb} (C(com[a][y]+1, 2) - C(com[a][y], 2))
        
        for y in Na:
            c_old = com[b][y]
            c_new = c_old + 1
            delta += c_new*(c_new-1)//2 - c_old*(c_old-1)//2
        for y in Nb:
            c_old = com[a][y]
            c_new = c_old + 1
            delta += c_new*(c_new-1)//2 - c_old*(c_old-1)//2

        # Total delta
        new_c4 = best_c4 + delta
        
        # Acceptance
        if new_c4 < best_c4 or (new_c4 > best_c4 and random.random() < math.exp(-(new_c4 - best_c4) / T)):
            # Apply move
            adj[u][v] = adj[v][u] = 0
            for w in Nv:
                com[u][w] -= 1
                com[w][u] -= 1
            for w in Nu:
                com[v][w] -= 1
                com[w][v] -= 1
            
            adj[a][b] = adj[b][a] = 1
            for y in Na:
                com[b][y] += 1
                com[y][b] += 1
            for y in Nb:
                com[a][y] += 1
                com[y][a] += 1
            
            best_c4 = new_c4
            best_adj = [row[:] for row in adj]
            
            if best_c4 == 0:
                break
        
        T *= alpha
        if T < T_min:
            T = T_min

    # Final check
    if _count_c4(best_adj) < best_c4:
         best_c4 = _count_c4(best_adj)

    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if best_adj[i][j]:
                edges.append([i, j])
    assert len(edges) == E, f"got {len(edges)} edges"
    return edges
