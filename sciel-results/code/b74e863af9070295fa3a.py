import random
import time
import math
import sys

N = 16
TARGET_EDGES = 65
DEADLINE = 135.0  # Safety margin before 150s kill

def _build_adj_and_deg():
    """Build adjacency matrix and degree list from a given edge set."""
    adj = [[False]*N for _ in range(N)]
    deg = [0]*N
    for u, v in edge_set:
        adj[u][v] = adj[v][u] = True
        deg[u] += 1
        deg[v] += 1
    return adj, deg

def _build_triangles(adj):
    """Build the set of all triangles as tuples (a,b,c) with a<b<c."""
    tris = set()
    for i in range(N):
        for j in range(i+1, N):
            if not adj[i][j]:
                continue
            for k in range(j+1, N):
                if adj[i][k] and adj[j][k]:
                    tris.add((i, j, k))
    return tris

def _max_sum(tris, deg):
    """Max degree-sum over all triangles."""
    if not tris:
        return 0
    return max(deg[a] + deg[b] + deg[c] for a, b, c in tris)

def _apply_swap(adj, deg, tris, e_rm, e_add):
    """
    Apply an edge swap: remove e_rm=(u,v), add e_add=(x,y).
    Returns (new_deg, new_tris).
    """
    u, v = e_rm
    x, y = e_add
    
    # Degree changes
    deg[u] -= 1
    deg[v] -= 1
    deg[x] += 1
    deg[y] += 1
    
    # Adjacency changes
    adj[u][v] = adj[v][u] = False
    adj[x][y] = adj[y][x] = True
    
    # Identify affected triangles: those containing (u,v) or (x,y)
    affected = set()
    for a, b, c in tris:
        tset = {a, b, c}
        if (u in tset and v in tset) or (x in tset and y in tset):
            affected.add((a, b, c))
    
    new_tris = set(tris - affected)
    
    # Check for newly formed triangles containing (x,y)
    for w in range(N):
        if w == x or w == y:
            continue
        if adj[x][w] and adj[y][w]:
            new_tris.add(tuple(sorted((x, y, w))))
    
    return deg, new_tris

def _undo_swap(adj, deg, e_rm, e_add, old_deg, old_tris):
    """Undo a swap, restoring old state."""
    u, v = e_rm
    x, y = e_add
    deg[u] = old_deg[u]
    deg[v] = old_deg[v]
    deg[x] = old_deg[x]
    deg[y] = old_deg[y]
    adj[u][v] = adj[v][u] = True
    adj[x][y] = adj[y][x] = False
    return deg, old_tris

def _make_k88_plus_edge(extra_edge):
    """Create K_{8,8} plus one extra edge."""
    adj = [[False]*N for _ in range(N)]
    for i in range(8):
        for j in range(8, 16):
            adj[i][j] = adj[j][i] = True
    u, v = extra_edge
    adj[u][v] = adj[v][u] = True
    deg = [0]*N
    for i in range(N):
        for j in range(N):
            if adj[i][j]:
                deg[i] += 1
    tris = _build_triangles(adj)
    return adj, deg, tris

def _make_circulant():
    """Create a circulant graph with 65 edges."""
    # Try different connection sets
    # Total edges = N * |S| / 2. We need 65 edges.
    # 65 / 16 = 4.0625. So we need a mix.
    # Let's try S = {1, 2, 3, 4, 5, 6, 7, 8} -> 16*8/2 = 64 edges.
    # Add one more edge, e.g., (0, 8) is already there if 8 in S.
    # Let's try S = {1, 2, 3, 4, 5, 6, 7, 9} -> 64 edges.
    # Actually, let's just generate a random graph with 65 edges for diversity.
    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(all_edges)
    edges = set(all_edges[:TARGET_EDGES])
    adj = [[False]*N for _ in range(N)]
    deg = [0]*N
    for u, v in edges:
        adj[u][v] = adj[v][u] = True
        deg[u] += 1
        deg[v] += 1
    tris = _build_triangles(adj)
    return adj, deg, tris

def _hill_climb(adj, deg, tris, max_time, start_temp=3.0, end_temp=0.05, seed=12345):
    """Simulated annealing with incremental triangle updates."""
    t0 = time.time()
    rng = random.Random(seed)
    
    cur_score = _max_sum(tris, deg)
    best_score = cur_score
    best_adj = [row[:] for row in adj]
    best_deg = deg[:]
    best_tris = set(tris)
    
    edges = [(i,j) for i in range(N) for j in range(i+1, N) if adj[i][j]]
    nonedges = [(i,j) for i in range(N) for j in range(i+1, N) if not adj[i][j]]
    
    max_iters = 1000000
    cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
    temp = start_temp
    iters = 0
    stall_count = 0
    stall_threshold = 10000
    
    while iters < max_iters:
        # Time check
        if iters % 1000 == 0:
            if time.time() - t0 > max_time:
                break
        
        u, v = edges[rng.randrange(len(edges))]
        x, y = nonedges[rng.randrange(len(nonedges))]
        
        if (u == x and v == y) or (u == y and v == x):
            iters += 1
            temp *= cooling_rate
            continue
        
        old_deg = deg[:]
        old_tris = tris
        new_deg, new_tris = _apply_swap(adj, deg, tris, (u,v), (x,y))
        new_score = _max_sum(new_tris, new_deg)
        
        if new_score < cur_score:
            accept = True
        elif new_score == cur_score:
            accept = rng.random() < 0.3
        else:
            delta = new_score - cur_score
            if temp < 1e-9:
                accept = False
            else:
                accept = math.exp(-delta / temp) > rng.random()
        
        if accept:
            deg, tris = new_deg, new_tris
            cur_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = [row[:] for row in adj]
                best_deg = deg[:]
                best_tris = set(tris)
                stall_count = 0
            else:
                stall_count += 1
            
            # Update edge lists
            edges.remove((u,v))
            nonedges.append((u,v))
            nonedges.remove((x,y))
            edges.append((x,y))
        else:
            # Undo
            adj[u][v] = adj[v][u] = True
            adj[x][y] = adj[y][x] = False
            deg = old_deg
            tris = old_tris
        
        iters += 1
        temp *= cooling_rate
        
        # Optional: early stop if stalled too long and temp is low
        if stall_count > stall_threshold and temp < 0.1:
            break
            
    return best_adj, best_deg, best_tris, best_score

def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    best_overall_score = float('inf')
    best_overall_adj = None
    
    # Phase 1: K_{8,8} + edge (0,1)
    adj, deg, tris = _make_k88_plus_edge((0, 1))
    best_adj, best_deg, best_tris, best_score = _hill_climb(adj, deg, tris, DEADLINE - (time.time()-t0), seed=100)
    if best_score < best_overall_score:
        best_overall_score = best_score
        best_overall_adj = best_adj
    
    if time.time() - t0 > DEADLINE:
        return {'n': N, 'edges': [[i, j] for i in range(N) for j in range(i+1, N) if best_overall_adj[i][j]]}
    
    # Phase 2: K_{8,8} + edge (0, 8) (cross-partition, but 0 and 8 are in different parts in K8,8? No, 0 is in 0-7, 8 is in 8-15. So (0,8) is a cross edge already. We need an internal edge to add to K8,8 to get 65.
    # K8,8 has 64 edges. Adding (0,1) gives 65.
    # Let's try a different base graph.
    
    # Phase 2: Random graph
    adj, deg, tris = _make_circulant()
    best_adj, best_deg, best_tris, best_score = _hill_climb(adj, deg, tris, DEADLINE - (time.time()-t0), seed=200)
    if best_score < best_overall_score:
        best_overall_score = best_score
        best_overall_adj = best_adj
    
    if time.time() - t0 > DEADLINE:
        return {'n': N, 'edges': [[i, j] for i in range(N) for j in range(i+1, N) if best_overall_adj[i][j]]}
    
    # Phase 3: Another random restart
    adj, deg, tris = _make_circulant()
    best_adj, best_deg, best_tris, best_score = _hill_climb(adj, deg, tris, DEADLINE - (time.time()-t0), seed=300)
    if best_score < best_overall_score:
        best_overall_score = best_score
        best_overall_adj = best_adj
    
    return {'n': N, 'edges': [[i, j] for i in range(N) for j in range(i+1, N) if best_overall_adj[i][j]]}
