import random
import time
import math
import numpy as np

N = 16
TARGET_EDGES = 65
DEADLINE = 135.0  # safety margin before the 150s kill

def _score_from_adj(adj):
    """O(n^3) score: max deg(u)+deg(v)+deg(w) over all triangles."""
    deg = [0]*N
    for i in range(N):
        deg[i] = sum(adj[i])
    best = 0
    for i in range(N):
        for j in range(i+1, N):
            if adj[i][j]:
                for k in range(j+1, N):
                    if adj[i][k] and adj[j][k]:
                        s = deg[i] + deg[j] + deg[k]
                        if s > best:
                            best = s
    return best

def _count_edges(adj):
    return sum(sum(row) for row in adj) // 2

def _get_edges_and_nonedges(adj):
    edges = []
    nonedges = []
    for i in range(N):
        for j in range(i+1, N):
            if adj[i][j]:
                edges.append((i,j))
            else:
                nonedges.append((i,j))
    return edges, nonedges

def _build_triangles_and_index(adj, deg):
    """
    Returns:
    triangles: list of (i, j, k) with i<j<k
    tri_index: dict mapping edge (u,v) (sorted) to list of triangle indices
    """
    triangles = []
    tri_index = {}
    for i in range(N):
        for j in range(i+1, N):
            if adj[i][j]:
                for k in range(j+1, N):
                    if adj[i][k] and adj[j][k]:
                        idx = len(triangles)
                        triangles.append((i, j, k))
                        
                        # Add to index for each edge in the triangle
                        for u, v in [(i, j), (i, k), (j, k)]:
                            key = (min(u, v), max(u, v))
                            if key not in tri_index:
                                tri_index[key] = []
                            tri_index[key].append(idx)
    return triangles, tri_index

def _compute_score(triangles, deg):
    if not triangles:
        return 0
    best = 0
    for (i, j, k) in triangles:
        s = deg[i] + deg[j] + deg[k]
        if s > best:
            best = s
    return best

def _try_swap_score(adj, deg, triangles, tri_index, cur_score, e_remove, e_add):
    """
    Compute new score if we swap e_remove for e_add.
    Returns (new_score, new_triangles, new_tri_index) or None if invalid.
    """
    u, v = e_remove
    x, y = e_add
    
    if u > v: u, v = v, u
    if x > y: x, y = y, x
    
    if not adj[u][v]:
        return None
    if adj[x][y]:
        return None
    
    # New degrees
    new_deg = deg[:]
    new_deg[u] -= 1
    new_deg[v] -= 1
    new_deg[x] += 1
    new_deg[y] += 1
    
    # Triangles to remove: those containing edge (u,v)
    remove_set = set()
    if (u, v) in tri_index:
        for idx in tri_index[(u, v)]:
            remove_set.add(idx)
            
    # Triangles to add: those that would be formed by adding (x,y)
    # A triangle (x, y, z) is formed if adj[x][z] and adj[y][z] are True.
    # Note: if z is u or v, we must check if the edge to z is still present.
    # Since we only remove (u,v), edges (x,u), (x,v), (y,u), (y,v) are unchanged 
    # unless x==u or y==u etc. But (x,y) is a non-edge, so x!=y.
    # If z==u, edge (x,u) exists? 
    # The swap removes (u,v) and adds (x,y).
    # Edges incident to x or y (other than x-y) are unchanged.
    # Edges incident to u or v (other than u-v) are unchanged.
    # So for any z != x, y:
    # adj[x][z] is original adj[x][z]
    # adj[y][z] is original adj[y][z]
    # The only change is adj[x][y] becomes True.
    # So a new triangle is (x,y,z) if adj[x][z] and adj[y][z] are True.
    
    add_list = []
    for z in range(N):
        if z == x or z == y:
            continue
        if adj[x][z] and adj[y][z]:
            a, b, c = sorted([x, y, z])
            new_tri = (a, b, c)
            # Check if this triangle already exists
            # It shouldn't, because if it existed, adj[x][y] would have been True
            # (since it's a triangle, all 3 edges must exist).
            # But wait, if z==u, and edge (x,u) and (y,u) exist, and we add (x,y), 
            # we form (x,y,u). This triangle did not exist before because (x,y) was missing.
            # So it's a new triangle.
            add_list.append(new_tri)
            
    # Build new triangle list
    # We need to preserve indices for the ones we don't remove
    # And append the new ones
    
    # Optimization: Instead of rebuilding the whole list and index, 
    # we can just compute the new score directly.
    # Score = max( max score of remaining triangles, max score of new triangles )
    
    # 1. Max score of remaining triangles
    # We have the old triangles. We know which ones are removed.
    # We need the max degree sum of the rest.
    # This is O(T) which is fine for N=16 (T is small).
    
    max_remaining = 0
    for idx, tri in enumerate(triangles):
        if idx in remove_set:
            continue
        i, j, k = tri
        s = new_deg[i] + new_deg[j] + new_deg[k]
        if s > max_remaining:
            max_remaining = s
            
    # 2. Max score of new triangles
    max_new = 0
    for tri in add_list:
        i, j, k = tri
        s = new_deg[i] + new_deg[j] + new_deg[k]
        if s > max_new:
            max_new = s
            
    new_score = max(max_remaining, max_new)
    
    # If we accept, we need to update triangles and tri_index.
    # To keep the state consistent for future swaps, we must return the new structures.
    
    new_triangles = []
    new_tri_index = {}
    
    # Copy remaining triangles
    for idx, tri in enumerate(triangles):
        if idx in remove_set:
            continue
        new_idx = len(new_triangles)
        new_triangles.append(tri)
        i, j, k = tri
        for u2, v2 in [(i, j), (i, k), (j, k)]:
            key = (min(u2, v2), max(u2, v2))
            if key not in new_tri_index:
                new_tri_index[key] = []
            new_tri_index[key].append(new_idx)
            
    # Add new triangles
    for tri in add_list:
        new_idx = len(new_triangles)
        new_triangles.append(tri)
        i, j, k = tri
        for u2, v2 in [(i, j), (i, k), (j, k)]:
            key = (min(u2, v2), max(u2, v2))
            if key not in new_tri_index:
                new_tri_index[key] = []
            new_tri_index[key].append(new_idx)
            
    return new_score, new_triangles, new_tri_index

def _init_adj(a, b, rng, needed):
    """Create initial adjacency list with K_{a,b} plus internal edges."""
    adj = [[False]*N for _ in range(N)]
    for i in range(a):
        for j in range(a, a+b):
            adj[i][j] = adj[j][i] = True
    
    if needed > 0:
        internal_pairs = []
        for i in range(a):
            for j in range(i+1, a):
                internal_pairs.append((i, j))
        for i in range(a, a+b):
            for j in range(i+1, a+b):
                internal_pairs.append((i, j))
        
        if len(internal_pairs) >= needed:
            selected = rng.sample(internal_pairs, needed)
            for (u, v) in selected:
                adj[u][v] = adj[v][u] = True
    
    return adj

def construct():
    t0 = time.time()
    rng = random.Random(12345)
    
    best_adj = None
    best_deg = None
    best_score = float('inf')
    
    # Multiple restarts with different initial structures
    num_restarts = 25
    
    for r in range(num_restarts):
        if time.time() - t0 > DEADLINE - 2:
            break
        
        # Initialize: K_{a,b} + internal edges
        splits = [(8, 8), (7, 9), (9, 7), (6, 10), (10, 6), (5, 11), (11, 5), (4, 12), (12, 4)]
        a, b = splits[r % len(splits)]
        
        needed = 65 - a * b
        
        if needed < 0 or needed > a*(a-1)//2 + b*(b-1)//2:
            a, b = 8, 8
            needed = 1
            
        adj = _init_adj(a, b, rng, needed)
        
        # Verify we have exactly 65 edges
        ec = _count_edges(adj)
        if ec != 65:
            adj = [[False]*N for _ in range(N)]
            for i in range(8):
                for j in range(8, 16):
                    adj[i][j] = adj[j][i] = True
            adj[0][1] = adj[1][0] = True
            
        deg = [0]*N
        for i in range(N):
            for j in range(N):
                if adj[i][j]:
                    deg[i] += 1
        
        cur_score = _score_from_adj(adj)
        
        if cur_score < best_score:
            best_score = cur_score
            best_adj = [row[:] for row in adj]
            best_deg = deg[:]
            
        # Get initial triangles and index
        triangles, tri_index = _build_triangles_and_index(adj, deg)
        
        edges, nonedges = _get_edges_and_nonedges(adj)
        
        # Simulated Annealing with incremental scoring
        start_temp = 5.0
        end_temp = 0.05
        max_iters = 50000  # Increased iterations due to faster eval
        cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
        temp = start_temp
        
        for it in range(max_iters):
            if time.time() - t0 > DEADLINE - 1:
                break
            
            # Pick random edge to remove
            e_remove = edges[rng.randrange(len(edges))]
            # Pick random non-edge to add
            e_add = nonedges[rng.randrange(len(nonedges))]
            
            u, v = e_remove
            x, y = e_add
            
            if (u == x and v == y) or (u == y and v == x):
                temp *= cooling_rate
                continue
            
            result = _try_swap_score(adj, deg, triangles, tri_index, cur_score, e_remove, e_add)
            
            if result is None:
                temp *= cooling_rate
                continue
            
            new_score, new_triangles, new_tri_index = result
            
            if new_score < cur_score:
                accept = True
            elif new_score == cur_score:
                accept = rng.random() < 0.5
            else:
                delta = new_score - cur_score
                if temp < 1e-9:
                    accept = False
                else:
                    accept = math.exp(-delta / temp) > rng.random()
            
            if accept:
                # Apply swap to adj and deg
                adj[u][v] = adj[v][u] = False
                deg[u] -= 1
                deg[v] -= 1
                adj[x][y] = adj[y][x] = True
                deg[x] += 1
                deg[y] += 1
                
                cur_score = new_score
                triangles = new_triangles
                tri_index = new_tri_index
                
                if new_score < best_score:
                    best_score = new_score
                    best_adj = [row[:] for row in adj]
                    best_deg = deg[:]
                
                # Update edge lists
                # Using lists is O(N) for remove, but N is small (120 pairs)
                edges.remove(e_remove)
                nonedges.append(e_remove)
                nonedges.remove(e_add)
                edges.append(e_add)
            
            temp *= cooling_rate
        
        if time.time() - t0 > DEADLINE - 5:
            break
    
    # Final hill-climbing polish on the best solution
    if best_adj is not None and time.time() - t0 < DEADLINE - 10:
        adj = [row[:] for row in best_adj]
        deg = [0]*N
        for i in range(N):
            for j in range(N):
                if adj[i][j]:
                    deg[i] += 1
        
        cur_score = _score_from_adj(adj)
        triangles, tri_index = _build_triangles_and_index(adj, deg)
        edges, nonedges = _get_edges_and_nonedges(adj)
        
        # Aggressive hill climbing: try all single swaps, accept if better
        improved = True
        while improved and time.time() - t0 < DEADLINE - 5:
            improved = False
            for ei in range(len(edges)):
                if time.time() - t0 > DEADLINE - 5:
                    break
                e_remove = edges[ei]
                for ni in range(len(nonedges)):
                    e_add = nonedges[ni]
                    u, v = e_remove
                    x, y = e_add
                    if (u == x and v == y) or (u == y and v == x):
                        continue
                    
                    result = _try_swap_score(adj, deg, triangles, tri_index, cur_score, e_remove, e_add)
                    if result is None:
                        continue
                    
                    new_score, new_triangles, new_tri_index = result
                    
                    if new_score < cur_score:
                        # Apply
                        adj[u][v] = adj[v][u] = False
                        deg[u] -= 1
                        deg[v] -= 1
                        adj[x][y] = adj[y][x] = True
                        deg[x] += 1
                        deg[y] += 1
                        
                        cur_score = new_score
                        triangles = new_triangles
                        tri_index = new_tri_index
                        
                        if cur_score < best_score:
                            best_score = cur_score
                            best_adj = [row[:] for row in adj]
                            best_deg = deg[:]
                        
                        edges.remove(e_remove)
                        nonedges.append(e_remove)
                        nonedges.remove(e_add)
                        edges.append(e_add)
                        improved = True
                        break
                if improved:
                    break
    
    if best_adj is None:
        adj = [[False]*N for _ in range(N)]
        for i in range(8):
            for j in range(8, 16):
                adj[i][j] = adj[j][i] = True
        adj[0][1] = adj[1][0] = True
        best_adj = adj
    
    ec = _count_edges(best_adj)
    if ec != TARGET_EDGES:
        adj = [[False]*N for _ in range(N)]
        for i in range(8):
            for j in range(8, 16):
                adj[i][j] = adj[j][i] = True
        adj[0][1] = adj[1][0] = True
        best_adj = adj
    
    return {'n': N, 'edges': [[i, j] for i in range(N) for j in range(i+1, N) if best_adj[i][j]]}
