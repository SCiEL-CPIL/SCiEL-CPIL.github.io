"""Rewritten program: hill-climbing + simulated annealing to minimize max triangle degree sum."""
import random
import time
import math

N = 16
TARGET_EDGES = 65
DEADLINE = 145.0  # safety margin before 150s kill

def _score(adj, deg):
    """Compute max degree sum over all triangles."""
    best = 0
    for i in range(N):
        for j in range(i+1, N):
            if not adj[i][j]:
                continue
            for k in range(j+1, N):
                if adj[i][k] and adj[j][k]:
                    s = deg[i] + deg[j] + deg[k]
                    if s > best:
                        best = s
    return best

def _swap(adj, deg, e_remove, e_add):
    """Perform an edge swap: remove e_remove, add e_add. Returns (new_adj, new_deg)."""
    u, v = e_remove
    x, y = e_add
    # Remove (u,v)
    adj[u][v] = adj[v][u] = False
    deg[u] -= 1
    deg[v] -= 1
    # Add (x,y)
    adj[x][y] = adj[y][x] = True
    deg[x] += 1
    deg[y] += 1
    return adj, deg

def _undo_swap(adj, deg, e_remove, e_add):
    """Undo an edge swap."""
    u, v = e_remove
    x, y = e_add
    # Undo add (x,y)
    adj[x][y] = adj[y][x] = False
    deg[x] -= 1
    deg[y] -= 1
    # Undo remove (u,v)
    adj[u][v] = adj[v][u] = True
    deg[u] += 1
    deg[v] += 1
    return adj, deg

def construct():
    t0 = time.time()
    global_best_score = float('inf')
    global_best_adj = None
    
    # Run multiple restarts
    for seed in range(10):
        if time.time() - t0 > DEADLINE:
            break
            
        rng = random.Random(seed * 12345 + 7)
        
        # Initialize: K_{8,8} + one internal edge
        adj = [[False]*N for _ in range(N)]
        for i in range(8):
            for j in range(8, 16):
                adj[i][j] = adj[j][i] = True
        adj[0][1] = adj[1][0] = True  # internal edge in part A
        
        deg = [0]*N
        for i in range(N):
            for j in range(N):
                if adj[i][j]:
                    deg[i] += 1
        
        cur_score = _score(adj, deg)
        best_adj = [row[:] for row in adj]
        best_deg = deg[:]
        best_score = cur_score
        
        # Efficient edge list management
        # We maintain a list of edges and a dict mapping edge -> index in list
        # And a list of non-edges and a dict mapping non-edge -> index in list
        
        edges = []
        nonedges = []
        edge_index = {}
        nonedge_index = {}
        
        for i in range(N):
            for j in range(i+1, N):
                if adj[i][j]:
                    edges.append((i, j))
                    edge_index[(i, j)] = len(edges) - 1
                else:
                    nonedges.append((i, j))
                    nonedge_index[(i, j)] = len(nonedges) - 1
        
        def remove_edge_from_list(e):
            """Remove edge e from edges list and add to nonedges."""
            idx = edge_index[e]
            last = len(edges) - 1
            last_edge = edges[last]
            
            # Swap with last
            edges[idx] = last_edge
            edge_index[last_edge] = idx
            
            # Pop last
            edges.pop()
            del edge_index[e]
            
            # Add to nonedges
            nonedges.append(e)
            nonedge_index[e] = len(nonedges) - 1
        
        def add_edge_to_list(e):
            """Remove edge e from nonedges list and add to edges."""
            idx = nonedge_index[e]
            last = len(nonedges) - 1
            last_nonedge = nonedges[last]
            
            # Swap with last
            nonedges[idx] = last_nonedge
            nonedge_index[last_nonedge] = idx
            
            # Pop last
            nonedges.pop()
            del nonedge_index[e]
            
            # Add to edges
            edges.append(e)
            edge_index[e] = len(edges) - 1

        # Simulated annealing parameters
        max_iters = 50000 # Per restart
        start_temp = 5.0
        end_temp = 0.1
        cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
        
        temp = start_temp
        iters_done = 0
        
        while iters_done < max_iters:
            if time.time() - t0 > DEADLINE:
                break
            
            # Find worst triangle for targeted search
            worst_tri = None
            worst_sum = 0
            for i in range(N):
                for j in range(i+1, N):
                    if not adj[i][j]:
                        continue
                    for k in range(j+1, N):
                        if adj[i][k] and adj[j][k]:
                            s = deg[i] + deg[j] + deg[k]
                            if s > worst_sum:
                                worst_sum = s
                                worst_tri = (i, j, k)
            
            # Decide which edge to remove
            # 50% chance to pick an edge from the worst triangle, otherwise random
            if worst_tri and rng.random() < 0.5:
                # Pick one of the 3 edges in the worst triangle
                i, j, k = worst_tri
                candidates = [(i, j), (i, k), (j, k)]
                e_remove = candidates[rng.randrange(3)]
            else:
                e_remove = edges[rng.randrange(len(edges))]
                
            # Pick random non-edge to add
            e_add = nonedges[rng.randrange(len(nonedges))]
            
            u, v = e_remove
            x, y = e_add
            
            # Ensure we're not removing and adding the same edge
            if (u == x and v == y) or (u == y and v == x):
                iters_done += 1
                temp *= cooling_rate
                continue
            
            # Perform swap
            adj, deg = _swap(adj, deg, e_remove, e_add)
            new_score = _score(adj, deg)
            
            # Decide whether to accept
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
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = [row[:] for row in adj]
                    best_deg = deg[:]
                
                # Update edge/non-edge lists efficiently
                remove_edge_from_list(e_remove)
                add_edge_to_list(e_add)
            else:
                # Undo
                adj, deg = _undo_swap(adj, deg, e_remove, e_add)
            
            iters_done += 1
            temp *= cooling_rate
        
        # Update global best
        if best_score < global_best_score:
            global_best_score = best_score
            global_best_adj = best_adj

    if global_best_adj is None:
        # Fallback to initial construction if something went wrong
        adj = [[False]*N for _ in range(N)]
        for i in range(8):
            for j in range(8, 16):
                adj[i][j] = adj[j][i] = True
        adj[0][1] = adj[1][0] = True
        global_best_adj = adj

    return {'n': N, 'edges': [[i, j] for i in range(N) for j in range(i+1, N) if global_best_adj[i][j]]}
