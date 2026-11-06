"""Stronger construction for 16-vertex, 65-edge triangle-degree-sum minimization."""
import time
import itertools
import random

# Deadline guard: keep a safety margin before the 150s wall-clock kill.
_DEADLINE = 150.0
_START = time.time()

def _remaining():
    return _DEADLINE - 15.0 - (time.time() - _START)  # 15s safety margin

def _score_graph(adj, n):
    """Compute the maximum degree sum over all triangles in the graph."""
    deg = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                deg[u] += 1
                deg[v] += 1
    best = 0
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u][v]:
                continue
            for w in range(v + 1, n):
                if adj[u][w] and adj[v][w]:
                    s = deg[u] + deg[v] + deg[w]
                    if s > best:
                        best = s
    return best

def _edges_from_adj(adj, n):
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                edges.append([u, v])
    return edges

def _make_adj(n, edges):
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = adj[v][u] = True
    return adj

def _local_search(adj, n, target_edges, max_iters=200000):
    """Local search: try adding/removing/moving edges to minimize max triangle degree sum."""
    best_score = _score_graph(adj, n)
    best_adj = [row[:] for row in adj]
    current_score = best_score
    no_improve = 0
    iter_count = 0
    
    while no_improve < 5000 and iter_count < max_iters:
        if _remaining() < 5.0:
            break
        iter_count += 1
        improved = False
        
        # Randomly pick a move type
        move_type = random.random()
        
        if move_type < 0.33:
            # Try adding an edge
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u][v]]
            if not non_edges:
                continue
            u, v = random.choice(non_edges)
            adj[u][v] = adj[v][u] = True
            new_score = _score_graph(adj, n)
            if new_score < current_score or (new_score == current_score and random.random() < 0.1):
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_adj = [row[:] for row in adj]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                adj[u][v] = adj[v][u] = False
                no_improve += 1
        elif move_type < 0.66:
            # Try removing an edge
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u][v]]
            if not edges:
                continue
            u, v = random.choice(edges)
            adj[u][v] = adj[v][u] = False
            new_score = _score_graph(adj, n)
            if new_score < current_score or (new_score == current_score and random.random() < 0.1):
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_adj = [row[:] for row in adj]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                adj[u][v] = adj[v][u] = True
                no_improve += 1
        else:
            # Try moving an edge: remove (u,v), add (u,w) or (v,w)
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u][v]]
            if not edges:
                continue
            u, v = random.choice(edges)
            # Pick a random third vertex
            w = random.choice([x for x in range(n) if x != u and x != v])
            # Remove (u,v)
            adj[u][v] = adj[v][u] = False
            # Add either (u,w) or (v,w)
            if random.random() < 0.5:
                a, b = u, w
            else:
                a, b = v, w
            if not adj[a][b]:
                adj[a][b] = adj[b][a] = True
            new_score = _score_graph(adj, n)
            if new_score < current_score or (new_score == current_score and random.random() < 0.1):
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_adj = [row[:] for row in adj]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                adj[a][b] = adj[b][a] = False
                adj[u][v] = adj[v][u] = True
                no_improve += 1
    
    return best_adj, best_score

def construct():
    n = 16
    target_edges = 65
    random.seed(42)
    
    # Start from K_{8,8} + one intra-part edge (the seed construction)
    base_edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    adj = _make_adj(n, base_edges)
    
    best_adj = [row[:] for row in adj]
    best_score = _score_graph(adj, n)
    
    # Run multiple local search iterations from different starting points
    num_restarts = 30
    for restart in range(num_restarts):
        if _remaining() < 10.0:
            break
        
        if restart == 0:
            # First restart uses the base construction
            cur_adj = [row[:] for row in adj]
        else:
            # Perturb the base: randomly add/remove a few intra-part edges
            cur_adj = [row[:] for row in adj]
            num_perturb = random.randint(1, 4)
            for _ in range(num_perturb):
                # Pick random vertices from same part
                part = random.choice([0, 1])
                if part == 0:
                    verts = list(range(8))
                else:
                    verts = list(range(8, 16))
                a, b = random.sample(verts, 2)
                if cur_adj[a][b]:
                    cur_adj[a][b] = cur_adj[b][a] = False
                else:
                    cur_adj[a][b] = cur_adj[b][a] = True
        
        # Ensure we have exactly target_edges edges; adjust if needed
        edge_count = sum(1 for u in range(n) for v in range(u + 1, n) if cur_adj[u][v])
        while edge_count < target_edges:
            # Add a random non-edge
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not cur_adj[u][v]]
            if not non_edges:
                break
            u, v = random.choice(non_edges)
            cur_adj[u][v] = cur_adj[v][u] = True
            edge_count += 1
        while edge_count > target_edges:
            # Remove a random edge (prefer intra-part edges to keep structure)
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if cur_adj[u][v]]
            # Prefer removing intra-part edges
            intra = [(u, v) for u, v in edges if u < 8 and v < 8 or u >= 8 and v >= 8]
            if intra and random.random() < 0.7:
                u, v = random.choice(intra)
            else:
                u, v = random.choice(edges)
            cur_adj[u][v] = cur_adj[v][u] = False
            edge_count -= 1
        
        # Local search
        search_adj, search_score = _local_search(cur_adj, n, target_edges, max_iters=50000)
        
        if search_score < best_score:
            best_score = search_score
            best_adj = [row[:] for row in search_adj]
        
        # Also try local search from the current best
        if _remaining() > 20.0:
            cur_adj2 = [row[:] for row in best_adj]
            search_adj2, search_score2 = _local_search(cur_adj2, n, target_edges, max_iters=30000)
            if search_score2 < best_score:
                best_score = search_score2
                best_adj = [row[:] for row in search_adj2]
    
    # Final polish: more iterations from the best found
    if _remaining() > 10.0:
        cur_adj3 = [row[:] for row in best_adj]
        search_adj3, search_score3 = _local_search(cur_adj3, n, target_edges, max_iters=100000)
        if search_score3 < best_score:
            best_score = search_score3
            best_adj = [row[:] for row in search_adj3]
    
    edges = _edges_from_adj(best_adj, n)
    return {'n': n, 'edges': edges}
