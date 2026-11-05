import random
import time
import math

N = 16
TARGET_EDGES = 65
DEADLINE = 145.0

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

def _apply_swap(adj, deg, e_remove, e_add):
    u, v = e_remove
    x, y = e_add
    adj[u][v] = adj[v][u] = False
    deg[u] -= 1
    deg[v] -= 1
    adj[x][y] = adj[y][x] = True
    deg[x] += 1
    deg[y] += 1

def _undo_swap(adj, deg, e_remove, e_add):
    u, v = e_remove
    x, y = e_add
    adj[x][y] = adj[y][x] = False
    deg[x] -= 1
    deg[y] -= 1
    adj[u][v] = adj[v][u] = True
    deg[u] += 1
    deg[v] += 1

def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    best_adj = None
    best_deg = None
    best_score = float('inf')
    
    # Run multiple restarts
    num_restarts = 20
    # Dynamically adjust iterations based on time
    # Start with a guess, then adjust?
    # Let's just use a fixed number of iterations per restart
    # 20 restarts * 5000 iters = 100,000 iters
    # If 1ms/iter, 100s. Safe.
    
    for r in range(num_restarts):
        if time.time() - t0 > DEADLINE:
            break
            
        # Initialize: K_{8,8} + 1 edge
        adj = [[False]*N for _ in range(N)]
        for i in range(8):
            for j in range(8, 16):
                adj[i][j] = adj[j][i] = True
        # Add 1 random internal edge
        # To increase diversity, pick random internal edge
        # Part A: 0-7, Part B: 8-15
        # Random pair in A or B
        part = rng.choice([range(8), range(8, 16)])
        u, v = rng.sample(list(part), 2)
        adj[u][v] = adj[v][u] = True
        
        deg = [0]*N
        for i in range(N):
            for j in range(N):
                if adj[i][j]:
                    deg[i] += 1
        
        cur_score = _score(adj, deg)
        
        if cur_score < best_score:
            best_score = cur_score
            best_adj = [row[:] for row in adj]
            best_deg = deg[:]
            
        edges, nonedges = _get_edges_and_nonedges(adj)
        
        # Simulated Annealing
        start_temp = 5.0
        end_temp = 0.1
        max_iters = 5000
        cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
        temp = start_temp
        
        for it in range(max_iters):
            if time.time() - t0 > DEADLINE:
                break
                
            # Pick random edge to remove
            e_remove = edges[rng.randrange(len(edges))]
            # Pick random non-edge to add
            e_add = nonedges[rng.randrange(len(nonedges))]
            
            u, v = e_remove
            x, y = e_add
            
            # Ensure not swapping same edge
            if (u == x and v == y) or (u == y and v == x):
                continue
                
            # Apply swap
            _apply_swap(adj, deg, e_remove, e_add)
            new_score = _score(adj, deg)
            
            # Acceptance
            if new_score < cur_score:
                accept = True
            elif new_score == cur_score:
                accept = rng.random() < 0.5
            else:
                delta = new_score - cur_score
                # Quick reject if delta is large
                if delta > 10:
                    accept = False
                elif temp < 1e-9:
                    accept = False
                else:
                    accept = math.exp(-delta / temp) > rng.random()
            
            if accept:
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = [row[:] for row in adj]
                    best_deg = deg[:]
                
                # Update lists
                # Remove e_remove from edges, add to nonedges
                # Remove e_add from nonedges, add to edges
                # Using list remove is O(M) which is fine
                edges.remove(e_remove)
                nonedges.append(e_remove)
                nonedges.remove(e_add)
                edges.append(e_add)
            else:
                _undo_swap(adj, deg, e_remove, e_add)
                
            temp *= cooling_rate

    # Return best found
    if best_adj is None:
        # Fallback
        adj = [[False]*N for _ in range(N)]
        for i in range(8):
            for j in range(8, 16):
                adj[i][j] = adj[j][i] = True
        adj[0][1] = adj[1][0] = True
        best_adj = adj

    return {'n': N, 'edges': [[i, j] for i in range(N) for j in range(i+1, N) if best_adj[i][j]]}
