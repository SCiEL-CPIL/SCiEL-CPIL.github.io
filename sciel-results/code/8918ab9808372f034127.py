import time
import random

N = 20
C = 20
DEADLINE = 140.0
START = time.time()

def elapsed():
    return time.time() - START

def make_adj(col_edges):
    adj = [[-1]*N for _ in range(N)]
    for u, v, c in col_edges:
        adj[u][v] = c
        adj[v][u] = c
    return adj

def all_four_cycles(adj):
    n = N
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
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

def _try_add(adj, u, v, c):
    if u == v:
        return False
    if adj[u][v] >= 0:
        return False
    for x in range(N):
        if x == v or adj[u][x] < 0:
            continue
        for y in range(N):
            if y == u or y == x or adj[x][y] < 0 or adj[y][v] < 0:
                continue
            cols = {c, adj[u][x], adj[x][y], adj[y][v]}
            if len(cols) != 4:
                return False
    return True

def _add_edge(col_edges, adj, u, v, c):
    col_edges.append([u, v, c])
    adj[u][v] = c
    adj[v][u] = c

def _remove_edge(col_edges, adj, u, v):
    for i, e in enumerate(col_edges):
        if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
            col_edges.pop(i)
            adj[u][v] = -1
            adj[v][u] = -1
            return True
    return False

def greedy_construct(seed_val):
    random.seed(seed_val)
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if elapsed() > DEADLINE - 5:
            break
        if adj[u][v] >= 0:
            continue
        
        colours = list(range(C))
        random.shuffle(colours)
        
        for c in colours:
            if _try_add(adj, u, v, c):
                _add_edge(col_edges, adj, u, v, c)
                break
    
    return col_edges

def local_improve(col_edges):
    adj = make_adj(col_edges)
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Phase 1: Add edges
    for attempt in range(20):
        random.shuffle(all_pairs)
        improved = False
        for u, v in all_pairs:
            if elapsed() > DEADLINE - 5:
                return col_edges
            if adj[u][v] >= 0:
                continue
            colours = list(range(C))
            random.shuffle(colours)
            for c in colours:
                if _try_add(adj, u, v, c):
                    _add_edge(col_edges, adj, u, v, c)
                    improved = True
                    break
        if not improved:
            break
    
    # Phase 2: Swap edges to escape local optima
    # If we are stuck, try removing an edge and adding a different one
    if elapsed() > DEADLINE - 10:
        return col_edges
        
    for swap_attempt in range(50):
        if elapsed() > DEADLINE - 5:
            return col_edges
            
        # Find missing edges
        missing = [(i, j) for i, j in all_pairs if adj[i][j] < 0]
        if not missing:
            return col_edges
            
        # Pick a random existing edge to remove
        if not col_edges:
            return col_edges
            
        idx = random.randint(0, len(col_edges) - 1)
        u_rem, v_rem, c_rem = col_edges[idx]
        
        # Pick a random missing edge to try adding
        u_add, v_add = random.choice(missing)
        
        # Backup state
        backup_edges = [e[:] for e in col_edges]
        backup_adj = [row[:] for row in adj]
        backup_count = len(col_edges)
        
        # Remove existing edge
        _remove_edge(col_edges, adj, u_rem, v_rem)
        
        # Try to add the new edge with a random colour
        success = False
        colours = list(range(C))
        random.shuffle(colours)
        for c in colours:
            if _try_add(adj, u_add, v_add, c):
                _add_edge(col_edges, adj, u_add, v_add, c)
                success = True
                break
        
        if success:
            # Try to fill more edges after the swap
            random.shuffle(all_pairs)
            for u, v in all_pairs:
                if elapsed() > DEADLINE - 5:
                    break
                if adj[u][v] >= 0:
                    continue
                colours = list(range(C))
                random.shuffle(colours)
                for c in colours:
                    if _try_add(adj, u, v, c):
                        _add_edge(col_edges, adj, u, v, c)
                        break
            
            new_count = len(col_edges)
            # Accept if better or equal (to explore)
            if new_count >= backup_count:
                if new_count > backup_count:
                    # Keep improvement
                    pass
                else:
                    # Equal count, keep with 50% probability to avoid cycling
                    if random.random() > 0.5:
                        col_edges = [e[:] for e in backup_edges]
                        adj = [row[:] for row in backup_adj]
            else:
                # Revert
                col_edges = [e[:] for e in backup_edges]
                adj = [row[:] for row in backup_adj]
        else:
            # Revert
            col_edges = [e[:] for e in backup_edges]
            adj = [row[:] for row in backup_adj]
            
    return col_edges

def try_structured_coloring():
    """Try the (u+v) mod C coloring on complete graph."""
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    
    for u in range(N):
        for v in range(u+1, N):
            c = (u + v) % C
            if _try_add(adj, u, v, c):
                _add_edge(col_edges, adj, u, v, c)
    
    return col_edges

def try_structured_coloring2():
    """Try (u-v) mod C coloring."""
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    
    for u in range(N):
        for v in range(u+1, N):
            c = (u - v) % C
            if _try_add(adj, u, v, c):
                _add_edge(col_edges, adj, u, v, c)
    
    return col_edges

def try_structured_coloring3():
    """Try (u*v) mod C coloring."""
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    
    for u in range(N):
        for v in range(u+1, N):
            c = (u * v) % C
            if _try_add(adj, u, v, c):
                _add_edge(col_edges, adj, u, v, c)
    
    return col_edges

def construct():
    best = []
    best_count = 0
    
    # Try structured colorings first
    for fn in [try_structured_coloring, try_structured_coloring2, try_structured_coloring3]:
        if elapsed() > DEADLINE - 30:
            break
        result = fn()
        count = count_edges(result)
        if count > best_count:
            best = result
            best_count = count
    
    # Run multiple greedy attempts with local search
    max_attempts = 30
    for i in range(max_attempts):
        if elapsed() > DEADLINE - 15:
            break
        
        result = greedy_construct(i)
        count = count_edges(result)
        
        # Run local improvement (add + swap)
        if elapsed() < DEADLINE - 20:
            improved = local_improve([e[:] for e in result])
            improved_count = count_edges(improved)
            if improved_count > count:
                result = improved
                count = improved_count
        
        if count > best_count:
            best = result
            best_count = count
    
    # Final verification
    if best and is_rainbow(best):
        return best
    
    # Fallback
    star = [[0, v, v] for v in range(1, N)]
    return star
