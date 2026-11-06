import time
import random
import sys
import math

N = 20
C = 20
DEADLINE = 140.0  # Evaluator kills at 150s, keep margin
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
    if not col_edges:
        return True
    adj = make_adj(col_edges)
    for cyc in all_four_cycles(adj):
        cols = set(e[2] for e in cyc)
        if len(cols) != 4:
            return False
    return True

def count_edges(col_edges):
    return len(col_edges)

def _try_add(adj, u, v, c):
    """Check if edge (u,v) with colour c can be added without violating rainbow 4-cycle constraint."""
    if u == v:
        return False
    if adj[u][v] >= 0:
        return False
    
    # Check all 4-cycles that would be created by adding edge (u,v,c)
    # A 4-cycle u-x-y-v-u requires: adj[u][x]>=0, adj[x][y]>=0, adj[y][v]>=0
    # with x != v, y != u, x != y
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

def _try_recolor(adj, u, v, new_c):
    """Check if recoloring edge (u,v) to new_c is valid."""
    if u == v:
        return False
    if adj[u][v] < 0:
        return False
    if adj[u][v] == new_c:
        return True
    
    # Check all 4-cycles containing edge (u,v)
    # Cycle: u-x-y-v-u
    # Edges: (u,v), (v,y), (y,x), (x,u)
    # Colours: new_c, adj[v][y], adj[y][x], adj[x][u]
    for x in range(N):
        if x == v or adj[u][x] < 0:
            continue
        for y in range(N):
            if y == u or y == x or adj[x][y] < 0 or adj[y][v] < 0:
                continue
            cols = {new_c, adj[u][x], adj[x][y], adj[y][v]}
            if len(cols) != 4:
                return False
    return True

def _add_edge(col_edges, adj, u, v, c):
    col_edges.append([u, v, c])
    adj[u][v] = c
    adj[v][u] = c

def _remove_edge(col_edges, adj, u, v):
    """Remove edge (u,v) from col_edges and adj."""
    for i, e in enumerate(col_edges):
        if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
            col_edges.pop(i)
            break
    adj[u][v] = -1
    adj[v][u] = -1

def find_violation(adj):
    """Find one violating 4-cycle if it exists. Returns list of 4 edges or None."""
    for a in range(N):
        for b in range(a+1, N):
            if adj[a][b] < 0: continue
            for c in range(b+1, N):
                if adj[b][c] < 0: continue
                for d in range(c+1, N):
                    # Check cycle a-b-c-d-a
                    if adj[c][d] >= 0 and adj[d][a] >= 0:
                        cols = {adj[a][b], adj[b][c], adj[c][d], adj[d][a]}
                        if len(cols) != 4:
                            return [(a,b,adj[a][b]), (b,c,adj[b][c]), (c,d,adj[c][d]), (d,a,adj[d][a])]
                    # Check cycle a-b-d-c-a
                    if adj[b][d] >= 0 and adj[d][c] >= 0:
                        cols = {adj[a][b], adj[b][d], adj[d][c], adj[c][a]}
                        if len(cols) != 4:
                            return [(a,b,adj[a][b]), (b,d,adj[b][d]), (d,c,adj[d][c]), (c,a,adj[c][a])]
                    # Check cycle a-c-b-d-a
                    if adj[a][c] >= 0 and adj[c][b] >= 0 and adj[b][d] >= 0 and adj[d][a] >= 0:
                        # Wait, a-c-b-d-a is not a simple 4-cycle on distinct vertices a,b,c,d in order?
                        # The vertices are a,c,b,d. Edges: (a,c), (c,b), (b,d), (d,a).
                        # This is a valid 4-cycle if a,c,b,d are distinct.
                        cols = {adj[a][c], adj[c][b], adj[b][d], adj[d][a]}
                        if len(cols) != 4:
                            return [(a,c,adj[a][c]), (c,b,adj[c][b]), (b,d,adj[b][d]), (d,a,adj[d][a])]
    return None

def repair_graph(col_edges, adj, max_iter=10000):
    """
    Start with a dense graph, find violations, and fix them by recoloring or removing edges.
    """
    # We assume col_edges and adj are consistent and represent a full or near-full graph.
    # We will try to resolve conflicts.
    
    for _ in range(max_iter):
        if elapsed() > DEADLINE - 30:
            break
            
        violation = find_violation(adj)
        if violation is None:
            return True
            
        # Identify conflicting edges in the violation
        # Find two edges with the same colour
        edge_colors = {}
        for e in violation:
            u, v, c = e
            key = (min(u,v), max(u,v))
            edge_colors[key] = c
            
        # Find a colour that appears more than once
        counts = {}
        for k, c in edge_colors.items():
            counts[c] = counts.get(c, 0) + 1
            
        conflicting_color = None
        for c, cnt in counts.items():
            if cnt > 1:
                conflicting_color = c
                break
                
        if conflicting_color is None:
            # Should not happen if violation exists
            break
            
        # Find the edges with this colour
        conflicting_edges = [k for k, c in edge_colors.items() if c == conflicting_color]
        
        # Try to recolor one of the conflicting edges
        fixed = False
        for edge_key in conflicting_edges:
            u, v = edge_key
            # Try all other colours
            for new_c in range(C):
                if new_c == conflicting_color:
                    continue
                if _try_recolor(adj, u, v, new_c):
                    # Apply recolor
                    for i, e in enumerate(col_edges):
                        if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                            col_edges[i][2] = new_c
                            break
                    adj[u][v] = new_c
                    adj[v][u] = new_c
                    fixed = True
                    break
            if fixed:
                break
                
        if not fixed:
            # Remove one of the conflicting edges
            u, v = conflicting_edges[0]
            _remove_edge(col_edges, adj, u, v)
            
    return find_violation(adj) is None

def local_improve(col_edges, adj):
    """Try to add more edges to the given graph using greedy pass."""
    adj = [row[:] for row in adj] # Work on copy
    col_edges = [list(e) for e in col_edges]
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    improved = True
    while improved:
        improved = False
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if elapsed() > DEADLINE - 10:
                return col_edges, adj
            if adj[u][v] >= 0:
                continue
            colours = list(range(C))
            random.shuffle(colours)
            for c in colours:
                if _try_add(adj, u, v, c):
                    _add_edge(col_edges, adj, u, v, c)
                    improved = True
                    break
    return col_edges, adj

def simulated_annealing(col_edges, adj, max_iters=50000, T_init=5.0, T_min=0.01, cooling=0.999):
    """
    Simulated annealing allowing add, remove, and recolor moves.
    """
    col_edges = [list(e) for e in col_edges]
    adj = [row[:] for row in adj]
    T = T_init
    best_edges = [list(e) for e in col_edges]
    best_adj = [row[:] for row in adj]
    best_count = len(col_edges)
    
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    for it in range(max_iters):
        if elapsed() > DEADLINE - 5:
            break
        
        move_type = random.random()
        
        if move_type < 0.6:
            # Try to add an edge
            u, v = random.choice(all_pairs)
            if adj[u][v] >= 0:
                continue
            colours = list(range(C))
            random.shuffle(colours)
            added = False
            for c in colours:
                if _try_add(adj, u, v, c):
                    _add_edge(col_edges, adj, u, v, c)
                    added = True
                    break
            if added:
                if len(col_edges) > best_count:
                    best_count = len(col_edges)
                    best_edges = [list(e) for e in col_edges]
                    best_adj = [row[:] for row in adj]
                    
        elif move_type < 0.9:
            # Try to recolor a random edge
            if not col_edges:
                continue
            idx = random.randint(0, len(col_edges)-1)
            u, v, old_c = col_edges[idx]
            new_c = random.randint(0, C-1)
            if new_c == old_c:
                continue
            
            if _try_recolor(adj, u, v, new_c):
                # Accept recoloring
                adj[u][v] = new_c
                adj[v][u] = new_c
                col_edges[idx][2] = new_c
                
        else:
            # Try to remove a random edge
            if not col_edges:
                continue
            idx = random.randint(0, len(col_edges)-1)
            u, v, c = col_edges[idx]
            
            old_edges = [list(e) for e in col_edges]
            old_adj = [row[:] for row in adj]
            
            _remove_edge(col_edges, adj, u, v)
            
            # Metropolis acceptance for removal (delta E = 1)
            if random.random() < math.exp(-1.0 / (T + 1e-10)):
                pass
            else:
                col_edges[:] = old_edges
                adj[:] = old_adj
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_edges, best_adj

def construct():
    best_edges = []
    best_adj = []
    best_count = 0
    
    # 1. Dense Start + Repair
    # Start with K_20 and random colours
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    for u in range(N):
        for v in range(u+1, N):
            c = random.randint(0, C-1)
            _add_edge(col_edges, adj, u, v, c)
            
    # Repair the graph to make it valid
    if elapsed() < DEADLINE - 60:
        repair_graph(col_edges, adj, max_iter=20000)
        
    if is_rainbow(col_edges):
        best_edges = col_edges
        best_adj = adj
        best_count = len(col_edges)
    else:
        # Fallback to greedy if repair fails completely
        col_edges, adj = greedy_construct(0)
        best_edges = col_edges
        best_adj = adj
        best_count = len(col_edges)
        
    # 2. Local Improvement
    if best_edges:
        best_edges, best_adj = local_improve(best_edges, best_adj)
        best_count = len(best_edges)
        
    # 3. Simulated Annealing
    if best_edges and elapsed() < DEADLINE - 20:
        sa_edges, sa_adj = simulated_annealing(best_edges, best_adj, max_iters=50000, T_init=10.0, T_min=0.01, cooling=0.9995)
        if len(sa_edges) > best_count:
            best_edges = sa_edges
            best_adj = sa_adj
            best_count = len(sa_edges)
            
    # 4. Final Local Improvement
    if best_edges and elapsed() < DEADLINE - 5:
        best_edges, best_adj = local_improve(best_edges, best_adj)
        best_count = len(best_edges)

    return best_edges

def greedy_construct(seed_val):
    """Run one greedy construction attempt."""
    random.seed(seed_val)
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if elapsed() > DEADLINE - 10:
            break
        if adj[u][v] >= 0:
            continue
        
        colours = list(range(C))
        random.shuffle(colours)
        
        for c in colours:
            if _try_add(adj, u, v, c):
                _add_edge(col_edges, adj, u, v, c)
                break
    
    return col_edges, adj

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} edges")
    if is_rainbow(res):
        print("Valid")
    else:
        print("Invalid")
