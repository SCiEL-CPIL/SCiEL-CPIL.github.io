import time
import random
import sys

N = 20
C = 20
DEADLINE = 150.0
START = time.time()

def elapsed():
    return time.time() - START

def make_adj(col_edges):
    adj = [[-1]*N for _ in range(N)]
    for u, v, c in col_edges:
        adj[u][v] = c
        adj[v][u] = c
    return adj

def build_conflict_counter(col_edges):
    """
    For each colour c, count how many times c appears on each of the 4 "positions"
    of each 4-cycle.  A 4-cycle is rainbow iff for every colour, no position has count>1.
    We store conf[c][a][b][c2][d] = number of 4-cycles where colour c is on edge (a,b)
    with the cycle going a->b->c2->d->a.
    """
    adj = make_adj(col_edges)
    # conf[c][a][b][c2][d] but we only need to know if any position has count>1
    # Actually, for validity check: a 4-cycle is non-rainbow iff some colour appears on 2+ edges of it.
    # Let's just count per colour per edge how many 4-cycles use that edge with that colour.
    # That's not quite right either. Let me think differently.
    
    # A 4-cycle is non-rainbow iff at least one colour appears on 2 or more of its 4 edges.
    # For each colour c, for each pair of edges that could share a 4-cycle, count how many 4-cycles
    # have colour c on both edges. If any such count > 0, we have a violation.
    
    # Simpler: for each colour c, build the subgraph of edges coloured c.
    # A 4-cycle is non-rainbow iff some colour c appears on 2+ edges of that 4-cycle.
    # This happens iff there exist two edges (u,v) and (x,y) both coloured c such that
    # u,v,x,y form a 4-cycle (in some order) with the other two edges present.
    
    # For fast checking, let's precompute for each colour c, the set of edges with colour c.
    # Then for each pair of edges in that set, check if they can be part of a 4-cycle.
    
    # Actually, the most direct approach for _try_add: when adding edge (u,v,c),
    # we need to check all 4-cycles containing (u,v). For each such 4-cycle, check if
    # colour c already appears on another edge of that 4-cycle, OR if any other colour
    # appears on two edges.
    
    # Let me just use the direct approach but optimize it.
    
    return adj

def is_rainbow(col_edges):
    if not col_edges:
        return True
    adj = make_adj(col_edges)
    n = N
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    for (x,y,z,w) in [(a,b,c,d), (a,b,d,c), (a,c,b,d)]:
                        if adj[x][y] >= 0 and adj[y][z] >= 0 and adj[z][w] >= 0 and adj[w][x] >= 0:
                            cols = [adj[x][y], adj[y][z], adj[z][w], adj[w][x]]
                            if len(set(cols)) != 4:
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
            break
    adj[u][v] = -1
    adj[v][u] = -1

def hadamard_init():
    """
    Initialize using a structured approach based on a 20x20 matrix.
    We use a simple algebraic construction: partition vertices into 4 groups of 5.
    Use colours based on group differences to minimize conflicts.
    """
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    
    # Group vertices: 0-4, 5-9, 10-14, 15-19
    # Within-group edges: use colour based on (i+j) % 5 + 15 (colours 15-19)
    # Between-group edges: use colour based on group difference
    
    groups = [[0,1,2,3,4], [5,6,7,8,9], [10,11,12,13,14], [15,16,17,18,19]]
    
    # Within-group: sparse, use distinct colours
    for g in range(4):
        for i in range(5):
            for j in range(i+1, 5):
                u, v = groups[g][i], groups[g][j]
                c = 15 + (i + j) % 5
                if _try_add(adj, u, v, c):
                    _add_edge(col_edges, adj, u, v, c)
    
    # Between-group: use colours 0-14 based on group pair
    for g1 in range(4):
        for g2 in range(g1+1, 4):
            # 6 group pairs, use colours 0-5 for base, vary within
            base_c = (g1 * 4 + g2) % 15
            for i in range(5):
                for j in range(5):
                    u, v = groups[g1][i], groups[g2][j]
                    c = (base_c + i + j) % 15
                    if _try_add(adj, u, v, c):
                        _add_edge(col_edges, adj, u, v, c)
    
    return col_edges

def greedy_construct(seed_val):
    """Run one greedy construction attempt with structured initialization."""
    random.seed(seed_val)
    
    # Start with hadamard-like initialization
    col_edges = hadamard_init()
    adj = make_adj(col_edges)
    
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
    
    return col_edges

def local_improve(col_edges, max_rounds=10):
    """Try to add more edges to the given graph."""
    adj = make_adj(col_edges)
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    for attempt in range(max_rounds):
        if elapsed() > DEADLINE - 10:
            break
        random.shuffle(all_pairs)
        improved = False
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
                    improved = True
                    break
        if not improved:
            break
    
    return col_edges

def simulated_annealing(col_edges, max_iters=20000, T_init=5.0, T_min=0.01, cooling=0.9995):
    """
    Simulated annealing with strategic edge removal.
    Key insight: removing a "blocking" edge can allow many new edges to be added.
    """
    col_edges = [list(e) for e in col_edges]
    adj = make_adj(col_edges)
    T = T_init
    best_edges = [list(e) for e in col_edges]
    best_count = len(col_edges)
    
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    for it in range(max_iters):
        if elapsed() > DEADLINE - 15:
            break
        
        move_type = random.random()
        
        if move_type < 0.4:
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
            if added and len(col_edges) > best_count:
                best_count = len(col_edges)
                best_edges = [list(e) for e in col_edges]
                
        elif move_type < 0.7:
            # Strategic removal: remove an edge that is "blocking" many potential additions
            if not col_edges:
                continue
            
            # Find edges whose removal would allow the most new edges
            # Approximate: for each edge, count how many non-edges become addable
            # This is expensive, so do it only sometimes
            if random.random() < 0.3:
                # Full analysis
                removal_scores = []
                for i, e in enumerate(col_edges):
                    u, v, c = e
                    # Temporarily remove
                    old_adj = [row[:] for row in adj]
                    _remove_edge(col_edges, adj, u, v)
                    
                    # Count how many new edges can be added
                    addable = 0
                    for pu, pv in all_pairs:
                        if adj[pu][pv] >= 0:
                            continue
                        for pc in range(C):
                            if _try_add(adj, pu, pv, pc):
                                addable += 1
                                break
                    
                    removal_scores.append((addable, i))
                    
                    # Restore
                    _add_edge(col_edges, adj, u, v, c)
                    adj[:] = old_adj
                    
                if removal_scores:
                    removal_scores.sort(reverse=True)
                    # Remove the most blocking edge with some probability
                    if random.random() < 0.5:
                        _, idx = removal_scores[0]
                        u, v, c = col_edges[idx]
                        _remove_edge(col_edges, adj, u, v)
                        # Try to add edges back
                        for pu, pv in all_pairs:
                            if adj[pu][pv] >= 0:
                                continue
                            for pc in range(C):
                                if _try_add(adj, pu, pv, pc):
                                    _add_edge(col_edges, adj, pu, pv, pc)
                                    break
                        if len(col_edges) > best_count:
                            best_count = len(col_edges)
                            best_edges = [list(e) for e in col_edges]
            else:
                # Simple random removal
                idx = random.randint(0, len(col_edges)-1)
                u, v, c = col_edges[idx]
                _remove_edge(col_edges, adj, u, v)
                # Try to add edges
                for pu, pv in all_pairs:
                    if adj[pu][pv] >= 0:
                        continue
                    for pc in range(C):
                        if _try_add(adj, pu, pv, pc):
                            _add_edge(col_edges, adj, pu, pv, pc)
                            break
                if len(col_edges) > best_count:
                    best_count = len(col_edges)
                    best_edges = [list(e) for e in col_edges]
                    
        else:
            # Recolor
            if not col_edges:
                continue
            idx = random.randint(0, len(col_edges)-1)
            u, v, old_c = col_edges[idx]
            new_c = random.randint(0, C-1)
            if new_c == old_c:
                continue
            
            old_adj = [row[:] for row in adj]
            _remove_edge(col_edges, adj, u, v)
            if _try_add(adj, u, v, new_c):
                _add_edge(col_edges, adj, u, v, new_c)
                if len(col_edges) > best_count:
                    best_count = len(col_edges)
                    best_edges = [list(e) for e in col_edges]
            else:
                adj[:] = old_adj
                # Re-add with old colour
                _add_edge(col_edges, adj, u, v, old_c)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final local improvement on best
    improved = local_improve([list(e) for e in best_edges], max_rounds=5)
    if count_edges(improved) > best_count:
        best_edges = improved
        best_count = count_edges(improved)
    
    return best_edges

def construct():
    best = []
    best_count = 0
    
    # Run multiple greedy attempts with different seeds
    max_attempts = 20
    for i in range(max_attempts):
        if elapsed() > DEADLINE - 30:
            break
        
        result = greedy_construct(i)
        count = count_edges(result)
        
        if count > best_count:
            best = result
            best_count = count
            
            # Try to improve locally
            if elapsed() < DEADLINE - 35:
                improved = local_improve([list(e) for e in best], max_rounds=5)
                improved_count = count_edges(improved)
                if improved_count > best_count:
                    best = improved
                    best_count = improved_count
        
        # Early exit if we've found a very dense graph
        if best_count >= 170:
            break
    
    # Run simulated annealing on the best solution found
    if best and elapsed() < DEADLINE - 40:
        sa_result = simulated_annealing(best, max_iters=15000, T_init=5.0, T_min=0.01, cooling=0.9995)
        sa_count = count_edges(sa_result)
        if sa_count > best_count:
            best = sa_result
            best_count = sa_count
    
    # Final local improvement
    if best and elapsed() < DEADLINE - 15:
        improved = local_improve([list(e) for e in best], max_rounds=10)
        improved_count = count_edges(improved)
        if improved_count > best_count:
            best = improved
            best_count = improved_count
    
    # Final verification
    if best and is_rainbow(best):
        return best
    
    # Fallback: try a simple star
    star = [[0, v, v-1] for v in range(1, N)]
    if is_rainbow(star):
        return star
    
    return star
