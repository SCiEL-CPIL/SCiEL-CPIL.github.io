import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 20
    max_edges = 190
    
    def get_neighbors(u):
        return [v for v in range(n) if g[u][v] != -1]

    def is_rainbow_cycle(u, v, w, x):
        # Edges: (u,v), (v,w), (w,x), (x,u)
        c1 = g[u][v]
        c2 = g[v][w]
        c3 = g[w][x]
        c4 = g[x][u]
        if c1 == -1 or c2 == -1 or c3 == -1 or c4 == -1:
            return True # Not a cycle, so not "bad"
        return len({c1, c2, c3, c4}) == 4

    def get_affected_cycles(u, v):
        # Returns list of (w, x) such that u-v-w-x-u is a potential cycle
        # We only consider w, x where edges (v,w), (w,x), (x,u) exist
        # Note: This list is used to check status BEFORE and AFTER change
        cycles = []
        # w must be neighbor of v (excluding u)
        for w in range(n):
            if w == u or w == v: continue
            if g[v][w] == -1: continue
            # x must be neighbor of w (excluding u, v) AND neighbor of u
            for x in range(n):
                if x == u or x == v or x == w: continue
                if g[w][x] == -1: continue
                if g[u][x] == -1: continue
                cycles.append((w, x))
        return cycles

    def calc_bad_count():
        bad = 0
        # Iterate over all potential 4-cycles
        # To avoid double counting, we can just iterate all u < v < w < x and check all 3 pairings?
        # Or just iterate all 4-tuples (u,v,w,x) distinct and check if it forms a cycle?
        # Simpler: Iterate all pairs (u,v) and (w,x) ?
        # Let's just iterate all 4 distinct vertices and check the 3 possible cycle structures?
        # Actually, the definition of "4-cycle as a subgraph" implies a specific set of 4 edges.
        # A set of 4 vertices {a,b,c,d} can form 3 distinct 4-cycles (squares).
        # We need ALL of them to be rainbow?
        # "Every 4-cycle (as a subgraph) must receive four distinct colours"
        # Yes, if the graph contains the edges of any 4-cycle, that cycle must be rainbow.
        
        # Efficient way:
        # For each pair of non-adjacent vertices? No.
        # Let's stick to the incremental update method for the search.
        # For initialization, we can do a full scan.
        
        for u in range(n):
            for v in range(u+1, n):
                if g[u][v] == -1: continue
                # Find cycles containing (u,v)
                # To avoid double counting, we only count cycles where (u,v) is the "first" edge in some canonical order?
                # This is tricky.
                # Alternative: Just count all bad cycles by iterating all 4-combinations.
                pass
        
        # Full scan is O(N^4) = 20^4 = 160,000.
        # For each 4-combination {a,b,c,d}, there are 3 cycles.
        # Total checks: 4845 * 3 = 14535.
        # This is fast enough for initialization.
        
        bad = 0
        from itertools import combinations
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            # Cycle 1: a-b-c-d-a
            # Edges: (a,b), (b,c), (c,d), (d,a)
            if g[a][b] != -1 and g[b][c] != -1 and g[c][d] != -1 and g[d][a] != -1:
                if len({g[a][b], g[b][c], g[c][d], g[d][a]}) != 4:
                    bad += 1
            # Cycle 2: a-b-d-c-a
            # Edges: (a,b), (b,d), (d,c), (c,a)
            if g[a][b] != -1 and g[b][d] != -1 and g[d][c] != -1 and g[c][a] != -1:
                if len({g[a][b], g[b][d], g[d][c], g[c][a]}) != 4:
                    bad += 1
            # Cycle 3: a-c-b-d-a
            # Edges: (a,c), (c,b), (b,d), (d,a)
            if g[a][c] != -1 and g[c][b] != -1 and g[b][d] != -1 and g[d][a] != -1:
                if len({g[a][c], g[c][b], g[b][d], g[d][a]}) != 4:
                    bad += 1
        return bad

    best_edges = 0
    best_graph = None

    # Run multiple restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        # Initialize graph
        # Strategy: Start with a sparse graph and add edges?
        # Or start dense and remove?
        # Let's start with a random graph with p=0.8
        g = [[-1]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.8:
                    c = random.randint(0, n-1)
                    g[u][v] = c
                    g[v][u] = c
        
        # Calculate initial metrics
        edge_count = sum(1 for u in range(n) for v in range(u+1, n) if g[u][v] != -1)
        bad_count = calc_bad_count()
        
        # Simulated Annealing
        T = 50.0
        cooling = 0.99
        min_T = 0.1
        
        # Local search loop
        steps = 0
        max_steps = 5000
        while steps < max_steps and time.time() < deadline:
            steps += 1
            
            # Choose a move
            move_type = random.random()
            
            if move_type < 0.3:
                # Add an edge
                # Pick random non-edge
                non_edges = [(u, v) for u in range(n) for v in range(u+1, n) if g[u][v] == -1]
                if not non_edges:
                    continue
                u, v = random.choice(non_edges)
                c_new = random.randint(0, n-1)
                
                # Calculate delta
                affected = get_affected_cycles(u, v)
                
                # Count bad cycles BEFORE
                bad_before = 0
                for w, x in affected:
                    if not is_rainbow_cycle(u, v, w, x):
                        bad_before += 1
                
                # Apply change
                g[u][v] = c_new
                g[v][u] = c_new
                edge_count += 1
                
                # Count bad cycles AFTER
                bad_after = 0
                for w, x in affected:
                    if not is_rainbow_cycle(u, v, w, x):
                        bad_after += 1
                
                delta_bad = bad_after - bad_before
                delta_edge = 1
                # Cost = bad * 1000 + (max - edges)
                delta_cost = delta_bad * 1000 - delta_edge
                
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    bad_count += delta_bad
                else:
                    # Reject
                    g[u][v] = -1
                    g[v][u] = -1
                    edge_count -= 1

            elif move_type < 0.6:
                # Remove an edge
                edges = [(u, v) for u in range(n) for v in range(u+1, n) if g[u][v] != -1]
                if not edges:
                    continue
                u, v = random.choice(edges)
                
                affected = get_affected_cycles(u, v)
                
                bad_before = 0
                for w, x in affected:
                    if not is_rainbow_cycle(u, v, w, x):
                        bad_before += 1
                
                c_old = g[u][v]
                g[u][v] = -1
                g[v][u] = -1
                edge_count -= 1
                
                bad_after = 0
                # After removal, no cycle containing (u,v) exists, so 0 bad cycles from these
                # Wait, is_rainbow_cycle returns True if edge missing.
                # So bad_after for these specific cycles will be 0.
                
                delta_bad = bad_after - bad_before
                delta_edge = -1
                delta_cost = delta_bad * 1000 - delta_edge
                
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    bad_count += delta_bad
                else:
                    # Reject
                    g[u][v] = c_old
                    g[v][u] = c_old
                    edge_count += 1

            else:
                # Change color of an edge
                edges = [(u, v) for u in range(n) for v in range(u+1, n) if g[u][v] != -1]
                if not edges:
                    continue
                u, v = random.choice(edges)
                c_old = g[u][v]
                c_new = random.randint(0, n-1)
                if c_new == c_old:
                    continue
                
                affected = get_affected_cycles(u, v)
                
                bad_before = 0
                for w, x in affected:
                    if not is_rainbow_cycle(u, v, w, x):
                        bad_before += 1
                
                g[u][v] = c_new
                g[v][u] = c_new
                
                bad_after = 0
                for w, x in affected:
                    if not is_rainbow_cycle(u, v, w, x):
                        bad_after += 1
                
                delta_bad = bad_after - bad_before
                delta_edge = 0
                delta_cost = delta_bad * 1000
                
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    bad_count += delta_bad
                else:
                    g[u][v] = c_old
                    g[v][u] = c_old

            # Update best
            if bad_count == 0 and edge_count > best_edges:
                best_edges = edge_count
                best_graph = [row[:] for row in g]
                
            # Cooling
            T *= cooling
            if T < min_T:
                break

        # If we found a perfect graph, we might try to add more edges?
        # The SA should handle that.
        
        if best_edges >= 150:
            break

    if best_graph is None:
        # Fallback
        return [[0, v, 0] for v in range(1, 20)]
        
    result = []
    for u in range(n):
        for v in range(u+1, n):
            if best_graph[u][v] != -1:
                result.append([u, v, best_graph[u][v]])
                
    return result
