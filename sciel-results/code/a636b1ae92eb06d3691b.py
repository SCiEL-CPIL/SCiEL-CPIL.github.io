import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    n = 20
    
    # Precompute all 4-cycles in K_n
    # A 4-cycle is a set of 4 vertices {a,b,c,d} and a permutation.
    # There are 3 distinct cycles for each set of 4 vertices.
    # Cycle 1: a-b-c-d-a
    # Cycle 2: a-b-d-c-a
    # Cycle 3: a-c-b-d-a
    # We represent a cycle as a tuple of 4 vertices in order.
    all_cycles = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        all_cycles.append((a, b, c, d))
        all_cycles.append((a, b, d, c))
        all_cycles.append((a, c, b, d))
        
    def count_bad_cycles(adj):
        count = 0
        for cycle in all_cycles:
            u, v, x, y = cycle
            if adj[u][v] == -1 or adj[v][x] == -1 or adj[x][y] == -1 or adj[y][u] == -1:
                continue
            c1 = adj[u][v]
            c2 = adj[v][x]
            c3 = adj[x][y]
            c4 = adj[y][u]
            if len({c1, c2, c3, c4}) < 4:
                count += 1
        return count

    def get_bad_cycles(adj):
        bad = []
        for cycle in all_cycles:
            u, v, x, y = cycle
            if adj[u][v] == -1 or adj[v][x] == -1 or adj[x][y] == -1 or adj[y][u] == -1:
                continue
            c1 = adj[u][v]
            c2 = adj[v][x]
            c3 = adj[x][y]
            c4 = adj[y][u]
            if len({c1, c2, c3, c4}) < 4:
                bad.append(cycle)
        return bad

    def edge_count(adj):
        count = 0
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v] != -1:
                    count += 1
        return count

    def make_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v] != -1:
                    edges.append([u, v, adj[u][v]])
        return edges

    best_edges = [[0, v, 0] for v in range(1, n)] # Star fallback
    best_count = len(best_edges)
    
    def solve_attempt(initializer):
        adj = [[-1]*n for _ in range(n)]
        
        # Initialize
        for u in range(n):
            for v in range(u+1, n):
                c = initializer(u, v)
                adj[u][v] = c
                adj[v][u] = c
        
        # Local search to remove bad cycles
        # We prioritize recoloring over deletion
        max_iterations = 1000
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            bad_cycles = get_bad_cycles(adj)
            if not bad_cycles:
                break
            
            # Pick a random bad cycle
            cycle = random.choice(bad_cycles)
            u, v, x, y = cycle
            
            # Try to recolor one of the edges in this cycle
            improved = False
            # Shuffle edges in cycle to try different ones first
            edges_in_cycle = [(u,v), (v,x), (x,y), (y,u)]
            random.shuffle(edges_in_cycle)
            
            current_bad_count = len(bad_cycles)
            
            for (a, b) in edges_in_cycle:
                old_c = adj[a][b]
                # Try all colors
                for new_c in range(n):
                    if new_c == old_c:
                        continue
                    adj[a][b] = new_c
                    adj[b][a] = new_c
                    
                    # Check if this reduces bad cycles
                    # We can just count bad cycles involving this edge or just total
                    # Total count is safer but O(N_cycles)
                    # Optimization: only check cycles involving (a,b)
                    # But for simplicity and correctness with n=20, total count is fine
                    # 14535 cycles * 4 edges * 20 colors = ~1M checks per iteration?
                    # No, we only do this if we pick a bad cycle.
                    # Let's optimize: check if the specific cycle is fixed, and if new bad cycles are created.
                    
                    # Quick check: is the specific cycle fixed?
                    c1 = adj[u][v]
                    c2 = adj[v][x]
                    c3 = adj[x][y]
                    c4 = adj[y][u]
                    if len({c1, c2, c3, c4}) < 4:
                        adj[a][b] = old_c
                        adj[b][a] = old_c
                        continue
                    
                    # Check if we created new bad cycles
                    # We need to check all cycles that include edge (a,b)
                    new_bad_count = 0
                    for cyc in all_cycles:
                        # Check if (a,b) or (b,a) is in cyc
                        # This is slow. 
                        # Alternative: Just count total bad cycles.
                        pass
                    
                    # For n=20, counting total bad cycles is 14535 iterations.
                    # 14535 * 4 ops = 58k ops.
                    # If we do this for 20 colors * 4 edges = 80 times, that's 4.6M ops per bad cycle fix attempt.
                    # If we have 1000 bad cycles, that's 4.6B ops. Too slow.
                    
                    # Better approach:
                    # 1. Identify all cycles involving (a,b).
                    # 2. Check those.
                    
                    # Let's precompute which cycles involve which edge.
                    # This is static for K_n.
                    # But the graph changes (edges removed).
                    # We only care about cycles where all 4 edges exist.
                    
                    # Let's stick to a simpler heuristic:
                    # If the specific cycle is fixed, accept the move if it doesn't seem to break others.
                    # To be safe, we will do a full count but only if the specific cycle is fixed.
                    # And we will limit the number of recoloring attempts per cycle.
                    
                    new_bad_count = count_bad_cycles(adj)
                    if new_bad_count < current_bad_count:
                        improved = True
                        break
                    adj[a][b] = old_c
                    adj[b][a] = old_c
                
                if improved:
                    break
            
            if not improved:
                # Remove one of the edges in the cycle
                # Pick the one that appears in the most bad cycles
                # We already have the list of bad cycles
                edge_bad_count = {}
                for cyc in bad_cycles:
                    uu, vv, xx, yy = cyc
                    for e in [(uu,vv), (vv,xx), (xx,yy), (yy,uu)]:
                        key = (min(e[0], e[1]), max(e[0], e[1]))
                        edge_bad_count[key] = edge_bad_count.get(key, 0) + 1
                
                # Find max
                max_bad = 0
                worst_edge = None
                for (a, b) in edges_in_cycle:
                    key = (min(a, b), max(a, b))
                    if edge_bad_count.get(key, 0) > max_bad:
                        max_bad = edge_bad_count.get(key, 0)
                        worst_edge = (a, b)
                
                if worst_edge:
                    a, b = worst_edge
                    adj[a][b] = -1
                    adj[b][a] = -1
                else:
                    # Fallback: remove first edge
                    a, b = edges_in_cycle[0]
                    adj[a][b] = -1
                    adj[b][a] = -1
        
        return adj

    # Define initializers
    def init_random(u, v):
        return random.randint(0, n-1)
    
    def init_mod_sum(u, v):
        return (u + v) % n
    
    def init_mod_prod(u, v):
        return (u * v) % n
        
    def init_structured(u, v):
        # Random offset per vertex
        # This requires global state, so we define it inside the attempt or use a fixed one
        # Let's use a fixed permutation for determinism in one attempt
        offsets = list(range(n))
        # Shuffle once outside? No, random inside is fine for multiple attempts
        # But this initializer is called for every edge. We need consistent offsets.
        # So we can't use a simple function.
        # We will handle structured init separately.
        return (u + v) % n # Fallback

    # Run multiple attempts with different initializers
    initializers = [init_random, init_mod_sum, init_mod_prod]
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Choose initializer
        if attempt < 20:
            init_func = init_random
        elif attempt < 30:
            init_func = init_mod_sum
        elif attempt < 40:
            init_func = init_mod_prod
        else:
            # Structured with random offsets
            offsets = list(range(n))
            random.shuffle(offsets)
            def init_structured_rand(u, v):
                return (offsets[u] + v) % n
            init_func = init_structured_rand
            
        adj = solve_attempt(init_func)
        
        if time.time() > deadline:
            break
            
        ec = edge_count(adj)
        if ec > best_count:
            best_count = ec
            best_edges = make_edges(adj)
            
    return best_edges
