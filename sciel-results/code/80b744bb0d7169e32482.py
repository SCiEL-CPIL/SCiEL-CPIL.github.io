import time
import random

def _luw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def construct(k, g):
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    # If not the specific case, use fallback
    if k != 3 or g != 16:
        return _luw_construct(k, g)
        
    # Heuristic search for k=3, g=16
    # Target n ~ 936 (known best)
    
    def gen_random_3_reg(n):
        """Generates a random simple 3-regular graph on n vertices.
        Returns adj list (list of sets) or None on failure.
        """
        if n < 3 or n % 2 != 0:
            return None
            
        for attempt in range(100):
            adj = [set() for _ in range(n)]
            stubs = [i for i in range(n) for _ in range(3)]
            random.shuffle(stubs)
            
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
                
            if valid:
                return adj
        return None

    def get_shortest_cycle(adj, n, max_girth):
        """Finds shortest cycle length and one edge in it using BFS.
        Returns (girth, (u, v)) or (inf, None).
        """
        min_girth = max_girth + 1
        min_edge = None
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            
            for u in queue:
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Cycle found
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            min_edge = (u, v)
                            if min_girth == 4: # Can't get much better for simple graph? No, 3 is impossible for 3-reg. 4 is min.
                                # Actually 3-reg can have girth 3? No, simple graph. Girth 3 is triangle.
                                # If we find 4, we can stop searching for shorter? 
                                # Yes, 3 is impossible in simple 3-reg? 
                                # Wait, K3 is 2-reg. 3-reg min girth is 4?
                                # No, consider K4. 3-reg, girth 3.
                                # So min is 3.
                                if min_girth == 3:
                                    return 3, min_edge
                                
                if min_girth == 3:
                    break
            if min_girth == 3:
                break
                
        return min_girth, min_edge

    def extract_edges(adj, n):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return edges

    # Candidate sizes
    # Known best: 936. 
    # If we fail at 936, try 938, 940...
    # But usually one n is enough if we have enough restarts.
    # Let's try 936 primarily.
    
    best_result = None
    
    # Try a few n values in increasing order
    for n_target in [936, 940, 950, 1000]:
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple restarts for this n
        for restart in range(50):
            if time.time() - start_time > time_limit:
                break
                
            adj = gen_random_3_reg(n_target)
            if not adj:
                continue
                
            cur_girth, cur_edge = get_shortest_cycle(adj, n_target, g)
            
            if cur_girth >= g:
                best_result = extract_edges(adj, n_target)
                return n_target, best_result
            
            # Local Search: Edge Swaps
            # Goal: Break short cycles.
            
            # We perform a number of swaps.
            # To avoid infinite loops, we limit steps per restart.
            max_steps = 2000
            stuck_counter = 0
            
            for step in range(max_steps):
                if time.time() - start_time > time_limit:
                    break
                    
                # Re-evaluate girth periodically? 
                # No, we know it's < g. We just try to fix.
                # But if we fix one, another might appear.
                # Let's just do random swaps on edges in the shortest cycle.
                
                # Find current shortest cycle
                cur_girth, cur_edge = get_shortest_cycle(adj, n_target, g)
                
                if cur_girth >= g:
                    best_result = extract_edges(adj, n_target)
                    return n_target, best_result
                    
                if not cur_edge:
                    break # Should not happen
                
                u1, v1 = cur_edge
                
                # Find a random edge e2 = (u2, v2) to swap with.
                # Constraints:
                # 1. e2 != e1
                # 2. Swapping (u1,v1) and (u2,v2) must not create loops/multiedges.
                #    New edges: (u1, u2) and (v1, v2) OR (u1, v2) and (v1, u2)
                
                swap_done = False
                # Try to find a valid e2
                for _ in range(20):
                    u2 = random.randint(0, n_target - 1)
                    if u2 in (u1, v1):
                        continue
                    if not adj[u2]:
                        continue
                    v2 = random.choice(list(adj[u2]))
                    if v2 in (u1, v1):
                        continue
                    if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                        continue
                        
                    # Check Option 1: (u1, u2), (v1, v2)
                    valid_swap = None
                    if u1 != u2 and v1 != v2:
                        if u2 not in adj[u1] and v2 not in adj[v1]:
                            # Also ensure we don't create multi-edge with other neighbors?
                            # adj sets handle uniqueness.
                            # Just need to ensure u1-u2 and v1-v2 are not existing edges.
                            valid_swap = ((u1, u2), (v1, v2))
                            
                    # Check Option 2: (u1, v2), (v1, u2)
                    if not valid_swap:
                        if u1 != v2 and v1 != u2:
                            if v2 not in adj[u1] and u2 not in adj[v1]:
                                valid_swap = ((u1, v2), (v1, u2))
                                
                    if valid_swap:
                        # Apply swap
                        ne1, ne2 = valid_swap
                        
                        # Remove old
                        adj[u1].remove(v1)
                        adj[v1].remove(u1)
                        adj[u2].remove(v2)
                        adj[v2].remove(u2)
                        
                        # Add new
                        a1, b1 = ne1
                        a2, b2 = ne2
                        adj[a1].add(b1)
                        adj[b1].add(a1)
                        adj[a2].add(b2)
                        adj[b2].add(a2)
                        
                        swap_done = True
                        break
                if not swap_done:
                    # Can't find any valid swap? 
                    # This implies high connectivity or we are stuck.
                    # Restart might be better.
                    # But let's try a few more steps.
                    pass
                    
        if best_result:
            break
            
    if best_result:
        return n_target, best_result
        
    # Fallback to LUW
    return _luw_construct(k, g)
