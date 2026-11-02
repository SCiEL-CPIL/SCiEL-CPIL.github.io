"""Small 4-regular graph of girth 10 (cage problem n(4,10))."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque, defaultdict

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) tuples.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # If k and g are not the specific case, fallback to general logic
    if k != 4 or g != 10:
        return _luw_construct(k, g)

    # Strategy: Try to find a graph with N < 512 using randomized local search.
    # We know LUW gives 512. We aim for 320 (best known).
    
    # Candidate N values to try, starting from the best known upper bound
    # and increasing. We stop if we find a valid graph or run out of time.
    # We prioritize smaller N.
    candidate_ns = [320, 340, 360, 380, 400, 420, 440, 460, 480, 500]
    
    best_result = None
    best_n = 512 # Fallback LUW size
    
    # Generate the fallback LUW graph first so we have a valid answer ready
    luw_n, luw_edges = _luw_construct(4, 10)
    if luw_n < best_n:
        best_n = luw_n
        best_result = (luw_n, luw_edges)
    
    # Try to improve upon the LUW graph
    for n in candidate_ns:
        if time.time() > deadline - 5:
            break
        if n >= best_n:
            continue
            
        # Time budget per N: try to spend at most 5-10 seconds per candidate
        # to ensure we cover multiple candidates or finish in time.
        # Actually, let's just run one big search for the smallest N (320)
        # with the remaining time. If it fails, we might not have time for others.
        # Better: Run a search for each N with a small time slice, or just
        # run a search for the smallest N with all remaining time.
        
        # Let's try to find a valid graph for the smallest N (320) with the
        # remaining time. If we find one, we return it.
        # If we don't find one in 320, we don't have time to try 340, etc.
        # So we just try 320.
        
        result = _local_search_for_n(4, 10, n, start_time, deadline)
        if result is not None:
            return result
            
    # If no better graph found, return the LUW graph
    return best_result

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction (original logic)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Define the rule for LUW construction
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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

def _has_girth_at_least(edges, n, g):
    """Check if graph has girth at least g using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_depth = g // 2
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] <= dist[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _count_short_cycles(adj, n):
    """Count cycles of length 4, 6, 8. Used as cost function."""
    # Count 4-cycles
    # For each edge (u,v), count common neighbors.
    # A 4-cycle is counted 4 times (once per edge) if we sum over edges?
    # No, a 4-cycle u-v-w-x-u has 4 edges.
    # Let's just count distinct 4-cycles.
    # This is O(N * degree^2).
    
    four_cycles = 0
    for u in range(n):
        for v in adj[u]:
            if u < v:
                common = len(set(adj[u]) & set(adj[v]))
                four_cycles += common
                
    # Each 4-cycle is counted once by each of its 4 edges?
    # If we iterate over edges (u,v) with u<v, and count common neighbors,
    # a cycle u-v-w-x-u will be found when we check (u,v) (common w,x -> 2),
    # when we check (v,w) (common u,x -> 2), etc.
    # So total count is 4 * (number of 4-cycles).
    four_cycles //= 4
    
    # Count 6-cycles is more complex. We skip it for speed in the heuristic,
    # or use a simpler proxy.
    # For now, just return four_cycles as the cost.
    return four_cycles

def _local_search_for_n(k, g, n, start_time, deadline):
    """Try to find a k-regular graph of girth >= g on exactly n vertices via local search."""
    if time.time() > deadline - 2:
        return None
        
    # Generate a random 4-regular graph
    edges = _gen_random_4reg(n)
    if edges is None:
        return None
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    if _has_girth_at_least(edges, n, g):
        return n, edges
        
    best_cost = _count_short_cycles(adj, n)
    best_edges = list(edges)
    best_adj = [list(a) for a in adj]
    
    temp = 50.0
    cooling = 0.99
    
    # We need to be fast.
    # Precompute adjacency sets for quick lookup?
    adj_set = [set(a) for a in adj]
    
    while time.time() < deadline - 2:
        # Pick two random edges
        e1_idx = random.randint(0, len(edges) - 1)
        e2_idx = random.randint(0, len(edges) - 1)
        if e1_idx == e2_idx:
            continue
            
        u1, v1 = edges[e1_idx]
        u2, v2 = edges[e2_idx]
        
        # Ensure distinct vertices
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            continue
            
        # Try swap 1: (u1, u2) and (v1, v2)
        new_e1 = (min(u1, u2), max(u1, u2))
        new_e2 = (min(v1, v2), max(v1, v2))
        
        if new_e1 != new_e2 and new_e1 not in adj_set[u1] and new_e2 not in adj_set[v1]:
             # Check if edges exist
             if new_e1[0] in adj_set[new_e1[1]]: continue
             if new_e2[0] in adj_set[new_e2[1]]: continue
             
             # Perform swap
             # Remove old edges
             adj[u1].remove(v1); adj[v1].remove(u1)
             adj[u2].remove(v2); adj[v2].remove(u2)
             adj_set[u1].discard(v1); adj_set[v1].discard(u1)
             adj_set[u2].discard(v2); adj_set[v2].discard(u2)
             
             # Add new edges
             adj[u1].append(u2); adj[u2].append(u1)
             adj[v1].append(v2); adj[v2].append(v1)
             adj_set[u1].add(u2); adj_set[u2].add(u1)
             adj_set[v1].add(v2); adj_set[v2].add(v1)
             
             edges[e1_idx] = new_e1
             edges[e2_idx] = new_e2
             
             new_cost = _count_short_cycles(adj, n)
             
             if new_cost < best_cost:
                 best_cost = new_cost
                 best_edges = list(edges)
                 best_adj = [list(a) for a in adj]
             elif random.random() < math.exp(-(new_cost - best_cost) / temp):
                 pass # Keep the move
             else:
                 # Revert
                 adj[u1].remove(u2); adj[u2].remove(u1)
                 adj[v1].remove(v2); adj[v2].remove(v1)
                 adj_set[u1].discard(u2); adj_set[u2].discard(u1)
                 adj_set[v1].discard(v2); adj_set[v2].discard(v1)
                 
                 adj[u1].append(v1); adj[v1].append(u1)
                 adj[u2].append(v2); adj[v2].append(u2)
                 adj_set[u1].add(v1); adj_set[v1].add(u1)
                 adj_set[u2].add(v2); adj_set[v2].add(u2)
                 
                 edges[e1_idx] = (u1, v1)
                 edges[e2_idx] = (u2, v2)
        
        # Try swap 2: (u1, v2) and (v1, u2)
        new_e1 = (min(u1, v2), max(u1, v2))
        new_e2 = (min(v1, u2), max(v1, u2))
        
        if new_e1 != new_e2 and new_e1 not in adj_set[u1] and new_e2 not in adj_set[v1]:
             if new_e1[0] in adj_set[new_e1[1]]: continue
             if new_e2[0] in adj_set[new_e2[1]]: continue
             
             adj[u1].remove(v1); adj[v1].remove(u1)
             adj[u2].remove(v2); adj[v2].remove(u2)
             adj_set[u1].discard(v1); adj_set[v1].discard(u1)
             adj_set[u2].discard(v2); adj_set[v2].discard(u2)
             
             adj[u1].append(v2); adj[v2].append(u1)
             adj[v1].append(u2); adj[u2].append(v1)
             adj_set[u1].add(v2); adj_set[v2].add(u1)
             adj_set[v1].add(u2); adj_set[u2].add(v1)
             
             edges[e1_idx] = new_e1
             edges[e2_idx] = new_e2
             
             new_cost = _count_short_cycles(adj, n)
             
             if new_cost < best_cost:
                 best_cost = new_cost
                 best_edges = list(edges)
                 best_adj = [list(a) for a in adj]
             elif random.random() < math.exp(-(new_cost - best_cost) / temp):
                 pass
             else:
                 adj[u1].remove(v2); adj[v2].remove(u1)
                 adj[v1].remove(u2); adj[u2].remove(v1)
                 adj_set[u1].discard(v2); adj_set[v2].discard(u1)
                 adj_set[v1].discard(u2); adj_set[u2].discard(v1)
                 
                 adj[u1].append(v1); adj[v1].append(u1)
                 adj[u2].append(v2); adj[v2].append(u2)
                 adj_set[u1].add(v1); adj_set[v1].add(u1)
                 adj_set[u2].add(v2); adj_set[v2].add(u2)
                 
                 edges[e1_idx] = (u1, v1)
                 edges[e2_idx] = (u2, v2)
                 
        temp *= cooling
        
        # Periodically check if we found a valid graph
        if best_cost == 0:
             if _has_girth_at_least(best_edges, n, g):
                 return n, best_edges

    # Final check on best found
    if _has_girth_at_least(best_edges, n, g):
        return n, best_edges
        
    return None

def _gen_random_4reg(n):
    """Generate a random 4-regular graph on n vertices."""
    if n < 4:
        return None
    if n * 4 % 2 != 0:
        return None
    
    for attempt in range(100):
        stubs = []
        for v in range(n):
            for _ in range(4):
                stubs.append(v)
        
        random.shuffle(stubs)
        
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            edge = (min(u,v), max(u,v))
            if edge in edges:
                valid = False
                break
            edges.add(edge)
        
        if valid:
            return list(edges)
    
    return None

# EVOLVE-BLOCK-END
