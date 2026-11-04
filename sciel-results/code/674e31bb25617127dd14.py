import time
import random
import itertools

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    # Try local search for a smaller graph
    # Target n=320 for k=4, g=10
    if k == 4 and g == 10:
        n_target = 320
        # Try to construct a 4-regular girth-10 graph of size 320
        result = _local_search_cage(n_target, k, g, deadline - start_time)
        if result is not None:
            n, edges = result
            if _verify(n, edges, k, g):
                return n, edges
        
    # Fallback: Lazebnik-Ustimenko-Woldar construction
    if time.time() > deadline:
        return 0, []  # Should not happen, but safe exit
        
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif q == 5:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    else:
        return 0, []
        
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
                
    n = len(order)
    edges_list = sorted(set(edges))
    
    if _verify(n, edges_list, k, g):
        return n, edges_list
    
    return n, edges_list

def _local_search_cage(n, k, g, time_budget):
    """
    Attempt to construct a k-regular graph of girth >= g with n vertices
    using local search (2-switches).
    Returns (n, edges) if successful, None otherwise.
    """
    if n < 1:
        return None
    
    # Generate random k-regular graph
    # Using a simple algorithm: create a random permutation of half-edges and pair them
    # This might create multi-edges or loops, so we need to clean up or use a better generator
    # For k=4, we can use the "configuration model" with rejection or repair
    
    def generate_random_regular(n, k):
        # Ensure n*k is even
        if (n * k) % 2 != 0:
            return None
        
        # Create list of half-edges
        half_edges = []
        for i in range(n):
            for _ in range(k):
                half_edges.append(i)
        
        edges = set()
        attempts = 0
        while attempts < 1000:
            random.shuffle(half_edges)
            edges = set()
            valid = True
            for i in range(0, len(half_edges), 2):
                u = half_edges[i]
                v = half_edges[i+1]
                if u == v:
                    valid = False
                    break
                if (u, v) in edges or (v, u) in edges:
                    valid = False
                    break
                edges.add((u, v))
            if valid:
                return list(edges)
            attempts += 1
        return None

    edges = generate_random_regular(n, k)
    if edges is None:
        return None
    
    # Convert to adjacency list for faster access
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Function to count short cycles (length < g)
    # For simplicity, we count cycles of length 3 to g-1
    # This is expensive. We will use a heuristic:
    # Just check if girth is < g using a fast BFS from a random edge?
    # Or count 4-cycles and 6-cycles.
    
    def count_short_cycles(adj, g):
        count = 0
        # Count 4-cycles
        # For each pair (u, v), count common neighbors
        # If common neighbors > 1, it forms 4-cycles
        # Actually, number of 4-cycles containing (u,v) is C(common, 2)
        # Sum over all pairs? No, each 4-cycle is counted 4 times (once for each edge)
        # So total 4-cycles = Sum_{u<v} C(common(u,v), 2) / 2 ? No.
        # A 4-cycle u-v-x-y-u has edges (u,v), (v,x), (x,y), (y,u).
        # The pairs of opposite vertices are (u,x) and (v,y).
        # So for each pair (u,x), the number of 4-cycles is C(common(u,x), 2).
        # Each 4-cycle is counted twice (once for (u,x) and once for (v,y)).
        # So total 4-cycles = 0.5 * Sum_{u!=x} C(common(u,x), 2)
        
        # This is O(N^3) roughly.
        
        # For now, let's just do a BFS to find *any* short cycle.
        # If we find one, return 1. If not, return 0.
        # This is not a good objective function for optimization, but good for checking.
        
        # Let's use a proper count for 4-cycles and 6-cycles if possible.
        # Or just use the "girth" as the objective, but it's discrete.
        
        # Heuristic: Count 4-cycles.
        c4 = 0
        for u in range(n):
            for v in range(u+1, n):
                common = set(adj[u]) & set(adj[v])
                if len(common) >= 2:
                    c4 += len(common) * (len(common) - 1) // 2
        # Each 4-cycle is counted 2 times (for the two pairs of opposite vertices)
        c4 //= 2
        
        # Count 6-cycles? Too expensive.
        # Let's just return c4 as a proxy.
        return c4

    # Local Search
    current_edges = edges
    current_adj = adj
    current_score = count_short_cycles(current_adj, g)
    best_edges = current_edges
    best_adj = current_adj
    best_score = current_score
    
    t = 1.0
    t_min = 0.01
    cooling = 0.99
    
    start_ls = time.time()
    
    while time.time() - start_ls < time_budget - 10: # Leave 10s for verification
        if current_score == 0:
            # Check if girth is actually >= g
            # We need to check for cycles of length 3, 5, 7, 9 as well
            # The count_short_cycles only checked 4-cycles.
            # So we must do a full girth check.
            if _verify(n, current_edges, k, g):
                return n, current_edges
            # If verify fails, it means there is an odd cycle or 6/8 cycle.
            # We need to find it.
        
        # Find a move
        # Pick a random edge (u, v)
        # Try to swap it with another edge (x, y)
        # A 2-switch replaces (u,v) and (x,y) with (u,x) and (v,y)
        # Conditions:
        # 1. u!=x, u!=y, v!=x, v!=y
        # 2. (u,x) not in edges, (v,y) not in edges
        # 3. u!=x, v!=y (no loops)
        
        # To make it effective, we should try to break short cycles.
        # Let's pick a random edge from the current graph.
        if not current_edges:
            break
            
        u, v = random.choice(current_edges)
        
        # Find candidates for (x, y)
        # We want (u, x) and (v, y) to be valid.
        # This is complex to sample directly.
        
        # Alternative: Pick a random vertex x != u, v.
        # Pick a random neighbor y of x? No.
        
        # Let's try a random 2-switch.
        # Pick two random edges (u,v) and (x,y).
        # If they are disjoint, try the switch.
        
        if len(current_edges) < 2:
            break
            
        (x, y) = random.choice(current_edges)
        
        # Check disjointness
        if u == x or u == y or v == x or v == y:
            continue
            
        # Try switch: (u,v), (x,y) -> (u,x), (v,y)
        # Check if (u,x) and (v,y) exist
        if (u, x) in current_adj[u] or (v, y) in current_adj[v]:
            continue
        if (x, u) in current_adj[x] or (y, v) in current_adj[y]:
            continue
            
        # Apply switch
        new_edges = list(current_edges)
        new_edges.remove((u, v))
        new_edges.remove((x, y))
        new_edges.append((u, x))
        new_edges.append((v, y))
        
        # Update adjacency
        new_adj = [list(nei) for nei in current_adj]
        # Remove old
        new_adj[u].remove(v)
        new_adj[v].remove(u)
        new_adj[x].remove(y)
        new_adj[y].remove(x)
        # Add new
        new_adj[u].append(x)
        new_adj[x].append(u)
        new_adj[v].append(y)
        new_adj[y].append(v)
        
        new_score = count_short_cycles(new_adj, g)
        
        delta = new_score - current_score
        if delta <= 0 or random.random() < pow(2.718, -delta / t):
            current_edges = new_edges
            current_adj = new_adj
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = current_edges
                best_adj = current_adj
        
        t *= cooling
        if t < t_min:
            t = t_min
            
    return None

def _verify(n, edges, k, g):
    """Return True iff the graph is simple, k-regular, connected, and has girth >= g."""
    if n < 1:
        return False
    
    # Check for duplicate edges
    if len(edges) != len(set(edges)):
        return False
        
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n:
            return False
        if a == b:
            return False
        adj[a].append(b)
        adj[b].append(a)
        
    for i in range(n):
        if len(adj[i]) != k:
            return False

    # connectivity
    seen = [False] * n
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                stack.append(w)
    if cnt != n:
        return False

    # Girth check using non-backtracking BFS from each edge
    # For each edge (u,v), find shortest non-backtracking path from u to v
    # Girth = min over edges of (shortest non-backtracking path length + 1)
    INF = 10 ** 9
    for (u, v) in edges:
        # BFS from u, avoiding backtracking through the edge (u,v)
        # State: (prev, cur, distance)
        dist = {}
        q = []
        for w in adj[u]:
            if w == v:
                continue  # Don't start by going back through (u,v)
            dist[(u, w)] = 1
            q.append((u, w, 1))
        head = 0
        best = INF
        while head < len(q):
            prev, cur, L = q[head]
            head += 1
            if cur == v:
                if L + 1 < best:
                    best = L + 1
                continue
            if L >= g:
                break
            for w in adj[cur]:
                if w == prev:
                    continue
                if (cur, w) not in dist:
                    dist[(cur, w)] = L + 1
                    q.append((cur, w, L + 1))
        if best < g:
            return False
    return True
