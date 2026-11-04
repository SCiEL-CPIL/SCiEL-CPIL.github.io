import time
import random
from collections import deque

def _validate_girth(n, edges, g):
    """
    Checks if the graph with n vertices and given edges has girth at least g.
    Uses non-backtracking walks from each vertex.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != 4: # Assuming k=4 for this specific task optimization, but make it general
            # The prompt says k is passed, but this helper is for the specific case or general?
            # Let's make it general k.
            pass

    # For speed, we assume k=4 in the main loop for this specific task, 
    # but the function should be general.
    
    # Girth check
    for start in range(n):
        # dist[(curr, prev)] = depth
        # We want to find if there is a cycle of length < g.
        # A cycle of length L through start corresponds to a non-backtracking walk 
        # from start to start of length L.
        # However, standard BFS on (node, prev) state space works.
        
        # Initialize
        dq = deque()
        dist = {}
        
        # Start from all neighbors of start
        for nb in adj[start]:
            state = (nb, start)
            dq.append(state)
            dist[state] = 1
            
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            
            # If we reached start, we found a cycle
            if curr == start:
                cycle_len = depth + 1
                if cycle_len < g:
                    return False
                # We can stop exploring this path if we found a cycle >= g?
                # No, there might be a shorter one. But BFS finds shortest.
                # So if the first time we hit start is >= g, then no shorter cycle exists?
                # Wait. BFS explores by depth.
                # If we hit start at depth d, the cycle length is d+1.
                # Since BFS is level-order, the first time we hit start is the shortest cycle.
                # So if d+1 >= g, we are safe for this start?
                # Yes, for this start.
                # But we must check all starts?
                # The girth is the minimum cycle length in the graph.
                # Every cycle passes through some start.
                # So if for all starts, the shortest cycle through start is >= g, then girth >= g.
                # Actually, if we check all starts, we are safe.
                # But we can break early if we find a cycle < g.
                continue

            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
                    
    return True

def _random_4_regular(n, rng):
    """
    Generates a random 4-regular simple graph on n vertices.
    Returns list of edges or None.
    """
    if n < 4:
        return None
    # Handshaking lemma: n * 4 must be even. n can be any integer >= 4?
    # 4n is always even.
    
    # Configuration Model
    stubs = []
    for v in range(n):
        for _ in range(4):
            stubs.append(v)
            
    edges = set()
    valid = True
    
    # Shuffle
    rng.shuffle(stubs)
    
    for i in range(0, n * 4, 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            valid = False
            break
        edge = (min(u, v), max(u, v))
        if edge in edges:
            valid = False
            break
        edges.add(edge)
        
    if not valid:
        return None
        
    return list(edges)

def construct(k, g):
    t_start = time.time()
    deadline = t_start + 110.0 # 10s margin
    
    # 1. Generate LUW baseline
    # The current program's LUW logic
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
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
                
    n_luw = len(order)
    
    # If k != 4 or g != 10, just return LUW (or whatever it is)
    # The task is specifically for k=4, g=10?
    # "The evaluator passes this task's degree k and girth g."
    # But the prompt title is "Small 4-regular graph of girth 10".
    # We assume k=4, g=10 is the target case.
    
    if k != 4 or g != 10:
        return n_luw, edges_luw
        
    # 2. Try to find smaller graph
    # Lower bound for (4,10) is 243.
    # Known best is 320.
    # Try n from 243 to 319.
    
    rng = random.Random(42)
    
    # We can optimize by trying smaller n first?
    # Or random n?
    # Smaller n is harder to find?
    # Actually, 4-regular graphs exist for all n >= 4.
    # But girth constraint is tight.
    
    # Let's try a few random graphs for n in [243, 320)
    # To maximize chance, we might want to try n close to 320 first?
    # No, we want the minimum n.
    # So we should search n=243, 244, ...
    # But 243 might be impossible.
    # We can just iterate n.
    
    best_n = n_luw
    best_edges = edges_luw
    
    # Time budget for search
    search_deadline = t_start + 100.0
    
    # We can also try to optimize the LUW graph? No, it's fixed.
    
    # Strategy:
    # For each n from 243 to best_n - 1:
    #   Try to find a valid graph.
    #   If found, update best and break?
    #   No, we want the *smallest*.
    #   So if we find one at 243, we return it.
    #   If we don't, we try 244, etc.
    
    # However, trying every n is slow.
    # We can parallelize? No.
    # We can try a few random trials per n.
    
    # Actually, the probability of a random 4-regular graph having girth 10 is very low.
    # For n=320, it might be non-zero.
    # For n=243, it might be zero.
    
    # Given the 100s budget, we can try maybe 1000-5000 random graphs.
    # Let's distribute them.
    
    # If we find ANY graph with n < n_luw, we return it.
    # To ensure we return the *smallest*, we should search in increasing order of n.
    # But if we don't find one at 243, we move to 244.
    # This might waste time on impossible n.
    
    # Alternative: Just try to find *any* graph with n < 320.
    # If we find one, return it.
    # But we want the minimum.
    # If we find one at 300, but there is one at 250, we missed it.
    # But 300 is better than 320.
    # The score is n. So 300 is better than 320.
    # So finding *any* smaller graph is an improvement.
    # Finding the *smallest* is better, but any improvement > 1.0.
    
    # So, let's just try to find a graph with n < 320.
    # We can sample n uniformly from [243, 319].
    
    for _ in range(5000): # Max trials
        if time.time() > search_deadline:
            break
            
        # Pick a random n
        n = rng.randint(243, 319)
        
        # Try to generate
        edges = _random_4_regular(n, rng)
        if edges is None:
            continue
            
        # Validate girth
        # This is the bottleneck.
        if _validate_girth(n, edges, 10):
            best_n = n
            best_edges = edges
            # Don't break, try to find smaller?
            # Or break?
            # If we break, we might not find a smaller one.
            # But we can continue searching for smaller n.
            # However, we don't know if smaller n exists.
            # Let's just keep the best found so far.
            
    return best_n, best_edges
