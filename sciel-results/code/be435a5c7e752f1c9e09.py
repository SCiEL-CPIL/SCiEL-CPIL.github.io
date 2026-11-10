"""
Strategy: Randomized Local Search (Simulated Annealing) for Cage Graphs.
Instead of constructive algebraic methods (LUW, Voltage), we start with a random 
3-regular graph and iteratively swap edges to eliminate short cycles.
This is a different design family: optimization over the space of regular graphs 
rather than construction via explicit formulas.
"""
import time
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k; returns (n, edges)."""
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _girth_at_least_nb(adj, n, g, t0=None, deadline=140.0):
    """Return True iff graph (adjacency list) has girth >= g."""
    if g <= 2:
        return True
    
    max_depth = g - 1
    from collections import deque
    
    for s in range(n):
        visited = set()
        queue = deque()
        for u in adj[s]:
            state = (u, s)
            visited.add(state)
            queue.append((u, s, 1))
        
        while queue:
            v, p, depth = queue.popleft()
            
            if t0 is not None and (time.time() - t0) > deadline:
                return False
            
            if depth >= max_depth:
                continue
            
            nd = depth + 1
            for u in adj[v]:
                if u == p:
                    continue
                if u == s:
                    if nd < g:
                        return False
                    continue
                state = (u, v)
                if state not in visited:
                    visited.add(state)
                    queue.append((u, v, nd))
    return True

def _generate_random_cubic_graph(n, seed=42):
    """Generate a random 3-regular graph on n vertices."""
    if n < 4 or n % 2 != 0:
        return None
    
    # Configuration model
    half_edges = []
    for i in range(n):
        for _ in range(3):
            half_edges.append(i)
    
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    
    # We need to pair up half_edges such that no loops or multi-edges are formed
    # Simple retry logic for small n
    max_attempts = 100
    for _ in range(max_attempts):
        rng.shuffle(half_edges)
        adj = [set() for _ in range(n)]
        success = True
        for i in range(0, len(half_edges), 2):
            u, v = half_edges[i], half_edges[i+1]
            if u == v:
                success = False
                break
            if v in adj[u]:
                success = False
                break
            adj[u].add(v)
            adj[v].add(u)
        
        if success:
            # Check if it's connected? Not strictly required for validity, 
            # but cages are usually connected. 
            # The problem doesn't explicitly forbid disconnected graphs, 
            # but "graph" usually implies connected in this context. 
            # Let's assume connectedness is preferred or required implicitly.
            # For now, let's just return it. If disconnected, girth is min of components.
            return [sorted(list(s)) for s in adj]
            
    return None

def _find_short_cycles(adj, n, max_girth):
    """Find cycles of length < max_girth. Returns a list of cycles (list of nodes)."""
    cycles = []
    for s in range(n):
        # BFS/DFS to find cycles starting at s
        # We only care about cycles of length < max_girth
        # To avoid duplicates, we can enforce that s is the smallest node in the cycle
        # and we only traverse nodes > s (except returning to s)
        
        stack = [(s, s, [s])] # (current, parent, path)
        visited_in_path = {s}
        
        while stack:
            v, p, path = stack.pop()
            if len(path) >= max_girth:
                continue
                
            for u in adj[v]:
                if u == p:
                    continue
                if u == s:
                    if len(path) < max_girth:
                        cycles.append(path[:])
                    continue
                if u < s:
                    continue # Only find cycles where s is the min node
                if u not in visited_in_path:
                    visited_in_path.add(u)
                    stack.append((u, v, path + [u]))
                    # Note: simple DFS with global visited_in_path doesn't work well for finding all cycles
                    # We need backtracking properly.
                    
    # The above DFS is flawed for finding all simple cycles efficiently.
    # Let's use a simpler heuristic: Random walk detection.
    # Or just return None and rely on the swap logic to find violations via local checks.
    # Actually, for the swap to be useful, we need to know WHERE the short cycles are.
    # Let's do a proper DFS with backtracking for small depths.
    
    cycles = []
    for s in range(n):
        # DFS from s
        # path: list of nodes
        # prev: previous node
        def dfs(v, prev, path):
            if len(path) >= max_girth:
                return
            for u in adj[v]:
                if u == prev:
                    continue
                if u == s:
                    if len(path) < max_girth:
                        # Check if s is the smallest node in the cycle to avoid duplicates
                        if min(path) == s:
                            cycles.append(path[:])
                    return
                if u < s:
                    continue
                # Only proceed if u is not in current path (simple cycle)
                if u in path:
                    continue
                dfs(u, v, path + [u])
        
        dfs(s, -1, [s])
        
    return cycles

def _local_search_cubic(n, g, t0, deadline):
    """
    Start with a random cubic graph and use edge swaps to increase girth.
    """
    # Start with a random graph
    # For n=512, generating a random cubic graph is fast.
    # We want to minimize n, but we must find a valid one.
    # Let's try a range of n values starting from the lower bound.
    # Moore bound for k=3, g=16:
    # n >= 2 * (1 + 2 + 4 + 8 + 16 + 32 + 64) = 2 * 127 = 254?
    # Wait, Moore bound for odd girth g:
    # n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
    # For k=3, g=16 (even):
    # n >= 2 * sum_{i=0}^{g/2 - 1} (k-1)^i = 2 * sum_{i=0}^7 2^i = 2 * (2^8 - 1) = 2 * 255 = 510.
    # So n must be at least 510. Since n must be even for cubic, n >= 512.
    
    start_n = 512
    # We can try to find a graph of size start_n. If it fails, we might need larger n.
    # But the task asks for "as few vertices as possible".
    # The seed program returns LUW which is ~1500.
    # If we can find one at 512, that's a huge win.
    # If not, maybe 514, etc.
    
    # Let's try n = 512 first.
    # Random search is stochastic. We need to run it for a while.
    
    best_adj = None
    best_girth_found = 0
    
    # We will try to improve a random instance.
    # Objective: Maximize the shortest cycle length.
    
    def get_girth(adj, n, limit=20):
        # Returns the girth (shortest cycle length) or limit if none found
        # Using BFS from each node is O(n * E) which is fine for n=512.
        # We can stop early if we find a cycle of length < current best.
        min_len = limit
        for s in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            
            while queue:
                v = queue.popleft()
                if dist[v] >= limit:
                    continue
                for u in adj[v]:
                    if dist[u] == -1:
                        dist[u] = dist[v] + 1
                        parent[u] = v
                        queue.append(u)
                    elif u != parent[v]:
                        # Found a cycle
                        cycle_len = dist[v] + dist[u] + 1
                        if cycle_len < min_len:
                            min_len = cycle_len
                            if min_len == 3:
                                return 3
        return min_len

    # Run local search
    # We try a few random starts
    for attempt in range(5):
        if time.time() - t0 > deadline:
            break
            
        adj = _generate_random_cubic_graph(n, seed=100 + attempt)
        if adj is None:
            continue
            
        current_girth = get_girth(adj, n, limit=g)
        
        # If we already have girth >= g, we are done!
        if current_girth >= g:
            return adj
            
        # Simulated Annealing
        # Temperature T starts high, decays.
        T = 1.0
        cooling = 0.999
        
        # We perform swaps. A swap replaces (u,v) and (x,y) with (u,x) and (v,y) or (u,y) and (v,x).
        # We must ensure no loops or multi-edges.
        
        steps = 0
        max_steps = 50000
        
        while steps < max_steps:
            if time.time() - t0 > deadline:
                break
                
            steps += 1
            T *= cooling
            
            # Pick a random edge (u, v)
            u = random.randint(0, n-1)
            v = random.choice(adj[u])
            
            # Pick another random edge (x, y)
            x = random.randint(0, n-1)
            y = random.choice(adj[x])
            
            # Check if edges are distinct
            if u == x or u == y or v == x or v == y:
                continue
            if (u, v) == (x, y) or (u, v) == (y, x):
                continue
            if (u, v) == (v, u) or (x, y) == (y, x): # Self loops not possible in simple graph gen
                 pass
            
            # Try swap 1: (u,x) and (v,y)
            # Check validity
            valid1 = True
            if x in adj[u] or y in adj[v]: # u-x or v-y already exists
                valid1 = False
            if u == x or v == y: # Loop
                valid1 = False
            
            if valid1:
                # Perform swap
                adj[u].remove(v)
                adj[v].remove(u)
                adj[x].remove(y)
                adj[y].remove(x)
                
                adj[u].add(x)
                adj[x].add(u)
                adj[v].add(y)
                adj[y].add(v)
                
                new_girth = get_girth(adj, n, limit=g)
                
                # Acceptance criterion
                if new_girth > current_girth or (T > 0.01 and random.random() < np.exp((new_girth - current_girth) / T)):
                    current_girth = new_girth
                    if current_girth >= g:
                        return adj
                else:
                    # Revert
                    adj[u].remove(x)
                    adj[x].remove(u)
                    adj[v].remove(y)
                    adj[y].remove(v)
                    
                    adj[u].add(v)
                    adj[v].add(u)
                    adj[x].add(y)
                    adj[y].add(x)
                    
            # Try swap 2: (u,y) and (v,x)
            valid2 = True
            if y in adj[u] or x in adj[v]:
                valid2 = False
            if u == y or v == x:
                valid2 = False
            
            if valid2:
                adj[u].remove(v)
                adj[v].remove(u)
                adj[x].remove(y)
                adj[y].remove(x)
                
                adj[u].add(y)
                adj[y].add(u)
                adj[v].add(x)
                adj[x].add(v)
                
                new_girth = get_girth(adj, n, limit=g)
                
                if new_girth > current_girth or (T > 0.01 and random.random() < np.exp((new_girth - current_girth) / T)):
                    current_girth = new_girth
                    if current_girth >= g:
                        return adj
                else:
                    # Revert
                    adj[u].remove(y)
                    adj[y].remove(u)
                    adj[v].remove(x)
                    adj[x].remove(v)
                    
                    adj[u].add(v)
                    adj[v].add(u)
                    adj[x].add(y)
                    adj[y].add(x)

    return None

def construct(k, g):
    t0 = time.time()
    HARD_LIMIT = 140.0

    # Try local search for k=3, g=16
    if k == 3 and g == 16:
        # Moore bound is 510, so try 512
        adj = _local_search_cubic(512, 16, t0, HARD_LIMIT - 10.0)
        if adj is not None:
            # Convert to edge list
            edges = []
            for i in range(512):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return 512, edges
        
        # If 512 failed, try slightly larger? 
        # But the task wants minimum n. 
        # If we can't find 512 in time, we should return the best known valid one.
        # The LUW graph is valid.
        
    # Fallback: LUW
    try:
        return _luw_construct(k, g)
    except Exception:
        raise

# EVOLVE-BLOCK-END
