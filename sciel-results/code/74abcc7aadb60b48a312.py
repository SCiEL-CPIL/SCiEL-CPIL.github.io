import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Fallback: L-U-W Construction
    def construct_luw(k, g):
        # (Copy of current code)
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

    luw_n, luw_edges = construct_luw(k, g)
    
    # Try Greedy Search
    def construct_greedy(k, g, deadline):
        # Moore Bound
        mb = 1
        term = 1
        for i in range((g - 3) // 2):
            mb += k * term
            term *= (k - 1)
        
        n_target = mb
        best_graph = None
        
        while time.time() < deadline:
            # Generate random k-regular graph on n_target vertices
            adj = generate_random_regular(k, n_target)
            if adj is None:
                n_target += 10
                continue
                
            # Try to increase girth
            success = False
            for _ in range(5000): # Max swaps per attempt
                if time.time() > deadline:
                    return None
                girth = check_girth(adj, g)
                if girth >= g:
                    success = True
                    break
                # Find a short cycle and try to break it
                # Simple heuristic: pick random edge, try to swap
                # This is slow. Better: find specific short cycle.
                # For now, just random swaps to see if we improve girth
                if not random_swap(adj, k):
                    break
            
            if success:
                edges = []
                for u in range(n_target):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n_target, edges
            
            # If not success, increase n_target
            n_target += 10
            
        return None

    result = construct_greedy(k, g, deadline)
    if result:
        n, edges = result
        if n < luw_n:
            return n, edges
            
    return luw_n, luw_edges

# Helper functions
def generate_random_regular(k, n):
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    random.shuffle(stubs)
    edges = set()
    adj = [set() for _ in range(n)]
    valid = True
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            # Self loop, try to fix by swapping with another stub
            valid = False
            break
        if v in adj[u]:
            # Multi edge
            valid = False
            break
        adj[u].add(v)
        adj[v].add(u)
        edges.add((min(u,v), max(u,v)))
    
    if not valid:
        # Retry with local swaps to fix multi-edges/self-loops
        # This is complex. For simplicity, just retry generation.
        return None
        
    return adj

def check_girth(adj, g):
    n = len(adj)
    for u in range(n):
        visited = {u: 0}
        queue = [(u, 0)]
        while queue:
            v, d = queue.pop(0)
            if d >= g // 2:
                continue
            for w in adj[v]:
                if w in visited:
                    if d + 1 + visited[w] < g:
                        return d + 1 + visited[w]
                else:
                    visited[w] = d + 1
                    queue.append((w, d + 1))
    return g

def random_swap(adj, k):
    n = len(adj)
    # Pick random edge (u, v)
    u = random.randint(0, n-1)
    if not adj[u]: return False
    v = random.choice(list(adj[u]))
    # Find another edge (x, y) to swap with
    x = random.randint(0, n-1)
    if not adj[x]: return False
    y = random.choice(list(adj[x]))
    
    # Try swap (u, v) and (x, y) -> (u, x) and (v, y)
    # Must ensure no self-loops or multi-edges
    if u == x or u == y or v == x or v == y:
        return False
    if x in adj[u] or y in adj[u] or x in adj[v] or y in adj[v]:
        return False
        
    adj[u].remove(v)
    adj[v].remove(u)
    adj[x].remove(y)
    adj[y].remove(x)
    
    adj[u].add(x)
    adj[x].add(u)
    adj[v].add(y)
    adj[y].add(v)
    return True
