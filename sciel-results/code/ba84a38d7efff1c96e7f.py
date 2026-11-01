import time
from collections import deque

def _verify_girth_fast(n, edges, k, g):
    """
    Fast girth verification using non-backtracking walks.
    Returns True if girth >= g, False otherwise.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False

    max_depth = g // 2

    for start in range(n):
        # visited[(v, u)] = depth of the walk ending at v coming from u
        visited = {}
        queue = deque()

        for v in adj[start]:
            queue.append((v, start, 1))
            visited[(v, start)] = 1

        while queue:
            u, prev, depth = queue.popleft()
            if depth == max_depth:
                continue
            for v in adj[u]:
                if v == prev:
                    continue
                if v == start:
                    if depth + 1 < g:
                        return False
                    continue
                state = (v, u)
                if state in visited:
                    if depth - visited[state] < g:
                        return False
                else:
                    visited[state] = depth
                    queue.append((v, u, depth + 1))
    return True

def _luw_construct(k, d):
    if k != 3:
        return None, None
    q = 3
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

def _gp_construct(n_half, k_step):
    """
    Construct Generalized Petersen Graph GP(n_half, k_step).
    Total vertices = 2 * n_half.
    """
    edges = []
    n = n_half
    for i in range(n):
        edges.append((i, (i + 1) % n))
        edges.append((n + i, n + (i + 1) % n))
        edges.append((i, n + (k_step * i) % n))
    return 2 * n, edges

def _lift_construct(m, k, g, trials=200, seed=0):
    """
    Construct a 3-regular graph as a lift of a small base graph.
    We try random lifts of size m (total vertices = m).
    Base is a 3-regular graph on m/3 vertices? No, a lift of size L over a base of size N gives NL vertices.
    Here we want total vertices n. Let n = m.
    We can view this as a voltage graph.
    Simpler: Just try random 3-regular graphs? No, that's expensive.
    
    Let's use the "randomized greedy tree-extension" idea from the prompt?
    Or maybe just random regular graphs?
    For n ~ 500-1000, generating random 3-regular graphs is possible.
    But verifying girth is O(n * g) ~ 500 * 16 = 8000 ops.
    200 trials * 8000 = 1.6M ops. Very fast.
    
    Let's implement a random 3-regular graph generator.
    Configuration model.
    """
    import random
    rng = random.Random(seed)
    
    # We need n to be even for 3-regular? No, 3n must be even, so n must be even.
    if m % 2 != 0:
        return None, None
        
    # Generate random 3-regular graph using configuration model
    # 1. Create list of stubs: [0,0,0, 1,1,1, ..., m-1, m-1, m-1]
    # 2. Shuffle
    # 3. Pair them up
    # 4. Check for multi-edges and self-loops. If found, retry.
    
    for _ in range(trials):
        stubs = []
        for i in range(m):
            stubs.extend([i, i, i])
        rng.shuffle(stubs)
        
        edges = []
        valid = True
        seen = set()
        for i in range(0, m, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            e = (min(u, v), max(u, v))
            if e in seen:
                valid = False
                break
            seen.add(e)
            edges.append(e)
            
        if not valid:
            continue
            
        # Check if it's 3-regular (should be by construction, but check for safety)
        # And check girth
        if _verify_girth_fast(m, edges, 3, g):
            return m, edges
            
    return None, None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if k != 3:
        # Fallback for non-3 regular (should not happen for this task)
        # Just try LUW for k=3 only.
        n_luw, edges_luw = _luw_construct(k, 7)
        if n_luw:
            return n_luw, edges_luw
        return 0, []

    best_n = float('inf')
    best_edges = []

    # 1. Try Generalized Petersen Graphs GP(n_half, k)
    # Scan n_half from 256 (Moore bound 512/2) to 468 (known best 936/2).
    # This is a very structured family.
    
    for n_half in range(256, 469):
        if time.time() > deadline - 20:
            break
        n_total = 2 * n_half
        if n_total >= best_n:
            break
            
        # Iterate k_step
        for k_step in range(2, n_half - 1):
            if time.time() > deadline - 10:
                break
            if k_step > n_half - k_step:
                continue
                
            n_gp, edges_gp = _gp_construct(n_half, k_step)
            if _verify_girth_fast(n_gp, edges_gp, 3, g):
                best_n = n_gp
                best_edges = edges_gp
                return best_n, best_edges

    # 2. Try Random Lifts / Random Regular Graphs
    # We know n must be >= 512.
    # The known best is 936.
    # Let's try to find a graph with n < 936 using random search.
    # We can try even n from 512 to 934.
    # But random search might not find one if it's rare.
    # However, the "Knowledge" says: "Increasing random trials for small lift sizes m (52-100) from 20 to 500 significantly increases the probability of finding a valid girth-16 lift with fewer vertices".
    # This implies that for smaller m, random search works well.
    # But m=52 is too small for girth 16 (Moore bound 512).
    # Maybe the knowledge refers to a different task or I misread?
    # "n(3,16)" Moore bound is 512.
    # So m must be >= 512.
    # Let's try random graphs for n in [512, 934] step 2.
    # But 200 trials per n is too many if we have many n's.
    # Let's be selective.
    # We can try a few n values with many trials.
    
    # Let's try n = 512, 514, ..., 550 with 50 trials each?
    # And n = 600, 700, 800, 900 with 20 trials each?
    
    import random
    rng = random.Random(123)
    
    # Strategy:
    # 1. Try n from 512 to 550 with 50 trials.
    # 2. If not found, try n from 552 to 934 with 10 trials each.
    
    # Time check
    if time.time() < deadline - 10:
        for n in range(512, 552, 2):
            if time.time() > deadline - 5:
                break
            res_n, res_edges = _lift_construct(n, 3, g, trials=50, seed=n)
            if res_n is not None:
                best_n = res_n
                best_edges = res_edges
                return best_n, best_edges

    if time.time() < deadline - 5:
        for n in range(554, 934, 2):
            if time.time() > deadline - 2:
                break
            # Reduce trials for larger n to save time
            trials = 10
            res_n, res_edges = _lift_construct(n, 3, g, trials=trials, seed=n)
            if res_n is not None:
                best_n = res_n
                best_edges = res_edges
                return best_n, best_edges

    # 3. Fallback to LUW(3, 7) which gives n=3^7 * 2? No, LUW(3,7) gives n=2 * 3^7 = 4374?
    # Let's check LUW size.
    # LUW(q, d) has q^(d+1) vertices?
    # For q=3, d=7: 3^8 = 6561?
    # Or 2 * 3^d?
    # Let's just run it and see.
    
    if time.time() < deadline - 2:
        n_luw, edges_luw = _luw_construct(3, 7)
        if n_luw and _verify_girth_fast(n_luw, edges_luw, 3, g):
            best_n = n_luw
            best_edges = edges_luw
    
    if best_n == float('inf'):
        # Last resort
        n_luw, edges_luw = _luw_construct(3, 8)
        return n_luw, edges_luw

    return best_n, best_edges

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"Constructed graph with {n} vertices")
