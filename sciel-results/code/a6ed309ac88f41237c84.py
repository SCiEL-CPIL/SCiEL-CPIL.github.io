import time
import random
import numpy as np

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    For k=4, g=10, attempts to find a graph with n < seed_n.
    The seed (LUW) for k=4, g=10 likely produces a specific n.
    We use a randomized greedy approach to build the graph layer by layer,
    checking girth constraints using BFS from each vertex.
    """
    
    # The evaluator passes k and g. 
    # For this specific task, k=4, g=10.
    # We need to return (n, edges).
    
    # Safety deadline
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    if k != 4 or g != 10:
        # Fallback to seed logic for other parameters if needed, 
        # but the task is specific. 
        # For now, return a trivial valid graph or seed behavior.
        # The seed behavior for general k, g is complex. 
        # We will implement a simple heuristic that works for k=4, g=10.
        pass
        
    # We will attempt to construct a 4-regular graph of girth 10.
    # The Moore bound for d=4, g=10 is 1 + 4 * sum_{i=0}^{4} 3^i = 1 + 4 * (1+3+9+27+81) = 1 + 4*121 = 485?
    # Wait, Moore bound for even girth g=2r: 1 + d * sum_{i=0}^{r-1} (d-1)^i.
    # r=5. 1 + 4 * (1 + 3 + 9 + 27 + 81) = 1 + 4 * 121 = 485.
    # The seed program (LUW) for q=4, d=5 (since g=10, d=g-5=5) 
    # LUW graph size is 2 * q^d. For q=4, d=5, size = 2 * 4^5 = 2048? 
    # Or is it smaller?
    # Let's check the seed code logic.
    # q = k = 4.
    # d = max(3, g-5 if (g-5)%2 else g-4). g=10 -> g-5=5 (odd) -> d=5.
    # Points: (0, y) where y is a vector of length d+1? 
    # y = [None, first] + [0]*(d-1). Length d.
    # Total points: q * q^d? No.
    # The code iterates `for first in range(q)`.
    # And y is a tuple.
    # The number of vertices in the connected component of the zero point is returned.
    # For LUW graphs, the graph is regular and usually connected or has few components.
    # The size of the LUW(d, q) graph is 2 * q^d.
    # For q=4, d=5, size = 2 * 1024 = 2048.
    # This is much larger than the known best (320).
    # So the seed is very weak.
    # We can easily beat 2048 by finding a graph of size ~300-400.
    
    # Strategy: Randomized construction with backtracking or just random greedy with restarts.
    # Given the small size target (~320), we can try to build it.
    # However, constructing a specific cage graph from scratch in 120s is hard.
    # But since the seed is 2048, ANY valid graph of size < 2048 will score > 1.
    # A random 4-regular graph of size n has girth ~ log_{d-1}(n).
    # For girth 10, n needs to be large enough.
    # Random 4-regular graphs of size 100-200 often have girth > 10.
    # Let's try to find a valid random 4-regular graph.
    
    def is_valid_graph(n, adj):
        # Check girth >= 10
        # BFS from each node, depth 4 (since g=10, radius 5, but non-backtracking)
        # Simpler: BFS up to depth 4 from each node, check for cycles.
        # If BFS finds a node at distance < 5 already visited, it might form a short cycle.
        # Actually, for girth g, BFS from any node up to depth (g-1)/2 (if g odd) or g/2 - 1 (if g even) should be a tree.
        # For g=10 (even), depth = g/2 - 1 = 4.
        # So BFS up to depth 4 from every node must not find a cycle (i.e., no node visited twice, no edge to a previously visited node in the current path except parent).
        
        for start in range(n):
            visited = [-1] * n # distance from start
            parent = [-1] * n
            queue = [(start, 0)]
            visited[start] = 0
            while queue:
                u, dist = queue.pop(0)
                if dist >= 4:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = dist + 1
                        parent[v] = u
                        queue.append((v, dist + 1))
                    else:
                        # v is already visited.
                        # If v is the parent of u, it's fine (back edge).
                        if v == parent[u]:
                            continue
                        # If v is visited and not parent, we found a cycle.
                        # Cycle length is dist(u) + dist(v) + 1.
                        # Since we are at depth <= 4, dist(u) <= 4, dist(v) <= 4.
                        # If dist(u) + dist(v) + 1 < 10, invalid.
                        # Actually, if any cycle is found in BFS tree up to depth 4, 
                        # its length is at most 2*4 + 1 = 9? 
                        # If we find a cross edge, the cycle length is dist(u) + dist(v) + 1.
                        # Max dist is 4. Max cycle length detected here is 4+4+1=9.
                        # So any such edge implies a cycle < 10.
                        return False
            # Also check for self loops? Simple graph assumed.
        return True

    def generate_random_4_regular(n):
        # Generate a random 4-regular graph on n vertices.
        # n must be even for 4-regular? No, sum of degrees must be even. 4n is always even.
        # But handshaking lemma: sum(deg) = 2E. 4n = 2E -> E = 2n.
        # We can use configuration model.
        # Half-edges: 4n stubs.
        # Match them randomly.
        
        if n < 10:
            return None
            
        # Stubs for each vertex
        stubs = []
        for i in range(n):
            for _ in range(4):
                stubs.append(i)
                
        random.shuffle(stubs)
        
        # Pair up stubs
        adj = {i: set() for i in range(n)}
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                # Self loop, invalid for simple graph in this context? 
                # Task says "simple (no loops)". 
                # So retry or handle. 
                # For now, if self loop, mark invalid.
                return None
            if v in adj[u]:
                # Multi-edge
                return None
            adj[u].add(v)
            adj[v].add(u)
            
        # Convert to list of lists for speed
        adj_list = [list(adj[i]) for i in range(n)]
        return adj_list

    best_n = 2048 # Seed size
    best_edges = []
    
    # We need to find a graph with n < 2048.
    # Let's try n from 300 to 600.
    # Random 4-regular graphs of size 300+ likely have girth > 10.
    
    # To be safe and fast, we'll test a few candidates.
    candidates = []
    # Known cage n(4,10) is 320.
    # We might not find 320, but 350 or 400 is easy.
    # Let's try sizes: 320, 322, ..., 400
    
    found = False
    # Iterate sizes
    for n in range(320, 500, 2):
        if time.time() > deadline:
            break
            
        # Try a few random instances of this size
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            adj = generate_random_4_regular(n)
            if adj is None:
                continue
                
            # Check girth
            if is_valid_graph(n, adj):
                # Convert to edge list
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                # Score is n. We want to minimize n.
                # Since we iterate n ascending, the first valid one is the best we found so far in this sweep.
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            # Since we are iterating n upwards, the first valid n is the minimum in this range.
            # We can stop if we found a very small one, or continue to see if smaller works?
            # We started at 320. If we found one at 320, that's optimal (known lower bound 243, best known 320).
            # If we found one at 320, we can't do better than 320 unless we find a graph of size 320 that is valid (which we did).
            # Wait, is 320 the best known? Yes.
            # So if we find a valid graph at n=320, we have matched the best known.
            # Can we find smaller? Unlikely if 320 is the cage.
            # But the seed was 2048. So 320 is a huge improvement.
            # We can break early to save time.
            break

    if not found:
        # If we didn't find a valid graph in the random search (unlikely for n>350),
        # fall back to a slightly larger n or return the seed?
        # We must return a valid graph.
        # Let's try a larger n, e.g., 600.
        for n in range(500, 800, 10):
            if time.time() > deadline:
                break
            for attempt in range(10):
                adj = generate_random_4_regular(n)
                if adj is None: continue
                if is_valid_graph(n, adj):
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    best_n = n
                    best_edges = edges
                    found = True
                    break
            if found:
                break

    if not found:
        # Absolute fallback: return the seed's graph?
        # We don't have the seed's graph handy to reconstruct exactly without running it.
        # But we can construct a trivial valid graph: a cycle is not 4-regular.
        # A 4-regular graph must have n >= 5.
        # If we fail, we should return the seed result to ensure validity and score 1.0.
        # However, the seed code is in the prompt. We can re-implement the seed logic as fallback.
        
        # Re-implementing seed logic quickly for fallback:
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
        best_n = len(order)
        best_edges = edges

    return best_n, best_edges
