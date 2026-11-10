import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth >= g on as few vertices as possible.
    Uses a randomized local search for small k, g (specifically k=4, g=10)
    to find a graph much smaller than the L-U-W construction.
    Falls back to L-U-W if the search fails or for other parameters.
    """
    t0 = time.time()
    DEADLINE = 110.0  # Safety margin
    
    def elapsed():
        return time.time() - t0

    # Specific optimization for k=4, g=10
    if k == 4 and g == 10:
        # Try to find a 4-regular graph of girth 10 on n vertices
        # Start with known best n=320, then try larger if needed
        # We restrict to bipartite graphs as the cage is bipartite
        
        def try_bipartite_local_search(n, time_limit):
            """
            Attempts to construct a 4-regular bipartite graph on n vertices
            (n/2 in each part) with girth >= 10.
            Returns (n, edges) if successful, else None.
            """
            if n % 2 != 0:
                return None
            
            half_n = n // 2
            # Vertices 0..half_n-1 are in Left, half_n..n-1 are in Right
            
            def generate_random_4_reg_bipartite():
                # Left vertices: 0..half_n-1
                # Right vertices: half_n..n-1
                # Each left vertex connects to 4 distinct right vertices.
                # Each right vertex connects to 4 distinct left vertices.
                
                # Method: Random permutation of right vertices for each left vertex?
                # No, that doesn't guarantee right degrees.
                # Use a random 4-regular bipartite graph generation:
                # 1. Create a list of 4 copies of each Left vertex.
                # 2. Shuffle this list.
                # 3. The first half_n*4 items are assigned to Right vertices?
                # No, that assigns 4 neighbors to each Right vertex, but they are 
                # specific Left vertices.
                # Let's do:
                # For each Right vertex, pick 4 distinct Left vertices.
                # This is hard to balance.
                
                # Better: 
                # 1. List of all Left vertices, repeated 4 times.
                # 2. Shuffle.
                # 3. Divide into chunks of 4. Each chunk is the neighbor set of a Right vertex.
                # This ensures each Right vertex has 4 neighbors.
                # And each Left vertex appears 4 times in the total list, so it has degree 4.
                
                left_pool = [i for i in range(half_n) for _ in range(4)]
                random.shuffle(left_pool)
                
                adj = [[] for _ in range(n)]
                # Assign neighbors for Right vertices
                for j in range(half_n):
                    right_v = half_n + j
                    neighbors = left_pool[j*4:(j+1)*4]
                    # Ensure distinct? The shuffle might put duplicates in a chunk if 
                    # we're not careful, but since we pick 4 from a pool of 4 copies,
                    # it's possible to have duplicates in a chunk?
                    # No, the pool has 4 copies of each left vertex.
                    # If a chunk contains [0, 0, 1, 2], that's a multi-edge.
                    # We need to ensure neighbors are distinct.
                    
                    # Retry logic for distinct neighbors
                    while len(set(neighbors)) < 4:
                        # Reshuffle or pick new ones
                        # Simplest: regenerate the whole graph? Or just fix this vertex?
                        # Let's just regenerate if collision happens, it's rare for large n.
                        left_pool = [i for i in range(half_n) for _ in range(4)]
                        random.shuffle(left_pool)
                        neighbors = left_pool[j*4:(j+1)*4]
                        # This loop might be slow if collisions are frequent.
                        # For n=320, half_n=160. 4 copies of 160 items.
                        # Probability of collision in a chunk of 4 is low.
                        
                    for l in neighbors:
                        adj[l].append(right_v)
                        adj[right_v].append(l)
                        
                return adj

            def get_girth(adj):
                """
                Returns the length of the shortest cycle in the graph.
                """
                n_vertices = len(adj)
                # BFS from each vertex
                # Since it's bipartite, we can just check for cycles.
                # Shortest cycle in bipartite graph is even.
                # We want >= 10.
                
                # Optimization: If we find a cycle of length < 10, return it.
                # We can stop early.
                
                for start in range(n_vertices):
                    # BFS
                    dist = [-1] * n_vertices
                    parent = [-1] * n_vertices
                    dist[start] = 0
                    queue = [start]
                    head = 0
                    while head < len(queue):
                        u = queue[head]
                        head += 1
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                queue.append(v)
                            elif parent[u] != v and parent[v] != u:
                                # Found a cycle
                                # Length = dist[u] + dist[v] + 1
                                # But this is not necessarily the shortest cycle through start.
                                # Actually, in BFS, the first time we see a back edge,
                                # it gives a cycle.
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < 10:
                                    return cycle_len
                    # If BFS finishes without finding a cycle < 10 from start,
                    # it doesn't mean the graph has no cycle < 10.
                    # We need to check all starts?
                    # Actually, if the girth is < 10, there is a cycle.
                    # BFS from every vertex in a cycle will detect it.
                    # So we must check all starts.
                    
                return 10  # Assume no short cycle found (or >= 10)

            def try_swap(adj):
                """
                Attempts a random 2-opt swap to improve girth.
                Returns True if a swap was made.
                """
                # Pick a random edge (u, v)
                # u in Left, v in Right
                # Pick another edge (u', v')
                # Swap to (u, v') and (u', v)
                
                # To be efficient, pick a random Left vertex u, and a neighbor v.
                u = random.randint(0, half_n - 1)
                if not adj[u]:
                    return False
                v = random.choice(adj[u])
                
                # Pick another Left vertex u'
                u_prime = random.randint(0, half_n - 1)
                if u_prime == u:
                    return False
                if not adj[u_prime]:
                    return False
                v_prime = random.choice(adj[u_prime])
                
                # Ensure we don't create multi-edges
                # New edges: (u, v_prime) and (u_prime, v)
                # Check if (u, v_prime) already exists
                if v_prime in adj[u]:
                    return False
                if v in adj[u_prime]:
                    return False
                
                # Perform swap
                adj[u].remove(v)
                adj[u].append(v_prime)
                adj[v].remove(u)
                adj[v].append(u_prime)
                
                adj[u_prime].remove(v_prime)
                adj[u_prime].append(v)
                adj[v_prime].remove(u_prime)
                adj[v_prime].append(u)
                
                return True

            # Main search loop
            best_girth = 0
            best_adj = None
            
            while time.time() - t0 < time_limit:
                adj = generate_random_4_reg_bipartite()
                
                # Initial girth check
                girth = get_girth(adj)
                
                if girth >= 10:
                    # Success
                    edges = []
                    for u in range(half_n):
                        for v in adj[u]:
                            if u < v: # Only add once
                                edges.append((u, v))
                    return n, edges
                
                # Local search
                current_girth = girth
                no_improve_count = 0
                while time.time() - t0 < time_limit:
                    if try_swap(adj):
                        new_girth = get_girth(adj)
                        if new_girth > current_girth:
                            current_girth = new_girth
                            no_improve_count = 0
                            if current_girth >= 10:
                                edges = []
                                for u in range(half_n):
                                    for v in adj[u]:
                                        if u < v:
                                            edges.append((u, v))
                                return n, edges
                        else:
                            no_improve_count += 1
                            if no_improve_count > 100:
                                break
                    else:
                        break
                        
            # If we exit the inner loop, we restart with a new random graph
            
            return None

        # Try a few n values
        # 320 is the known cage. Try it first.
        # If it fails, try slightly larger n where it might be easier?
        # Or just keep trying 320?
        # Let's try 320 for 60s.
        # If fail, try 324 for 20s.
        # If fail, try 328 for 20s.
        # If fail, fallback.
        
        n_candidates = [320, 324, 328, 332, 336, 340, 350, 400]
        
        for n in n_candidates:
            if time.time() - t0 > DEADLINE - 10:
                break
            
            # Allocate time based on remaining time
            # Simple: 10 seconds per candidate, max 10 candidates
            # But 320 is most important. Give it more time.
            time_limit = min(30, DEADLINE - (time.time() - t0) - 5)
            if time_limit < 1:
                continue
                
            result = try_bipartite_local_search(n, time_limit)
            if result:
                return result

    # Fallback: L-U-W Construction
    # (Same as before)
    
    q = k
    if g % 2 == 1:
        d = max(3, g - 5)
        if d % 2 == 0:
            d += 1
    else:
        d = max(4, g - 4)
        if d % 2 == 1:
            d += 1
    
    if q not in [2, 3, 4, 5, 7, 11, 13]:
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        q = 4
        while not (is_prime(q) or q == 4):
            q += 1
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y):
            return x ^ y
        def sub(x, y):
            return x ^ y
        def mul(x, y):
            return table[x][y]
    else:
        def add(x, y):
            return (x + y) % q
        def sub(x, y):
            return (x - y) % q
        def mul(x, y):
            return (x * y) % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
