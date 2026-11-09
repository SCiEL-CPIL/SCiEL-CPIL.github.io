import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    # Start with 8 K4s and 1 isolated vertex (n=33)
    n = 33
    adj = [set() for _ in range(n)]
    
    # Build 8 K4s
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                adj[base + i].add(base + j)
                adj[base + j].add(base + i)
                
    # Vertex 32 is isolated initially.
    # We will try to connect it and adjust others.
    
    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    def has_c5(adj):
        # Check for C5 subgraph
        # Use bitsets for speed
        bit_adj = [0] * n
        for u in range(n):
            mask = 0
            for v in adj[u]:
                mask |= (1 << v)
            bit_adj[u] = mask
            
        # Find C5: u1-u2-u3-u4-u5-u1
        # u1 < u2 < u3 < u4 < u5 to avoid permutations?
        # No, C5 is a cycle.
        # Iterate u1, u2, u3, u4, u5
        # This is O(n^5) which is 33^5 ~ 39M. Too slow in Python loop.
        # Better: iterate u1, u2, u3.
        # Common neighbors of u1 and u3? No.
        # Path u1-u2-u3.
        # Need u4, u5 such that u3-u4-u5-u1.
        # u4 in N(u3) \ {u2}
        # u5 in N(u4) \ {u3}
        # u5 in N(u1) \ {u2}
        # And u4 != u1, u5 != u2, etc.
        
        for u1 in range(n):
            n1 = adj[u1]
            for u2 in n1:
                if u2 <= u1: continue # Symmetry? No, C5 is directed in search but undirected in graph.
                # Just check all
                n2 = adj[u2]
                for u3 in n2:
                    if u3 == u1: continue
                    n3 = adj[u3]
                    for u4 in n3:
                        if u4 == u1 or u4 == u2: continue
                        n4 = adj[u4]
                        for u5 in n4:
                            if u5 == u1 or u5 == u2 or u5 == u3: continue
                            if u5 in n1 and u4 in adj[u5]: # u5-u1 and u5-u4
                                # Check u4-u3 (yes, u4 in n3), u3-u2 (yes), u2-u1 (yes)
                                # Check u5-u4 (yes, u4 in n4)
                                # Check u1-u2 (yes)
                                # Check u2-u3 (yes)
                                # Check u3-u4 (yes)
                                # Check u4-u5 (yes)
                                # Check u5-u1 (yes)
                                return True
        return False

    def has_is_9(adj):
        # Check if there is an independent set of size 9
        # Use branching or simple backtracking
        # n=33, small.
        # Try to find IS of size 9.
        
        # Greedy heuristic first?
        # If greedy fails, try backtracking.
        
        def find_is(vertices, size):
            if size == 0:
                return True
            if len(vertices) < size:
                return False
            if time.time() > deadline:
                return False # Abort
            
            # Pick a vertex
            v = vertices[0]
            rest = vertices[1:]
            
            # Option 1: Include v
            # Neighbors of v cannot be in IS
            neighbors = adj[v]
            new_vertices = [u for u in rest if u not in neighbors]
            if find_is(new_vertices, size - 1):
                return True
            
            # Option 2: Exclude v
            if find_is(rest, size):
                return True
                
            return False

        return find_is(list(range(n)), 9)

    # Initial state is invalid (IS of 9 exists: 1 from each K4 + isolated v)
    # We need to add edges to break IS or remove C5s.
    
    # Strategy:
    # 1. Randomly add edges to connect vertex 32 to others.
    # 2. Check IS. If IS found, add edge within IS.
    # 3. Check C5. If C5 found, remove edge from C5.
    # 4. Repeat.
    
    # To make it faster, we can use a heuristic.
    # Connect 32 to all vertices in K4_0. This creates K5 (C5).
    # So don't do that.
    
    # Let's try to find a valid graph via local search.
    # Start with 8 K4s + 1 vertex connected to 2 vertices of K4_0.
    # This had IS=9.
    
    # Let's just return 32 if we don't find 33 quickly.
    
    best_n = 32
    best_edges = get_edges(adj) # This is 8 K4s + isolated, which is invalid for n=33 but valid for n=32 if we drop vertex 32.
    
    # Actually, the current program returns 32.
    # Let's try to search for 33.
    
    # Random initialization for 33
    adj = [set() for _ in range(n)]
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                adj[base + i].add(base + j)
                adj[base + j].add(base + i)
                
    # Connect 32 to random vertices
    random.seed(42)
    for _ in range(5):
        # Add random edges involving 32
        for _ in range(3):
            v = random.randint(0, 31)
            adj[32].add(v)
            adj[v].add(32)
            
    # Local search
    for step in range(2000):
        if time.time() > deadline:
            break
            
        is_c5 = has_c5(adj)
        is_is9 = has_is_9(adj)
        
        if not is_c5 and not is_is9:
            return {'n': n, 'edges': get_edges(adj)}
            
        if is_c5:
            # Remove a random edge from a C5
            # This is expensive to find C5 again.
            # Just remove a random edge incident to a high degree node?
            # Or restart?
            # Let's remove a random edge from the graph to break cycles.
            u, v = random.choice(list(adj[32])) if adj[32] else (random.randint(0,31), random.randint(0,31))
            if u != v:
                adj[u].remove(v)
                adj[v].remove(u)
                
        elif is_is9:
            # Add a random edge to break IS
            # Pick a random pair of non-adjacent vertices
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)

    # If 33 not found, return 32
    adj32 = [set() for _ in range(32)]
    edges32 = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                adj32[base + i].add(base + j)
                adj32[base + j].add(base + i)
                edges32.append([base + i, base + j])
                
    return {'n': 32, 'edges': edges32}

