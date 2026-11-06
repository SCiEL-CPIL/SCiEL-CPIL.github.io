import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    n = 20
    # C[u][v] stores the number of common neighbors of u and v
    C = [[0] * n for _ in range(n)]
    # Adj[u][v] is True if edge (u,v) exists
    Adj = [[False] * n for _ in range(n)]
    
    def is_valid_add(u, v):
        # Check if adding (u,v) keeps the graph C4-free
        # Condition 1: C[u][v] <= 1
        if C[u][v] > 1:
            return False
        # Condition 2: For all x in N(u), C[x][v] must be 0
        for x in range(n):
            if Adj[u][x] and C[x][v] > 0:
                return False
        # Condition 3: For all y in N(v), C[u][y] must be 0
        for y in range(n):
            if Adj[v][y] and C[u][y] > 0:
                return False
        return True

    def add_edge(u, v):
        Adj[u][v] = True
        Adj[v][u] = True
        # Update C
        # C[u][v] and C[v][u] remain the same value (count of common neighbors)
        # But the "potential" for future additions changes.
        # Actually, C[u][v] is the count of common neighbors.
        # When we add (u,v), u becomes a common neighbor of x and v for all x in N(u).
        # So C[x][v] += 1 for all x in N(u).
        # And C[u][y] += 1 for all y in N(v).
        for x in range(n):
            if Adj[u][x]:
                C[x][v] += 1
                C[v][x] += 1
        for y in range(n):
            if Adj[v][y]:
                C[u][y] += 1
                C[y][u] += 1
        # Note: C[u][v] itself does not change immediately in terms of value,
        # but the logic for "is_valid_add" uses C[u][v] as the current count.
        # Wait, if we add (u,v), do we update C[u][v]?
        # C[u][v] is the number of common neighbors.
        # The edge (u,v) itself is not a neighbor.
        # So C[u][v] stays the same.
        # However, the conditions for *other* edges change.

    def remove_edge(u, v):
        # Reverse of add_edge
        for x in range(n):
            if Adj[u][x] and x != v: # x is neighbor of u, x != v
                C[x][v] -= 1
                C[v][x] -= 1
        for y in range(n):
            if Adj[v][y] and y != u: # y is neighbor of v, y != u
                C[u][y] -= 1
                C[y][u] -= 1
        Adj[u][v] = False
        Adj[v][u] = False

    def get_edges():
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if Adj[u][v]:
                    edges.append([u, v, 0]) # Color 0 is fine
        return edges

    best_edge_count = 0
    best_edges = []

    # Strategy: Multiple restarts with random greedy + local search
    num_restarts = 5000
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Reset
        for i in range(n):
            for j in range(n):
                C[i][j] = 0
                Adj[i][j] = False
        
        # Randomized Greedy Construction
        # Shuffle vertices to break symmetry
        vertices = list(range(n))
        random.shuffle(vertices)
        
        # Try to add edges in random order
        # To be faster, we can just iterate all pairs and try to add if valid
        # This is O(n^2 * n) per step, which is fast.
        
        # Greedy Phase
        while True:
            added = False
            # Randomize the order of pairs to explore different solutions
            pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if not Adj[u][v]:
                    if is_valid_add(u, v):
                        add_edge(u, v)
                        added = True
            if not added:
                break
        
        current_count = sum(sum(row) for row in Adj) // 2
        if current_count > best_edge_count:
            best_edge_count = current_count
            best_edges = get_edges()
            
        # Local Search Phase (optional, might be slow)
        # Try to swap edges to increase count?
        # Since greedy usually gets close to max for C4-free, we might skip heavy LS.
        # But let's try a simple "remove one, add two" if stuck?
        # Given time constraints, just many restarts is safer.

    return best_edges
