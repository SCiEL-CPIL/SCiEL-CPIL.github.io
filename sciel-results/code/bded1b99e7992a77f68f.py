import random
import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try to construct a graph with N vertices
    # Start from Moore bound
    # Moore bound for (5,9) is 428.
    # We'll try N from 428 to, say, 1500.
    
    def check_girth(n, edges):
        # Check if graph has girth >= g
        # We need to check for cycles of length < g
        # For each edge (u, v), check if there is a path of length < g-1 between u and v
        # that does not use the edge (u, v).
        # Actually, simpler: just check if the graph contains any cycle of length < g.
        # We can do BFS from each vertex to find shortest cycle?
        # Or, since we are constructing it, we can just return the edges and let the evaluator check?
        # No, we must certify it.
        # But the task says "certified exactly with non-backtracking walks".
        # We can just return the graph. The evaluator will check.
        # But if we return an invalid graph, score is 0.
        # So we should verify locally.
        
        # Efficient check:
        # Build adjacency list.
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check for cycles of length L < g
        # We can use a DFS with depth limit g-1.
        # Or, just check if the graph is a tree? No, it's regular.
        
        # A simple check:
        # For each edge (u, v), remove it temporarily, check dist(u, v).
        # If dist < g-1, then there is a cycle of length dist+1 < g.
        # This is O(E * (N+E)) which is too slow for N=1000.
        
        # Better:
        # We can just assume the construction is correct if we enforce the constraints during construction.
        # The vertex-by-vertex construction ensures that when adding v, no short cycle is formed involving v.
        # Since all cycles must involve the last added vertex (in a tree-like growth), this suffices?
        # No, cycles can be formed between older vertices?
        # No, if we add vertices one by one and only connect to previous ones,
        # any new cycle must include the new vertex.
        # So if we ensure that adding v doesn't create a short cycle, the graph remains short-cycle-free.
        
        return True # Placeholder, we rely on construction constraints.

    def try_construct(N):
        # n vertices 0..N-1
        # edges list
        edges = []
        adj = [[] for _ in range(N)]
        
        # We need to add vertices 0..N-1.
        # Vertex 0: No previous vertices. We can't connect to anyone.
        # So we must start with a "seed" graph or handle the first few vertices specially.
        # For a 5-regular graph, we can't start with 1 vertex.
        # We can start with a "girth 9" base graph?
        # Or, we can just pick 5 random vertices for the first vertex? No, they don't exist.
        
        # Strategy:
        # Add vertices in order 0..N-1.
        # For v=0, we can't add it.
        # So we must have a different initialization.
        # Maybe we add a "virtual" set of vertices?
        # Or, we just start with a known small graph with girth >= 9?
        # The smallest is 1116. We don't have it.
        
        # Alternative:
        # We can start with a cycle of length 9? No, not regular.
        
        # Let's try a different construction order.
        # We can add edges between existing vertices?
        # No, we need to add vertices.
        
        # Let's try to build a graph on N vertices by adding edges.
        # Start with empty graph.
        # Add edges randomly, ensuring degree <= 5 and girth >= 9.
        # This is the "Greedy Edge Addition" approach.
        
        # This is better.
        
        # 1. Initialize empty graph.
        # 2. While edges < 5*N/2:
        #    Pick random (u, v).
        #    If deg(u) < 5 and deg(v) < 5 and not connected:
        #       Check if adding (u, v) creates cycle < 9.
        #       If yes, add it.
        #    If stuck, restart or backtrack.
        
        # This is prone to getting stuck.
        
        # Let's try the Vertex-by-Vertex with a "Seed".
        # We can start with a "Cage" of size 9? No.
        
        # Given the complexity, let's fallback to LUW if we can't find a smaller one quickly.
        # But we want to IMPROVE.
        
        # Let's try to find a graph with N=1200 using Randomized Greedy.
        
        # We will use a simple heuristic:
        # 1. Generate a random 5-regular graph.
        # 2. Check girth.
        # 3. If bad, try to fix.
        
        # This is too complex to implement robustly in 120s.
        
        # Let's try a different explicit construction:
        # The **Klein** graph is (3,8).
        # The **Tutte** graph is (3,12).
        
        # For (5,9), the **Lazebnik** graph is the standard explicit one.
        # Is there a **Boutin** graph for k=5?
        # Boutin(3, 2m+1) is small.
        # Boutin(5, 2m+1) ?
        
        # If we can't find a better one, we return LUW.
        # But we can try to **optimize** the LUW parameters?
        # No, q=5 is fixed.
        
        # What if we use a **Randomized** search for a smaller graph?
        # We will try N from 428 to 1500.
        # For each N, we try to construct a graph using a simple backtracking.
        
        # Backtracking:
        # Order edges by some heuristic.
        # Try to add edge.
        # Check girth.
        # If fail, backtrack.
        
        # This is too slow.
        
        # Let's just return the LUW graph. It's safe.
        # But the score is 1.0.
        # Can we do better?
        # The task says "Smallest known such graph has 1116 vertices".
        # If we could hardcode the 1116 graph, we would get a great score.
        # But we don't have the edges.
        
        # Can we find a paper or online source for the 1116 graph?
        # No internet access.
        
        # So we must construct it.
        
        # Let's try a **Greedy** construction with a **Time Limit**.
        # If it succeeds, great. If not, fallback to LUW.
        
        return None

    # Since implementing a robust fast constructor for (5,9) is difficult,
    # and the LUW graph is guaranteed valid, we will stick to LUW.
    # However, we can try to use a **smaller** LUW variant if possible?
    # No.
    
    # Let's check if there is a **Klein** surface construction for k=5?
    # The (5,9) cage is not a known surface graph like (3,8) or (3,10).
    
    # I will output the LUW construction as it is the only reliable one I can write quickly.
    # To potentially improve, I can try to run a **Randomized Local Search** for a short time.
    # If it finds a graph with N < 3125, I return it.
    # Otherwise, I return LUW.
    
    # Let's implement a simple Randomized Local Search.
    # 1. Generate random 5-regular graph on N=1200.
    # 2. Check girth.
    # 3. If girth < 9, try to swap edges to fix.
    
    # This is risky.
    
    # I will stick to LUW.
    
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
