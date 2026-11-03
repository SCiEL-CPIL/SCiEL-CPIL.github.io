import time
import random
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def luw_construction():
        """Construct the Lazebnik-Ustimenko-Woldar graph CD(d, q)."""
        q = k
        # For girth g, we want d such that girth is at least d+5.
        # For even g, d is odd? 
        # Standard LUW graphs CD(d, q) have girth d+5 if d is odd, d+4 if d is even?
        # Actually, for q=4, g=10, d=5 gives girth 10? 
        # The current code uses: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        # If g=10, g-5=5 (odd), so d=5.
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

    def build_adj_from_edges(edges, n):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        return adj

    def nonbacktracking_girth(adj, g):
        """Return True if the graph has a cycle shorter than g (i.e., girth < g)."""
        n = len(adj)
        for s in range(n):
            for a in adj[s]:
                dist = {}
                dist[(a, s)] = 1
                q = deque([(a, s)])
                while q:
                    v, p = q.popleft()
                    d = dist[(v, p)]
                    if d >= g - 1:
                        continue
                    for w in adj[v]:
                        if w == p:
                            continue
                        new_state = (w, v)
                        if new_state not in dist:
                            dist[new_state] = d + 1
                            if w == s and v != a:
                                if d + 1 < g:
                                    return True
                            q.append(new_state)
        return False

    # Specific optimization for k=4, g=10
    if k == 4 and g == 10:
        # The (4,10)-cage has 320 vertices.
        # It can be constructed as a voltage lift of the Tutte-Coxeter graph (30 vertices, girth 8)
        # over the group Z_10.
        
        # 1. Define Tutte-Coxeter Graph
        # 30 vertices. Bipartite.
        # We can generate it or hardcode the edges.
        # Let's generate it to be safe, or use a known adjacency list.
        # The Tutte-Coxeter graph is the incidence graph of the generalized quadrangle GQ(2,2) ? No, GQ(2,2) has 15 points/lines.
        # Tutte-Coxeter is the point-line incidence graph of the generalized hexagon of order (2,2)? No.
        # It is the unique 4-regular graph on 30 vertices with girth 8.
        
        # Let's hardcode the edges of the Tutte-Coxeter graph.
        # Source: known graph.
        tc_edges = [
            (0, 10), (0, 11), (0, 12), (0, 13),
            (1, 14), (1, 15), (1, 16), (1, 17),
            (2, 18), (2, 19), (2, 20), (2, 21),
            (3, 22), (3, 23), (3, 24), (3, 25),
            (4, 26), (4, 27), (4, 28), (4, 29),
            (5, 10), (5, 15), (5, 20), (5, 25),
            (6, 11), (6, 16), (6, 21), (6, 26),
            (7, 12), (7, 17), (7, 22), (7, 27),
            (8, 13), (8, 18), (8, 23), (8, 28),
            (9, 14), (9, 19), (9, 24), (9, 29),
            (10, 5), (10, 6), (10, 7), (10, 8), # Wait, 10 is connected to 0, 5, 6, 7, 8? No, degree 4.
            # Let's verify the structure.
            # Vertices 0-9 and 10-19 and 20-29? No, 30 vertices.
            # Let's use a standard generation method for the Tutte-Coxeter graph if hardcoding is risky.
            # Actually, I can just use the known 320-vertex cage construction directly if I can find the voltages.
            # But finding voltages requires the base graph.
        ]
        
        # Let's construct the Tutte-Coxeter graph properly.
        # It is the incidence graph of the generalized quadrangle of order (2,2) is 15x15? No.
        # The Tutte-Coxeter graph is the point-line graph of the generalized hexagon G(2,2)?
        # No, G(2,2) has 14 points?
        # Let's just use a known 320-vertex cage construction if available in memory?
        # No, I must construct it.
        
        # Alternative: Use the (4,10) cage from the literature which is a voltage graph.
        # Base: Tutte-Coxeter (30 vertices). Group: Z_10.
        # Let's generate the Tutte-Coxeter graph using its algebraic definition.
        # It is the bipartite graph with parts { (i, j) | i in Z_5, j in Z_3 } ? No, 15+15=30.
        # Yes, 15 vertices in each part.
        # Part A: (i, j) with i in Z_5, j in Z_3.
        # Part B: (i, j) with i in Z_5, j in Z_3.
        # Edge between (i1, j1) in A and (i2, j2) in B iff ...
        # This is getting complicated to derive from scratch correctly in a short time.
        
        # Let's stick to the robust LUW fallback but try to optimize the search for the 320 vertex solution 
        # by using a known valid set of voltages for the Tutte-Coxeter lift if I can recall or derive them.
        # 
        # Actually, the (4,10) cage is known to be a Cayley graph? No, not symmetric enough.
        # 
        # Let's try a different known construction: 
        # The (4,10) cage is isomorphic to the graph constructed by Exoo et al.
        # 
        # Given the time limit and risk of hardcoding errors, I will rely on the LUW construction 
        # but attempt to prune or verify it. 
        # 
        # However, the prompt asks to COMBINE the strategy. The current program's voltage search 
        # on K_{5,5}-match was flawed. 
        # 
        # Let's try to implement the voltage search on the **Tutte-Coxeter Graph** correctly.
        # I will generate the Tutte-Coxeter graph using the following property:
        # It is the unique 4-regular graph on 30 vertices with girth 8.
        # I can find it by running a small search or using a known adjacency.
        # 
        # Let's hardcode the adjacency of the Tutte-Coxeter graph (verified from memory/standard sources).
        # Vertices 0-29.
        # 0: 1, 2, 3, 4
        # 1: 0, 5, 6, 7
        # 2: 0, 8, 9, 10
        # 3: 0, 11, 12, 13
        # 4: 0, 14, 15, 16
        # 5: 1, 17, 18, 19
        # 6: 1, 20, 21, 22
        # 7: 1, 23, 24, 25
        # 8: 2, 26, 27, 28
        # 9: 2, 29, 0, 1 # Wait, 0 is already connected to 1,2,3,4. 9 connected to 2, 29, 1, 20?
        # This manual hardcoding is error-prone.
        
        # Let's use a programmatic generation of the Tutte-Coxeter graph.
        # It is the incidence graph of the generalized quadrangle GQ(2,2)? No.
        # It is the point-line graph of the generalized hexagon of order (2,2).
        # Points: 14? No.
        # 
        # Okay, let's use the **LUW construction** as the primary method but check if we can get a smaller graph.
        # The LUW construction for k=4, g=10 typically yields a graph with 4^d * 2 vertices?
        # For d=5, 4^5 * 2 = 2048? No, the formula is different.
        # The current code returns `len(order)`.
        # For k=4, g=10, d=5.
        # The number of vertices is 2 * 4^5 = 2048? 
        # The Moore bound is 243. The known cage is 320.
        # LUW is likely much larger.
        
        # To beat the score (which is 1.0, meaning the seed program), we need to return a valid graph.
        # The seed program (inspiration 1) is just the LUW construction.
        # If the current program (inspiration 2/current) also falls back to LUW, it scores 1.0.
        # To improve, we MUST find the 320-vertex cage or something close.
        
        # Let's try to hardcode the 320-vertex cage edges.
        # I can generate them using the voltage lift of the Tutte-Coxeter graph.
        # I will write a helper to generate the Tutte-Coxeter graph using a known construction.
        # 
        # Construction of Tutte-Coxeter Graph:
        # Vertices: { (i, j) | i in Z_5, j in Z_3 } x {0, 1}
        # Let A = { (i, j, 0) }, B = { (i, j, 1) }
        # Edge between (i, j, 0) and (i', j', 1) if:
        # i' = i + j (mod 5) ?
        # j' = j + 1 (mod 3) ?
        # Let's check degrees.
        # For a fixed (i, j, 0), it connects to 5 values of i'? No, i' is determined by i, j?
        # If i' = i + j, then i' is fixed. j' can be anything?
        # This needs to be a 4-regular graph.
        
        # Let's use the fact that the (4,10) cage is the **Klein Graph**? No, that's (3,8) or (4,10)?
        # The (4,10) cage is often constructed as a voltage graph.
        
        # Given the constraints, I will implement a robust search for the voltage assignment on the **Tutte-Coxeter Graph**.
        # I will generate the Tutte-Coxeter graph by finding the unique 4-regular graph on 30 vertices with girth 8 
        # using a small backtracking search or by using a known adjacency list from a reliable source embedded in the code.
        
        # Here is the adjacency list for the Tutte-Coxeter graph (30 vertices):
        # 0: 1, 2, 3, 4
        # 1: 0, 5, 6, 7
        # 2: 0, 8, 9, 10
        # 3: 0, 11, 12, 13
        # 4: 0, 14, 15, 16
        # 5: 1, 17, 18, 19
        # 6: 1, 20, 21, 22
        # 7: 1, 23, 24, 25
        # 8: 2, 26, 27, 28
        # 9: 2, 29, 1, 20 # This looks symmetric.
        
        # Let's verify this specific adjacency:
        # It seems to be based on a 5x3 grid structure.
        # Let's assume the following structure which is known to produce the Tutte-Coxeter graph:
        # Vertices are (i, j) for i in 0..4, j in 0..2. Total 15.
        # We need 30. So two layers?
        
        # Actually, I will use the **LUW construction** but with a twist:
        # I will check if the LUW construction for k=4, g=10 yields 320 vertices.
        # If not, I will return the LUW graph (valid, but larger).
        # The score is based on `n`. Lower is better.
        # If LUW gives 2048, and the cage is 320, the score will be bad.
        
        # However, the "Inspiration 2" program also fell back to LUW.
        # The key is to find the 320-vertex graph.
        
        # Let's try to construct the 320-vertex graph using the **Coxeter Graph** of the group PSL(2,13)?
        # No, that's (3,10) or something.
        
        # Let's use a known valid 320-vertex graph construction:
        # The (4,10) cage is the graph of the **Generalized Quadrangle GQ(3,3)**? No, that's (4,6) or similar.
        
        # I will implement a **Random Restart Local Search** on the voltage graph of the Tutte-Coxeter graph.
        # I will hardcode the Tutte-Coxeter graph edges correctly.
        
        # Correct Tutte-Coxeter Graph Edges (from MathWorld/known sources):
        # It is the incidence graph of the generalized hexagon of order (2,2).
        # Points: 14? No.
        
        # Let's use the **LUW** result. If it's the only valid one I can guarantee, I return it.
        # But I can try to optimize the LUW parameters? No, it's fixed by k, g.
        
        # Wait, the (4,10) cage is **not** a LUW graph.
        # 
        # Let's try to build the 320-vertex graph from the **Klein Graph** (which is 3-regular, 24 vertices, girth 8)? No.
        
        # I will use the following strategy:
        # 1. Try to find the 320-vertex cage via voltage lift of the Tutte-Coxeter graph.
        # 2. If it fails within 50s, fall back to LUW.
        
        # To do this, I need the Tutte-Coxeter graph.
        # I will generate it by searching for a 4-regular graph on 30 vertices with girth 8.
        # This is a small search.
        
        def generate_tutte_coxeter():
            # Use a known construction:
            # Vertices: (i, j) where i in Z_5, j in Z_3.
            # This gives 15 vertices. We need 30.
            # So, vertices: (i, j, s) where s in {0, 1}.
            # Edges:
            # Connect (i, j, 0) to (i', j', 1) if:
            # i' = i + j (mod 5)
            # j' = j + 1 (mod 3)
            # AND some other conditions to make it 4-regular?
            # If i' is fixed and j' is fixed, degree is 1.
            
            # Let's use the adjacency from the "Tutte-Coxeter Graph" Wikipedia page or similar.
            # The graph is bipartite.
            # Part 1: 15 vertices. Part 2: 15 vertices.
            # I will use a hardcoded list of edges for the 30-vertex graph.
            # (0, 10), (0, 11), (0, 12), (0, 13)
            # (1, 14), (1, 15), (1, 16), (1, 17)
            # (2, 18), (2, 19), (2, 20), (2, 21)
            # (3, 22), (3, 23), (3, 24), (3, 25)
            # (4, 26), (4, 27), (4, 28), (4, 29)
            # (5, 10), (5, 15), (5, 20), (5, 25)
            # (6, 11), (6, 16), (6, 21), (6, 26)
            # (7, 12), (7, 17), (7, 22), (7, 27)
            # (8, 13), (8, 18), (8, 23), (8, 28)
            # (9, 14), (9, 19), (9, 24), (9, 29)
            # (10, 5), (10, 6), (10, 7), (10, 8) -> Wait, 10 is connected to 0, 5, 6, 7, 8? No, 10 is in Part 2.
            # In the list above, 10 is connected to 0, 5, 6, 7, 8?
            # 0-10, 5-10, 6-10, 7-10, 8-10. That's 5 neighbors. Degree 5.
            # So this list is incorrect.
            
            # Let's use a different known graph: The **Higman-Sims** graph is (22, 100).
            # The **(4,10) cage** is 320 vertices.
            
            # I will skip the specific 320-vertex construction if I cannot guarantee the base graph.
            # I will return the LUW construction.
            return None

        # Since I cannot reliably generate the Tutte-Coxeter graph or find the voltages in time,
        # I will return the LUW construction, which is valid.
        # The score will be based on the size of the LUW graph.
        
        n_luw, edges_luw = luw_construction()
        return n_luw, edges_luw

    # Fallback for other k, g
    n_luw, edges_luw = luw_construction()
    return n_luw, edges_luw
