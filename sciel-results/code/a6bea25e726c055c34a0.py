import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    def check_girth(n, edges, min_girth):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Verify degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check for simple graph properties (no self loops, no multi edges)
        # The construction ensures this, but just in case.
        # We assume edges are unique and no loops.
        
        # BFS on non-backtracking states
        # State: (prev, curr)
        # We start from every directed edge (u, v)
        # We want to find a closed walk of length < min_girth
        
        for u0 in range(n):
            for u1 in adj[u0]:
                # Start state (u0, u1)
                # We are looking for a path (u0, u1) -> ... -> (u_last, u0)
                # where u_last != u1
                # The length of the path in state graph is L.
                # The cycle length in G is L + 1.
                # We need L + 1 < min_girth => L < min_girth - 1.
                
                q = deque()
                visited = set()
                # State: (prev, curr, length)
                q.append((u0, u1, 0))
                visited.add((u0, u1))
                
                while q:
                    pu, cu, l = q.popleft()
                    
                    if l >= min_girth - 1:
                        continue
                        
                    for nu in adj[cu]:
                        if nu == pu:
                            continue
                            
                        # New state is (cu, nu)
                        # Check if it closes a cycle
                        if nu == u0 and cu != u1:
                            # Cycle found of length l + 1
                            # Since l < min_girth - 1, length < min_girth
                            return False
                            
                        if (cu, nu) not in visited:
                            visited.add((cu, nu))
                            q.append((cu, nu, l + 1))
                            
        return True

    def luw_construct(d, q):
        # GF(q) operations
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y) % q
            sub = lambda x,y: (x-y) % q
            mul = lambda x,y: x*y % q
        
        # Rule for LUW
        rule = {}
        # Initialize rules for c=2 to d
        # The seed had specific initialization. Let's replicate.
        # Seed:
        # rule = {2: (1,1), 3: (2,1), 4: (1,2)}
        # for c in range(5, d+1): ...
        
        # To be safe, let's generate rules for 2 to d consistently.
        # The pattern in seed:
        # c=2: (1,1)
        # c=3: (2,1)
        # c=4: (1,2)
        # c=5: r=(0), c0=5, [(1,3),(4,1),(5,1),(1,4)][0] = (1,3)
        # c=6: r=(1), c0=5, [(1,3),(4,1),(5,1),(1,4)][1] = (4,1)
        # c=7: r=(2), c0=5, [(1,3),(4,1),(5,1),(1,4)][2] = (5,1)
        # c=8: r=(3), c0=5, [(1,3),(4,1),(5,1),(1,4)][3] = (1,4)
        
        # Let's implement this general rule.
        for c in range(2, d+1):
            if c == 2:
                rule[c] = (1, 1)
            elif c == 3:
                rule[c] = (2, 1)
            elif c == 4:
                rule[c] = (1, 2)
            else:
                r = (c-5) % 4
                c0 = c - r
                rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        
        start = (0, (None,) + (0,)*d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for i, vertex in enumerate(order):
            if time.time() > deadline:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        return n, edges

    best_n = float('inf')
    best_edges = []
    
    # Try d from 2 to 6
    # d=2: n=250
    # d=3: n=1250
    # d=4: n=6250
    # d=5: n=31250
    # d=6: n=156250
    
    for d in range(2, 7):
        if time.time() > deadline:
            break
            
        try:
            n, edges = luw_construct(d, k)
            if n < best_n:
                # Check girth
                if check_girth(n, edges, g):
                    best_n = n
                    best_edges = edges
                    # Since n increases with d, and we want min n, 
                    # we can break here if we found a valid one with small d?
                    # Yes, d=2 is smallest. If d=2 works, it's the best.
                    # But we must ensure it's valid.
                    return n, edges
        except Exception as e:
            print(f"Error for d={d}: {e}")
            continue
            
    # If no small d worked, return the largest constructed or seed's
    if best_n == float('inf'):
        # Fallback to seed's logic (d=5 for g=9)
        d = g - 5 if (g - 5) % 2 == 1 else g - 4
        if d < 3: d = 3
        n, edges = luw_construct(d, k)
        return n, edges

    return best_n, best_edges
