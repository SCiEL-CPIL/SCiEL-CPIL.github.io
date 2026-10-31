import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145
    
    if k != 3:
        return _luw_construct(k, g)
        
    if g != 16:
        # Fallback for other girths if needed, but task is specific
        return _luw_construct(k, g)

    # Base graph K_{3,3}
    # Vertices 0,1,2 (Left) and 3,4,5 (Right)
    # Edges (i, 3+j) for i in 0..2, j in 0..2
    # We model voltages a[i][j] for edge i -> 3+j
    # Voltage 3+j -> i is -a[i][j]
    
    # Enumerate closed walks of length < g in K_{3,3}
    # Walk is a sequence of vertices.
    # We need voltage sums.
    
    # Let's generate all walks of length L from 4 to g-1
    # Since K_{3,3} is small, we can do DFS.
    
    def get_walks(max_len):
        walks = []
        # Start at 0 (L0)
        # State: (current_node, path)
        # We need to close back to 0.
        # Actually, voltage is sum over edges.
        # We can just generate all sequences of length L that start and end at 0,
        # with adjacent nodes in K_{3,3}.
        
        # K_{3,3} adj:
        # 0: 3,4,5
        # 1: 3,4,5
        # 2: 3,4,5
        # 3: 0,1,2
        # 4: 0,1,2
        # 5: 0,1,2
        
        def dfs(curr, path, length):
            if length == max_len:
                if curr == 0:
                    # Convert path to voltage coefficients
                    # Path is [v0, v1, ..., vL]
                    # Edges are (v0,v1), (v1,v2), ...
                    # Voltage of edge (u,v):
                    # if u in {0,1,2} and v in {3,4,5}: a[u][v-3]
                    # if u in {3,4,5} and v in {0,1,2}: -a[v][u-3]
                    coeff = [0]*9
                    for i in range(len(path)-1):
                        u = path[i]
                        v = path[i+1]
                        if u < 3 and v >= 3:
                            idx = u*3 + (v-3)
                            coeff[idx] += 1
                        elif u >= 3 and v < 3:
                            idx = v*3 + (u-3)
                            coeff[idx] -= 1
                    # We only care if sum is 0 mod m
                    # We can store the coeff tuple
                    walks.append(tuple(coeff))
                return
            if length >= max_len:
                return
            # Neighbors
            if curr < 3:
                nexts = [3,4,5]
            else:
                nexts = [0,1,2]
            for nxt in nexts:
                dfs(nxt, path + [nxt], length + 1)
        
        # We want walks of specific lengths
        # But we can just generate all up to g-1 and filter
        # To avoid duplicates or too many, we can generate per length
        for L in range(4, g, 2): # Bipartite, only even
            dfs(0, [0], 0)
            
        # Remove duplicates?
        # Different walks can have same coeff?
        # Yes. We only need to check if ANY walk has sum 0.
        # So we just need the set of coeff vectors.
        return list(set(walks))

    # Optimization: 
    # The number of walks grows. 
    # L=4: 9 cycles * 4 rotations? 
    # Let's just run it.
    
    all_walks = get_walks(g)
    
    # For each walk, we have a coeff vector c.
    # We need sum(c[i] * a[i]) != 0 mod m.
    
    # Search for a[i] in Z_m
    # Fix a[0]=0, a[1]=1, a[2]=2 to reduce symmetry
    
    def check_assignment(a, m, walks):
        for c in walks:
            s = 0
            for i in range(9):
                s += c[i] * a[i]
            if s % m == 0:
                return False
        return True

    found = False
    best_m = None
    best_a = None
    
    # Try m around 156
    for m in [156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170]:
        if time.time() > deadline - 5:
            break
        # Random search
        random.seed(m)
        for _ in range(5000):
            a = [random.randint(0, m-1) for _ in range(9)]
            a[0] = 0
            a[1] = 1
            a[2] = 2
            if check_assignment(a, m, all_walks):
                found = True
                best_m = m
                best_a = a
                break
        if found:
            break
            
    if not found:
        # Try larger m or more attempts
        # Or just use the largest m tried
        # Actually, if m=156 fails, try m=156 with more attempts?
        # Or try m=157..
        # Let's try a more directed search for m=156
        pass

    if not found:
        # Fallback: Use LUW or random lift of larger size
        # But we want to return a valid graph.
        # Let's try m=200
        for m in range(171, 250):
            if time.time() > deadline - 5:
                break
            random.seed(m)
            for _ in range(1000):
                a = [random.randint(0, m-1) for _ in range(9)]
                a[0] = 0
                a[1] = 1
                a[2] = 2
                if check_assignment(a, m, all_walks):
                    found = True
                    best_m = m
                    best_a = a
                    break
            if found:
                break

    if not found:
        # Absolute fallback: return a large random graph that is likely valid?
        # No, must be valid.
        # Use LUW construction which is guaranteed to have large girth.
        return _luw_construct(3, 16)

    # Build the graph
    m = best_m
    a = best_a
    n = 6 * m
    edges = []
    for i in range(3):
        for j in range(3):
            for fiber in range(m):
                u = i * m + fiber
                v = (3+j) * m + (fiber + a[i*3+j]) % m
                edges.append((u, v))
    return n, edges

def _luw_construct(k, g):
    # Standard LUW fallback
    if k != 3:
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
    else:
        q = 3
        d = 11 # g-5
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
