import numpy as np
from scipy.sparse import csr_matrix
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct():
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

    # Try to find a smaller graph using random search
    # Moore bound for k=5, g=9 is 426
    # We try n from 430 upwards
    
    best_n, best_edges = luw_construct()
    
    for n in range(430, 1200, 10):
        if time.time() > deadline:
            break
            
        # Try to generate a random k-regular graph with girth >= g
        found = False
        attempts = 0
        max_attempts = 50  # Limit attempts per n
        
        while attempts < max_attempts and not found and time.time() < deadline:
            attempts += 1
            # Generate random k-regular graph
            # Using configuration model
            stubs = [i for i in range(n) for _ in range(k)]
            np.random.shuffle(stubs)
            
            # Pair stubs
            edges_set = set()
            valid = True
            adj = {i: set() for i in range(n)}
            
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                edges_set.add((min(u, v), max(u, v)))
                adj[u].add(v)
                adj[v].add(u)
            
            if not valid:
                continue
                
            # Check girth
            # Build sparse adjacency matrix
            row = []
            col = []
            for u, v in edges_set:
                row.append(u)
                col.append(v)
                row.append(v)
                col.append(u)
                
            A = csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
            
            # Check for cycles of length 3 to 8
            # Triangles: diag(A^2) == 0
            A2 = A @ A
            if np.any(A2.diagonal() > 0):
                continue
                
            # Check for cycles of length L: for all edges (u,v), (A^{L-2})[u,v] == 0
            # We need A^1 to A^7
            # A^1 is A itself. For L=3, we check A^1? No, L=3 means path of length 1 between adjacent nodes?
            # Cycle of length L means a closed walk of length L.
            # A cycle of length 3 (triangle) u-v-w-u. Path from u to v of length 2 is u-w-v.
            # So for edge (u,v), if (A^2)[u,v] > 0, there is a triangle.
            # Wait, A2[u,v] counts paths of length 2 from u to v.
            # If u and v are adjacent, and A2[u,v] > 0, then there is a path u-x-v of length 2, forming a triangle.
            # So checking diag(A2) is for triangles?
            # No, diag(A2)[u] counts common neighbors of u. If > 0, there is a triangle u-x-y-u? No, u-x and u-y and x-y.
            # Yes, diag(A2) > 0 implies triangle.
            
            # For 4-cycles: u-v-w-x-u. Path from u to v of length 3 is u-x-w-v.
            # So if (A^3)[u,v] > 0 for an edge (u,v), there is a 4-cycle.
            
            # So we need to check:
            # L=3: A^1? No, path length L-2.
            # Triangle (L=3): path length 1. A^1[u,v] is 1 for all edges. This doesn't help.
            # Wait, a triangle is a cycle of length 3.
            # If we have edge (u,v), and a path of length 2 from u to v, that's a triangle.
            # So check A^2[u,v] for all edges (u,v).
            # But we already checked diag(A2). Is that sufficient?
            # No. diag(A2) checks if u has a common neighbor with itself? No, common neighbors of u.
            # If u and v are adjacent, and they share a neighbor w, then u-v-w-u is a triangle.
            # This is detected by A2[u,v] > 0.
            # But diag(A2)[u] > 0 means u has a common neighbor with itself? No.
            # A2[u,u] = number of length 2 walks from u to u. This is the number of common neighbors of u.
            # If u and v are adjacent, and w is a common neighbor, then A2[u,v] >= 1.
            # So we should check A2[u,v] for all edges (u,v).
            
            # Let's re-evaluate.
            # Cycle of length L.
            # L=3: Triangle. Edge (u,v). Path u-w-v. Length 2. A2[u,v] > 0.
            # L=4: Square. Edge (u,v). Path u-w-x-v. Length 3. A3[u,v] > 0.
            # L=5: Pentagon. Edge (u,v). Path u-w-x-y-v. Length 4. A4[u,v] > 0.
            # ...
            # L=8: Octagon. Edge (u,v). Path u-...-v. Length 7. A7[u,v] > 0.
            
            # So we need to check A2, A3, ..., A7 on the edges.
            
            ok = True
            # We already have A2
            for u, v in edges_set:
                if A2[u, v] > 0:
                    ok = False
                    break
            if not ok:
                continue
                
            A3 = A2 @ A
            for u, v in edges_set:
                if A3[u, v] > 0:
                    ok = False
                    break
            if not ok:
                continue
                
            A4 = A3 @ A
            for u, v in edges_set:
                if A4[u, v] > 0:
                    ok = False
                    break
            if not ok:
                continue
                
            A5 = A4 @ A
            for u, v in edges_set:
                if A5[u, v] > 0:
                    ok = False
                    break
            if not ok:
                continue
                
            A6 = A5 @ A
            for u, v in edges_set:
                if A6[u, v] > 0:
                    ok = False
                    break
            if not ok:
                continue
                
            A7 = A6 @ A
            for u, v in edges_set:
                if A7[u, v] > 0:
                    ok = False
                    break
            if not ok:
                continue
                
            if ok:
                found = True
                best_n = n
                best_edges = list(edges_set)
                break
                
    return best_n, best_edges
