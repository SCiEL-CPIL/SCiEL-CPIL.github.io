import time
import random
import numpy as np

_DEADLINE = 110.0

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
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

    # Target: Find a 4-regular graph with girth >= 10 on 320 vertices
    if k == 4 and g == 10:
        n = 320
        t0 = time.time()
        
        # Generate random 4-regular graph
        def gen_random_4reg(n):
            while True:
                stubs = []
                for i in range(n):
                    for _ in range(4):
                        stubs.append(i)
                random.shuffle(stubs)
                edges = []
                valid = True
                seen = set()
                for i in range(0, len(stubs), 2):
                    u = stubs[i]
                    v = stubs[i+1]
                    if u == v:
                        valid = False
                        break
                    key = (min(u, v), max(u, v))
                    if key in seen:
                        valid = False
                        break
                    seen.add(key)
                    edges.append(key)
                if valid:
                    return edges

        edges = gen_random_4reg(n)
        
        # Build adjacency matrix
        A = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
            
        # Cost function: Sum of traces of A^k for k=3..9
        # If this is 0, no cycles of length 3..9 exist.
        def calc_cost(A):
            cost = 0
            Ak = A.copy()
            for k in range(3, 10):
                Ak = Ak @ A
                cost += np.trace(Ak)
            return cost

        current_cost = calc_cost(A)
        
        # Simulated Annealing
        T = 1.0
        cooling = 0.999
        best_cost = current_cost
        best_edges = edges[:]
        
        # Precompute edge list for fast access
        # We will modify A directly and keep a list of edges for reconstruction if needed
        # But for SA, we just need to update A and check cost.
        
        while time.time() - t0 < _DEADLINE:
            if current_cost == 0:
                # Found a valid graph!
                # Reconstruct edges from A
                final_edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if A[i, j] == 1:
                            final_edges.append((i, j))
                return n, final_edges

            # Pick a random move
            # Select two edges to swap
            # To ensure we don't pick adjacent edges or same edge, we can just pick random vertices
            # A simpler move: Pick a random edge (u,v) and a random edge (x,y) from the list
            # But maintaining the list while swapping is tricky.
            # Alternative: Pick 4 distinct vertices u,v,x,y.
            # Check if (u,v) and (x,y) are edges.
            # Swap to (u,x) and (v,y).
            
            # Efficient move generation:
            # Pick random u, v, x, y
            # If A[u,v] and A[x,y] are 1, and u,v,x,y distinct, and (u,x) and (v,y) are 0
            # Then swap is valid.
            
            max_attempts = 100
            moved = False
            for _ in range(max_attempts):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                
                if u == v or x == y: continue
                if u in (x, y) or v in (x, y): continue
                if u == x or u == y or v == x or v == y: continue
                
                if A[u, v] == 0 or A[x, y] == 0: continue
                if A[u, x] == 1 or A[v, y] == 1: continue
                # Also check (u,y) and (v,x) for the other swap option?
                # We'll just try one swap type: (u,v),(x,y) -> (u,x),(v,y)
                
                # Perform swap
                A[u, v] = 0; A[v, u] = 0
                A[x, y] = 0; A[y, x] = 0
                A[u, x] = 1; A[x, u] = 1
                A[v, y] = 1; A[y, v] = 1
                
                new_cost = calc_cost(A)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_edges = []
                        for i in range(n):
                            for j in range(i+1, n):
                                if A[i, j] == 1:
                                    best_edges.append((i, j))
                    moved = True
                    break
                else:
                    # Revert
                    A[u, x] = 0; A[x, u] = 0
                    A[v, y] = 0; A[y, v] = 0
                    A[u, v] = 1; A[v, u] = 1
                    A[x, y] = 1; A[y, x] = 1
                    # SA acceptance
                    if random.random() < np.exp(-(new_cost - current_cost) / T):
                        # Accept
                        A[u, v] = 0; A[v, u] = 0
                        A[x, y] = 0; A[y, x] = 0
                        A[u, x] = 1; A[x, u] = 1
                        A[v, y] = 1; A[y, v] = 1
                        current_cost = new_cost
                        moved = True
                        break
            
            if not moved:
                # Try to restart or change strategy?
                pass
                
            T *= cooling
            if T < 0.01:
                T = 0.01

        # If loop finishes without finding cost 0, return best found?
        # But best found might have cost > 0 (invalid).
        # We MUST return a valid graph.
        # If we didn't find a valid graph on 320 vertices, we should fall back to LUW.
        
    return luw_construct(k, g)
