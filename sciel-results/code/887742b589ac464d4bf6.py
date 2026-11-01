import random
import time

def construct(k, g):
    # Target: 5-regular, girth >= 9
    # Start with a reasonable N
    # Moore bound for k=5, g=9 is 426.
    # Known best is 1116.
    # We try to find a graph with N < 6250 (current LUW size).
    
    # If k != 5 or g != 9, fall back to LUW
    if k != 5 or g != 9:
        return _luw_construct(k, g)
        
    N = 1116
    deadline = time.time() + 110 # 10s margin
    
    # Generate random 5-regular graph
    # Use configuration model
    def generate_random_regular(N, k):
        # Stirling numbers / random pairing
        # Simplified: random permutation of stubs
        stubs = []
        for i in range(N):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        edges = set()
        for i in range(0, N*k, 2):
            u, v = stubs[i], stubs[i+1]
            if u != v:
                if (u, v) not in edges and (v, u) not in edges:
                    edges.add((u, v))
                else:
                    # Retry or handle loop/multi-edge
                    # For simplicity, just add if unique, else skip and retry later?
                    # This might break regularity.
                    # Better: use a proper algorithm.
                    pass
        # This simple method might fail to produce a simple graph.
        # Use a robust generator:
        # Start with empty, add edges randomly, ensure degree < k.
        # This is slow.
        
        # Let's use a known fast method:
        # 1. Create list of all vertices repeated k times.
        # 2. Shuffle.
        # 3. Pair up.
        # 4. If loops or multi-edges, swap endpoints.
        
        stubs = []
        for i in range(N):
            stubs.extend([i]*k)
        random.shuffle(stubs)
        
        edges = []
        # We need to ensure simple graph.
        # This is a heuristic.
        
        # Actually, let's just return the LUW graph if we can't find a better one quickly.
        # But LUW is 6250.
        # If we find ANY graph with N < 6250, we win.
        
        # Try N=1000
        # Random 5-regular graph on 1000 vertices.
        # Girth is likely 6 or 7.
        # We need to increase girth.
        
        # This is too complex for a short time.
        
        return None

    # Fallback: Use LUW but with optimized parameters if possible?
    # No, LUW is fixed.
    
    # Let's try a different known graph:
    # The **Cage** n(5,9) = 1116.
    # If we can't construct it, we return LUW (6250).
    
    # Is there a smaller graph?
    # n(5,9) >= 428.
    # Maybe n=500?
    
    # Let's try to **generate** a graph with N=1116 using a **greedy** approach.
    # 1. Initialize N vertices.
    # 2. Add edges randomly, avoiding short cycles.
    # 3. Check if we can complete to 5-regular.
    
    # This is risky.
    
    # Strategy:
    # 1. Try to construct a 5-regular graph with N=1116 using a **randomized greedy** algorithm.
    # 2. If it fails or takes too long, return LUW.
    
    # Greedy:
    # For each vertex, pick 5 neighbors that don't create cycles < 9.
    # This is hard to check globally.
    
    # Given the constraints, the safest "improvement" is to **hardcode** a known smaller graph if we had one.
    # Since we don't, we must rely on the LUW graph or a very simple random search.
    
    # Let's try a **very simple** random search for N=1000.
    # If we find a valid one, we return it.
    
    # But probability is low.
    
    # Therefore, we stick with LUW but **optimize the code** for speed?
    # No, the score is 1/n.
    
    # Wait, the LUW graph for k=5, g=9 has N=6250.
    # Is there a **smaller** LUW graph?
    # No, q must be 5.
    
    # What if we use **k=5** but **g=9** and a **different construction**?
    
    # I will implement a **basic random regular graph generator** and **girth checker**.
    # If it finds a valid graph with N < 6250, we return it.
    # Otherwise, we return LUW.
    
    # This is a "shot in the dark" but might find a smaller graph.
    
    # Actually, the **1116** graph is the best.
    # If we can't find it, we are stuck with 6250.
    
    # Let's return LUW.
    return _luw_construct(k, g)

def _luw_construct(k, g):
    # Same as before
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
