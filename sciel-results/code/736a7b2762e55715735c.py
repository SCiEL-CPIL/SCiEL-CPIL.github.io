import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 256
    S = set(range(1, n + 1))
    
    # Generate all violating triples (a, b, c) with a < b <= c and a | (b+c)
    edges = []
    for a in range(1, n + 1):
        # b > a, c >= b
        # b+c is a multiple of a
        # min b+c = (a+1) + (a+1) = 2a+2
        # max b+c = n + n = 2n
        # multiples of a in [2a+2, 2n]
        start_mult = (2 * a + 2 + a - 1) // a + 1  # ceil((2a+2)/a)
        # Actually simpler: iterate k such that k*a is in range
        # k*a >= 2a+2 => k >= 2 + 2/a
        # k*a <= 2n => k <= 2n/a
        
        min_k = 2
        # ensure k*a > 2a => k > 2
        if 2 * a <= 2 * a + 2:
             min_k = 3 if a < 2 else 3 # For a=1, 2*1=2, min sum 4. k=4?
             # Let's just iterate b and c? No, too slow.
             # Iterate k.
        
        # k must satisfy:
        # 1. k*a >= 2(a+1)
        # 2. k*a <= 2n
        # 3. There exist b,c in S such that b+c=k*a, a < b <= c.
        #    b >= a+1, c <= n.
        #    b <= k*a - b => 2b <= k*a => b <= k*a/2
        #    c = k*a - b >= b => b <= k*a/2
        #    c <= n => k*a - b <= n => b >= k*a - n
        #    b >= a+1
        #    So b in [max(a+1, k*a-n), floor(k*a/2)]
        
        min_k = (2 * (a + 1) + a - 1) // a + 1
        max_k = (2 * n) // a
        
        for k in range(min_k, max_k + 1):
            s = k * a
            # b range
            b_min = max(a + 1, s - n)
            b_max = s // 2
            
            if b_min <= b_max:
                # Add edges for all valid b
                for b in range(b_min, b_max + 1):
                    c = s - b
                    # b <= c is guaranteed by b <= s//2
                    # a < b is guaranteed by b >= a+1
                    # c <= n is guaranteed by b >= s-n
                    # b <= n? b <= s/2 <= n. Yes.
                    edges.append((a, b, c))

    # Map vertex to list of edge indices
    vertex_to_edges = {v: [] for v in range(1, n + 1)}
    for i, e in enumerate(edges):
        for v in e:
            vertex_to_edges[v].append(i)
            
    # Degree of vertex in active graph
    degree = [0] * (n + 1)
    active_edges = [True] * len(edges)
    
    # Initialize degrees
    for i, e in enumerate(edges):
        for v in e:
            degree[v] += 1
            
    # To speed up finding max degree, we could use a heap, but n is small (256).
    # Just scan for max degree each step.
    # Number of steps <= n.
    # Scanning 256 items is trivial.
    
    while True:
        if time.time() > deadline:
            break
            
        # Find vertex with max degree
        max_deg = -1
        max_v = -1
        for v in S:
            if degree[v] > max_deg:
                max_deg = degree[v]
                max_v = v
                
        if max_deg <= 0:
            break
            
        # Remove max_v
        S.remove(max_v)
        
        # Update degrees for all edges containing max_v
        for i in vertex_to_edges[max_v]:
            if active_edges[i]:
                active_edges[i] = False
                e = edges[i]
                # Decrease degree of other vertices
                for v in e:
                    if v != max_v:
                        degree[v] -= 1
                        
    return list(S)
