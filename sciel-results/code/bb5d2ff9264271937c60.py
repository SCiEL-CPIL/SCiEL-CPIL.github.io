import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin

    N = 256

    # Build conflict sets
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1

    # conflict[a] = set of b in [1,N] such that a+b is a perfect square
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                conflict[a].add(b)

    # Numbers where 2a is a perfect square are forbidden (self-conflict)
    forbidden = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            forbidden.add(a)

    valid_numbers = [x for x in range(1, N + 1) if x not in forbidden]
    M = len(valid_numbers)
    num_to_idx = {x: i for i, x in enumerate(valid_numbers)}
    idx_to_num = valid_numbers

    # Build adjacency as bitmasks for the COMPLEMENT graph (G')
    # Edge in G' if a and b are NOT in conflict (and a != b)
    adj_comp = [0] * M
    for i, a in enumerate(valid_numbers):
        mask = (1 << M) - 1
        mask &= ~(1 << i) # Remove self
        for b in conflict[a]:
            if b in num_to_idx:
                j = num_to_idx[b]
                mask &= ~(1 << j)
        adj_comp[i] = mask

    best_size = 0
    best_clique = []
    
    def time_exceeded():
        return time.time() > deadline

    def color_and_order(P):
        uncolored = P
        colors = [0] * M
        color = 0
        
        while uncolored:
            color += 1
            in_class = 0
            for i in range(M):
                if not (uncolored >> i) & 1:
                    continue
                if (adj_comp[i] & in_class) == 0:
                    in_class |= (1 << i)
                    colors[i] = color
                    uncolored &= ~(1 << i)
                    
        vertices = [i for i in range(M) if (P >> i) & 1]
        vertices.sort(key=lambda x: colors[x], reverse=True)
        
        return colors, vertices

    def expand(P, size, clique):
        nonlocal best_size, best_clique
        
        if time_exceeded():
            return
            
        if P == 0:
            if size > best_size:
                best_size = size
                best_clique = clique[:]
            return
            
        colors, vertices = color_and_order(P)
        
        if not vertices:
            return
            
        max_color = colors[vertices[0]]
        if size + max_color <= best_size:
            return
            
        P_temp = P
        
        for v in vertices:
            if time_exceeded():
                return
                
            if size + colors[v] <= best_size:
                break
                
            new_clique = clique + [v]
            P_next = P_temp & adj_comp[v]
            P_next &= ~(1 << v)
            
            expand(P_next, size + 1, new_clique)
            
            P_temp &= ~(1 << v)
            
    initial_P = (1 << M) - 1
    expand(initial_P, 0, [])
    
    if best_clique:
        result = [idx_to_num[i] for i in best_clique]
    else:
        result = [valid_numbers[0]]
        
    return sorted(result)
