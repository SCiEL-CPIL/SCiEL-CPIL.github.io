import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    N = 256
    # Precompute squares
    squares = set()
    k = 2
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
        
    # Build adjacency bitsets
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if i + j in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    all_vertices = (1 << (N + 1)) - 2
    
    # Greedy initial solution
    def greedy_initial():
        S = []
        P = all_vertices
        while P:
            best_v = -1
            min_deg = N + 1
            p = P
            while p:
                lsb = p & -p
                v = lsb.bit_length() - 1
                deg = (adj[v] & P).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                p -= lsb
            S.append(best_v)
            P &= ~adj[best_v]
            P &= ~(1 << best_v)
        return S
        
    S_init = greedy_initial()
    best_size = len(S_init)
    best_set = S_init.copy()
    
    # Calculate initial candidate set P
    P_init = all_vertices
    for v in S_init:
        P_init &= ~adj[v]
        P_init &= ~(1 << v)
        
    def color_graph(P):
        U = P
        color = 0
        order = []
        while U:
            color += 1
            V = U
            while V:
                lsb = V & -V
                v = lsb.bit_length() - 1
                order.append((v, color))
                V &= ~adj[v]
                V &= ~(1 << v)
                U &= ~(1 << v)
        return color, order
        
    def search(P, S, size):
        nonlocal best_size, best_set
        if time.time() > deadline:
            return False
        
        if not P:
            if size > best_size:
                best_size = size
                best_set = S.copy()
            return True
            
        bound, order = color_graph(P)
        if size + bound <= best_size:
            return True
            
        # Select vertex with max color
        # order is sorted by color ascending, so last has max color
        v, _ = order[-1]
        
        # Branch 1: Include v
        new_P = P & ~adj[v] & ~(1 << v)
        S.append(v)
        if search(new_P, S, size + 1):
            return True
        S.pop()
        
        # Branch 2: Exclude v
        new_P = P & ~(1 << v)
        if search(new_P, S, size):
            return True
            
        return False
        
    search(P_init, S_init.copy(), best_size)
    
    return best_set
