import time
import random

def construct():
    """
    Returns a valid 2-coloring of K_n that is K_{2,2,2}-free in both colors.
    Attempts to find a coloring for n=30 via local search.
    Falls back to n=29 (Paley graph) if not found in time.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    def get_paley_29():
        q = 29
        squares = set()
        for x in range(1, q):
            squares.add((x * x) % q)
        colors = []
        for i in range(q):
            for j in range(i + 1, q):
                d = (j - i) % q
                if d in squares:
                    colors.append(1)
                else:
                    colors.append(0)
        return {'n': q, 'colors': colors}
    
    # Try to find n=30
    n = 30
    q = 29
    
    squares = set()
    for x in range(1, q):
        squares.add((x * x) % q)
    
    # Build Paley-29 adjacency
    adj0 = [0] * n
    adj1 = [0] * n
    
    for i in range(q):
        for j in range(i + 1, q):
            d = (j - i) % q
            if d in squares:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
            else:
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
    
    # Precompute neighbor lists for fast iteration
    def get_neighbors(adj, v):
        """Get list of neighbors of v in adjacency bitmask array"""
        result = []
        temp = adj[v]
        while temp:
            lsb = temp & -temp
            result.append(lsb.bit_length() - 1)
            temp ^= lsb
        return result
    
    # Count K222 containing a specific edge (u,v) in color c
    # Uses precomputed structure for speed
    def count_k222_edge(u, v, c, adj0, adj1):
        if c == 0:
            adj = adj0
        else:
            adj = adj1
            
        # S = N_c(u) ∩ N_c(v)
        S = adj[u] & adj[v]
        
        # Get neighbors of u and v in color c
        # a' ∈ N_c(v) \ {u}, b' ∈ N_c(u) \ {v}
        # For each pair (a', b') with (a',b') in color c:
        #   add C(|N_c(a') ∩ N_c(b') ∩ S|, 2)
        
        count = 0
        neighbors_v = get_neighbors(adj, v)
        neighbors_u = get_neighbors(adj, u)
        
        for a in neighbors_v:
            if a == u:
                continue
            for b in neighbors_u:
                if b == v or b == a:
                    continue
                # Check if (a, b) is in color c
                if (adj[a] >> b) & 1:
                    inter = adj[a] & adj[b] & S
                    size = inter.bit_count()
                    if size >= 2:
                        count += size * (size - 1) // 2
                        
        return count
    
    # Total cost = sum over all edges of count_k222_edge
    # Each K222 is counted 12 times (once per edge in the K222)
    def total_cost(adj0, adj1):
        cost = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj0[u] >> v) & 1:
                    cost += count_k222_edge(u, v, 0, adj0, adj1)
                else:
                    cost += count_k222_edge(u, v, 1, adj0, adj1)
        return cost
    
    # Try deterministic extensions for vertex 29
    # The idea: vertex 29 should connect to a subset of {0,...,28}
    # such that no monochromatic K222 is formed
    
    best_result = None
    best_cost = float('inf')
    
    # Try several deterministic patterns for vertex 29
    # Pattern: connect to vertices where some property holds
    patterns = []
    
    # Pattern 1: connect to all even-indexed vertices
    patterns.append(set(range(0, q, 2)))
    # Pattern 2: connect to all odd-indexed vertices  
    patterns.append(set(range(1, q, 2)))
    # Pattern 3: connect to vertices that are quadratic residues mod 29
    patterns.append(set(squares))
    # Pattern 4: connect to non-residues
    patterns.append(set(range(1, q)) - squares)
    # Pattern 5: first 15 vertices
    patterns.append(set(range(15)))
    # Pattern 6: last 15 vertices
    patterns.append(set(range(14, 29)))
    
    # Also try random patterns
    for _ in range(20):
        pattern = set(random.sample(range(q), random.randint(10, 20)))
        patterns.append(pattern)
    
    for pattern in patterns:
        if time.time() > deadline - 5:
            break
            
        test_adj0 = adj0[:]
        test_adj1 = adj1[:]
        
        for i in range(q):
            if i in pattern:
                test_adj0[29] |= (1 << i)
                test_adj0[i] |= (1 << 29)
            else:
                test_adj1[29] |= (1 << i)
                test_adj1[i] |= (1 << 29)
        
        cost = total_cost(test_adj0, test_adj1)
        
        if cost < best_cost:
            best_cost = cost
            best_result = (test_adj0[:], test_adj1[:])
            
            if cost == 0:
                # Found valid coloring!
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (test_adj0[i] >> j) & 1:
                            colors.append(0)
                        else:
                            colors.append(1)
                return {'n': n, 'colors': colors}
    
    # If no zero-cost found, start local search from best initial
    if best_result is None:
        # Fallback: random initialization
        for i in range(q):
            if random.random() < 0.5:
                adj0[29] |= (1 << i)
                adj0[i] |= (1 << 29)
            else:
                adj1[29] |= (1 << i)
                adj1[i] |= (1 << 29)
        current_adj0 = adj0[:]
        current_adj1 = adj1[:]
        current_cost = total_cost(current_adj0, current_adj1)
    else:
        current_adj0, current_adj1 = best_result
        current_cost = best_cost
    
    # Local search with simulated annealing
    T = 1.0
    cooling = 0.995
    min_T = 0.05
    
    # Precompute edge list for faster iteration
    edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    while time.time() < deadline:
        # Pick random edge
        u, v = random.choice(edges)
        
        # Current color
        if (current_adj0[u] >> v) & 1:
            c = 0
        else:
            c = 1
            
        # Compute delta cost
        old_count = count_k222_edge(u, v, c, current_adj0, current_adj1)
        
        # Flip the edge
        if c == 0:
            current_adj0[u] &= ~(1 << v)
            current_adj0[v] &= ~(1 << u)
            current_adj1[u] |= (1 << v)
            current_adj1[v] |= (1 << u)
        else:
            current_adj1[u] &= ~(1 << v)
            current_adj1[v] &= ~(1 << u)
            current_adj0[u] |= (1 << v)
            current_adj0[v] |= (1 << u)
            
        new_count = count_k222_edge(u, v, 1 - c, current_adj0, current_adj1)
        delta = new_count - old_count
        new_cost = current_cost + delta
        
        if new_cost < current_cost or random.random() < pow(2.718, -delta / T):
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_adj0 = current_adj0[:]
                best_adj1 = current_adj1[:]
                
                if best_cost == 0:
                    colors = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (best_adj0[i] >> j) & 1:
                                colors.append(0)
                            else:
                                colors.append(1)
                    return {'n': n, 'colors': colors}
        else:
            # Revert
            if c == 0:
                current_adj0[u] |= (1 << v)
                current_adj0[v] |= (1 << u)
                current_adj1[u] &= ~(1 << v)
                current_adj1[v] &= ~(1 << u)
            else:
                current_adj1[u] |= (1 << v)
                current_adj1[v] |= (1 << u)
                current_adj0[u] &= ~(1 << v)
                current_adj0[v] &= ~(1 << u)
        
        T *= cooling
        if T < min_T:
            T = min_T
    
    # Return best found or fallback
    if best_cost == 0:
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if (best_adj0[i] >> j) & 1:
                    colors.append(0)
                else:
                    colors.append(1)
        return {'n': n, 'colors': colors}
    else:
        return get_paley_29()
