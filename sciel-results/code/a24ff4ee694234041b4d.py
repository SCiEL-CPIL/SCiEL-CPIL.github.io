import time

def _is_paley_valid(p):
    """
    Checks if the Paley tournament on p vertices satisfies the property:
    Every set of 4 vertices has a common dominator.
    This is equivalent to: For every vertex v, the subgraph induced by its in-neighbors is transitive (acyclic).
    """
    if p < 5:
        return False
    
    # Squares mod p
    squares = set((x * x) % p for x in range(1, p))
    
    # For each vertex v, get in-neighbors
    for v in range(p):
        in_neighbors = []
        for u in range(p):
            if u == v:
                continue
            # u -> v iff (v - u) % p is in squares
            # Wait, definition: i -> j iff (j - i) % p in squares.
            # So u -> v iff (v - u) % p in squares.
            if (v - u) % p in squares:
                in_neighbors.append(u)
        
        # Check if tournament on in_neighbors is transitive
        # Transitive iff no 3-cycles.
        # A 3-cycle exists iff there are a, b, c such that a->b, b->c, c->a.
        # We can check for each pair (a, b) in in_neighbors:
        # If a -> b, then for any c in in_neighbors:
        #   if b -> c and c -> a, then we have a cycle a->b->c->a.
        
        n_in = len(in_neighbors)
        # If n_in < 3, no 3-cycle possible.
        if n_in < 3:
            continue
            
        # Build adjacency for in_neighbors for fast lookup
        # Map vertex to index? Or just use sets?
        # Since p is small, we can use a dictionary or list of sets.
        # adj[a] = set of b such that a -> b
        
        adj = {u: set() for u in in_neighbors}
        for i in range(n_in):
            a = in_neighbors[i]
            for j in range(i + 1, n_in):
                b = in_neighbors[j]
                # Check direction between a and b
                if (b - a) % p in squares:
                    # a -> b
                    adj[a].add(b)
                else:
                    # b -> a
                    adj[b].add(a)
                    
        # Check for 3-cycles
        # For each a in in_neighbors:
        #   For each b in adj[a]:
        #     For each c in adj[b]:
        #       if c in adj[a]: # c -> a
        #           return False (found cycle)
        
        for a in in_neighbors:
            for b in adj[a]:
                for c in adj[b]:
                    if c in adj[a]:
                        return False
                        
    return True

def construct():
    start_time = time.time()
    deadline = 110
    
    # Primes p ≡ 3 (mod 4)
    # 3, 7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83
    candidates = [7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83]
    
    best_n = None
    best_orientation = None
    
    for p in candidates:
        if time.time() > deadline:
            break
            
        if _is_paley_valid(p):
            best_n = p
            # Construct orientation
            squares = set((x * x) % p for x in range(1, p))
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    if (j - i) % p in squares:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            best_orientation = orientation
            break # Found smallest valid prime
            
    if best_n is None:
        # Fallback to 67 if something went wrong or time out
        p = 67
        squares = set((x * x) % p for x in range(1, p))
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return {'n': p, 'orientation': orientation}
        
    return {'n': best_n, 'orientation': best_orientation}
