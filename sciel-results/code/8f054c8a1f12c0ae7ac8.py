import random
import time

def construct():
    n = 40
    target_edges = 401
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    def get_score(adj):
        max_book = 0
        for u in range(n):
            # Mask out bits <= u
            v_bits = adj[u] & (~((1 << (u + 1)) - 1))
            while v_bits:
                lsb = v_bits & -v_bits
                v = lsb.bit_length() - 1
                common = adj[u] & adj[v]
                book_size = common.bit_count()
                if book_size > max_book:
                    max_book = book_size
                v_bits ^= lsb
        return max_book

    def generate_random_20_reg():
        # Start with K_20,20
        adj = [0] * n
        for u in range(20):
            for v in range(20, 40):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Randomize with 2-switches
        # We need to pick random edges.
        # To make it faster, we can just pick random vertices and check.
        for _ in range(1000):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or not (adj[u] & (1 << v)):
                continue
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if x == y or not (adj[x] & (1 << y)):
                continue
            
            # Check if (u,v) and (x,y) are distinct and not sharing vertices in a way that prevents switch
            # Actually, 2-switch requires (u,v) and (x,y) to be disjoint edges?
            # No, standard 2-switch on regular graphs:
            # Remove (u,v) and (x,y), add (u,x) and (v,y).
            # Requires (u,x) and (v,y) to be non-edges.
            # Also u!=x, v!=y etc to avoid self loops.
            
            if u == x or u == y or v == x or v == y:
                continue
                
            if (adj[u] & (1 << x)) or (adj[v] & (1 << y)):
                continue
                
            # Perform switch
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            adj[x] ^= (1 << y)
            adj[y] ^= (1 << x)
            
            adj[u] |= (1 << x)
            adj[x] |= (1 << u)
            adj[v] |= (1 << y)
            adj[y] |= (1 << v)
            
        return adj

    best_adj = None
    best_score = float('inf')
    
    while time.time() < deadline:
        # Generate new candidate
        adj = generate_random_20_reg()
        
        # Add one random edge
        # Find a non-edge
        added = False
        for _ in range(100):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and not (adj[u] & (1 << v)):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added = True
                break
        if not added:
            # Fallback: add (0,1) if possible, else skip
            if not (adj[0] & (1 << 1)):
                adj[0] |= (1 << 1)
                adj[1] |= (1 << 0)
            else:
                continue

        current_score = get_score(adj)
        
        # Local search
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            # Pick random edge to remove
            u1 = random.randint(0, n - 1)
            v1 = random.randint(0, n - 1)
            if u1 == v1 or not (adj[u1] & (1 << v1)):
                continue
            
            # Pick random non-edge to add
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2 or (adj[u2] & (1 << v2)):
                continue
            
            if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                continue

            # Swap
            adj[u1] ^= (1 << v1)
            adj[v1] ^= (1 << u1)
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            
            new_score = get_score(adj)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj[:]
            else:
                # Revert
                adj[u1] ^= (1 << v1)
                adj[v1] ^= (1 << u1)
                adj[u2] ^= (1 << v2)
                adj[v2] ^= (1 << u2)
                no_improve_count += 1
                
        if best_score <= 7:
            break

    if best_adj is None:
        # Fallback to seed
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]

    # Reconstruct edge list
    edges = []
    for u in range(n):
        v_bits = best_adj[u] & (~((1 << (u + 1)) - 1))
        while v_bits:
            lsb = v_bits & -v_bits
            v = lsb.bit_length() - 1
            edges.append([u, v])
            v_bits ^= lsb
            
    return edges
