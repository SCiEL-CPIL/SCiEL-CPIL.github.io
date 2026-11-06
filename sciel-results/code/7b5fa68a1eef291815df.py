import random
import time
import math

def construct():
    deadline = time.time() + 145.0
    n = 40
    target_edges = 401
    
    # Bitmask representation
    # adj[u] is an integer, bit v is 1 if edge (u,v) exists
    
    def init_k20_20_plus_1():
        adj = [0] * n
        for a in range(20):
            mask = 0
            for b in range(20, 40):
                mask |= (1 << b)
            adj[a] = mask
            for b in range(20, 40):
                adj[b] |= (1 << a)
        # Add edge (0, 1)
        adj[0] |= (1 << 1)
        adj[1] |= (1 << 0)
        return adj

    def init_random():
        adj = [0] * n
        edges = set()
        while len(edges) < target_edges:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            if (u,v) in edges: continue
            edges.add((u,v))
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def compute_books(adj):
        books = {}
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    books[(u,v)] = common.bit_count()
        return books

    def max_book_val(books):
        if not books: return 0
        return max(books.values())

    def get_edge_list(adj):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    edges.append((u,v))
        return edges

    best_adj = None
    best_max = float('inf')
    
    num_restarts = 20 # Adjust based on time
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            adj = init_k20_20_plus_1()
        else:
            # Mix of K20,20+1 and Random
            if r % 2 == 0:
                adj = init_k20_20_plus_1()
                # Perturb
                for _ in range(10):
                    # Random swap
                    edges = get_edge_list(adj)
                    if not edges: break
                    u, v = random.choice(edges)
                    # Find non-edge
                    found = False
                    for _ in range(50):
                        a = random.randint(0, n-1)
                        b = random.randint(0, n-1)
                        if a == b: continue
                        if a > b: a, b = b, a
                        if (a,b) == (u,v): continue
                        if not ((adj[a] >> b) & 1):
                            # Swap
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            adj[a] |= (1 << b)
                            adj[b] |= (1 << a)
                            found = True
                            break
            else:
                adj = init_random()
                
        books = compute_books(adj)
        current_max = max_book_val(books)
        
        if current_max < best_max:
            best_max = current_max
            best_adj = adj[:]
            
        # Greedy Descent
        improved = True
        no_improve_count = 0
        while improved and time.time() < deadline:
            improved = False
            books = compute_books(adj)
            current_max = max_book_val(books)
            
            if current_max < best_max:
                best_max = current_max
                best_adj = adj[:]
                
            # Find edge with max book
            max_edge = None
            max_val = -1
            for e, b in books.items():
                if b > max_val:
                    max_val = b
                    max_edge = e
            
            if max_edge is None:
                break
                
            u, v = max_edge
            
            # Try swaps involving (u,v) or its neighbors to reduce max_val
            # Candidate moves:
            # 1. Swap (u,v) with random non-edge
            # 2. Swap (u,w) with random non-edge where w is common neighbor
            # 3. Swap (v,w) with random non-edge where w is common neighbor
            
            best_move = None
            best_new_max = current_max
            
            # Get common neighbors of u and v
            common_uv = adj[u] & adj[v]
            common_list = [i for i in range(n) if (common_uv >> i) & 1]
            
            # Sample some moves
            moves = []
            
            # Move type 1: Remove (u,v), add random
            for _ in range(5):
                a = random.randint(0, n-1)
                b = random.randint(0, n-1)
                if a == b: continue
                if a > b: a, b = b, a
                if (a,b) == (u,v): continue
                if (adj[a] >> b) & 1: continue
                moves.append(((u,v), (a,b)))
                
            # Move type 2: Remove (u,w), add random
            if common_list:
                for w in random.sample(common_list, min(5, len(common_list))):
                    for _ in range(2):
                        a = random.randint(0, n-1)
                        b = random.randint(0, n-1)
                        if a == b: continue
                        if a > b: a, b = b, a
                        if (a,b) == (u,w): continue
                        if (adj[a] >> b) & 1: continue
                        moves.append(((u,w), (a,b)))
                        
            # Move type 3: Remove (v,w), add random
            if common_list:
                for w in random.sample(common_list, min(5, len(common_list))):
                    for _ in range(2):
                        a = random.randint(0, n-1)
                        b = random.randint(0, n-1)
                        if a == b: continue
                        if a > b: a, b = b, a
                        if (a,b) == (v,w): continue
                        if (adj[a] >> b) & 1: continue
                        moves.append(((v,w), (a,b)))
            
            # Evaluate moves
            for rem, add in moves:
                ru, rv = rem
                au, av = add
                
                # Apply
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                adj[au] |= (1 << av)
                adj[av] |= (1 << au)
                
                # Compute new max
                new_books = compute_books(adj)
                new_max = max_book_val(new_books)
                
                # Revert
                adj[ru] |= (1 << rv)
                adj[rv] |= (1 << ru)
                adj[au] &= ~(1 << av)
                adj[av] &= ~(1 << au)
                
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_move = (rem, add)
            
            if best_move:
                rem, add = best_move
                ru, rv = rem
                au, av = add
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                adj[au] |= (1 << av)
                adj[av] |= (1 << au)
                improved = True
                no_improve_count = 0
            else:
                # Kick: random swap
                edges = get_edge_list(adj)
                if edges:
                    u, v = random.choice(edges)
                    for _ in range(50):
                        a = random.randint(0, n-1)
                        b = random.randint(0, n-1)
                        if a == b: continue
                        if a > b: a, b = b, a
                        if (a,b) == (u,v): continue
                        if (adj[a] >> b) & 1: continue
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                        break
                no_improve_count += 1
                if no_improve_count > 50:
                    break # Restart

    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
