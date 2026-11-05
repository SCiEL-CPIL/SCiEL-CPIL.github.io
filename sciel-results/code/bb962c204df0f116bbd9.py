import random
import numpy as np

def construct():
    n = 40
    target_edges = 401
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # State: list of booleans indicating if edge is present
    # Or set of edges.
    # For speed, use bitmask or boolean array.
    # adj[i] is bitmask of neighbors.
    
    def get_book_sizes(adj):
        books = [0] * n
        # We need max book, but for SA we need to know which edges are hot.
        # Let's just compute max book for now.
        max_book = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    b = bin(adj[u] & adj[v]).count('1')
                    if b > max_book:
                        max_book = b
        return max_book

    def build_initial_graph():
        random.seed(42) # Fixed seed for reproducibility in init? 
        # Better to use random initialization
        indices = list(range(len(all_edges)))
        random.shuffle(indices)
        selected_indices = indices[:target_edges]
        adj = [0] * n
        for idx in selected_indices:
            u, v = all_edges[idx]
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def simulate_annealing(adj, iterations=50000, T_start=10.0, T_end=0.1):
        current_adj = adj[:]
        current_cost = get_book_sizes(current_adj)
        best_adj = current_adj[:]
        best_cost = current_cost
        
        # Precompute edge list for current graph to speed up random selection
        # Actually, we can just pick random u, v
        
        for i in range(iterations):
            T = T_start * (T_end / T_start) ** (i / iterations)
            
            # Pick a random edge to remove
            # We need a list of current edges.
            # Maintaining a list is O(m) to update.
            # Let's just pick random u, v. If not edge, try again?
            # Or maintain a list.
            
            # To be efficient, let's maintain a list of edges.
            # But updating list on swap is O(m) if we search.
            # Given m=401, O(401) is fine.
            
            # Get current edges
            edges = []
            for u in range(n):
                mask = current_adj[u]
                while mask:
                    lsb = mask & -mask
                    v = lsb.bit_length() - 1
                    if u < v:
                        edges.append((u, v))
                    mask ^= lsb
            
            if not edges:
                break
                
            # Pick random edge to remove
            u, v = random.choice(edges)
            
            # Pick random non-edge to add
            # Try a few times
            added = False
            for _ in range(10):
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)
                if x == y:
                    continue
                if x > y:
                    x, y = y, x
                if not (current_adj[x] & (1 << y)):
                    # Check if it's the same edge we're removing
                    if (x == u and y == v):
                        continue
                    # Valid non-edge
                    break
            else:
                continue # Couldn't find non-edge? Unlikely
            
            # Perform swap: remove (u,v), add (x,y)
            
            # 1. Remove (u,v)
            # Identify common neighbors of u and v
            common_mask = current_adj[u] & current_adj[v]
            
            # Update book sizes conceptually?
            # We just recompute cost after swap.
            
            # Modify adj
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            
            # 2. Add (x,y)
            current_adj[x] |= (1 << y)
            current_adj[y] |= (1 << x)
            
            # 3. Calculate new cost
            new_cost = get_book_sizes(current_adj)
            
            # 4. Accept/Reject
            if new_cost < current_cost or (T > 0 and random.random() < np.exp((current_cost - new_cost) / T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = current_adj[:]
            else:
                # Rollback
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_adj[x] &= ~(1 << y)
                current_adj[y] &= ~(1 << x)
                
        return best_adj, best_cost

    # Run SA multiple times with different seeds
    best_adj = None
    best_cost = float('inf')
    
    for seed in range(20):
        random.seed(seed)
        adj = build_initial_graph()
        adj, cost = simulate_annealing(adj, iterations=20000)
        if cost < best_cost:
            best_cost = cost
            best_adj = adj[:]
            
        # If we found the theoretical minimum, we can stop
        if best_cost == 7:
            break

    # Convert best_adj to edge list
    if best_adj is None:
        # Fallback
        adj = build_initial_graph()
        best_adj = adj
        best_cost = get_book_sizes(adj)

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
