import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 20
    total_pairs = n * (n - 1) // 2
    
    # Precompute bit masks for vertices
    # adj[i] is a bitmask of neighbors of i
    
    def is_valid_addition(adj, u, v):
        if adj[u] & (1 << v):
            return False
        
        # Check triples containing u: {u, a, b}
        # If v is in N(a) and N(b), then v is added to N({u, a, b}).
        # We need old |N({u, a, b})| <= 1.
        # N({u, a, b}) = N(u) & N(a) & N(b)
        # But N(u) does not include v yet.
        # So we check if (N(u) & N(a) & N(b)).bit_count() <= 1
        # If it is 2, adding v makes it 3 -> Invalid.
        
        # Optimization: Iterate a, b
        # If v is not in N(a) or not in N(b), then v is not a common neighbor of {u,a,b}, 
        # so the set N({u,a,b}) doesn't change size? 
        # Wait. N({u,a,b}) is the set of common neighbors.
        # Before adding (u,v): Common neighbors are N(u)_old & N(a) & N(b).
        # After adding (u,v): N(u)_new = N(u)_old U {v}.
        # New Common neighbors = (N(u)_old U {v}) & N(a) & N(b)
        # = (N(u)_old & N(a) & N(b)) U ({v} & N(a) & N(b))
        
        # Case 1: v is NOT in N(a) or NOT in N(b).
        # Then {v} & N(a) & N(b) is empty.
        # New Common neighbors = N(u)_old & N(a) & N(b).
        # Size is unchanged. Since graph was valid, size <= 2.
        # So we only need to worry if v IS in N(a) AND v IS in N(b).
        
        for a in range(n):
            if a == u or a == v:
                continue
            if not (adj[a] & (1 << v)):
                continue
            
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (adj[b] & (1 << v)):
                    continue
                
                # v is in N(a) and N(b).
                # So v is a common neighbor of {u, a, b} after addition.
                # We need the total count of common neighbors to be <= 2.
                # So we need count(N(u)_old & N(a) & N(b)) <= 1.
                count = (adj[u] & adj[a] & adj[b]).bit_count()
                if count > 1:
                    return False
                        
        # Check triples containing v: {v, a, b}
        # Similarly, if u is in N(a) and N(b), then u is added to N({v, a, b}).
        for a in range(n):
            if a == u or a == v:
                continue
            if not (adj[a] & (1 << u)):
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (adj[b] & (1 << u)):
                    continue
                
                count = (adj[v] & adj[a] & adj[b]).bit_count()
                if count > 1:
                    return False
                        
        return True

    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2

    def greedy_fill(adj):
        """Try to add all possible edges."""
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            # Randomize order to break symmetry? Or just iterate.
            # Iterating in fixed order is faster but might get stuck in a specific local optimum.
            # For greedy fill, we just want to maximize.
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if is_valid_addition(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            changed = True

    def local_search(adj):
        """
        Perform local search:
        1. Greedy fill
        2. Swap moves (remove k, add k+1)
        3. Kick and refill
        """
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        best_adj = current_adj[:]
        best_edges = current_edges
        
        # 1. Greedy Fill
        greedy_fill(current_adj)
        current_edges = count_edges(current_adj)
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = current_adj[:]

        # 2. Swap Moves
        # Try to remove 1 edge and add 2
        # Try to remove 2 edges and add 3
        max_swap_steps = 500
        for step in range(max_swap_steps):
            if time.time() > deadline:
                break
            
            improved = False
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if current_adj[i] & (1 << j)]
            random.shuffle(edges)
            
            # Attempt 1-in-1-out (0 net gain, but might enable future gains? No, we want max edges)
            # We only care about net gain.
            
            # Attempt 2-in-1-out
            for u, v in edges:
                if time.time() > deadline:
                    break
                if improved:
                    break
                
                # Remove (u,v)
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                
                # Try to add 2 edges
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j)) and (i != u or j != v) and (i != v or j != u)]
                random.shuffle(missing)
                
                added = []
                for a, b in missing:
                    if time.time() > deadline:
                        break
                    if is_valid_addition(current_adj, a, b):
                        current_adj[a] |= (1 << b)
                        current_adj[b] |= (1 << a)
                        added.append((a, b))
                        if len(added) == 2:
                            break
                
                if len(added) == 2:
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
                    improved = True
                elif len(added) == 1:
                    # Net 0, but we are in a different state. 
                    # We should probably revert to keep edge count high for now, 
                    # unless we do a kick later. 
                    # For now, revert to maintain best known edge count in current_adj?
                    # Actually, if we don't improve, we should revert to the state before the swap attempt
                    # to try the next edge.
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    # Undo the added edge
                    a, b = added[0]
                    current_adj[a] &= ~(1 << b)
                    current_adj[b] &= ~(1 << a)
                else:
                    # Revert
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
            
            if not improved:
                # Try 3-in-2-out?
                # This is more complex. Let's try Kick instead if no improvement.
                break

        # 3. Kick and Refill
        # Randomly remove some edges, then greedy fill
        kick_size = 5
        for _ in range(10):
            if time.time() > deadline:
                break
            
            kick_adj = current_adj[:]
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if kick_adj[i] & (1 << j)]
            random.shuffle(edges)
            
            # Remove kick_size edges
            for i in range(min(kick_size, len(edges))):
                u, v = edges[i]
                kick_adj[u] &= ~(1 << v)
                kick_adj[v] &= ~(1 << u)
            
            # Greedy fill
            greedy_fill(kick_adj)
            kick_edges = count_edges(kick_adj)
            
            if kick_edges > current_edges:
                current_adj = kick_adj[:]
                current_edges = kick_edges
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = current_adj[:]
                # Continue local search from this new state?
                # For simplicity, we return best_adj found so far.
                # We could loop back to swap phase, but time is tight.
                
        return best_adj, best_edges

    best_adj = [0] * n
    best_edges = 0
    
    # Generate initial candidates
    # Use a few different strategies for initialization
    # 1. Random Greedy
    # 2. Structured Greedy (e.g., complete bipartite K10,10 is not K3,3 free? 
    #    K10,10 contains K3,3. So we need to be careful.)
    #    A random graph G(20, p) with p ~ 0.7 might be a good start.
    
    num_restarts = 2000 # Reduced to allow more time per restart for local search
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize with random edges
        adj = [0] * n
        if i % 2 == 0:
            # Random sparse start
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        else:
            # Random dense start (might be invalid, but greedy fill will clean up? 
            # No, greedy fill only adds. We need a valid start or clean up invalid ones.
            # Better to start with a valid graph.
            # Let's stick to random sparse and let greedy fill do the work.
            # Or use a specific valid seed.
            pass
            
        adj, edges = local_search(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
        # If we found a very good solution, do more local search on it
        if best_edges >= 145 and time.time() < deadline - 10:
             # Polishing
             for _ in range(5):
                if time.time() > deadline - 2:
                    break
                adj, edges = local_search(best_adj)
                if edges > best_edges:
                    best_edges = edges
                    best_adj = adj
                else:
                    break

    result_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result_edges.append([i, j])
                
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
