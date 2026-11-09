import random
import time

def count_c4(neighbors, n):
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            common = neighbors[u] & neighbors[v]
            k = bin(common).count('1')
            total += k * (k - 1) // 2
    return total

def construct():
    n = 31
    m = 91
    best_c4 = float('inf')
    best_edges = None
    
    # Use a fixed seed for reproducibility, or random. 
    # Since we need to beat the seed, we should try to find a good configuration.
    # We will run several random restarts.
    
    start_time = time.time()
    time_limit = 115  # Leave some buffer
    
    while time.time() - start_time < time_limit:
        # 1. Initialize random graph
        edges = set()
        # Generate random pairs
        while len(edges) < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                if u > v:
                    u, v = v, u
                edges.add((u, v))
        
        # 2. Build neighbor bitmasks
        neighbors = [0] * n
        for u, v in edges:
            neighbors[u] |= (1 << v)
            neighbors[v] |= (1 << u)
            
        # 3. Initial C4 count
        current_c4 = count_c4(neighbors, n)
        
        # 4. Local Search: Swap edges
        improved = True
        while improved:
            improved = False
            # Create a list of edges to iterate over to avoid modification during iteration
            # We will try all possible swaps
            # To make it faster, we can just pick a random edge to remove and a random non-edge to add?
            # Systematic search is better for small n.
            
            # There are ~91 edges and ~374 non-edges. 91*374 ~ 34k swaps to check per pass.
            # Each check is fast.
            
            # We need to efficiently compute delta C4.
            # Let's just recompute C4 if we accept a swap? 
            # Recomputing C4 takes O(n^2) = 961 ops. 34k swaps * 961 ops = 32M ops.
            # This might be too slow in Python if we do a full pass.
            # We should optimize the delta calculation or limit the number of swaps checked.
            
            # Alternative: Randomized Local Search
            # Pick random edge (u,v) and random non-edge (x,y).
            # Check if swap improves.
            
            # Let's try a limited systematic search or a random walk with hill climbing.
            
            for _ in range(1000): # Limit iterations per restart
                if time.time() - start_time > time_limit:
                    break
                    
                # Pick random edge to remove
                u, v = random.choice(list(edges))
                
                # Pick random non-edge to add
                # We need to find a non-edge (x,y)
                # To make it efficient, we can pick random x, y and check if non-edge.
                # Or maintain a list of non-edges? Updating it is costly.
                # Let's just sample.
                
                found = False
                for _ in range(10): # Try a few times to find a non-edge
                    x = random.randint(0, n - 1)
                    y = random.randint(0, n - 1)
                    if x == y:
                        continue
                    if x > y:
                        x, y = y, x
                    if (x, y) not in edges:
                        # Check if (x,y) is the same as (u,v) or shares vertices?
                        # If it shares vertices, it's a "move" not a "swap".
                        # Swap: remove (u,v), add (x,y). 
                        # We should ensure (x,y) is not (u,v).
                        if (x, y) == (u, v):
                            continue
                        found = True
                        break
            
                if not found:
                    continue
                
                # Tentatively swap
                # Remove (u,v)
                edges.remove((u, v))
                neighbors[u] &= ~(1 << v)
                neighbors[v] &= ~(1 << u)
                
                # Add (x,y)
                edges.add((x, y))
                neighbors[x] |= (1 << y)
                neighbors[y] |= (1 << x)
                
                new_c4 = count_c4(neighbors, n)
                
                if new_c4 < current_c4:
                    current_c4 = new_c4
                    improved = True
                else:
                    # Revert
                    edges.remove((x, y))
                    neighbors[x] &= ~(1 << y)
                    neighbors[y] &= ~(1 << x)
                    
                    edges.add((u, v))
                    neighbors[u] |= (1 << v)
                    neighbors[v] |= (1 << u)
            
        if current_c4 < best_c4:
            best_c4 = current_c4
            best_edges = set(edges)
            # Optional: If we find a very good one, we can tighten the search?
            # For now, just keep going.
            
    # Convert best_edges to list of lists
    return [list(e) for e in best_edges]
