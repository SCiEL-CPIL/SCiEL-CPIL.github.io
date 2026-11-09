import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    target_edges = 401
    
    # 1. Initialize with the known good structure (Book Size 9)
    A = list(range(9))
    B = list(range(9, 40))
    
    edges = set()
    for a in A:
        for b in B:
            edges.add((a, b))
            
    B1 = list(range(9, 24))
    B2 = list(range(24, 40))
    needed = 122
    added = 0
    deg_b1 = {v: 0 for v in B1}
    deg_b2 = {v: 0 for v in B2}
    
    pairs = [(u, v) for u in B1 for v in B2]
    random.seed(42)
    random.shuffle(pairs)
    
    for u, v in pairs:
        if added == needed:
            break
        if deg_b1[u] < 9 and deg_b2[v] < 9:
            edges.add((u, v))
            deg_b1[u] += 1
            deg_b2[v] += 1
            added += 1
            
    if len(edges) != target_edges:
        # Fallback
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        edges = set()
        for u, v in all_pairs:
            if len(edges) == target_edges:
                break
            edges.add((u, v))

    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def get_max_book():
        max_b = 0
        for u, v in edges:
            b = len(adj[u].intersection(adj[v]))
            if b > max_b:
                max_b = b
        return max_b

    current_max_book = get_max_book()
    best_max_book = current_max_book
    best_edges = set(edges)
    
    # Local Search Loop
    while time.time() < deadline:
        # Find an edge with current max book size to remove
        curr_edges_list = list(edges)
        curr_max = 0
        worst_edges = []
        for u, v in curr_edges_list:
            b = len(adj[u].intersection(adj[v]))
            if b > curr_max:
                curr_max = b
                worst_edges = [(u,v)]
            elif b == curr_max:
                worst_edges.append((u,v))
                
        if not worst_edges:
            break
            
        u, v = random.choice(worst_edges)
        
        # Remove (u,v)
        edges.remove((u,v))
        adj[u].remove(v)
        adj[v].remove(u)
        
        # Find a non-edge to add
        x, y = -1, -1
        # Try a few times to find a valid non-edge
        for _ in range(10):
            rx = random.randint(0, n-1)
            ry = random.randint(0, n-1)
            if rx == ry: continue
            rx, ry = min(rx, ry), max(rx, ry)
            if (rx, ry) not in edges:
                x, y = rx, ry
                break
                
        if x == -1:
            # Revert if we couldn't find a non-edge (unlikely)
            edges.add((u,v))
            adj[u].add(v)
            adj[v].add(u)
            continue
            
        # Add (x,y)
        edges.add((x,y))
        adj[x].add(y)
        adj[y].add(x)
        
        # Calculate new max book
        new_max = 0
        for a, b in edges:
            nb = len(adj[a].intersection(adj[b]))
            if nb > new_max:
                new_max = nb
                
        if new_max < best_max_book:
            best_max_book = new_max
            best_edges = set(edges)
        elif new_max == best_max_book:
            # Accept move to explore
            pass
        else:
            # Revert if new_max > best_max_book? 
            # Or revert if new_max > curr_max?
            # Standard hill climbing: revert if worse than current.
            # But we are tracking best.
            # If new_max > curr_max, it's definitely worse than current state.
            # If new_max == curr_max, it's equal.
            # If new_max < curr_max, it's better.
            
            # Let's use: Accept if new_max <= curr_max.
            # Revert if new_max > curr_max.
            
            if new_max > curr_max:
                edges.remove((x,y))
                adj[x].remove(y)
                adj[y].remove(x)
                edges.add((u,v))
                adj[u].add(v)
                adj[v].add(u)
            else:
                # new_max <= curr_max
                # If new_max < best_max_book, we already updated best.
                # If new_max == best_max_book, we keep the new state.
                pass

    result = [[u, v] for u, v in best_edges]
    return result
