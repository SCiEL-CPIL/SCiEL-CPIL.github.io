import time
import random
import numpy as np

def construct():
    deadline = time.time() + 115
    n = 40
    
    # Adjacency as list of bitmasks
    adj = [0] * n
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def total_edges():
        return sum(bin(a).count('1') for a in adj) // 2
        
    def has_c6(u, v):
        """Check if adding edge (u,v) creates a C6."""
        # Neighbors of u excluding v
        nu = adj[u] & ~(1 << v)
        # Neighbors of v excluding u
        nv = adj[v] & ~(1 << u)
        
        a = nu
        while a:
            la = a & -a
            a ^= la
            ai = la.bit_length() - 1
            
            d = nv
            while d:
                ld = d & -d
                d ^= ld
                di = ld.bit_length() - 1
                
                # Check if there is a vertex b connected to both ai and di
                # b cannot be u, v, ai, di
                common = adj[ai] & adj[di]
                common &= ~(1 << u) & ~(1 << v) & ~(1 << ai) & ~(1 << di)
                if common:
                    return True
        return False

    # Initialize with 8 K5s
    for k in range(8):
        base = k * 5
        for i in range(5):
            for j in range(i+1, 5):
                add_edge(base + i, base + j)
                
    best_adj = list(adj)
    best_count = total_edges()
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Local Search
    while time.time() < deadline - 5:
        # Find all valid missing edges
        valid_edges = []
        for u, v in all_pairs:
            if not (adj[u] & (1 << v)):
                if not has_c6(u, v):
                    valid_edges.append((u, v))
                    
        if not valid_edges:
            # Stuck, try perturbation
            # Remove a random edge
            if time.time() > deadline - 3:
                break
            current_edges = [(u, v) for u, v in all_pairs if (adj[u] & (1 << v))]
            if not current_edges:
                break
            ru, rv = random.choice(current_edges)
            remove_edge(ru, rv)
            continue
            
        # Add all valid edges? Or just the best one?
        # Adding all valid edges at once might create C6s between the new edges?
        # No, has_c6 checks against the *current* graph.
        # If we add edge e1, it might create a C6 with e2 if e1 and e2 interact.
        # So we must add them one by one or check carefully.
        # Given the density, adding one by one is safer.
        # However, we can try to add as many as possible in a greedy fashion.
        
        added_any = False
        # Sort by some heuristic? Or just random order?
        # Random order helps escape local minima.
        random.shuffle(valid_edges)
        
        for u, v in valid_edges:
            if time.time() > deadline - 3:
                break
            # Re-check if still missing and valid (might have changed if we added other edges)
            if adj[u] & (1 << v):
                continue
            if not has_c6(u, v):
                add_edge(u, v)
                added_any = True
                
        current_count = total_edges()
        if current_count > best_count:
            best_count = current_count
            best_adj = list(adj)
            
        if not added_any:
            # Perturbation
            if time.time() > deadline - 3:
                break
            current_edges = [(u, v) for u, v in all_pairs if (adj[u] & (1 << v))]
            if not current_edges:
                break
            # Remove 1 or 2 random edges to allow new structure
            for _ in range(random.randint(1, 3)):
                ru, rv = random.choice(current_edges)
                remove_edge(ru, rv)
                
    # Final polish
    if time.time() < deadline - 2:
        # Try to add any remaining edges
        for u, v in all_pairs:
            if time.time() > deadline - 1:
                break
            if not (adj[u] & (1 << v)):
                if not has_c6(u, v):
                    add_edge(u, v)
        current_count = total_edges()
        if current_count > best_count:
            best_count = current_count
            best_adj = list(adj)

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
