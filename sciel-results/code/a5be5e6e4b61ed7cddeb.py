import time
import random
import numpy as np

_N = 40
_M = 401

def _build_initial_graph():
    """
    Construct the graph from inspirations:
    K20,20 minus 9 edges incident to vertex 0 in A, plus a star of 10 in A centered at 0.
    Vertices 0..19 in A, 20..39 in B.
    """
    edges = []
    edge_set = set()
    
    # Add all bipartite edges except the 9 removed ones
    for u in range(20):
        for v in range(20, 40):
            if u == 0 and 20 <= v <= 28:
                continue
            edges.append((u, v))
            edge_set.add((u, v))
    
    # Add internal star edges centered at 0
    for i in range(1, 11):
        edges.append((0, i))
        edge_set.add((0, i))
        
    return edges, edge_set

def _compute_books(edges, adj_mask):
    """Compute book sizes for all edges given adjacency bitmasks."""
    books = {}
    for u, v in edges:
        common = adj_mask[u] & adj_mask[v]
        books[(u, v)] = bin(common).count('1')
    return books

def _max_book(books):
    if not books:
        return 0
    return max(books.values())

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    rng = random.Random(42)
    
    # 1. Initialize with the strong structural candidate
    edges, edge_set = _build_initial_graph()
    
    # Build adjacency bitmasks
    adj_mask = [0] * _N
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Compute initial books
    books = _compute_books(edges, adj_mask)
    cur_max = _max_book(books)
    best_max = cur_max
    best_edges = edges.copy()
    
    # List of non-edges for addition
    non_edges = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if (u, v) not in edge_set:
                non_edges.append((u, v))
                
    # Simulated Annealing parameters
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # Local search
    while time.time() < deadline and T > T_min:
        if not edges or not non_edges:
            break
            
        u, v = rng.choice(edges)
        x, y = rng.choice(non_edges)
        
        # Identify affected edges
        # Removal of (u,v) affects (u,w) and (v,w) for w in N(u)∩N(v)
        # Addition of (x,y) affects (x,z) and (y,z) for z in N(x)∩N(y)
        
        common_uv = adj_mask[u] & adj_mask[v]
        common_xy = adj_mask[x] & adj_mask[y]
        
        affected = set()
        affected.add((u, v))
        for w in range(_N):
            if common_uv & (1 << w):
                if w != u and w != v:
                    affected.add((min(u, w), max(u, w)))
                    affected.add((min(v, w), max(v, w)))
                    
        affected.add((x, y))
        for z in range(_N):
            if common_xy & (1 << z):
                if z != x and z != y:
                    affected.add((min(x, z), max(x, z)))
                    affected.add((min(y, z), max(y, z)))

        # Backup values
        backup = {}
        for e in affected:
            if e in books:
                backup[e] = books[e]
            
        # Apply changes to books
        # 1. Remove (u,v)
        del books[(u, v)]
        
        for w in range(_N):
            if common_uv & (1 << w):
                if w == u or w == v: continue
                key_uw = (min(u, w), max(u, w))
                if key_uw in books:
                    books[key_uw] -= 1
                key_vw = (min(v, w), max(v, w))
                if key_vw in books:
                    books[key_vw] -= 1

        # 2. Add (x,y)
        books[(x, y)] = bin(common_xy).count('1')
        
        for z in range(_N):
            if common_xy & (1 << z):
                if z == x or z == y: continue
                key_xz = (min(x, z), max(x, z))
                if key_xz in books:
                    books[key_xz] += 1
                key_yz = (min(y, z), max(y, z))
                if key_yz in books:
                    books[key_yz] += 1

        # Update adjacency masks
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        adj_mask[x] |= (1 << y)
        adj_mask[y] |= (1 << x)

        # Update edge lists
        edges.remove((u, v))
        edges.append((x, y))
        non_edges.remove((x, y))
        non_edges.append((u, v))
        edge_set.remove((u, v))
        edge_set.add((x, y))

        # Calculate new max
        new_max = _max_book(books)
        
        # Acceptance
        delta = new_max - cur_max
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            cur_max = new_max
            if cur_max < best_max:
                best_max = cur_max
                best_edges = edges.copy()
        else:
            # Revert
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            adj_mask[x] &= ~(1 << y)
            adj_mask[y] &= ~(1 << x)
            
            edges.remove((x, y))
            edges.append((u, v))
            non_edges.remove((u, v))
            non_edges.append((x, y))
            edge_set.remove((x, y))
            edge_set.add((u, v))
            
            # Restore books
            if (x, y) in books:
                del books[(x, y)]
            if (u, v) in backup:
                books[(u, v)] = backup[(u, v)]
                
            for e in affected:
                if e == (u, v) or e == (x, y):
                    continue
                if e in backup:
                    books[e] = backup[e]

        T *= cooling

    # Return the best solution found
    return [[u, v] for (u, v) in best_edges]
