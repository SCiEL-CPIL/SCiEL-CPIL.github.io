import random
import time
import math

def construct():
    n = 31
    target_edges = 91
    
    start_time = time.time()
    time_limit = 115  # Safety margin for 120s limit
    
    # 1. Construct the Polarity Graph of PG(2,5)
    # Points are non-zero vectors in F_5^3 modulo scaling
    # We represent each point by a normalized tuple (x, y, z) where the first non-zero component is 1
    
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: find first non-zero
                if x != 0:
                    norm = (1, y * pow(x, -1, 5), z * pow(x, -1, 5))
                elif y != 0:
                    norm = (0, 1, z * pow(y, -1, 5))
                else:
                    norm = (0, 0, 1)
                
                if norm not in seen:
                    seen.add(norm)
                    points.append(norm)
                    
    # points should have length 31
    # Map point index to vector
    # 2. Build adjacency for the polarity graph
    # P ~ Q if dot(P, Q) == 0 in F_5 and P != Q
    
    def dot(u, v):
        return (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % 5
        
    neighbors = [0] * n
    edge_set = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            if dot(points[i], points[j]) == 0:
                neighbors[i] |= (1 << j)
                neighbors[j] |= (1 << i)
                edge_set.add((i, j))
                
    # Verify edge count
    current_edges = list(edge_set)
    # Should be 90 edges
    
    # 3. Find the best edge to add
    # Calculate common neighbors for all non-edges
    best_add = None
    min_c = float('inf')
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for u, v in all_pairs:
        if (u, v) in edge_set:
            continue
        common = (neighbors[u] & neighbors[v]).bit_count()
        if common < min_c:
            min_c = common
            best_add = (u, v)
            
    # Add the best edge
    if best_add:
        u, v = best_add
        neighbors[u] |= (1 << v)
        neighbors[v] |= (1 << u)
        edge_set.add((u, v))
        current_edges.append((u, v))
        
    # Current C4 count
    def count_c4s(nbrs):
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (nbrs[i] >> j) & 1:
                    common = (nbrs[i] & nbrs[j]).bit_count()
                    total += common * (common - 1) // 2
        return total // 4
        
    current_c4s = count_c4s(neighbors)
    best_c4_count = current_c4s
    best_edges = current_edges[:]
    best_neighbors = neighbors[:]
    
    # 4. Local Search
    # Try to improve by swapping edges
    
    improved = True
    local_time_limit = time_limit - (time.time() - start_time) - 3
    local_start = time.time()
    
    while improved and time.time() - local_start < local_time_limit:
        improved = False
        # Randomize the order of edges to try
        random.shuffle(current_edges)
        
        for i in range(len(current_edges)):
            if time.time() - local_start > local_time_limit:
                break
            u, v = current_edges[i]
            # Remove edge (u,v)
            saved_nu = neighbors[u]
            saved_nv = neighbors[v]
            neighbors[u] = saved_nu & ~(1 << v)
            neighbors[v] = saved_nv & ~(1 << u)
            edge_set.discard((u, v))
            
            # Try adding a different edge
            best_alt_cost = current_c4s
            best_alt_pair = None
            
            # To speed up, we can limit the search to edges that are "close" or just random sample?
            # Full search is 465 pairs.
            for a, b in all_pairs:
                if (a, b) in edge_set:
                    continue
                common = (neighbors[a] & neighbors[b]).bit_count()
                cost = common * (common - 1) // 2
                if cost < best_alt_cost:
                    best_alt_cost = cost
                    best_alt_pair = (a, b)
                    
            # Restore
            neighbors[u] = saved_nu
            neighbors[v] = saved_nv
            edge_set.add((u, v))
            
            if best_alt_pair is not None and best_alt_cost < current_c4s:
                a, b = best_alt_pair
                current_edges[i] = (a, b)
                edge_set.add((a, b))
                neighbors[a] |= (1 << b)
                neighbors[b] |= (1 << a)
                current_c4s = best_alt_cost
                improved = True
                break # Restart loop with new state
            
        if current_c4s < best_c4_count:
            best_c4_count = current_c4s
            best_edges = current_edges[:]
            best_neighbors = neighbors[:]
            
            if best_c4_count == 1:
                break
                
    if best_edges is None:
        return [[i, j] for i, j in all_pairs[:91]]
    
    return [[u, v] for u, v in best_edges]

if hasattr(construct, '_best_edges'):
    del construct._best_edges
