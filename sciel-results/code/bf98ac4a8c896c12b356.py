import random
import time
import itertools
import numpy as np

def construct():
    n = 20
    # Use a numpy adjacency matrix for efficient common-neighbor calculations
    # adj[u, v] = 1 if edge exists, 0 otherwise
    adj = np.zeros((n, n), dtype=np.int8)
    
    def can_add(u, v):
        if u == v:
            return False
        if adj[u, v] == 1:
            return False
        
        # common[u, v] = sum(adj[u, :] * adj[v, :])
        # We need common[u, v] <= 2
        if np.dot(adj[u], adj[v]) > 2:
            return False
        
        # Check 2: For all y in N(v), common[u, y] must be <= 1
        # This is equivalent to: for all y where adj[v, y] == 1 and y != u, 
        # we need common[u, y] <= 1.
        # common[u, y] = dot(adj[u], adj[y])
        # We can compute all common[u, y] for y in N(v) at once.
        # Let Nv be the vector adj[v] with adj[v, u] set to 0 (to exclude u).
        # We want to check if max(common[u, y] for y in Nv) <= 1.
        # common[u, y] for all y is adj @ adj[u].
        # So we want max((adj @ adj[u])[y] for y in Nv) <= 1.
        
        common_u = adj @ adj[u] # common[u, y] for all y
        
        # Mask for N(v) excluding u
        mask_v = adj[v].copy()
        mask_v[u] = 0
        mask_v[v] = 0 # v is not a neighbor of itself in simple graph logic, but adj[v,v]=0 usually
        
        # Check if any common[u, y] > 1 for y in N(v)
        # We can just take the max over the masked positions.
        # However, we only care about positions where mask_v is 1.
        # If we set common_u where mask_v is 0 to -1, then max will ignore them (unless all are -1).
        # But simpler: just loop over neighbors of v? No, we want vectorization.
        # Actually, for n=20, a small loop is fine, but let's try to be smart.
        # The condition is: for all y in N(v)\{u}, common[u, y] <= 1.
        # This is equivalent to: max(common_u[y] for y in N(v)\{u}) <= 1.
        
        # Let's just iterate over neighbors of v. It's at most 19 neighbors.
        # Is it faster than numpy overhead? Probably for n=20, the overhead of 
        # creating masks and dot products might be higher than a simple python loop 
        # over the set of neighbors if we stored them. But we are using numpy matrix.
        # Let's stick to the vectorized check if possible.
        
        # Create a temporary vector to hold common_u values only for neighbors of v.
        # We can use np.where
        # valid_commons = np.where(mask_v == 1, common_u, -1)
        # if np.max(valid_commons) > 1: return False
        
        # This creates a new array of size 20.
        # Let's optimize: 
        # We know we only care about y where adj[v,y]=1 and y!=u.
        
        # Actually, let's revert to a hybrid approach for clarity and speed on small n:
        # The bottleneck in the original code was likely the set operations and python loops 
        # for every single check in the 500 attempts.
        # With numpy, np.dot is fast.
        
        # Let's do the check for neighbors of v.
        # neighbors_v = np.nonzero(adj[v])[0]
        # for y in neighbors_v:
        #    if y == u: continue
        #    if common_u[y] > 1: return False
        
        # This is O(deg(v)) which is small.
        
        # Let's try the fully vectorized max approach first.
        # It's O(n^2) for the dot product, which is 400 ops.
        # The loop is O(deg(v)) ~ 10 ops.
        # The loop is likely faster for n=20.
        
        neighbors_v = np.nonzero(adj[v])[0]
        for y in neighbors_v:
            if y == u:
                continue
            if common_u[y] > 1:
                return False
        
        # Check 3: Symmetric
        common_v = adj @ adj[v]
        neighbors_u = np.nonzero(adj[u])[0]
        for y in neighbors_u:
            if y == v:
                continue
            if common_v[y] > 1:
                return False
        
        return True

    def add_edge(u, v):
        adj[u, v] = 1
        adj[v, u] = 1

    def remove_edge(u, v):
        adj[u, v] = 0
        adj[v, u] = 0

    def get_edge_count():
        return np.sum(adj) // 2

    def build_clebsch():
        # Clebsch graph on vertices 0-15
        # Vertices are 4-bit strings.
        # Adjacency: Hamming distance 1 or 4.
        for u in range(16):
            for v in range(16):
                if u >= v:
                    continue
                # Calculate Hamming distance
                diff = u ^ v
                popcount = bin(diff).count('1')
                if popcount == 1 or popcount == 4:
                    add_edge(u, v)

    def build_initial_graph():
        # Build Clebsch graph
        build_clebsch()
        
        # Find all edges in Clebsch graph to use as neighbor sets for new vertices
        edges_in_clebsch = []
        for u in range(16):
            for v in range(16):
                if u < v and adj[u, v] == 1:
                    edges_in_clebsch.append((u, v))
        
        # Greedily pick 4 disjoint edges
        used_vertices = set()
        selected_pairs = []
        for u, v in edges_in_clebsch:
            if u not in used_vertices and v not in used_vertices:
                selected_pairs.append((u, v))
                used_vertices.add(u)
                used_vertices.add(v)
                if len(selected_pairs) == 4:
                    break
            
        # Connect new vertices 16-19 to these pairs
        for i, (u, v) in enumerate(selected_pairs):
            new_v = 16 + i
            add_edge(new_v, u)
            add_edge(new_v, v)
            
        # Connect new vertices among themselves (K4)
        for i in range(4):
            for j in range(i + 1, 4):
                u = 16 + i
                v = 16 + j
                if can_add(u, v):
                    add_edge(u, v)

    start_time = time.time()
    time_limit = 115.0 # Safety margin for 120s limit
    
    # Build initial structure
    build_initial_graph()
    
    best_adj = adj.copy()
    best_count = get_edge_count()
    
    # Local search
    while time.time() - start_time < time_limit:
        # Try to add a random edge
        improved = False
        # Try many random additions
        for _ in range(500):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if can_add(u, v):
                add_edge(u, v)
                improved = True
                break
        
        if not improved:
            # Stuck. Try to remove an edge to allow new ones
            if get_edge_count() > 0:
                # Pick a random edge to remove
                # Get list of edges
                edges_list = []
                for u in range(n):
                    for v in range(u+1, n):
                        if adj[u, v] == 1:
                            edges_list.append((u, v))
                
                if edges_list:
                    u, v = random.choice(edges_list)
                    remove_edge(u, v)
                    # Try to add edges again
                    for _ in range(500):
                        u2 = random.randint(0, n - 1)
                        v2 = random.randint(0, n - 1)
                        if can_add(u2, v2):
                            add_edge(u2, v2)
                            improved = True
                            break
                    if improved:
                        curr_count = get_edge_count()
                        if curr_count > best_count:
                            best_count = curr_count
                            best_adj = adj.copy()
            else:
                # Empty graph, restart
                pass

    # Convert best_adj to list of edges
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v] == 1:
                best_edges.append([u, v])

    if not best_edges:
        # Fallback to seed if something went wrong
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    # Verification step
    adj_check = np.zeros((n, n), dtype=np.int8)
    for u, v in best_edges:
        adj_check[u, v] = 1
        adj_check[v, u] = 1
    
    valid = True
    # Check for K2,3: no two vertices have > 2 common neighbors
    # common = adj_check @ adj_check
    # We need all off-diagonal elements of common to be <= 2
    common = adj_check @ adj_check
    # Mask out diagonal
    np.fill_diagonal(common, 0)
    if np.max(common) > 2:
        valid = False
        
    if not valid:
        # Fallback
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    return best_edges
