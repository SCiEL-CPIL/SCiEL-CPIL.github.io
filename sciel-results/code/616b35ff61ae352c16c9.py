import random
import time
import itertools
import numpy as np

def construct():
    n = 20
    # adj is a binary numpy matrix: adj[u,v] = 1 if edge exists
    adj = np.zeros((n, n), dtype=np.int32)
    # common[u][v] is the number of common neighbors of u and v
    common = np.zeros((n, n), dtype=np.int32)
    
    def can_add_vectorized():
        """
        Returns a boolean matrix where can_add[i,j] = True if edge (i,j) can be added.
        Uses fully vectorized NumPy operations for speed.
        """
        # Start with all potential edges (upper triangle, no self-loops)
        can = np.ones((n, n), dtype=bool)
        np.fill_diagonal(can, False)
        can = can & (adj == 0)  # Only consider non-edges
        
        # Check 1: common[i,j] <= 2
        can = can & (common <= 2)
        
        # Check 2 & 3: For edge (u,v), for all y in N(v)\{u}, common[u,y] <= 1
        # and for all y in N(u)\{v}, common[v,y] <= 1
        
        # We need to check: for each (u,v) candidate, is there any y such that
        # adj[v,y]==1, y!=u, y!=v, and common[u,y]>1?
        # AND is there any y such that adj[u,y]==1, y!=u, y!=v, and common[v,y]>1?
        
        # This is complex to vectorize directly. Let's use a different approach:
        # For each vertex u, compute for each potential neighbor v whether adding (u,v) 
        # would violate the constraint.
        
        # Actually, let's use a more direct vectorized check:
        # For a candidate edge (u,v) to be valid:
        # 1. common[u,v] <= 2
        # 2. max(common[u, N(v)\{u,v}]) <= 1  (if N(v)\{u,v} is non-empty)
        # 3. max(common[v, N(u)\{u,v}]) <= 1  (if N(u)\{u,v} is non-empty)
        
        # Let's compute these conditions vectorized.
        
        # Condition 2: For each (u,v), check if any y in N(v) (excluding u,v) has common[u,y] > 1
        # We can create a 3D tensor or use matrix operations.
        
        # Alternative: For each u, compute a vector bad_u where bad_u[v] = 1 if 
        # adding edge (u,v) would violate condition 2 or 3.
        
        # Let me think differently. The constraint for adding edge (u,v) is:
        # - common[u,v] <= 2
        # - For all y: if adj[v,y]==1 and y!=u and y!=v, then common[u,y] <= 1
        # - For all y: if adj[u,y]==1 and y!=u and y!=v, then common[v,y] <= 1
        
        # Let's precompute for each vertex u, a matrix M[u] where M[u][v] = 
        # max_{y in N(v)\{u,v}} common[u,y] if N(v)\{u,v} non-empty, else 0
        # and N[u][v] = max_{y in N(u)\{u,v}} common[v,y] if N(u)\{u,v} non-empty, else 0
        
        # This is still complex. Let's use a simpler approach: iterate over candidates
        # but use vectorized checks for each candidate.
        
        # Actually, let's just do a vectorized check for all candidates at once.
        # For each pair (u,v), we need to check:
        # A) common[u,v] <= 2
        # B) No y exists with adj[v,y]==1, y!=u, y!=v, and common[u,y]>1
        # C) No y exists with adj[u,y]==1, y!=u, y!=v, and common[v,y]>1
        
        # For B: For fixed u, consider all v. We need: for all y, NOT(adj[v,y]==1 AND y!=u AND y!=v AND common[u,y]>1)
        # This is equivalent to: for all y, (adj[v,y]==0 OR y==u OR y==v OR common[u,y]<=1)
        
        # Let's define for each u, a matrix V_u where V_u[v] = 1 if condition B holds for (u,v)
        # V_u[v] = 1 if for all y: adj[v,y]==0 or y==u or y==v or common[u,y]<=1
        
        # This can be written as: V_u[v] = min_y (1 - adj[v,y]*(y!=u)*(y!=v)*(common[u,y]>1))
        # But this is still a min over y, which is not easily vectorized in a simple way.
        
        # Let's try a different strategy: use the fact that n=20 is small.
        # We can vectorize over the 190 possible edges.
        
        # Create arrays of all possible (u,v) pairs with u<v
        u_arr, v_arr = np.triu_indices(n, k=1)
        num_candidates = len(u_arr)
        
        # Initialize can_add for these candidates
        can_vec = np.ones(num_candidates, dtype=bool)
        
        # Check 1: common[u,v] <= 2
        common_uv = common[u_arr, v_arr]
        can_vec = can_vec & (common_uv <= 2)
        
        # Check 2: For each candidate (u,v), no y in N(v)\{u} has common[u,y] > 1
        # For each candidate i, u=u_arr[i], v=v_arr[i]
        # We need: for all y, adj[v,y]==0 or y==u or common[u,y]<=1
        
        # Let's create a 2D array: for each candidate i and each y, 
        # bad2[i,y] = 1 if adj[v_arr[i],y]==1 and y!=u_arr[i] and common[u_arr[i],y]>1
        # Then condition 2 holds for candidate i iff sum_y bad2[i,y] == 0
        
        # adj_v_y = adj[v_arr, :]  # shape (num_candidates, n)
        adj_v_y = adj[v_arr, :]  # (num_candidates, n)
        # y != u_arr[i]: create a mask
        y_indices = np.arange(n)
        u_expanded = u_arr[:, np.newaxis]  # (num_candidates, 1)
        y_mask = y_indices[np.newaxis, :]  # (1, n)
        y_neq_u = (y_mask != u_expanded)  # (num_candidates, n)
        
        common_u_y = common[u_arr, :]  # (num_candidates, n)
        common_gt_1 = (common_u_y > 1)  # (num_candidates, n)
        
        bad2 = (adj_v_y == 1) & y_neq_u & common_gt_1  # (num_candidates, n)
        can_vec = can_vec & (bad2.sum(axis=1) == 0)
        
        # Check 3: For each candidate (u,v), no y in N(u)\{v} has common[v,y] > 1
        adj_u_y = adj[u_arr, :]  # (num_candidates, n)
        y_neq_v = (y_mask != v_arr[:, np.newaxis])  # (num_candidates, n)
        common_v_y = common[v_arr, :]  # (num_candidates, n)
        common_v_gt_1 = (common_v_y > 1)  # (num_candidates, n)
        
        bad3 = (adj_u_y == 1) & y_neq_v & common_v_gt_1  # (num_candidates, n)
        can_vec = can_vec & (bad3.sum(axis=1) == 0)
        
        return u_arr, v_arr, can_vec
    
    def add_edge(u, v):
        adj[u, v] = 1
        adj[v, u] = 1
        # Update common counts
        # For y in N(v) (excluding u): common[u][y] += 1
        neighbors_v = np.where(adj[v] == 1)[0]
        mask = np.ones(n, dtype=bool)
        mask[u] = False
        neighbors_v_masked = neighbors_v[mask[neighbors_v]]
        if len(neighbors_v_masked) > 0:
            common[u, neighbors_v_masked] += 1
            common[neighbors_v_masked, u] += 1
            
        # For y in N(u) (excluding v): common[v][y] += 1
        neighbors_u = np.where(adj[u] == 1)[0]
        mask = np.ones(n, dtype=bool)
        mask[v] = False
        neighbors_u_masked = neighbors_u[mask[neighbors_u]]
        if len(neighbors_u_masked) > 0:
            common[v, neighbors_u_masked] += 1
            common[neighbors_u_masked, v] += 1

    def remove_edge(u, v):
        adj[u, v] = 0
        adj[v, u] = 0
        # Revert updates
        # For y in N(v) (excluding u): common[u][y] -= 1
        neighbors_v = np.where(adj[v] == 1)[0]
        if len(neighbors_v) > 0:
            common[u, neighbors_v] -= 1
            common[neighbors_v, u] -= 1
        # For y in N(u) (excluding v): common[v][y] -= 1
        neighbors_u = np.where(adj[u] == 1)[0]
        if len(neighbors_u) > 0:
            common[v, neighbors_u] -= 1
            common[neighbors_u, v] -= 1

    def get_edge_count():
        return int(adj.sum()) // 2

    def get_edges_list():
        """Get list of all edges as (u, v) with u < v"""
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v] == 1:
                    edges.append((u, v))
        return edges

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
        edges_in_clebsch = get_edges_list()
        
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
            
        # If we didn't find 4 disjoint edges (unlikely for Clebsch), we can pick overlapping ones
        # but that might violate constraints if not careful. 
        # However, for now, let's assume we found 4.
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
                # Use vectorized check for this single edge
                u_arr, v_arr, can_vec = can_add_vectorized()
                # Find index of (u,v)
                for idx in range(len(u_arr)):
                    if u_arr[idx] == u and v_arr[idx] == v:
                        if can_vec[idx]:
                            add_edge(u, v)
                        break

    start_time = time.time()
    time_limit = 115.0 # Safety margin for 120s limit
    
    # Build initial structure
    build_initial_graph()
    
    best_edges = [[u, v] for u, v in get_edges_list()]
    best_count = get_edge_count()
    
    # Local search with vectorized checks
    while time.time() - start_time < time_limit:
        # Get all valid edges using vectorized check
        u_arr, v_arr, can_vec = can_add_vectorized()
        
        if np.any(can_vec):
            # Add a random valid edge
            valid_indices = np.where(can_vec)[0]
            idx = random.choice(valid_indices.tolist())
            add_edge(u_arr[idx], v_arr[idx])
            
            curr_count = get_edge_count()
            if curr_count > best_count:
                best_count = curr_count
                best_edges = [[u, v] for u, v in get_edges_list()]
        else:
            # No edges can be added. Try to remove an edge to allow new ones.
            curr_count = get_edge_count()
            if curr_count > 0:
                # Pick a random edge to remove
                edges_list = get_edges_list()
                if edges_list:
                    u, v = random.choice(edges_list)
                    remove_edge(u, v)
                    # Try to add edges again in the next iteration
            else:
                # Empty graph, restart
                pass

    # Final check: ensure the best_edges is valid
    if not best_edges:
        # Fallback to seed if something went wrong
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    # Verification step (optional but good for safety)
    adj_check = [set() for _ in range(n)]
    for u, v in best_edges:
        adj_check[u].add(v)
        adj_check[v].add(u)
    
    valid = True
    for u in range(n):
        for v in range(u + 1, n):
            common_count = len(adj_check[u] & adj_check[v])
            if common_count > 2:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    return best_edges
