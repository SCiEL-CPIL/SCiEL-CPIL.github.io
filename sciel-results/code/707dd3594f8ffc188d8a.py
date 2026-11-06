"""Improved program: use polarity graph of PG(2,5) with a different bilinear form,
then perform aggressive local search to minimize C4 count."""
import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s limit
    
    n = 31
    m = 91
    
    # Build the polarity graph of PG(2,5)
    # Points of PG(2,5): nonzero vectors (x,y,z) mod 5, identified up to scalar.
    # Representative: fix one coordinate to be 1 if possible.
    
    points = []
    # Type 1: z != 0, scale so z=1: (x,y,1) for x,y in GF(5) -> 25 points
    for x in range(5):
        for y in range(5):
            points.append((x, y, 1))
    # Type 2: z = 0, x != 0, scale so x=1: (1,y,0) for y in GF(5) -> 5 points
    for y in range(5):
        points.append((1, y, 0))
    # Type 3: z = 0, x = 0, y != 0, scale so y=1: (0,1,0) -> 1 point
    points.append((0, 1, 0))
    
    assert len(points) == 31
    
    # Define the polarity using a symmetric bilinear form.
    # For PG(2,q) with q odd, a standard polarity is given by:
    # B((x1,y1,z1), (x2,y2,z2)) = x1*x2 + y1*y2 + z1*z2
    # But we can also use other symmetric forms. Let's try a few.
    
    def build_adjacency(form_type):
        """Build adjacency matrix based on the chosen bilinear form."""
        adj = np.zeros((n, n), dtype=np.int8)
        
        if form_type == 0:
            # Standard: x1*x2 + y1*y2 + z1*z2
            for i in range(n):
                for j in range(i+1, n):
                    val = (points[i][0]*points[j][0] + 
                           points[i][1]*points[j][1] + 
                           points[i][2]*points[j][2]) % 5
                    if val == 0:
                        adj[i, j] = 1
                        adj[j, i] = 1
        elif form_type == 1:
            # x1*x2 + y1*y2 + z1*z2 + x1*y2 + y1*x2 (symmetric)
            for i in range(n):
                for j in range(i+1, n):
                    val = (points[i][0]*points[j][0] + 
                           points[i][1]*points[j][1] + 
                           points[i][2]*points[j][2] +
                           points[i][0]*points[j][1] +
                           points[i][1]*points[j][0]) % 5
                    if val == 0:
                        adj[i, j] = 1
                        adj[j, i] = 1
        elif form_type == 2:
            # x1*x2 + y1*y2 + z1*z2 + x1*z2 + z1*x2
            for i in range(n):
                for j in range(i+1, n):
                    val = (points[i][0]*points[j][0] + 
                           points[i][1]*points[j][1] + 
                           points[i][2]*points[j][2] +
                           points[i][0]*points[j][2] +
                           points[i][2]*points[j][0]) % 5
                    if val == 0:
                        adj[i, j] = 1
                        adj[j, i] = 1
        elif form_type == 3:
            # x1*x2 + y1*y2 + z1*z2 + y1*z2 + z1*y2
            for i in range(n):
                for j in range(i+1, n):
                    val = (points[i][0]*points[j][0] + 
                           points[i][1]*points[j][1] + 
                           points[i][2]*points[j][2] +
                           points[i][1]*points[j][2] +
                           points[i][2]*points[j][1]) % 5
                    if val == 0:
                        adj[i, j] = 1
                        adj[j, i] = 1
        
        return adj
    
    def count_c4s(adj):
        """Count the number of 4-cycles in the graph using matrix multiplication."""
        # A 4-cycle u-v-w-x-u exists if:
        # u~v, v~w, w~x, x~u, and u!=w, v!=x
        # Using adjacency matrix A, the number of closed walks of length 4 is trace(A^4)
        # But this counts each 4-cycle multiple times.
        # 
        # Better approach: For each pair of vertices (u,v), count common neighbors.
        # A 4-cycle is determined by two pairs of vertices that share exactly 2 common neighbors... 
        # Actually, a C4 is a set of 4 vertices {a,b,c,d} with edges ab, bc, cd, da.
        # 
        # Number of C4s = sum over all pairs (u,v) of C(common_neighbors(u,v), 2) / 2
        # Wait, that's not quite right either.
        # 
        # Let's use: for each pair of vertices (u,v), let c(u,v) be the number of common neighbors.
        # Then the number of 4-cycles is (1/4) * sum_{u<v} c(u,v) * (c(u,v) - 1)
        # No, that's not right.
        # 
        # Correct formula: Number of C4s = (1/2) * sum_{u} sum_{v > u} C(c(u,v), 2)
        # where c(u,v) is the number of common neighbors of u and v.
        # But this counts each C4 twice (once for each diagonal pair).
        # Actually, each C4 has 2 diagonals, so we divide by 2.
        # Wait, let me think again.
        # 
        # A C4: a-b-c-d-a. The diagonals are (a,c) and (b,d).
        # For diagonal (a,c), the common neighbors are b and d, so C(2,2) = 1.
        # For diagonal (b,d), the common neighbors are a and c, so C(2,2) = 1.
        # So sum over all pairs gives 2 per C4.
        # Thus, number of C4s = (1/2) * sum_{u<v} C(c(u,v), 2)
        
        A = adj.astype(np.int64)
        # Compute common neighbors: (A @ A)[i,j] = number of common neighbors of i and j
        common = A @ A
        # For each pair (i,j), C(common[i,j], 2) = common[i,j]*(common[i,j]-1)/2
        # Sum over all i<j
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                c = common[i, j]
                if c >= 2:
                    total += c * (c - 1) // 2
        return total // 2
    
    def get_edges(adj):
        """Get list of edges from adjacency matrix."""
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    edges.append((i, j))
        return edges
    
    def edge_to_idx(u, v):
        """Map edge (u,v) to an index in the edge list."""
        # We'll use a dictionary for fast lookup
        pass
    
    # Try different bilinear forms and pick the one with fewest C4s
    best_adj = None
    best_c4 = float('inf')
    
    for form_type in range(4):
        adj = build_adjacency(form_type)
        num_edges = int(np.sum(adj) // 2)
        
        if num_edges == m:
            c4 = count_c4s(adj)
            if c4 < best_c4:
                best_c4 = c4
                best_adj = adj.copy()
        elif num_edges > m:
            # Need to remove edges
            # Find edges that are in the most C4s and remove them
            adj_copy = adj.copy()
            edges = get_edges(adj_copy)
            to_remove = num_edges - m
            
            # Compute C4 contribution of each edge
            # An edge (u,v) is in a C4 if there exist a,b such that u~a, a~b, b~v, and a!=v, b!=u
            # This is equivalent to: a is a common neighbor of u and some neighbor of v... 
            # Actually, let's just compute it directly.
            
            def count_c4s_containing_edge(u, v, adj_mat):
                count = 0
                neighbors_u = np.where(adj_mat[u] == 1)[0]
                neighbors_v = np.where(adj_mat[v] == 1)[0]
                for a in neighbors_u:
                    if a == v:
                        continue
                    for b in neighbors_v:
                        if b == u or b == a:
                            continue
                        if adj_mat[a, b]:
                            count += 1
                return count
            
            # For each edge, compute how many C4s it's in
            edge_c4_counts = {}
            for (u, v) in edges:
                edge_c4_counts[(u, v)] = count_c4s_containing_edge(u, v, adj_copy)
            
            # Remove edges with highest C4 count
            for _ in range(to_remove):
                if time.time() > DEADLINE - 20:
                    break
                # Find edge with max C4 count
                max_edge = max(edge_c4_counts.items(), key=lambda x: x[1])
                u, v = max_edge[0]
                adj_copy[u, v] = 0
                adj_copy[v, u] = 0
                del edge_c4_counts[(u, v)]
                # Update C4 counts for remaining edges (this is expensive, so we skip for now)
            
            c4 = count_c4s(adj_copy)
            if c4 < best_c4:
                best_c4 = c4
                best_adj = adj_copy.copy()
    
    if best_adj is None:
        # Fallback: use form 0
        best_adj = build_adjacency(0)
    
    # Now perform local search to minimize C4 count
    # Strategy: try swapping edges (remove one, add another) to reduce C4 count
    
    def compute_c4_delta_remove_add(adj, remove_edge, add_edge):
        """
        Compute the change in C4 count when removing remove_edge and adding add_edge.
        This is an approximation; we compute the full C4 count before and after.
        """
        # For efficiency, we'll compute the C4 count directly
        pass
    
    # Local search: try to improve by swapping edges
    current_adj = best_adj.copy()
    current_c4 = count_c4s(current_adj)
    best_c4_found = current_c4
    best_adj_found = current_adj.copy()
    
    # Precompute common neighbors for efficiency
    def get_common_neighbors(adj_mat, u, v):
        return np.where(adj_mat[u] & adj_mat[v] == 1)[0]
    
    # Try random swaps
    random.seed(42)
    edges = get_edges(current_adj)
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_set = set(edges)
    
    num_swaps_attempted = 0
    max_swaps = 5000  # Limit the number of swaps
    
    while time.time() < DEADLINE - 30 and num_swaps_attempted < max_swaps:
        # Pick a random edge to remove
        if len(edges) == 0:
            break
        remove_idx = random.randint(0, len(edges) - 1)
        u, v = edges[remove_idx]
        
        # Pick a random non-edge to add
        available = [e for e in all_possible_edges if e not in edge_set]
        if not available:
            break
        add_idx = random.randint(0, len(available) - 1)
        w, x = available[add_idx]
        
        # Apply the swap
        current_adj[u, v] = 0
        current_adj[v, u] = 0
        current_adj[w, x] = 1
        current_adj[x, w] = 1
        
        new_c4 = count_c4s(current_adj)
        
        if new_c4 < best_c4_found:
            best_c4_found = new_c4
            best_adj_found = current_adj.copy()
        else:
            # Revert
            current_adj[u, v] = 1
            current_adj[v, u] = 1
            current_adj[w, x] = 0
            current_adj[x, w] = 0
        
        num_swaps_attempted += 1
    
    # Use the best adjacency found
    final_adj = best_adj_found
    final_edges = get_edges(final_adj)
    
    # Ensure we have exactly m edges
    if len(final_edges) != m:
        # This shouldn't happen, but just in case
        if len(final_edges) > m:
            final_edges = final_edges[:m]
        else:
            # Add random edges
            edge_set = set(final_edges)
            while len(final_edges) < m:
                u, v = random.choice(all_possible_edges)
                if (u, v) not in edge_set:
                    final_edges.append((u, v))
                    edge_set.add((u, v))
    
    return [list(e) for e in final_edges]
