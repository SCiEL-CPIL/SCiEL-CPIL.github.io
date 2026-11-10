import math
import time

def construct():
    N = 512
    limit = 512
    
    # 1. Build the conflict graph complement (valid pairs graph)
    # adj[i] is a bitmask of vertices j such that lcm(i, j) <= 512
    adj = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            # Calculate lcm(i, j)
            # lcm(i, j) = i * j // gcd(i, j)
            g = math.gcd(i, j)
            l = (i // g) * j
            if l <= limit:
                # Set bit j in adj[i] and bit i in adj[j]
                # We use 0-indexed bits for the mask, so vertex v corresponds to bit (v-1)
                adj[i] |= (1 << (j - 1))
                adj[j] |= (1 << (i - 1))
                
    # Initial candidate set: all vertices 1..N
    # Bitmask for all vertices
    all_vertices = (1 << N) - 1
    
    best_size = 0
    best_set_mask = 0
    
    # Tomita Algorithm for Maximum Clique on the graph defined by adj
    # We represent the current candidate set P as a bitmask.
    # We need to order the vertices in P for the search.
    # A common optimization is to order by degree in the subgraph induced by P.
    
    # To make the recursion efficient, we can use a simple recursive function.
    # Since N=512, recursion depth is at most 512, which is safe.
    
    # We will use a helper to get the list of vertices from a mask,
    # sorted by their degree within the mask.
    # However, sorting at every step is expensive. 
    # Instead, we can use a static ordering based on global degree, 
    # or simply iterate and rely on the bound `len(P)`.
    # For better performance, we can pre-compute a global degree order 
    # and filter the mask. But dynamic degree is better for pruning.
    
    # Let's try a simpler approach first: iterate through bits in P.
    # To improve pruning, we can use a greedy coloring bound, 
    # but for N=512, the simple `len(P)` bound might be sufficient 
    # if the graph is sparse enough (which it is, for valid pairs).
    
    # Actually, the "valid pairs" graph (complement of conflict) is what we want to find a clique in.
    # Let's check the density.
    # For i > 256, lcm(i, j) > 512 for most j.
    # The graph of valid pairs is likely sparse.
    
    def expand(P_mask, current_size, current_set_mask):
        nonlocal best_size, best_set_mask
        
        if current_size > best_size:
            best_size = current_size
            best_set_mask = current_set_mask
            
        if P_mask == 0:
            return
        
        # Upper bound: current_size + number of candidates
        if current_size + P_mask.bit_count() <= best_size:
            return
            
        # To improve pruning, we can order the candidates.
        # We extract the list of vertices in P_mask.
        # Sorting by degree in the subgraph induced by P_mask is best.
        
        # Optimization: If the number of candidates is small, we can brute force?
        # No, let's stick to the standard loop.
        
        # We iterate through the vertices in P_mask.
        # We remove the first vertex from P_mask for the next iterations in the loop?
        # Standard Tomita:
        # while P is not empty:
        #   u = first vertex in P
        #   P = P \ {u}
        #   if current_size + |P| <= best: return
        #   expand(P & N(u), current_size + 1, current_set | {u})
        
        # To get the "first" vertex, we can just take the lowest set bit.
        # However, the order matters for efficiency. 
        # Using the lowest bit index is fast.
        
        while P_mask:
            # Get the lowest set bit (vertex with smallest index)
            # This corresponds to the smallest integer in the set.
            # Is this a good order? Not necessarily. 
            # High degree vertices first is better.
            
            # Let's try to pick a vertex with high degree in P_mask.
            # This requires scanning. For N=512, scanning 512 bits is cheap.
            # But doing it at every node might be slow.
            
            # Alternative: Pre-sort vertices by global degree descending.
            # Then iterate through this static order, checking if the vertex is in P_mask.
            
            # Let's implement the static order approach.
            # Pre-compute order once.
            # We need to access this inside the recursion.
            
            # Let's just use the lowest bit for now and see if it's fast enough.
            # Given the constraints, the search space might be small.
            
            lsb = P_mask & -P_mask
            u_idx = lsb.bit_length() - 1
            u_val = u_idx + 1
            
            P_mask ^= lsb # Remove u from P
            
            if current_size + P_mask.bit_count() <= best_size:
                return
                
            # Recurse with neighbors of u in the valid graph
            new_P = P_mask & adj[u_val]
            expand(new_P, current_size + 1, current_set_mask | lsb)
            
    # Initialize
    expand(all_vertices, 0, 0)
    
    # Convert best_set_mask back to list of integers
    result = []
    for i in range(N):
        if (best_set_mask >> i) & 1:
            result.append(i + 1)
            
    # Ensure non-empty
    if not result:
        return [1]
        
    return result

# To improve performance, let's add a global ordering based on degree.
# We will modify the expand function to use a pre-sorted list of vertices.

def construct():
    N = 512
    limit = 512
    
    # Build adjacency bitmask for valid pairs (complement of conflict)
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            l = (i // g) * j
            if l <= limit:
                adj[i] |= (1 << (j - 1))
                adj[j] |= (1 << (i - 1))
                
    # Pre-compute global degree for each vertex
    degrees = [0] * (N + 1)
    for i in range(1, N + 1):
        degrees[i] = adj[i].bit_count()
        
    # Create a static order of vertices sorted by degree descending
    # This helps in finding large cliques earlier
    order = list(range(1, N + 1))
    order.sort(key=lambda x: degrees[x], reverse=True)
    
    # Map vertex value to its position in the order? No, we just iterate through order.
    
    all_vertices = (1 << N) - 1
    best_size = 0
    best_set_mask = 0
    
    def expand(P_mask, current_size, current_set_mask):
        nonlocal best_size, best_set_mask
        
        if current_size > best_size:
            best_size = current_size
            best_set_mask = current_set_mask
            
        if P_mask == 0:
            return
            
        if current_size + P_mask.bit_count() <= best_size:
            return
            
        # Iterate through vertices in the static order
        # We need to check if the vertex is in P_mask
        for u_val in order:
            u_bit = 1 << (u_val - 1)
            if not (P_mask & u_bit):
                continue
                
            # Remove u from P for subsequent iterations in this level
            P_mask ^= u_bit
            
            if current_size + P_mask.bit_count() <= best_size:
                return
                
            new_P = P_mask & adj[u_val]
            expand(new_P, current_size + 1, current_set_mask | u_bit)
            
    expand(all_vertices, 0, 0)
    
    result = []
    for i in range(N):
        if (best_set_mask >> i) & 1:
            result.append(i + 1)
            
    if not result:
        return [1]
        
    return result
