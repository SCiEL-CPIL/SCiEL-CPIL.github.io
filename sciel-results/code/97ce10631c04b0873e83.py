import math
import sys

def construct():
    n_max = 256
    
    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    # 1. Identify valid nodes
    valid_nodes = []
    for i in range(1, n_max + 1):
        if not is_perfect_square(2 * i):
            valid_nodes.append(i)
            
    N = len(valid_nodes)
    
    # 2. Build Complement Graph Adjacency Masks
    # adj[i] is a bitmask of j such that valid_nodes[i] + valid_nodes[j] is NOT a square
    adj = [0] * N
    for i in range(N):
        vi = valid_nodes[i]
        mask = 0
        for j in range(N):
            if i == j:
                continue
            vj = valid_nodes[j]
            if not is_perfect_square(vi + vj):
                mask |= (1 << j)
        adj[i] = mask
        
    # 3. Branch and Bound with Coloring
    
    best_clique_mask = 0
    best_size = 0
    
    sys.setrecursionlimit(10000)
    
    # Greedy Coloring for bound
    # We color the subgraph induced by P_mask.
    # We return the maximum color used.
    def get_color_bound(P_mask):
        if P_mask == 0:
            return 0
            
        colors = {}
        # To get a tighter bound, we should order vertices.
        # Let's just iterate through set bits.
        # For better bounds, we might want to pick vertices with high degree in P first?
        # Simple greedy:
        
        # Extract list of vertices
        vertices = []
        temp = P_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            vertices.append(idx)
            temp ^= lsb
            
        # Sort by degree in P (descending) to potentially get tighter coloring?
        # Or just keep index order.
        # Let's try index order for simplicity first, then optimize if too slow.
        
        for v in vertices:
            # Find used colors by neighbors of v in 'vertices' that are already colored
            used = set()
            # Neighbors of v in P
            neigh_mask = adj[v] & P_mask
            # Check which neighbors are in 'vertices' and already colored
            # Since we are iterating, 'colors' contains previously processed vertices.
            for u in vertices:
                if u in colors:
                    if (neigh_mask >> u) & 1:
                        used.add(colors[u])
                        
            # Assign smallest available color
            c = 1
            while c in used:
                c += 1
            colors[v] = c
            
        return max(colors.values()) if colors else 0

    def expand(R_size, R_mask, P_mask, X_mask):
        nonlocal best_clique_mask, best_size
        
        if P_mask == 0:
            if R_size > best_size:
                best_size = R_size
                best_clique_mask = R_mask
            return
            
        # Bound
        max_color = get_color_bound(P_mask)
        if R_size + max_color <= best_size:
            return
            
        # Iterate through vertices in P
        # We can iterate in a specific order.
        # Let's iterate through all bits in P_mask.
        
        temp = P_mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            temp ^= lsb
            
            # Recurse
            new_R_mask = R_mask | (1 << v)
            new_P_mask = P_mask & adj[v]
            new_X_mask = X_mask & adj[v]
            
            expand(R_size + 1, new_R_mask, new_P_mask, new_X_mask)
            
            # Move v from P to X
            P_mask ^= (1 << v)
            X_mask |= (1 << v)
            
    # Initial call
    initial_P = (1 << N) - 1
    expand(0, 0, initial_P, 0)
    
    # Map back to original values
    result = []
    temp = best_clique_mask
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        result.append(valid_nodes[idx])
        temp ^= lsb
        
    # Verification
    def is_valid(s):
        for i in range(len(s)):
            for j in range(i, len(s)):
                if is_perfect_square(s[i] + s[j]):
                    return False
        return True
        
    if not is_valid(result):
        result = []
        
    return sorted(result)
