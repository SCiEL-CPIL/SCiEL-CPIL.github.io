import time
import sys

def construct():
    # Target n=26
    n = 26
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # colors[i] = color of edge i
    colors = [-1] * num_edges
    
    # Adjacency bitsets for each color
    # adj[c][u] is a set of neighbors of u in color c
    # Using Python sets or integers as bitsets?
    # n=26, so 26 bits. Integers are efficient.
    adj = [[0] * n for _ in range(5)]
    
    # Edge index map
    edge_idx = {}
    for k, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = k
        edge_idx[(j, i)] = k # Symmetric access
        
    def get_idx(i, j):
        return edge_idx[(i, j)] if i < j else edge_idx[(j, i)]

    def check_c4(u, v, c):
        # Check if adding edge (u,v) with color c creates a C4
        # We need to check if there is a path u-x-y-v in color c
        # x must be in adj[c][u]
        # y must be in adj[c][x] and adj[c][v]
        # Note: x and y must be distinct from u, v, and each other?
        # In a simple graph, x != y.
        # If x == y, we have u-x-v which is a path of length 2, not 3.
        # C4 requires distinct vertices u, x, y, v.
        # So x != y.
        
        # Iterate through neighbors x of u in color c
        # For each x, check if there is a neighbor y of x in color c that is also a neighbor of v in color c
        # And y != x (implied if x is not neighbor of itself)
        # And y != u, y != v (implied by simple graph if no self loops)
        
        # Bitset logic:
        # N_u = adj[c][u]
        # N_v = adj[c][v]
        # For each x in N_u:
        #   if (adj[c][x] & N_v) != 0:
        #       # There exists y in N_x and N_v
        #       # We must ensure y != x
        #       # Since x is in N_u, and graph is simple, x != u.
        #       # If y == x, then x is in N_v.
        #       # So if x is in N_v, and x is in N_u, we have u-x-v.
        #       # This is a path of length 2.
        #       # Does it form a C4 with the new edge (u,v)?
        #       # Cycle: u-x-v-u. Length 3. Not a C4.
        #       # So we need y != x.
        #       # If (adj[c][x] & N_v) has a bit set, let's find one.
        #       # If the only common neighbor is x, then no C4.
        #       # If there is another common neighbor y, then C4.
        
        N_u = adj[c][u]
        N_v = adj[c][v]
        
        # Iterate through set bits of N_u
        # To iterate bits efficiently in Python, we can use while loop
        temp = N_u
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            x = lsb.bit_length() - 1
            temp ^= lsb
            
            # Check common neighbors of x and v
            common = adj[c][x] & N_v
            if common:
                # Check if common contains only x
                # If common == (1 << x), then the only common neighbor is x.
                # This corresponds to path u-x-v.
                # We need y != x.
                if common != (1 << x):
                    return True
                # If common has other bits, return True
                # Actually, if common has any bit other than x, it's a C4.
                # If common is 0, no.
                # If common is (1<<x), no C4 (only triangle u-x-v).
                # Wait, if common has bit y where y != x, then u-x-y-v is a path.
                # So if (common & ~(1 << x)) != 0:
                #     return True
            # Optimization: if common has more than 1 bit, or 1 bit not x, return True.
            # My check: if common != 0 and common != (1 << x): return True.
            # But what if common has bit x AND bit y?
            # Then common != (1 << x) is True. Correct.
            # What if common has bit y only?
            # Then common != 0 and common != (1 << x) (since y!=x). Correct.
            # So the condition `if common and common != (1 << x): return True` is correct.
            
            # Wait, I need to be careful.
            # If common is (1 << x), it means x is a neighbor of v.
            # This forms triangle u-x-v.
            # Adding u-v creates a triangle, not C4.
            # So we ignore x if it's the only common neighbor.
            
            # If common has any other bit, we found y.
            # So:
            if common and (common ^ (1 << x)) != 0:
                return True
            # Or simply:
            # if common:
            #    if common != (1 << x): return True
            
        return False

    # Backtracking
    start_time = time.time()
    
    def backtrack(k):
        if k == num_edges:
            return True
        if time.time() - start_time > 110:
            return False
            
        u, v = edges[k]
        
        # Try colors 0..4
        # Heuristic: try colors that are most constrained? Or just 0..4
        for c in range(5):
            if not check_c4(u, v, c):
                # Add edge
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                colors[k] = c
                
                if backtrack(k + 1):
                    return True
                
                # Undo
                adj[c][u] ^= (1 << v)
                adj[c][v] ^= (1 << u)
                colors[k] = -1
        return False

    if backtrack(0):
        return {'n': n, 'colors': colors}
    
    # Fallback: try n=25? Or return invalid?
    # The task says maximize n. If we fail 26, we should try smaller.
    # But backtracking for 26 should work if the instance is satisfiable.
    # If it times out, we might need a heuristic.
    # For now, if it fails, return a known small valid solution?
    # The current program returns n=11 if it fails.
    # I will return n=11 to be safe, or try to repair.
    
    n_seed = 11
    return {'n': n_seed, 'colors': [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]}
