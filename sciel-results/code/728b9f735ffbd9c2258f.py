import time
import random
import sys

def construct():
    n = 16
    deadline = time.time() + 110
    
    # Helper: Check if adding edge (u,v) creates a triangle
    def creates_triangle(adj, u, v):
        # Check common neighbors
        common = adj[u] & adj[v]
        return common != 0

    # Helper: Fast 3-colorability check
    def is_3_colorable(adj):
        color = [-1] * n
        # Precompute degree order? For n=16, static order with bitmask pruning is usually fast enough.
        # However, dynamic ordering (MRV - Minimum Remaining Values) is much faster for hard instances.
        # Let's use a simple recursive backtracking with bitmasks.
        
        def backtrack(idx):
            if idx == n:
                return True
            
            u = idx
            # Calculate available colors
            used_colors = 0
            m = adj[u]
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                if color[v] != -1:
                    used_colors |= (1 << color[v])
            
            avail = 7 ^ used_colors # 0b111
            while avail:
                lsb = avail & -avail
                c = lsb.bit_length() - 1
                avail ^= lsb
                
                color[u] = c
                if backtrack(idx + 1):
                    return True
                color[u] = -1
            
            return False
        
        return backtrack(0)

    def get_edge_count(adj):
        count = 0
        for i in range(n):
            count += adj[i].bit_count()
        return count // 2

    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    def make_initial_graph():
        # Create a Grötzsch graph (11 vertices) and 5 isolated vertices
        # Grötzsch graph is triangle-free and 4-chromatic
        adj = [0] * n
        # Cycle 0-1-2-3-4-0
        cycle = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        for u, v in cycle:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Connect to 5-9
        # Standard Grötzsch construction:
        # 5 connected to 0,1; 6 to 1,2; 7 to 2,3; 8 to 3,4; 9 to 4,0
        # This creates 5-cycles 0-1-5, etc? No, standard is:
        # Vertices 0-4 form C5.
        # Vertices 5-9:
        # 5 connected to 0, 1
        # 6 connected to 1, 2
        # 7 connected to 2, 3
        # 8 connected to 3, 4
        # 9 connected to 4, 0
        # Wait, this is not the standard Grötzsch.
        # Standard Grötzsch:
        # C5: 0,1,2,3,4
        # 5 connected to 0,1
        # 6 connected to 1,2
        # 7 connected to 2,3
        # 8 connected to 3,4
        # 9 connected to 4,0
        # 10 connected to 5,6,7,8,9
        # Let's verify triangle freeness.
        # 10-5-0-4-9-10? 10-5 and 10-9. 5-0, 9-0. No triangle 10-5-9.
        # 10-5-1-6-10? 10-5, 10-6. 5-1, 6-1. No triangle.
        # This graph is triangle-free.
        
        adj[5] |= (1 << 0) | (1 << 1)
        adj[0] |= (1 << 5)
        adj[1] |= (1 << 5)
        
        adj[6] |= (1 << 1) | (1 << 2)
        adj[1] |= (1 << 6)
        adj[2] |= (1 << 6)
        
        adj[7] |= (1 << 2) | (1 << 3)
        adj[2] |= (1 << 7)
        adj[3] |= (1 << 7)
        
        adj[8] |= (1 << 3) | (1 << 4)
        adj[3] |= (1 << 8)
        adj[4] |= (1 << 8)
        
        adj[9] |= (1 << 4) | (1 << 0)
        adj[4] |= (1 << 9)
        adj[0] |= (1 << 9)
        
        adj[10] |= (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9)
        adj[5] |= (1 << 10)
        adj[6] |= (1 << 10)
        adj[7] |= (1 << 10)
        adj[8] |= (1 << 10)
        adj[9] |= (1 << 10)
        
        return adj

    # Initialize
    adj = make_initial_graph()
    
    # Verify initial graph is valid (TF and not 3-colorable)
    # It should be.
    if is_3_colorable(adj):
        # Fallback: Random restart
        for _ in range(100):
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        if not creates_triangle(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
            if not is_3_colorable(adj):
                break
    
    best_adj = adj[:]
    best_count = get_edge_count(adj)
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.995
    
    while time.time() < deadline:
        if T < 0.01:
            T = 0.01
            
        # Propose move
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
            
        if adj[u] & (1 << v):
            # Remove edge
            new_adj = adj[:]
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            delta_e = -1
        else:
            # Add edge
            if creates_triangle(adj, u, v):
                continue
            new_adj = adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            delta_e = 1
        
        # Check validity of new_adj
        # 1. Triangle free: Ensured by move logic for add. Remove always keeps TF.
        # 2. Not 3-colorable
        if is_3_colorable(new_adj):
            # Invalid state. Reject.
            # Or accept with very low prob to jump? 
            # Let's just reject to stay in valid space.
            continue
            
        # Acceptance
        if delta_e > 0 or random.random() < (2.71828 ** (delta_e / T)):
            adj = new_adj
            current_count = get_edge_count(adj)
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
                
        T *= cooling
        
    return get_edges_from_adj(best_adj)

if __name__ == "__main__":
    # Quick test
    edges = construct()
    print(f"Found {len(edges)} edges")
