    import time

    def construct():
        # Build G0 = K2
        n = 2
        adj = [0] * n
        adj[0] |= (1 << 1)
        adj[1] |= (1 << 0)
        
        # Apply Mycielski 4 times
        for _ in range(4):
            new_n = 2 * n + 1
            new_adj = [0] * new_n
            
            # 1. Copy original edges
            for i in range(n):
                new_adj[i] = adj[i]
            
            # 2. Add copy edges
            # u' (index n+u) connects to neighbors of u
            for u in range(n):
                u_prime = n + u
                neighbors = adj[u]
                # Add edges between u_prime and neighbors
                while neighbors:
                    lsb = neighbors & -neighbors
                    v = lsb.bit_length() - 1
                    new_adj[u_prime] |= (1 << v)
                    new_adj[v] |= (1 << u_prime)
                    neighbors ^= lsb
                    
            # 3. Add root edges
            root = 2 * n
            for u in range(n):
                u_prime = n + u
                new_adj[root] |= (1 << u_prime)
                new_adj[u_prime] |= (1 << root)
                
            adj = new_adj
            n = new_n
            
        # Extract edges
        edges = []
        for i in range(n):
            a = adj[i]
            while a:
                lsb = a & -a
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                a ^= lsb
                
        return {'n': n, 'edges': edges}
    