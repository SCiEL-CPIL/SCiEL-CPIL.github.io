<<<<<<< SEARCH
    # Bitset-based adjacency for fast C6 detection
    # adj_mask[i] is a 40-bit integer where bit j is 1 if (i,j) is an edge
    
    def would_create_c6(u, v, adj_mask):
        """Check if adding edge (u,v) creates a C6 using bitset operations.
        
        A C6 exists if there's a path u-a-b-c-d-v of length 5.
        Equivalently: there exists a path of length 4 from u to some node d
        such that d is adjacent to v, and the path is simple.
        
        Using bitsets:
        - N1 = neighbors of u (excluding v)
        - N2 = union of neighbors of nodes in N1 (excluding u, v, and N1)
        - N3 = union of neighbors of nodes in N2 (excluding u, v, N1, N2)
        - N4 = union of neighbors of nodes in N3 (excluding u, v, N1, N2, N3)
        - If N4 intersects with neighbors of v (excluding u, N1, N2, N3), then C6 exists.
        """
        # Get neighbors of u as a bitset
        mask_u = adj_mask[u]
        mask_v = adj_mask[v]
        
        # Remove u and v from consideration
        mask_u &= ~(1 << u)
        mask_v &= ~(1 << v)
        
        # N1: neighbors of u (excluding v)
        n1 = mask_u & ~(1 << v)
        if n1 == 0:
            return False
        
        # N2: neighbors of nodes in N1, excluding u, v, and N1
        n2 = 0
        temp = n1
        while temp:
            # Get lowest set bit
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            temp ^= lsb
            # Add neighbors of idx, excluding u, v, and N1
            n2 |= (adj_mask[idx] & ~(1 << u) & ~(1 << v) & ~n1)
        
        if n2 == 0:
            return False
        
        # N3: neighbors of nodes in N2, excluding u, v, N1, and N2
        n3 = 0
        temp = n2
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            temp ^= lsb
            n3 |= (adj_mask[idx] & ~(1 << u) & ~(1 << v) & ~n1 & ~n2)
        
        if n3 == 0:
            return False
        
        # N4: neighbors of nodes in N3, excluding u, v, N1, N2, and N3
        n4 = 0
        temp = n3
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            temp ^= lsb
            n4 |= (adj_mask[idx] & ~(1 << u) & ~(1 << v) & ~n1 & ~n2 & ~n3)
        
        if n4 == 0:
            return False
        
        # Check if N4 intersects with neighbors of v (excluding u, N1, N2, N3)
        v_neighbors = mask_v & ~(1 << u) & ~n1 & ~n2 & ~n3
        if n4 & v_neighbors:
            return True
        
        return False
=======
    # Bitset-based adjacency for fast C6 detection
    # adj_mask[i] is a 40-bit integer where bit j is 1 if (i,j) is an edge
    
    # Precompute masks for each vertex
    single_bit = [1 << i for i in range(n)]
    
    def would_create_c6(u, v, adj_mask):
        """Check if adding edge (u,v) creates a C6 using bitset operations.
        
        A C6 exists if there's a path u-a-b-c-d-v of length 5.
        Using bitsets:
        - N1 = neighbors of u (excluding v)
        - N2 = union of neighbors of nodes in N1 (excluding u, v, and N1)
        - N3 = union of neighbors of nodes in N2 (excluding u, v, N1, N2)
        - N4 = union of neighbors of nodes in N3 (excluding u, v, N1, N2, N3)
        - If N4 intersects with neighbors of v (excluding u, N1, N2, N3), then C6 exists.
        """
        mask_u = adj_mask[u]
        mask_v = adj_mask[v]
        
        # N1: neighbors of u (excluding v and u itself)
        n1 = mask_u & ~single_bit[v] & ~single_bit[u]
        if n1 == 0:
            return False
        
        # N2: neighbors of nodes in N1, excluding u, v, and N1
        n2 = 0
        temp = n1
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            temp ^= lsb
            n2 |= (adj_mask[idx] & ~single_bit[u] & ~single_bit[v] & ~n1)
        
        if n2 == 0:
            return False
        
        # N3: neighbors of nodes in N2, excluding u, v, N1, and N2
        n3 = 0
        temp = n2
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            temp ^= lsb
            n3 |= (adj_mask[idx] & ~single_bit[u] & ~single_bit[v] & ~n1 & ~n2)
        
        if n3 == 0:
            return False
        
        # N4: neighbors of nodes in N3, excluding u, v, N1, N2, and N3
        n4 = 0
        temp = n3
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            temp ^= lsb
            n4 |= (adj_mask[idx] & ~single_bit[u] & ~single_bit[v] & ~n1 & ~n2 & ~n3)
        
        if n4 == 0:
            return False
        
        # Check if N4 intersects with neighbors of v (excluding u, N1, N2, N3)
        v_neighbors = mask_v & ~single_bit[u] & ~n1 & ~n2 & ~n3
        return bool(n4 & v_neighbors)
>>>>>>> REPLACE

