    def construct(k, g):
        # For k=3, g=16
        # Search for a circulant graph C_n(1, a, b)
        # n must be such that 3-regular.
        # Girth >= g.
        
        # Moore bound for n(3, 16) is 512.
        # Known cage is 936.
        
        # We will try n from 512 to 1500.
        # For each n, try small a, b.
        
        def get_girth(n, edges):
            # BFS from each node to find shortest cycle
            # This is O(n * (n+m)) which might be slow for n=1000.
            # 1000 * 3000 = 3,000,000 ops. Fast.
            pass
        
        # Implementation
    