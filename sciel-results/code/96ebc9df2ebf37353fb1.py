        def generate_strassen(n):
            # Returns U, V, W for n x n x n
            if n == 1:
                return [[1]], [[1]], [[1]]
            
            k = n // 2
            U_sub, V_sub, W_sub = generate_strassen(k)
            
            # U_sub is k*k x Rank_sub
            # We need to form U for n*n x Rank_total
            # Rank_total = 7 * Rank_sub
            
            U = [[0] * (7 * len(U_sub[0])) for _ in range(n * n)]
            V = [[0] * (7 * len(V_sub[0])) for _ in range(n * n)]
            W = [[0] * (7 * len(W_sub[0])) for _ in range(n * n)]
            
            # The 7 products:
            # P1 = (A00 + A11) (B00 + B11)
            # P2 = (A10 + A11) B00
            # P3 = A00 (B01 - B11)
            # P4 = A11 (B10 - B00)
            # P5 = (A00 + A01) B11
            # P6 = (A10 - A00) (B00 + B01)
            # P7 = (A01 - A11) (B10 + B11)
            
            # We need to define the linear combinations for A and B blocks.
            # A block indices:
            # A00: rows 0..k-1, cols 0..k-1
            # A01: rows 0..k-1, cols k..2k-1
            # A10: rows k..2k-1, cols 0..k-1
            # A11: rows k..2k-1, cols k..2k-1
            
            # For each of the 7 products, we iterate through the rank of the sub-decomposition.
            for prod_idx in range(7):
                # Define the linear forms for the A-blocks and B-blocks involved
                # We need to add contributions to U, V, W
                
                # This logic is tricky to implement generically without a symbolic algebra system
                # or very careful index tracking.
                pass
        