    def build_haar_basis(n):
        # n is power of 2
        if n == 1:
            return np.ones((1,1))
        
        # Split into two halves
        h = n // 2
        # Recursively build basis for half
        B_half = build_haar_basis(h)
        
        # The basis for n consists of:
        # 1. The scaling vector: [B_half, B_half] / sqrt(2) ? No.
        # The first vector is the average.
        # The rest are differences.
        
        B = np.zeros((n, n))
        
        # Vector 0: Average of the two halves
        # It should be orthogonal to the difference vectors.
        # Standard Haar:
        # c_0 = (x_0 + x_1)/sqrt(2)
        # d_0 = (x_0 - x_1)/sqrt(2)
        
        # For the basis matrix:
        # Row 0 of B is the scaling function.
        # Rows 1..n/2 are the detail functions of the left half.
        # Rows n/2..n-1 are the detail functions of the right half.
        
        # Actually, let's just use the property that H is orthogonal.
        # I can generate it recursively.
        
        # Vector 0: 1/sqrt(n) everywhere
        B[0, :] = 1.0 / np.sqrt(n)
        
        # The remaining n-1 vectors can be constructed.
        # Left half details:
        # Take B_half, put it in left half, 0 in right half, scale by sqrt(2)?
        # No, B_half is orthonormal for size h.
        # The vectors in B_half are orthogonal to the all-ones vector of size h.
        # So [B_half, 0] is orthogonal to [1, 1] (scaled).
        
        # Let's do it properly:
        # v_0 = [1, 1, ..., 1] / sqrt(n)
        
        # For k=1 to n-1:
        # If k < n/2:
        #   v_k = [B_half[k-1], 0] / sqrt(2)  ? No.
        #   The vectors in B_half are unit norm.
        #   [B_half[k-1], 0] has norm 1.
        #   We need them to be orthogonal to v_0.
        #   B_half[k-1] sums to 0 (since k-1 >= 1).
        #   So sum of [B_half[k-1], 0] is 0.
        #   So it is orthogonal to v_0.
        #   And they are orthogonal to each other.
        
        # So:
        for k in range(1, n):
            if k < n//2:
                # Left half detail
                B[k, :h] = B_half[k-1]
                # Right half is 0
            else:
                # Right half detail
                B[k, h:] = B_half[k-1-(n//2)]
                # Left half is 0
                
        # Wait, the norm of B[k] is 1.
        # But are they orthogonal to v_0?
        # v_0 is all 1/sqrt(n).
        # Dot product of B[k] and v_0 is sum(B[k])/sqrt(n).
        # sum(B[k]) = 0 for k>0.
        # So yes.
        
        return B
