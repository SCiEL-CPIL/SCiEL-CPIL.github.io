"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    # Winograd Fourier Transform Technique (WFT) over Z_2 x Z_7
    # For 2x4x7 matrix multiplication
    # n=2, m=4, p=7
    # We use the cyclic group Z_2 x Z_7 which has order 14
    # The idea is to embed the multiplication into a convolution over this group
    # and exploit the structure of the smaller dimension (n=2)
    
    n, m, p = DIMENSIONS
    
    # For WFT over Z_2 x Z_7:
    # The group has 14 elements: (a, b) where a in Z_2, b in Z_7
    # Characters are chi_{(u,v)}(a,b) = omega_2^{u*a} * omega_7^{v*b}
    # where omega_2 = -1, omega_7 = exp(2*pi*i/7)
    
    # For exact decomposition with integer/half-integer entries, we need to be careful.
    # The standard WFT gives complex entries. However, for Z_2 x Z_7, we can work
    # with the real structure since Z_2 contributes only +/-1.
    
    # Let's think about this differently. The tensor for matrix multiplication C = A*B
    # where A is 2x4, B is 4x7, C is 2x7.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Using WFT over Z_2 x Z_7:
    # We index A's rows by i in Z_2 (since n=2), and we need to handle m=4 and p=7.
    # Actually, let's reconsider the approach.
    
    # A simpler WFT approach: Use the fact that we can decompose the multiplication
    # using the structure of the cyclic groups.
    
    # For n=2, m=4, p=7:
    # Let's use the standard schoolbook as a baseline and see if we can do better.
    # The lower bound for matrix multiplication is max(n*m, m*p, p*n) = max(8, 28, 14) = 28.
    # The upper bound from schoolbook is n*m*p = 56.
    
    # Using WFT over Z_2 x Z_7 (order 14):
    # The rank would be related to the number of non-zero Fourier coefficients.
    
    # Let me try a different approach: use the fact that 4 divides into the structure
    # and 7 is prime.
    
    # Actually, let's use a known result: for matrix multiplication (n x m) * (m x p),
    # if we use WFT over Z_n x Z_p, the rank is n*p if m <= n and m <= p, or similar.
    
    # For n=2, m=4, p=7: m=4 is larger than n=2, so this doesn't directly apply.
    
    # Let's try a block decomposition approach.
    # Split the 4-dimensional middle into two 2-dimensional blocks.
    
    # A = [A1 | A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x7
    # C = A1*B1 + A2*B2
    
    # Each 2x2 * 2x7 multiplication can be done with rank 2*7 = 14 (schoolbook for that subproblem)
    # But we can use WFT for 2x2*2x7.
    
    # For 2x2*2x7: n=2, m=2, p=7
    # Using WFT over Z_2 x Z_7 (order 14):
    # The rank should be 14 if we can achieve it.
    
    # Let me construct the WFT decomposition for 2x2*2x7.
    
    def wft_2x2x7():
        """WFT decomposition for (2x2) * (2x7) -> (2x7)"""
        n_sub, m_sub, p_sub = 2, 2, 7
        
        # Group G = Z_2 x Z_7, order 14
        # Elements: (a, b) for a in {0,1}, b in {0,...,6}
        # Characters: chi_{(u,v)}(a,b) = (-1)^(u*a) * exp(2*pi*i*v*b/7)
        
        # For exact integer/half-integer decomposition, we need to avoid complex numbers.
        # This is tricky. Let me use a different approach.
        
        # Actually, the standard Winograd scheme for small matrices uses specific
        # integer-friendly decompositions.
        
        # For 2x2 * 2x2, the Winograd algorithm uses rank 5 (instead of 8).
        # For 2x2 * 2x7, let's try to extend this.
        
        # Alternative: use the fact that we can do 2x2*2x7 with rank 2*7 = 14
        # using a simple block approach, which is the same as schoolbook.
        
        # Let's just use schoolbook for the subproblems and see the total rank.
        # 2x2*2x7 schoolbook rank: 2*2*7 = 28
        # Two such blocks: 28 + 28 = 56, same as original schoolbook.
        
        # This doesn't help. Let me think more carefully.
        
        # The key insight for WFT: when one dimension is small (n=2), we can
        # use the structure to reduce the rank.
        
        # Let me try a direct construction using the group Z_2 x Z_7.
        
        # The multiplication tensor T_{i,j,k} = delta_{j, j'} for appropriate indices.
        # For C[i,k] = sum_j A[i,j] * B[j,k]
        
        # In the WFT framework, we embed A and B into functions on G = Z_2 x Z_7.
        
        # This is getting complex. Let me try a simpler improvement first.
        
        # Known result: for (n x m) * (m x p) with n=2, the rank is at most
        # 2*p + m - 1 or similar formulas from the literature.
        
        # Actually, let's use the following approach:
        # For n=2, we can write:
        # C[0,k] = sum_j A[0,j] * B[j,k]
        # C[1,k] = sum_j A[1,j] * B[j,k]
        
        # Let D[j,k] = A[0,j] * B[j,k] + A[1,j] * B[j,k] (sum)
        # Let E[j,k] = A[0,j] * B[j,k] - A[1,j] * B[j,k] (difference)
        
        # Then C[0,k] = (D[k] + E[k]) / 2, C[1,k] = (D[k] - E[k]) / 2
        # where D[k] = sum_j (A[0,j] + A[1,j]) * B[j,k]
        # and E[k] = sum_j (A[0,j] - A[1,j]) * B[j,k]
        
        # This doesn't directly reduce rank since we still need to compute
        # sums over j.
        
        # Let me just return the schoolbook decomposition for now, as a safe fallback.
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return wft_2x2x7()
# EVOLVE-BLOCK-END
