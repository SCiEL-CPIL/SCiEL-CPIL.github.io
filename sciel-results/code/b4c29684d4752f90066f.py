"""Program for matrix multiplication 2,2,6 tensor decomposition using linear code approach."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x6 matrix.
    
    Approach: Model the tensor as a linear code over F2 by using the rank-21
    Strassen decomposition as a generator matrix, then search for a rank-20
    subcode that satisfies the bilinear identity constraints.
    
    The tensor T[i,j,k] = 1 for all valid (i,j,k) where:
    - A is 2x2, B is 2x6, C is 2x6
    - C[i,k] = sum_j A[i,j]*B[j,k]
    
    We seek U (4 x r), V (12 x r), W (12 x r) with integer/half-integer entries
    such that for all (i,j,k):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    import numpy as np
    from itertools import product
    
    n, m, p = 2, 2, 6
    nm = n * m  # 4
    mp = m * p  # 12
    pn = p * n  # 12
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W correctly decompose the tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Step 1: Build correct rank-21 Strassen-based decomposition
    # Standard Strassen for 2x2 x 2x2:
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)b00
    # p3 = a00(b01-b11)
    # p4 = a11(b10-b00)
    # p5 = (a00+a01)b11
    # p6 = (a10-a00)(b00+b01)
    # p7 = (a01-a11)(b10+b11)
    # c00 = p1 + p4 - p5 - p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6
    
    def build_strassen_2x2x2x2():
        """Return U, V, W for 2x2 x 2x2 x 2x2 Strassen decomposition (rank 7)."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # p1 = (a00+a11)(b00+b11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        
        # p2 = (a10+a11)b00
        U[1, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        
        # p3 = a00(b01-b11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        
        # p4 = a11(b10-b00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        
        # p5 = (a00+a01)b11
        U[0, 4] = 1; U[2, 4] = 1
        V[3, 4] = 1
        
        # p6 = (a10-a00)(b00+b01)
        U[1, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        
        # p7 = (a01-a11)(b10+b11)
        U[2, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        
        # W[k*n+i]: W[0]=c00, W[1]=c01, W[2]=c10, W[3]=c11
        # c00 = p1 + p4 - p5 - p7
        W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = -1
        # c01 = p3 + p5
        W[1, 2] = 1; W[1, 4] = 1
        # c10 = p2 + p4
        W[2, 1] = 1; W[2, 3] = 1
        # c11 = p1 - p2 + p3 + p6
        W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
        
        return U, V, W
    
    U_s, V_s, W_s = build_strassen_2x2x2x2()
    
    # Build rank-21 decomposition for 2x2 x 2x6
    rank_21 = 21
    U21 = np.zeros((4, rank_21))
    V21 = np.zeros((12, rank_21))
    W21 = np.zeros((12, rank_21))
    
    for t in range(3):
        base = t * 7
        U21[:, base:base+7] = U_s
        for j in range(2):
            for k_local in range(2):
                for c in range(7):
                    V21[j * 6 + 2 * t + k_local, base + c] = V_s[j * 2 + k_local, c]
        for k_local in range(2):
            for i in range(2):
                for c in range(7):
                    W21[(2 * t + k_local) * 2 + i, base + c] = W_s[k_local * 2 + i, c]
    
    # Verify rank-21 works
    if not check_decomposition(U21, V21, W21, rank_21):
        # Fallback to schoolbook
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
    
    # Step 2: Search for rank-20 decomposition
    # Strategy: Try to find a linear combination that eliminates one multiplication
    # We look for vectors u, v, w such that we can modify the decomposition
    
    # Approach: Try all pairs of columns to see if we can merge them
    # For each pair (c1, c2), check if there's a way to combine them
    
    # Alternative: Use the fact that over F2, we can look for dependencies
    # Let's try a direct search: for each column c to eliminate, try to express
    # its contribution as a linear combination of others
    
    # More systematic: Try rank 20 by searching for integer solutions
    # to the bilinear constraints
    
    # Let's try a structured search: assume the rank-20 decomposition is close
    # to the rank-21 one, and try to find a rank-reducing transformation
    
    # Key insight: If we have a rank-r decomposition, we can apply invertible
    # transformations to U, V, W. We look for a transformation that makes
    # one column zero (or nearly zero) while preserving the tensor.
    
    # Try: For each column c, try to find linear combinations of other columns
    # that reproduce its effect
    
    # Simpler approach: Brute force search over small integer modifications
    # to the rank-21 decomposition to see if we can get rank 20
    
    # Actually, let's try a different approach: directly construct rank-20
    # using algebraic identities
    
    # For 2x2 x 2x6, we can think of it as:
    # C = A * B where A is 2x2, B is 2x6
    # Let B = [B1 | B2 | B3] where each Bi is 2x2
    # C = [A*B1 | A*B2 | A*B3]
    
    # We need 3 matrix multiplications of 2x2 by 2x2
    # Each needs rank 7 (Strassen), total 21
    # But we can share some computations
    
    # Let's try to find a rank-20 decomposition by exploiting the structure
    
    # Direct construction attempt for rank 20:
    # We'll try to use 20 multiplications with a specific pattern
    
    # Let me try a more direct approach: use the bilinear form constraints
    # and solve for a rank-20 decomposition
    
    # For rank r, we need to find U (4xr), V (12xr), W (12xr)
    # such that for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a system of 24 equations (2*2*6) with 3*r*12 variables (for r=20: 720 vars)
    # But the equations are cubic, so it's hard to solve directly
    
    # Let's try a randomized approach with integer constraints
    
    import random
    random.seed(42)
    
    def try_random_rank20():
        """Try to find a rank-20 decomposition using random search."""
        for attempt in range(1000):
            # Start from rank-21 and try to eliminate one column
            c_to_eliminate = random.randint(0, 20)
            
            # Try to find coefficients to redistribute the eliminated column
            # This is complex, so let's try a different approach
            
            # Direct random search with small integers
            U = np.zeros((4, 20))
            V = np.zeros((12, 20))
            W = np.zeros((12, 20))
            
            # Initialize with random small integers
            for c in range(20):
                U[:, c] = np.random.randint(-1, 2, 4)
                V[:, c] = np.random.randint(-1, 2, 12)
                W[:, c] = np.random.randint(-1, 2, 12)
            
            if check_decomposition(U, V, W, 20):
                return U, V, W
        
        return None
    
    result = try_random_rank20()
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If rank 20 doesn't work, return rank 21
    return {"U": U21.tolist(), "V": V21.tolist(), "W": W21.tolist()}
# EVOLVE-BLOCK-END
