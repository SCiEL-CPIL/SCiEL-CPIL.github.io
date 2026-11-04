"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    deadline = time.time() + 120
    
    # Schoolbook baseline
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to build a better decomposition using Strassen for 2x2 blocks
    # For the middle dimension m=4, group as (0,1) and (2,3)
    # For each group, use Strassen's 7-term decomposition for the 2x2x2 sub-problem
    
    # The idea: for rows i in {0,1} and cols k in some set, 
    # we can use Strassen for the 2x2 middle block.
    
    # Let me try a concrete construction:
    # Split the problem into:
    # - 2x2x5 sub-problem (rows 0-1, all cols)
    # - 1x4x5 sub-problem (row 2, all cols)
    
    # For the 2x2x5: use Strassen for each pair of middle indices
    # For the 1x4x5: schoolbook (rank 20)
    
    # For 2x2x5: split middle dim into pairs (0,1) and (2,3)
    # Each 2x2x2 sub-problem has rank 7 (Strassen)
    # But we have 5 cols, so we need to handle them separately
    
    # Actually, for 2x2xp, the rank is 7*ceil(p/2) if we pair cols?
    # No, that's not right either.
    
    # Let me just try a direct construction for 2x2x5.
    # The tensor for 2x2x5 has 2*2*5 = 20 entries in schoolbook.
    # Can we do better?
    
    # For 2x2x2: rank 7
    # For 2x2x3: rank 11 (best known)
    # For 2x2x4: rank 14? (7*2 if we split into two 2x2x2)
    # For 2x2x5: rank 18? (7*2 + 4 for the last col?)
    
    # Let me construct 2x2x5 using Strassen for cols 0-1 and 2-3, and schoolbook for col 4.
    # That gives 7 + 7 + 2 = 16 for the 2x2x5 part.
    # Plus 1x4x5 = 20 for the row 2 part.
    # Total: 16 + 20 = 36.
    
    # Let me verify this construction.
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition."""
        # Strassen's 7 products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        U = [
            [1, 1, 0, 0, 0, 0, 0],  # A00
            [0, 0, 0, 0, 1, 1, 0],  # A01
            [0, 0, 1, 0, 0, 1, 0],  # A10
            [1, 1, 0, 0, 0, 0, 1],  # A11
        ]
        V = [
            [1, 0, 0, 0, 0, 1, 1],  # B00
            [0, 1, -1, 1, 0, 0, 1],  # B01
            [1, 0, 1, 0, 0, 1, 0],  # B10
            [1, 1, 0, -1, 1, 0, 0],  # B11
        ]
        W = [
            [1, 0, 1, 0, 0, 0, 0],  # C00 row 0
            [0, 1, 0, 0, 1, 0, 1],  # C01 row 0
            [0, 0, 0, 1, 0, 1, 0],  # C10 row 1
            [1, 0, 0, 1, -1, 0, 0],  # C11 row 1
        ]
        return U, V, W
    
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition is correct."""
        rank = len(U[0])
        # Build the tensor from decomposition
        T_decomp = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            u = np.array(U[:, r])
            v = np.array(V[:, r])
            w = np.array(W[:, r])
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        T_decomp[i, j, k] += u[i] * v[j] * w[k]
        
        # Build the target tensor
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1
        
        return np.allclose(T_decomp, T_target, atol=1e-9)
    
    # Try the hybrid construction
    # 2x2x5 part: Strassen for (0,1) and (2,3) middle indices, schoolbook for index 4
    # 1x4x5 part: schoolbook
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Construct 2x2x5 decomposition
    # Middle indices: 0,1,2,3,4
    # Use Strassen for (0,1) and (2,3), schoolbook for 4
    
    U_225 = []
    V_225 = []
    W_225 = []
    
    # Strassen for middle pair (0,1)
    U_s1, V_s1, W_s1 = strassen_2x2x2()
    # Map to 2x2x5: rows 0-1, middle 0-1, cols 0-4
    # For each col k in 0-4, we need a 2x2x2 Strassen
    # But Strassen is for 2x2x2, so we apply it per col pair
    
    # Actually, for 2x2xp, we can use Strassen for each pair of middle indices.
    # For middle indices (0,1): use Strassen, gives 7 terms
    # For middle indices (2,3): use Strassen, gives 7 terms  
    # For middle index 4: schoolbook, gives 2 terms
    # Total: 16 terms for 2x2x5
    
    # Let me construct this properly.
    # For 2x2x5, the tensor has entries T[i,j,k] for i in 0-1, j in 0-4, k in 0-4
    # We want to decompose this into rank-1 terms.
    
    # For the (0,1) middle pair:
    # Use Strassen's 7 products for each output col k
    # But Strassen is for 2x2x2, so it handles 2x2 input and 2x2 output.
    # For 2x2x5, we have 2x2 input and 2x5 output.
    
    # I think the right way is: for each pair of output cols, use Strassen.
    # But that's not quite right either.
    
    # Let me just use schoolbook for now and return the baseline.
    # The search for better decompositions is complex and may not finish in time.
    
    return best_result
# EVOLVE-BLOCK-END
