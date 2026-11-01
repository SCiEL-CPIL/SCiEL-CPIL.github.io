"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # We'll build a decomposition of rank 56 using recursive Strassen-like splits
    # M_{2,3,10} <= 2 * M_{2,3,5}
    # M_{2,3,5} <= M_{2,3,2} + M_{2,3,3}
    # M_{2,3,2} <= 11, M_{2,3,3} <= 17
    # Total: 2*(11+17) = 56
    
    # Each "multiplication" in the decomposition corresponds to a rank-1 tensor
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # where c is the column (rank index)
    
    # Let's build this step by step
    # We'll collect (U_idx, V_idx, W_idx) triples for each rank-1 component
    
    components = []  # list of (i, j, k) for the rank-1 tensors
    
    # Helper: add a simple product A[i,j]*B[j,k] contributing to C[i,k]
    def add_simple(i, j, k):
        components.append((i, j, k))
    
    # M_{2,3,2}: A is 2x3, B is 3x2, C is 2x2
    # Split: A = [a | A2] where a is 2x1 (col 0), A2 is 2x2 (cols 1,2)
    # B = [b; B2] where b is 1x2 (row 0), B2 is 2x2 (rows 1,2)
    # C = a*b + A2*B2
    # a*b: 4 products: (0,0,0), (0,0,1), (1,0,0), (1,0,1)
    # A2*B2: 7 Strassen products for 2x2x2
    
    def strassen_2x2x2(offset_i, offset_j, offset_k):
        """Add 7 Strassen rank-1 components for 2x2x2 multiplication
        with A starting at row offset_i, B starting at col offset_j, C starting at col offset_k
        A is 2x2: rows offset_i, offset_i+1; cols offset_j, offset_j+1
        B is 2x2: rows offset_j, offset_j+1; cols offset_k, offset_k+1
        C is 2x2: rows offset_i, offset_i+1; cols offset_k, offset_k+1
        """
        # Strassen for 2x2:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B11)
        # M4 = A11*(B01-B10)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B10)
        # M7 = (A01-A11)*(B01+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        i0, i1 = offset_i, offset_i + 1
        j0, j1 = offset_j, offset_j + 1
        k0, k1 = offset_k, offset_k + 1
        
        # M1 = (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        # This is a sum of 4 products, but in tensor decomposition, 
        # each "multiplication" in Strassen corresponds to a single rank-1 term
        # where the U and V vectors have +/-1 entries
        
        # Actually, in the tensor decomposition, each Strassen "product" 
        # Mx is a single rank-1 component where U and V have specific patterns.
        
        # Let me reconsider. In the standard tensor decomposition of matrix mult,
        # each rank-1 term is: U[i,j] * V[j,k] * W[k,i]
        # For Strassen, we have 7 terms, each of which is a rank-1 tensor
        # where U, V, W have specific non-zero patterns.
        
        # For M1 = (A00+A11)*(B00+B11):
        # This contributes to C00, C01, C10, C11
        # U has +1 at (i0,j0) and +1 at (i1,j1)
        # V has +1 at (j0,k0) and +1 at (j1,k1)
        # W has +1 at (k0,i0), +1 at (k0,i1), +1 at (k1,i0), +1 at (k1,i1)
        # But wait, that's not right. Let me think again.
        
        # Actually, the Strassen decomposition is:
        # C = sum_{t=1}^{7} U_t * V_t * W_t
        # where each (U_t, V_t, W_t) is a rank-1 tensor.
        
        # For the 2x2x2 case:
        # t=1: U has +1 at (0,0) and (1,1); V has +1 at (0,0) and (1,1); W has +1 at all 4 positions
        #   This gives: (A00+A11)(B00+B11) * (1 for all C entries)
        #   C00 += (A00+A11)(B00+B11)
        #   C01 += (A00+A11)(B00+B11)
        #   C10 += (A00+A11)(B00+B11)
        #   C11 += (A00+A11)(B00+B11)
        #   But we only want this to contribute to specific C entries with specific signs.
        
        # I think I'm overcomplicating this. Let me use a different approach.
        
        # In the tensor decomposition, a rank-1 term is specified by:
        # U[i*m+j] = u_{ij}, V[j*p+k] = v_{jk}, W[k*n+i] = w_{ki}
        # and the product u_{ij} * v_{jk} * w_{ki} contributes to the coefficient of A[i,j]*B[j,k] in C[i,k]
        
        # For Strassen 2x2x2, the 7 rank-1 terms are:
        # (Using 0-indexed local coordinates)
        
        # Term 1: U: (0,0)=1, (1,1)=1; V: (0,0)=1, (1,1)=1; W: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1
        #   Contribution: A00*B00*C00 + A00*B00*C01 + A00*B00*C10 + A00*B00*C11
        #                + A11*B11*C00 + A11*B11*C01 + A11*B11*C10 + A11*B11*C11
        
        # This is getting very complex to track manually. Let me just hardcode a known 
        # rank-56 decomposition.
        
        pass
    
    # Given the complexity, let me just use the schoolbook rank-60 for now,
    # but the measured knowledge says rank 55 is achievable.
    # Let me implement a straightforward rank-60 and see.
    
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
# EVOLVE-BLOCK-END
