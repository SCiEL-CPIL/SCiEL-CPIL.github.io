"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2():
    """Return (U, V, W) for 2x2 * 2x2 = 2x2 using Strassen's 7-term algorithm.
    
    Blocks: A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    C = A*B, where C[i,j] is a 1x1 block (scalar).
    
    Strassen:
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    For each term, we need U, V, W entries.
    U row index: i*2 + j for A[i,j] block
    V row index: j*2 + k for B[j,k] block
    W row index: k*2 + i for C[k,i]... wait, W[k*n+i, r] convention.
    
    For 2x2: n=2, m=2, p=2. W[k*2+i, r].
    
    Let's build the decomposition term by term.
    Each term M_t corresponds to a product of linear combinations.
    We need to express M_t = sum_u (U[i,j,u] * V[j,k,u]) where the sum over u
    gives the coefficient for C[i,k].
    
    Actually, let me think more carefully. The tensor decomposition says:
    C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j.
    
    Wait, no. The standard matrix multiplication tensor is:
    T[i,j,k] = A[i,j] * B[j,k] -> C[i,k]
    
    The decomposition is: C[i,k] = sum_r (sum_j U[i*m+j, r]) * V[j*p+k, r] * W[k*n+i, r]
    
    Hmm, that's not quite right either. Let me reconsider.
    
    The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k (in the sense
    that C[i,k] = sum_j A[i,j] B[j,k]).
    
    A rank-r decomposition means: T[i,j,k] = sum_{r=1}^{r} U[i,j,r] * V[j,k,r] * W[i,k,r]
    where U is (n*m, r), V is (m*p, r), W is (p*n, r).
    
    Wait, the problem says W[k*n+i, r]. So W is indexed by (k*n+i) for the output
    position. Let me re-read: "W[k*n+i,r]".
    
    So the decomposition is:
    For each output position (i, k) where i is row of A and k is column of B:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    In tensor form, the multiplication tensor M has M[i,j,k] = 1 (it's the identity
    in the sense that it defines the bilinear map).
    
    A rank-r decomposition: M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    This means for each r, we have a "slice" that contributes to the tensor.
    
    For Strassen on 2x2 blocks, we're decomposing the 2x2x2 tensor (which has
    8 entries, all 1) into 7 rank-1 tensors.
    
    Let me just directly construct the 2x2x2 Strassen decomposition.
    
    The 2x2x2 tensor T has T[i,j,k] = 1 for i,j,k in {0,1}.
    
    Strassen's 7 terms:
    Term 1: (A00+A11)(B00+B11) -> contributes to C00, C11
    Term 2: (A10+A11)B00 -> contributes to C10, C11
    Term 3: A00(B01-B11) -> contributes to C01, C11
    Term 4: A11(B10-B00) -> contributes to C00, C10
    Term 5: (A00+A01)B11 -> contributes to C00, C01
    Term 6: (A10-A00)(B00+B01) -> contributes to C11
    Term 7: (A01-A11)(B10+B11) -> contributes to C00
    
    For each term, the U, V, W entries are determined by the linear combinations.
    
    Let me define the indices:
    U row: i*2+j (i in 0..1, j in 0..1) -> rows 0,1,2,3
    V row: j*2+k (j in 0..1, k in 0..1) -> rows 0,1,2,3
    W row: k*2+i (k in 0..1, i in 0..1) -> rows 0,1,2,3
    
    Term 1: U has 1 at (0,0) and (1,1) [i=0,j=0 and i=1,j=1]
             V has 1 at (0,0) and (1,1) [j=0,k=0 and j=1,k=1]
             W has 1 at (0,0) and (1,1) [k=0,i=0 and k=1,i=1]
             Wait, W[k*n+i] = W[k*2+i]. For C00: k=0,i=0 -> W[0]. For C11: k=1,i=1 -> W[3].
    
    Hmm, this is getting complex. Let me just directly verify by constructing
    the full 4x5x6 decomposition using block Strassen.
    
    Actually, let me take a simpler approach. I'll construct the decomposition
    by combining smaller block decompositions.
    """
    pass

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Split into blocks:
    # A is 4x5: split rows into 2+2, cols into 2+3
    # B is 5x6: split rows into 2+3, cols into 3+3
    # C is 4x6: split rows into 2+2, cols into 3+3
    
    # Block structure:
    # A = [[A00 (2x2), A01 (2x3)],
    #      [A10 (2x2), A11 (2x3)]]
    # B = [[B00 (2x3), B01 (2x3)],
    #      [B10 (3x3), B11 (3x3)]]
    # C = [[C00 (2x3), C01 (2x3)],
    #      [C10 (2x3), C11 (2x3)]]
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # We need decompositions for:
    # 2x2 * 2x3 -> 2x3
    # 2x3 * 3x3 -> 2x3
    
    # Let me build these block decompositions.
    
    # For 2x2 * 2x3:
    # A is 2x2, B is 2x3, C is 2x3
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # We can use a decomposition. Let me think about what's minimal.
    # Schoolbook for 2x2*2x3 has rank 2*2*3 = 12.
    # Can we do better?
    
    # Actually, let me just use schoolbook for the block multiplications
    # and see what total rank we get.
    
    # 2x2 * 2x3: rank 12
    # 2x3 * 3x3: rank 2*3*3 = 18
    
    # Total: 4 block multiplications, each with its rank.
    # But we need to combine them properly.
    
    # Let me construct the full decomposition by building each block's
    # decomposition and then combining.
    
    # This is getting complex. Let me try a different approach:
    # Use the known result that M(4,5,6) <= some value.
    
    # Actually, let me just implement a clean block decomposition.
    
    # I'll create helper functions to build block decompositions.
    
    def build_block_decomp(n1, m1, p1):
        """Build schoolbook decomposition for n1 x m1 * m1 x p1 -> n1 x p1.
        Returns (U, V, W) as lists of lists with integer entries.
        U: (n1*m1) x rank, V: (m1*p1) x rank, W: (p1*n1) x rank
        """
        rank = n1 * m1 * p1
        U = [[0] * rank for _ in range(n1 * m1)]
        V = [[0] * rank for _ in range(m1 * p1)]
        W = [[0] * rank for _ in range(p1 * n1)]
        col = 0
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    U[i * m1 + j][col] = 1
                    V[j * p1 + k][col] = 1
                    W[k * n1 + i][col] = 1
                    col += 1
        return U, V, W
    
    # Now combine block decompositions.
    # The full matrix multiplication 4x5 * 5x6 -> 4x6 can be decomposed
    # into block multiplications.
    
    # Let me use a simpler approach: just use the full schoolbook but
    # see if I can find a better structure.
    
    # Actually, let me try to implement Strassen-style for the 2x2 sub-blocks.
    
    # For the 2x2 * 2x2 part (A00 * B00), I can use Strassen with rank 7
    # instead of schoolbook rank 8.
    
    # Let me build the full decomposition manually.
    
    # First, let me define the Strassen 2x2 decomposition.
    
    def strassen_2x2():
        """Return (U, V, W) for 2x2 * 2x2 = 2x2, rank 7.
        U: 4x7, V: 4x7, W: 4x7
        """
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        
        # Term 1: (A00+A11)(B00+B11) -> C00, C11
        # U: A00 (row 0), A11 (row 3)
        # V: B00 (row 0), B11 (row 3)
        # W: C00 (row 0), C11 (row 3)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # Term 2: (A10+A11)B00 -> C10, C11
        # U: A10 (row 2), A11 (row 3)
        # V: B00 (row 0)
        # W: C10 (row 2), C11 (row 3)
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        
        # Term 3: A00(B01-B11) -> C01, C11
        # U: A00 (row 0)
        # V: B01 (row 1), -B11 (row 3)
        # W: C01 (row 1), C11 (row 3)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # Term 4: A11(B10-B00) -> C00, C10
        # U: A11 (row 3)
        # V: B10 (row 2), -B00 (row 0)
        # W: C00 (row 0), C10 (row 2)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # Term 5: (A00+A01)B11 -> C00, C01
        # U: A00 (row 0), A01 (row 1)
        # V: B11 (row 3)
        # W: C00 (row 0), C01 (row 1)
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = 1; W[1][4] = 1
        
        # Term 6: (A10-A00)(B00+B01) -> C11
        # U: A10 (row 2), -A00 (row 0)
        # V: B00 (row 0), B01 (row 1)
        # W: C11 (row 3)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # Term 7: (A01-A11)(B10+B11) -> C00
        # U: A01 (row 1), -A11 (row 3)
        # V: B10 (row 2), B11 (row 3)
        # W: C00 (row 0)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W
    
    # Now I need to combine these into the full 4x5*5x6 decomposition.
    # This is quite involved. Let me just use schoolbook for now and
    # see if I can improve incrementally.
    
    # For now, return schoolbook as baseline.
    n, m, p = DIMENSIONS
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
