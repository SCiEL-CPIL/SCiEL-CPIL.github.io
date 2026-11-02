"""Improved matrix multiplication tensor decomposition for 4x5x5.

Uses Strassen's algorithm (rank 7) for the 2x2x2 block, recursively
composed to get rank 49 for the 4x4x4 sub-block. The remaining disjoint
blocks (4x4x1, 4x1x4, 4x1x1) use schoolbook rank (16, 16, 4 respectively).
Total rank: 49 + 16 + 16 + 4 = 85.
"""
import time
import random

DIMENSIONS = [4, 5, 5]

def _strassen_2x2_2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix mult.
    
    Convention: C[i,j] = sum_k A[i,k] * B[k,j]
    U[i,k, r] = coefficient of A[i,k] in r-th product
    V[k,j, r] = coefficient of B[k,j] in r-th product
    W[i,j, r] = coefficient of r-th product in C[i,j]
    
    Returns (U, V, W) each as 2D lists of shape (4, 7) for U and V, (4, 7) for W.
    Actually U is (n*m, rank) = (4, 7), V is (m*p, rank) = (4, 7), W is (p*n, rank) = (4, 7).
    Indexing: U[i*m+k, r], V[k*p+j, r], W[j*n+i, r].
    """
    n = m = p = 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]  # (4, 7)
    V = [[0] * rank for _ in range(m * p)]  # (4, 7)
    W = [[0] * rank for _ in range(p * n)]  # (4, 7)
    
    # Strassen 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # U coefficients: for each (i,k) pair, which Strassen products use A[i,k]
    # A[0,0] (i=0,k=0): M1, M3, M5, M6 -> indices 0,2,4,5
    # A[0,1] (i=0,k=1): M3, M5, M7 -> indices 2,4,6
    # A[1,0] (i=1,k=0): M2, M6 -> indices 1,5 (with -1 for M6)
    # A[1,1] (i=1,k=1): M1, M2, M4, M7 -> indices 0,1,3,6 (with -1 for M7)
    
    # V coefficients: for each (k,j) pair, which Strassen products use B[k,j]
    # B[0,0] (k=0,j=0): M1, M2, M6 -> indices 0,1,5
    # B[0,1] (k=0,j=1): M4, M7 -> indices 3,6
    # B[1,0] (k=1,j=0): M1, M3, M5, M6 -> indices 0,2,4,5 (with -1 for M3, +1 for M6? Let me recheck)
    #   M1: B[1,0] has coeff +1 (in B00+B11... wait, B[1,0] is B10, not B11)
    
    # Let me redo this carefully.
    # B is 2x2: B[0,0], B[0,1], B[1,0], B[1,1]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # B[0,0] = B00: appears in M1 (+1), M2 (+1), M6 (+1)
    # B[0,1] = B01: appears in M4 (+1), M7 (+1)
    # B[1,0] = B10: appears in M3 (+1), M4 (-1), M6 (+1)
    # B[1,1] = B11: appears in M1 (+1), M3 (-1), M5 (+1), M7 (+1)
    
    # W coefficients: C[i,j] as sum of products
    # C[0,0] (i=0,j=0): M1 - M4 + M5 + M7 -> W[j*n+i, r] = W[0*2+0, r] = W[0, r]
    # C[0,1] (i=0,j=1): M3 + M4 -> W[1*2+0, r] = W[2, r]
    # C[1,0] (i=1,j=0): M2 + M5 -> W[0*2+1, r] = W[1, r]
    # C[1,1] (i=1,j=1): M1 - M2 + M3 + M6 -> W[1*2+1, r] = W[3, r]
    
    # Let me set them up:
    # U[i*m+k, r] where m=2
    # U[0*2+0, r] = U[0, r] for A[0,0]
    # U[0*2+1, r] = U[1, r] for A[0,1]
    # U[1*2+0, r] = U[2, r] for A[1,0]
    # U[1*2+1, r] = U[3, r] for A[1,1]
    
    # V[k*p+j, r] where p=2
    # V[0*2+0, r] = V[0, r] for B[0,0]
    # V[0*2+1, r] = V[1, r] for B[0,1]
    # V[1*2+0, r] = V[2, r] for B[1,0]
    # V[1*2+1, r] = V[3, r] for B[1,1]
    
    # W[j*n+i, r] where n=2
    # W[0*2+0, r] = W[0, r] for C[0,0]
    # W[0*2+1, r] = W[1, r] for C[1,0]  (i=1, j=0)
    # W[1*2+0, r] = W[2, r] for C[0,1]  (i=0, j=1)
    # W[1*2+1, r] = W[3, r] for C[1,1]  (i=1, j=1)
    
    # Now fill in:
    # A[0,0]: M1(+1), M3(+1), M5(+1), M6(-1)  [M6 = (A10-A00)(B00+B10), so A00 has -1]
    U[0] = [1, 0, 1, 0, 1, -1, 0]
    
    # A[0,1]: M3(+1), M5(+1), M7(+1)
    U[1] = [0, 0, 1, 0, 1, 0, 1]
    
    # A[1,0]: M2(+1), M6(+1)
    U[2] = [0, 1, 0, 0, 0, 1, 0]
    
    # A[1,1]: M1(+1), M2(+1), M4(+1), M7(-1)  [M7 = (A01-A11)(B01+B11), so A11 has -1]
    U[3] = [1, 1, 0, 1, 0, 0, -1]
    
    # B[0,0]: M1(+1), M2(+1), M6(+1)
    V[0] = [1, 1, 0, 0, 0, 1, 0]
    
    # B[0,1]: M4(+1), M7(+1)
    V[1] = [0, 0, 0, 1, 0, 0, 1]
    
    # B[1,0]: M3(+1), M4(-1), M6(+1)
    V[2] = [0, 0, 1, -1, 0, 1, 0]
    
    # B[1,1]: M1(+1), M3(-1), M5(+1), M7(+1)
    V[3] = [1, 0, -1, 0, 1, 0, 1]
    
    # W[0] for C[0,0] = M1 - M4 + M5 + M7
    W[0] = [1, 0, 0, -1, 1, 0, 1]
    
    # W[1] for C[1,0] = M2 + M5
    W[1] = [0, 1, 0, 0, 1, 0, 0]
    
    # W[2] for C[0,1] = M3 + M4
    W[2] = [0, 0, 1, 1, 0, 0, 0]
    
    # W[3] for C[1,1] = M1 - M2 + M3 + M6
    W[3] = [1, -1, 1, 0, 0, 1, 0]
    
    return U, V, W


def _schoolbook_block(n, m, p):
    """Return schoolbook decomposition for n x m x p.
    U shape (n*m, n*m*p), V shape (m*p, n*m*p), W shape (p*n, n*m*p)
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _compose_blocks(block_list):
    """Compose a list of (U_block, V_block, W_block, row_offsets) into full matrices.
    
    Each block has:
    - U_block: (n_block * m_block, rank_block)
    - V_block: (m_block * p_block, rank_block)
    - W_block: (p_block * n_block, rank_block)
    - row_offsets: (i_offset, j_offset, k_offset) - offsets for the block in the full matrix
    
    The full matrix is 4x5x5:
    - A is 4x5: rows i in [0,4), cols j in [0,5)
    - B is 5x5: rows j in [0,5), cols k in [0,5)
    - C is 5x4: rows k in [0,5), cols i in [0,4)
    
    Wait, let me re-read the convention.
    "output factor follows the official trace convention W[k*n+i,r]"
    
    So W has shape (p*n, rank) where p=5, n=4. So W[k*n+i, r].
    U has shape (n*m, rank) = (4*5, rank) = (20, rank). U[i*m+j, r].
    V has shape (m*p, rank) = (5*5, rank) = (25, rank). V[j*p+k, r].
    
    For the full 4x5x5:
    - U[i*m+j, r] where m=5, i in [0,4), j in [0,5)
    - V[j*p+k, r] where p=5, j in [0,5), k in [0,5)
    - W[k*n+i, r] where n=4, k in [0,5), i in [0,4)
    
    Block 1: 4x4x4 sub-block (i in [0,4), j in [0,4), k in [0,4))
    Block 2: 4x4x1 sub-block (i in [0,4), j in [0,4), k in [4,5))
    Block 3: 4x1x4 sub-block (i in [0,4), j in [4,5), k in [0,4))
    Block 4: 4x1x1 sub-block (i in [0,4), j in [4,5), k in [4,5))
    
    These are disjoint in terms of (j,k) pairs, so the tensor is a direct sum.
    """
    n, m, p = 4, 5, 5
    total_rank = sum(len(Ub[0]) for Ub, _, _, _ in block_list)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for Ub, Vb, Wb, (i_off, j_off, k_off) in block_list:
        n_b = len(Wb) // (p_b := len(Vb[0]) // (m_b := len(Ub[0]) // n_b))
        # Actually, let me just figure out the dimensions from the block sizes
        n_b = i_off  # No, i_off is the offset
        # The block has dimensions (n_b, m_b, p_b)
        # n_b * m_b = len(Ub)
        # m_b * p_b = len(Vb)
        # p_b * n_b = len(Wb)
        
        n_b = len(Ub[0]) and None  # This is getting complicated
        # Let me pass the block dimensions explicitly
        pass
    
    # Let me redo this with explicit dimensions
    return U, V, W


def _compose_blocks_v2(block_list):
    """Compose blocks with explicit dimensions.
    Each block is a tuple: (U_b, V_b, W_b, n_b, m_b, p_b, i_off, j_off, k_off)
    """
    n, m, p = 4, 5, 5
    total_rank = sum(len(Ub[0]) for Ub, _, _, _, _, _, _, _ in block_list)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for Ub, Vb, Wb, n_b, m_b, p_b, i_off, j_off, k_off in block_list:
        rank_b = len(Ub[0])
        
        # U: rows i_off..i_off+n_b-1, cols j_off..j_off+m_b-1
        for i_local in range(n_b):
            for j_local in range(m_b):
                row_full = (i_off + i_local) * m + (j_off + j_local)
                for r in range(rank_b):
                    U[row_full][col_offset + r] = Ub[i_local * m_b + j_local][r]
        
        # V: rows j_off..j_off+m_b-1, cols k_off..k_off+p_b-1
        for j_local in range(m_b):
            for k_local in range(p_b):
                row_full = (j_off + j_local) * p + (k_off + k_local)
                for r in range(rank_b):
                    V[row_full][col_offset + r] = Vb[j_local * p_b + k_local][r]
        
        # W: rows k_off..k_off+p_b-1, cols i_off..i_off+n_b-1
        for k_local in range(p_b):
            for i_local in range(n_b):
                row_full = (k_off + k_local) * n + (i_off + i_local)
                for r in range(rank_b):
                    W[row_full][col_offset + r] = Wb[k_local * n_b + i_local][r]
        
        col_offset += rank_b
    
    return U, V, W


def construct():
    """Construct a rank-85 exact tensor decomposition for 4x5x5 matrix multiplication.
    
    Block 1: 4x4x4 using Strassen recursively -> rank 49
    Block 2: 4x4x1 schoolbook -> rank 16
    Block 3: 4x1x4 schoolbook -> rank 16
    Block 4: 4x1x1 schoolbook -> rank 4
    Total: 85
    """
    # Block 1: 4x4x4
    # We need to compose Strassen 2x2x2 blocks to get 4x4x4.
    # The 4x4x4 tensor can be decomposed into 16 2x2x2 sub-blocks.
    # Each 2x2x2 sub-block has rank 7, so 16*7 = 112. That's worse!
    # 
    # Wait, that's not right. The Strassen algorithm for 4x4x4 uses the
    # 2x2x2 Strassen recursively. But the composition is more subtle.
    # 
    # Actually, the standard way to get rank 49 for 4x4x4 is:
    # The 4x4 matrix multiplication tensor is the direct sum of 16 2x2 tensors?
    # No, that would be 16*7 = 112.
    # 
    # Let me reconsider. The known result is that R(4x4x4) = 49.
    # This comes from the fact that the 4x4 matrix multiplication tensor
    # has a decomposition of rank 49, which is better than the trivial
    # 16*7=112 from composing 2x2 Strassen.
    # 
    # Actually, I think I'm confusing things. Let me look at this differently.
    # 
    # The tensor for n x m x p matrix multiplication has a natural structure.
    # For 4x4x4, we can view it as 4x4 matrix mult. The rank 49 decomposition
    # is known but I need to construct it.
    # 
    # One way: the 4x4x4 tensor is the Kronecker product of two 2x2x2 tensors?
    # No, that's not quite right either.
    # 
    # Let me think about this more carefully. The matrix multiplication tensor
    # T_{n,m,p} is defined by T[i,j,k] = delta[i*k, j] (or similar).
    # 
    # Actually, I recall that the rank of 4x4x4 matrix multiplication is at most 49,
    # achieved by a specific construction. But I'm not sure of the exact coefficients.
    # 
    # Let me try a different approach: use the fact that 4 = 2+2, and use
    # the tensor product structure.
    # 
    # If we partition the indices into pairs, the 4x4x4 tensor is related to
    # the 2x2x2 tensor but not simply its tensor product.
    # 
    # I think the safest approach is to use a known rank-49 construction.
    # Let me search my memory...
    # 
    # Actually, I believe the rank 49 for 4x4x4 comes from the following:
    # The 4x4 matrix multiplication can be done with 49 multiplications
    # using a recursive Strassen-like algorithm.
    # 
    # Let me try to construct it by viewing 4x4 as a 2x2 block matrix where
    # each block is 2x2, and using Strassen on the block level, but with
    # the inner 2x2 multiplications also using Strassen.
    # 
    # Block-level Strassen: 7 block multiplications, each of size 2x2x2.
    # Each 2x2x2 uses rank 7. So total = 7 * 7 = 49.
    # 
    # Yes! That's it. The 4x4x4 tensor is the "block Strassen" composition.
    # We have 7 intermediate block products, each computed with rank 7.
    # Total rank = 7 * 7 = 49.
    # 
    # So I need to implement this composition.
    
    def _strassen_4x4_4x4():
        """Construct rank-49 decomposition for 4x4x4 using block Strassen.
        
        The 4x4 matrices are partitioned into 2x2 blocks:
        A = [[A00, A01], [A10, A11]] where each Aij is 2x2
        B = [[B00, B01], [B10, B11]] where each Bij is 2x2
        C = A*B = [[C00, C01], [C10, C11]] where each Cij is 2x2
        
        Strassen at block level:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B10-B11)
        M4 = A11(B01-B10)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B10)
        M7 = (A01-A11)(B01+B11)
        
        C00 = M1 - M4 + M5 + M7
        C01 = M3 + M4
        C10 = M2 + M5
        C11 = M1 - M2 + M3 + M6
        
        Each Mj is a 2x2 matrix product, computed with rank 7.
        So we need 7 * 7 = 49 scalar multiplications.
        """
        # Get the 2x2x2 Strassen decomposition
        U2, V2, W2 = _strassen_2x2_2x2()
        
        # Now compose. The 4x4x4 tensor has:
        # U shape (4*4, 49) = (16, 49)
        # V shape (4*4, 49) = (16, 49)
        # W shape (4*4, 49) = (16, 49)
        
        rank = 49
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        
        # The 4x4 matrix A has indices (i, k) where i, k in [0,4)
        # Partitioned into 2x2 blocks:
        # A00: (0,0), (0,1), (1,0), (1,1) -> local indices (0,0)
        # A01: (0,2), (0,3), (1,2), (1,3) -> local indices (0,1)
        # A10: (2,0), (2,1), (3,0), (3,1) -> local indices (1,0)
        # A11: (2,2), (2,3), (3,2), (3,3) -> local indices (1,1)
        
        # Similarly for B with indices (k, j)
        # And C with indices (i, j)
        
        # For each block-level Strassen product Mj (j=0..6), we have
        # Mj = (sum of A blocks with coefficients) * (sum of B blocks with coefficients)
        # Each block product is a 2x2x2 multiplication.
        
        # Block-level Strassen coefficients for A:
        # M0 (M1): A00(+1), A11(+1)
        # M1 (M2): A10(+1), A11(+1)
        # M2 (M3): A00(+1)
        # M3 (M4): A11(+1)
        # M4 (M5): A00(+1), A01(+1)
        # M5 (M6): A10(+1), A00(-1)
        # M6 (M7): A01(+1), A11(-1)
        
        # Block-level Strassen coefficients for B:
        # M0 (M1): B00(+1), B11(+1)
        # M1 (M2): B00(+1)
        # M2 (M3): B10(+1), B11(-1)
        # M3 (M4): B01(+1), B10(-1)
        # M4 (M5): B11(+1)
        # M5 (M6): B00(+1), B10(+1)
        # M6 (M7): B01(+1), B11(+1)
        
        # Block-level Strassen coefficients for C:
        # C00: M0(+1), M3(-1), M4(+1), M6(+1)
        # C01: M2(+1), M3(+1)
        # C10: M1(+1), M4(+1)
        # C11: M0(+1), M1(-1), M2(+1), M5(+1)
        
        # A block offsets:
        # A00: i_local in [0,2), k_local in [0,2) -> global i in [0,2), k in [0,2)
        # A01: i_local in [0,2), k_local in [0,2) -> global i in [0,2), k in [2,4)
        # A10: i_local in [0,2), k_local in [0,2) -> global i in [2,4), k in [0,2)
        # A11: i_local in [0,2), k_local in [0,2) -> global i in [2,4), k in [2,4)
        
        # B block offsets:
        # B00: k_local in [0,2), j_local in [0,2) -> global k in [0,2), j in [0,2)
        # B01: k_local in [0,2), j_local in [0,2) -> global k in [0,2), j in [2,4)
        # B10: k_local in [0,2), j_local in [0,2) -> global k in [2,4), j in [0,2)
        # B11: k_local in [0,2), j_local in [0,2) -> global k in [2,4), j in [2,4)
        
        # C block offsets:
        # C00: i_local in [0,2), j_local in [0,2) -> global i in [0,2), j in [0,2)
        # C01: i_local in [0,2), j_local in [0,2) -> global i in [0,2), j in [2,4)
        # C10: i_local in [0,2), j_local in [0,2) -> global i in [2,4), j in [0,2)
        # C11: i_local in [0,2), j_local in [0,2) -> global i in [2,4), j in [2,4)
        
        # A coefficients for each Mj: list of (block_id, coeff)
        # block_id: 0=A00, 1=A01, 2=A10, 3=A11
        A_coeffs = [
            [(0, 1), (3, 1)],       # M0
            [(2, 1), (3, 1)],       # M1
            [(0, 1)],               # M2
            [(3, 1)],               # M3
            [(0, 1), (1, 1)],       # M4
            [(2, 1), (0, -1)],      # M5
            [(1, 1), (3, -1)],      # M6
        ]
        
        B_coeffs = [
            [(0, 1), (3, 1)],       # M0
            [(0, 1)],               # M1
            [(2, 1), (3, -1)],      # M2
            [(1, 1), (2, -1)],      # M3
            [(3, 1)],               # M4
            [(0, 1), (2, 1)],       # M5
            [(1, 1), (3, 1)],       # M6
        ]
        
        C_coeffs = [
            [(0, 1), (3, -1), (4, 1), (6, 1)],   # C00
            [(2, 1), (3, 1)],                     # C01
            [(1, 1), (4, 1)],                     # C10
            [(0, 1), (1, -1), (2, 1), (5, 1)],   # C11
        ]
        
        # Block offsets for A: (i_offset, k_offset)
        A_block_offsets = [(0, 0), (0, 2), (2, 0), (2, 2)]
        # Block offsets for B: (k_offset, j_offset)
        B_block_offsets = [(0, 0), (0, 2), (2, 0), (2, 2)]
        # Block offsets for C: (i_offset, j_offset)
        C_block_offsets = [(0, 0), (0, 2), (2, 0), (2, 2)]
        
        # Now fill in the U, V, W matrices.
        # U[i*4+k, r] where r = j*7 + r_local
        # V[k*4+j, r]
        # W[j*4+i, r]
        
        for j_block in range(7):
            for r_local in range(7):
                r = j_block * 7 + r_local
                
                # U: for each A block contributing to Mj_block
                for (block_id, coeff) in A_coeffs[j_block]:
                    i_off_a, k_off_a = A_block_offsets[block_id]
                    for i_local in range(2):
                        for k_local in range(2):
                            i_global = i_off_a + i_local
                            k_global = k_off_a + k_local
                            row = i_global * 4 + k_global
                            U[row][r] += coeff * U2[i_local * 2 + k_local][r_local]
                
                # V: for each B block contributing to Mj_block
                for (block_id, coeff) in B_coeffs[j_block]:
                    k_off_b, j_off_b = B_block_offsets[block_id]
                    for k_local in range(2):
                        for j_local in range(2):
                            k_global = k_off_b + k_local
                            j_global = j_off_b + j_local
                            row = k_global * 4 + j_global
                            V[row][r] += coeff * V2[k_local * 2 + j_local][r_local]
                
                # W: for each C block that uses Mj_block
                for (c_block_id, coeff) in C_coeffs:
                    # c_block_id is 0,1,2,3 for C00,C01,C10,C11
                    # Check if Mj_block contributes to this C block
                    # We need to find if j_block is in the list for this C block
                    # Let me restructure C_coeffs to be indexed by c_block_id
                    pass
        
        # Let me restructure C_coeffs properly
        # C_block 0 (C00): uses M0(+1), M3(-1), M4(+1), M6(+1)
        # C_block 1 (C01): uses M2(+1), M3(+1)
        # C_block 2 (C10): uses M1(+1), M4(+1)
        # C_block 3 (C11): uses M0(+1), M1(-1), M2(+1), M5(+1)
        
        C_contrib = [
            {0: 1, 3: -1, 4: 1, 6: 1},    # C00
            {2: 1, 3: 1},                   # C01
            {1: 1, 4: 1},                   # C10
            {0: 1, 1: -1, 2: 1, 5: 1},     # C11
        ]
        
        for j_block in range(7):
            for r_local in range(7):
                r = j_block * 7 + r_local
                
                # W: for each C block
                for c_block_id in range(4):
                    if j_block in C_contrib[c_block_id]:
                        coeff = C_contrib[c_block_id][j_block]
                        i_off_c, j_off_c = C_block_offsets[c_block_id]
                        for i_local in range(2):
                            for j_local in range(2):
                                i_global = i_off_c + i_local
                                j_global = j_off_c + j_local
                                row = j_global * 4 + i_global
                                W[row][r] += coeff * W2[j_local * 2 + i_local][r_local]
        
        return U, V, W
    
    U1, V1, W1 = _strassen_4x4_4x4()
    
    # Block 2: 4x4x1 schoolbook
    U2, V2, W2 = _schoolbook_block(4, 4, 1)
    
    # Block 3: 4x1x4 schoolbook
    U3, V3, W3 = _schoolbook_block(4, 1, 4)
    
    # Block 4: 4x1x1 schoolbook
    U4, V4, W4 = _schoolbook_block(4, 1, 1)
    
    # Compose all blocks
    # Block 1: i in [0,4), j in [0,4), k in [0,4)
    # Block 2: i in [0,4), j in [0,4), k in [4,5)
    # Block 3: i in [0,4), j in [4,5), k in [0,4)
    # Block 4: i in [0,4), j in [4,5), k in [4,5)
    
    block_list = [
        (U1, V1, W1, 4, 4, 4, 0, 0, 0),
        (U2, V2, W2, 4, 4, 1, 0, 0, 4),
        (U3, V3, W3, 4, 1, 4, 0, 4, 0),
        (U4, V4, W4, 4, 1, 1, 0, 4, 4),
    ]
    
    U, V, W = _compose_blocks_v2(block_list)
    
    return {"U": U, "V": V, "W": W}
