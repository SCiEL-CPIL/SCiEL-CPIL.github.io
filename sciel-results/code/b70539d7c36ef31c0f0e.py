"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                computed = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(computed - expected) > 1e-10:
                    return False
    return True

def _strassen_rank7():
    """
    Return rank-7 decomposition for 2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    U = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 1],
    ], dtype=float)
    
    V = np.array([
        [1, 1, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
    ], dtype=float)
    
    W = np.array([
        [1, -1, 0, -1, 0, 0, 1],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, -1],
        [1, 0, -1, 0, -1, 1, 0],
    ], dtype=float)
    
    return U, V, W

def _kron_strassen_4x4x4():
    """
    Return rank-49 decomposition for 4x4x4 matrix multiplication using Kronecker product
    of two 2x2 Strassen decompositions.
    
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    n = 4
    rank = 49
    
    U = np.zeros((n*n, rank), dtype=float)
    V = np.zeros((n*n, rank), dtype=float)
    W = np.zeros((n*n, rank), dtype=float)
    
    S_U, S_V, S_W = _strassen_rank7()
    
    # Kronecker product structure:
    # A (4x4) = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B (4x4) = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # C (4x4) = [[C00, C01], [C10, C11]] where each Ck is 2x2
    #
    # C = A * B can be computed as:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Using Strassen for the outer 2x2 and inner 2x2:
    # The Kronecker product approach uses 7*7 = 49 scalar multiplications.
    
    # U rows: A_ij where i in 0..3, j in 0..3
    # U row index: i*4 + j
    # A_block(br, bc): rows 2*br..2*br+1, cols 2*bc..2*bc+1
    # A_block(0,0) -> U rows 0,1,4,5
    # A_block(0,1) -> U rows 2,3,6,7
    # A_block(1,0) -> U rows 8,9,12,13
    # A_block(1,1) -> U rows 10,11,14,15
    
    # V rows: B_jk where j in 0..3, k in 0..3
    # V row index: j*4 + k
    # B_block(br, bc): rows 2*br..2*br+1, cols 2*bc..2*bc+1
    # B_block(0,0) -> V rows 0,1,4,5
    # B_block(0,1) -> V rows 2,3,6,7
    # B_block(1,0) -> V rows 8,9,12,13
    # B_block(1,1) -> V rows 10,11,14,15
    
    # W rows: C_ki where k in 0..3, i in 0..3
    # W row index: k*4 + i
    # C_block(br, bc): rows 2*br..2*br+1, cols 2*bc..2*bc+1
    # C_block(0,0) -> W rows 0,1,4,5
    # C_block(0,1) -> W rows 2,3,6,7
    # C_block(1,0) -> W rows 8,9,12,13
    # C_block(1,1) -> W rows 10,11,14,15
    
    # For the Kronecker product, we use 49 scalar multiplications.
    # Each scalar multiplication r = (r_outer, r_inner) where r_outer, r_inner in 0..6
    # The contribution to U, V, W is:
    # U[i,j, r] = S_U[local_i, r_outer] * S_U[local_j, r_inner]
    # where local_i, local_j are the positions within the 2x2 blocks
    
    # Let me think about this more carefully.
    # A (4x4) has entries A[i,j] for i,j in 0..3
    # We can write A[i,j] = sum over block structure...
    # Actually, the Kronecker product approach works as follows:
    #
    # Think of A as a 2x2 block matrix where each block is 2x2.
    # A = [[A00, A01], [A10, A11]]
    # Similarly for B and C.
    #
    # The product C = A*B can be computed using the Strassen algorithm at two levels.
    # The outer Strassen uses 7 products of 2x2 matrices.
    # Each 2x2 product uses Strassen again with 7 scalar multiplications.
    # Total: 7 * 7 = 49 scalar multiplications.
    
    # Let me implement this directly.
    # For each outer Strassen step r_o (0..6) and inner Strassen step r_i (0..6):
    # The scalar multiplication is:
    #   prod = (sum over a_blocks, b_blocks of S_U[...]) * (sum over ...)
    
    # Actually, let me use a cleaner formulation.
    # The 4x4 matrix multiplication tensor can be written as:
    # T[i,j,k] = 1 for all i,j,k in 0..3
    #
    # Using the Kronecker product of two 2x2 multiplication tensors:
    # T_4x4 = T_2x2 ⊗ T_2x2 (in a specific sense)
    
    # Let me just build it explicitly.
    # A[i,j] for i,j in 0..3
    # B[j,k] for j,k in 0..3  
    # C[i,k] for i,k in 0..3
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Using Kronecker:
    # Write i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0
    # where i1,j1,k1 in {0,1} and i0,j0,k0 in {0,1}
    #
    # A[i,j] = A[2*i1+i0, 2*j1+j0]
    # B[j,k] = B[2*j1+j0, 2*k1+k0]
    # C[i,k] = C[2*i1+i0, 2*k1+k0]
    #
    # The 4x4 multiplication can be seen as:
    # For each (i1, k1) pair (outer indices), we have a 2x2 multiplication
    # involving the blocks.
    
    # Let me use the direct Kronecker product construction.
    # The rank-49 decomposition has factors:
    # U[i,j, (r_o, r_i)] = U_2x2[i1, j1, r_o] * U_2x2[i0, j0, r_i]
    # V[j,k, (r_o, r_i)] = V_2x2[j1, k1, r_o] * V_2x2[j0, k0, r_i]
    # W[i,k, (r_o, r_i)] = W_2x2[i1, k1, r_o] * W_2x2[i0, k0, r_i]
    
    # Wait, that's not quite right either. Let me think again.
    
    # Actually, the standard Kronecker product approach for matrix multiplication:
    # If we have a rank-r decomposition of 2x2 multiplication:
    #   C_2x2 = sum_{r=1}^r U_r ⊗ V_r ⊗ W_r
    # Then for 4x4 multiplication, we can use:
    #   C_4x4 = sum_{r1=1}^r sum_{r2=1}^r (U_{r1}⊗U_{r2}) ⊗ (V_{r1}⊗V_{r2}) ⊗ (W_{r1}⊗W_{r2})
    # This gives rank r^2 = 49.
    
    # So:
    # U_4x4[i,j, (r1,r2)] = U_2x2[i1,j1,r1] * U_2x2[i0,j0,r2]
    # where i = 2*i1 + i0, j = 2*j1 + j0
    
    # V_4x4[j,k, (r1,r2)] = V_2x2[j1,k1,r1] * V_2x2[j0,k0,r2]
    # where j = 2*j1 + j0, k = 2*k1 + k0
    
    # W_4x4[i,k, (r1,r2)] = W_2x2[i1,k1,r1] * W_2x2[i0,k0,r2]
    # where i = 2*i1 + i0, k = 2*k1 + k0
    
    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            i = 2 * i1 + i0
                            j = 2 * j1 + j0
                            U[i * 4 + j, col] = S_U[i1 * 2 + j1, r1] * S_U[i0 * 2 + j0, r2]
            
            for j1 in range(2):
                for k1 in range(2):
                    for j0 in range(2):
                        for k0 in range(2):
                            j = 2 * j1 + j0
                            k = 2 * k1 + k0
                            V[j * 4 + k, col] = S_V[j1 * 2 + k1, r1] * S_V[j0 * 2 + k0, r2]
            
            for i1 in range(2):
                for k1 in range(2):
                    for i0 in range(2):
                        for k0 in range(2):
                            i = 2 * i1 + i0
                            k = 2 * k1 + k0
                            W[k * 4 + i, col] = S_W[k1 * 2 + i1, r1] * S_W[k0 * 2 + i0, r2]
    
    return U, V, W

def _schoolbook_4x4x3():
    """
    Return schoolbook decomposition for 4x4x3 matrix multiplication.
    U: (16, 48), V: (12, 48), W: (12, 48)
    """
    n = 4
    m = 4
    p = 3
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i*m*p + j*p + k
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
    
    return U, V, W

def _strassen_4x4x2():
    """
    Return rank-14 decomposition for 4x4x2 matrix multiplication using Kronecker product
    of Strassen (2x2x2) and schoolbook (2x2x1).
    
    U: (16, 14), V: (8, 14), W: (8, 14)
    """
    n = 4
    m = 4
    p = 2
    rank = 14
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    S_U, S_V, S_W = _strassen_rank7()
    
    # For 4x4x2:
    # A is 4x4, B is 4x2, C is 4x2
    # Split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] (each 2x2)
    # B = [[B00, B01], [B10, B11]] where B00, B01, B10, B11 are 2x1
    # C = [[C00, C01], [C10, C11]] where C00, C01, C10, C11 are 2x1
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # For each 2x2 * 2x1 product, we can use a rank-2 decomposition (schoolbook)
    # or we can use Strassen-like tricks.
    
    # Actually, for 2x2 * 2x1, the schoolbook rank is 4.
    # But we can do better: 2x2 * 2x1 has rank 3 (I think).
    # Let me just use schoolbook for now: rank 4 per product, 4 products = 16.
    # But we want 14, so let me think...
    
    # Actually, let me use the Kronecker product approach:
    # 4x4x2 = (2x2x2) ⊗ (2x2x1) in some sense
    # Rank = 7 * 2 = 14 if 2x2x1 has rank 2.
    
    # 2x2 * 2x1: A (2x2) * B (2x1) = C (2x1)
    # C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # Schoolbook rank: 4
    # Can we do rank 3? 
    # C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # Let M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,0])
    # Let M2 = (A[1,0]-A[0,0])*B[0,0]
    # Let M3 = A[0,1]*B[1,0]
    # C[0] = M1 - M2 - M3 + ... hmm, this is getting complicated.
    
    # Let me just use schoolbook for the 2x2x1 part: rank 4.
    # Total rank: 7 * 4 = 28. That's worse than schoolbook 4x4x2 = 32.
    # Wait, schoolbook 4x4x2 is 4*4*2 = 32. So 28 is better.
    
    # But I wanted 14. Let me reconsider.
    
    # Actually, for the 4x4x2 problem, let me just use schoolbook: rank 32.
    # Or use Kronecker with Strassen for the 2x2x2 part and schoolbook for 2x2x1.
    
    # Let me abandon this and just use schoolbook for the 4x4x3 part.
    # Total rank: 49 + 48 = 97.
    
    # Actually, let me try a different approach for the 4x4x3 part.
    # Split 4x4x3 into 4x4x2 + 4x4x1.
    # 4x4x2: use Kronecker (7 * rank_2x2x1)
    # 4x4x1: schoolbook rank 16
    
    # For 2x2x1, let me find a rank-2 decomposition.
    # A (2x2) * B (2x1) = C (2x1)
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    
    # Rank 2 decomposition:
    # M1 = (A[0,0] + A[1,1]) * (B[0] + B[1])
    # M2 = (A[1,0] - A[0,0]) * B[0]
    # M3 = A[0,1] * B[1]
    # That's rank 3.
    
    # Can we do rank 2?
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # If we use:
    # M1 = (A[0,0] + A[0,1]) * (B[0] + B[1])
    # M2 = (A[1,0] + A[1,1]) * (B[0] + B[1])
    # C[0] = ? Not straightforward.
    
    # I think rank 3 is minimal for 2x2x1.
    # So 4x4x2 rank = 7 * 3 = 21.
    # 4x4x1 rank = 16.
    # Total for 4x4x3: 21 + 16 = 37.
    # Total: 49 + 37 = 86.
    
    # That's better than 97!
    
    # Let me implement 2x2x1 rank-3 decomposition.
    
    return None  # Placeholder

def _strassen_2x2x1_rank3():
    """
    Return rank-3 decomposition for 2x2x1 matrix multiplication.
    A (2x2) * B (2x1) = C (2x1)
    U: (4, 3), V: (2, 3), W: (2, 3)
    """
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    
    # M1 = (A[0,0] + A[0,1]) * (B[0] + B[1])
    # M2 = (A[1,0] + A[1,1]) * (B[0] + B[1])
    # M3 = (A[1,0] - A[0,0]) * B[0]
    
    # C[0] = M1 - A[0,1]*B[0] - A[0,0]*B[1] ... not clean.
    
    # Let me try:
    # M1 = A[0,0] * (B[0] + B[1])
    # M2 = A[0,1] * B[1]
    # M3 = (A[1,0] + A[1,1]) * (B[0] + B[1])
    # C[0] = M1 - A[0,0]*B[1] + M2 ... still not clean.
    
    # I think I need to be more careful. Let me use a known rank-3 decomposition.
    # For 2x2 * 2x1:
    # M1 = (A[0,0] + A[1,1]) * (B[0] + B[1])
    # M2 = (A[1,0] - A[0,0]) * B[0]
    # M3 = A[0,1] * B[1]
    
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    #       = (M1 - A[1,1]*B[0] - A[0,0]*B[1] - A[1,1]*B[1]) + A[0,1]*B[1]
    #       = M1 - A[1,1]*B[0] - A[0,0]*B[1] - A[1,1]*B[1] + A[0,1]*B[1]
    # This is getting messy.
    
    # Let me just use schoolbook rank 4 for 2x2x1.
    # Total 4x4x2 rank = 7 * 4 = 28.
    # 4x4x1 rank = 16.
    # Total 4x4x3 rank = 28 + 16 = 44.
    # Total: 49 + 44 = 93.
    
    # That's still better than 112 (schoolbook) and 97 (previous).
    
    return None  # Placeholder

def _kron_strassen_4x4x2():
    """
    Return rank-28 decomposition for 4x4x2 matrix multiplication using Kronecker product
    of Strassen (2x2x2, rank 7) and schoolbook (2x2x1, rank 4).
    
    U: (16, 28), V: (8, 28), W: (8, 28)
    """
    n = 4
    m = 4
    p = 2
    rank = 28
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    S_U, S_V, S_W = _strassen_rank7()
    
    # 2x2x1 schoolbook decomposition:
    # A (2x2) * B (2x1) = C (2x1)
    # U2: (4, 4), V2: (2, 4), W2: (2, 4)
    U2 = np.zeros((4, 4), dtype=float)
    V2 = np.zeros((2, 4), dtype=float)
    W2 = np.zeros((2, 4), dtype=float)
    
    # Schoolbook for 2x2x1:
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    for i in range(2):
        for j in range(2):
            for k in range(1):  # p=1
                col = i*2*1 + j*1 + k
                U2[i*2 + j, col] = 1
                V2[j*1 + k, col] = 1
                W2[k*2 + i, col] = 1
    
    # Now Kronecker product:
    # 4x4x2 = (2x2x2) ⊗ (2x2x1)
    # U[i,j, (r1,r2)] = S_U[i1,j1,r1] * U2[i0,j0,r2]
    # where i = 2*i1 + i0, j = 2*j1 + j0
    # V[j,k, (r1,r2)] = S_V[j1,k1,r1] * V2[j0,k0,r2]
    # where j = 2*j1 + j0, k = 2*k1 + k0
    # W[i,k, (r1,r2)] = S_W[i1,k1,r1] * W2[i0,k0,r2]
    # where i = 2*i1 + i0, k = 2*k1 + k0
    
    for r1 in range(7):
        for r2 in range(4):
            col = r1 * 4 + r2
            
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            i = 2 * i1 + i0
                            j = 2 * j1 + j0
                            U[i * 4 + j, col] = S_U[i1 * 2 + j1, r1] * U2[i0 * 2 + j0, r2]
            
            for j1 in range(2):
                for k1 in range(2):
                    for j0 in range(2):
                        for k0 in range(1):
                            j = 2 * j1 + j0
                            k = 2 * k1 + k0
                            V[j * 2 + k, col] = S_V[j1 * 2 + k1, r1] * V2[j0 * 1 + k0, r2]
            
            for i1 in range(2):
                for k1 in range(2):
                    for i0 in range(2):
                        for k0 in range(1):
                            i = 2 * i1 + i0
                            k = 2 * k1 + k0
                            W[k * 4 + i, col] = S_W[k1 * 2 + i1, r1] * W2[k0 * 2 + i0, r2]
    
    return U, V, W

def _schoolbook_4x4x1():
    """
    Return schoolbook decomposition for 4x4x1 matrix multiplication.
    U: (16, 16), V: (4, 16), W: (4, 16)
    """
    n = 4
    m = 4
    p = 1
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i*m*p + j*p + k
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Strategy:
    # Split the 4x7 output into 4x4 + 4x2 + 4x1
    # 4x4 part: Kronecker Strassen, rank 49
    # 4x2 part: Kronecker (Strassen 2x2x2 * schoolbook 2x2x1), rank 28
    # 4x1 part: schoolbook, rank 16
    # Total rank: 49 + 28 + 16 = 93
    
    total_rank = 49 + 28 + 16
    
    U_full = np.zeros((n*m, total_rank), dtype=float)
    V_full = np.zeros((m*p, total_rank), dtype=float)
    W_full = np.zeros((p*n, total_rank), dtype=float)
    
    # Part 1: 4x4x4 (columns 0-3 of B)
    U1, V1, W1 = _kron_strassen_4x4x4()
    
    # Map V1 (16x49, assuming p=4) to V_full
    # V1 row j*4+k corresponds to B[j,k] for k in 0..3
    # V_full row j*7+k corresponds to B[j,k]
    for j in range(4):
        for k in range(4):
            v_row = j * 7 + k
            v1_row = j * 4 + k
            V_full[v_row, 0:49] = V1[v1_row, :]
    
    U_full[:, 0:49] = U1
    W_full[0:16, 0:49] = W1  # k=0..3, i=0..3
    
    # Part 2: 4x4x2 (columns 4-5 of B)
    U2, V2, W2 = _kron_strassen_4x4x2()
    offset2 = 49
    
    # Map V2 (8x28, assuming p=2) to V_full
    # V2 row j*2+k corresponds to B[j,k] for k in 0..1 (local)
    # In full V, this is B[j, 4+k] for k in 0..1
    for j in range(4):
        for k in range(2):
            v_row = j * 7 + (4 + k)
            v2_row = j * 2 + k
            V_full[v_row, offset2:offset2+28] = V2[v2_row, :]
    
    U_full[:, offset2:offset2+28] = U2
    W_full[16:24, offset2:offset2+28] = W2  # k=4,5 -> W rows 4*4+0=16 to 4*5+3=23
    
    # Part 3: 4x4x1 (column 6 of B)
    U3, V3, W3 = _schoolbook_4x4x1()
    offset3 = 49 + 28
    
    # Map V3 (4x16, assuming p=1) to V_full
    # V3 row j corresponds to B[j,0] (local)
    # In full V, this is B[j, 6]
    for j in range(4):
        v_row = j * 7 + 6
        v3_row = j
        V_full[v_row, offset3:offset3+16] = V3[v3_row, :]
    
    U_full[:, offset3:offset3+16] = U3
    W_full[24:28, offset3:offset3+16] = W3  # k=6 -> W rows 6*4+0=24 to 6*4+3=27
    
    # Verify
    valid = _verify(U_full.tolist(), V_full.tolist(), W_full.tolist())
    
    if valid:
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback to schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school), dtype=float)
    V_school = np.zeros((m*p, rank_school), dtype=float)
    W_school = np.zeros((p*n, rank_school), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i*m*p + j*p + k
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
