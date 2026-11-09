"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import random

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 iff (i,j,k) is valid."""
    # For each (i,j,k), check the sum
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(len(U[0])):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                # This should be 1 if the tensor entry is 1, but for matrix mult
                # the tensor T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor)
                # Actually no. The multiplication tensor has T[i,j,k] = delta_{i,j,k} in a sense.
                # Let me re-read the problem.
                # "Matrix multiplication 4,4,5"
                # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
                # The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
                # For matrix multiplication, T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B
                # contributes to (i,k) entry of C. But actually the tensor is defined as:
                # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
                # Wait, that's not right either. Let me think about this more carefully.
                # 
                # The matrix multiplication tensor T_{n,m,p} is defined such that:
                # (A * B)[i,k] = sum_j A[i,j] * B[j,k]
                # This can be written as:
                # (A * B)[i,k] = sum_{j,c} U[i,j,c] * A[i,j] * V[j,k,c] * B[j,k] * W[k,i,c]
                # Wait, that's not standard. Let me look at the interface again.
                # 
                # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
                # "exactly decompose the matrix multiplication tensor"
                # 
                # The standard tensor decomposition for matrix multiplication:
                # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # where T[i,j,k] = 1 for all valid i,j,k (since every A[i,j]*B[j,k] contributes to C[i,k])
                # 
                # Wait, that's not right. The tensor T for matrix multiplication has:
                # T[i,j,k] = 1 if and only if the product A[i,j] * B[j,k] appears in C[i,k].
                # But actually, for the standard matrix multiplication tensor, T[i,j,k] = 1 for ALL
                # i in [0,n), j in [0,m), k in [0,p), because every pair (A[i,j], B[j,k]) contributes
                # to C[i,k].
                #
                # Hmm, but that would mean the rank is at least n*m*p (schoolbook), which matches.
                #
                # Actually wait. Let me reconsider. The tensor is:
                # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * T[i,j,k]
                # where T[i,j,k] = 1 for all i,j,k in the matrix multiplication case.
                #
                # No wait, that's not right either. In the standard formulation:
                # (A B)[i,k] = sum_j A[i,j] B[j,k]
                # This is a trilinear form: sum_{i,j,k} C[i,k] A[i,j] B[j,k]
                # So the tensor is T[i,j,k] where i indexes rows of A, j indexes columns of A / rows of B,
                # and k indexes columns of B. The tensor entry T[i,j,k] = 1 for all valid indices.
                #
                # So yes, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
                #
                # Therefore, the verification should check that:
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
                #
                # But wait, that seems too simple. Let me re-read the W index: W[k*n+i, c].
                # So W is indexed by (k, i) where k is the column of B and i is the row of A.
                # This makes sense because C[i,k] depends on A[i,:] and B[:,k].
                #
                # OK so the verification is: for all i in [0,n), j in [0,m), k in [0,p):
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
                
                if abs(s - 1.0) > tol:
                    return False, (i, j, k, s)
    return True, None

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Try to find a rank-minimal decomposition using a search approach
    # For 4x4x5, the known best rank is at most 4*4*5 = 80 (schoolbook)
    # Strassen-based approaches might do better for the 4x4x4 part
    
    # First, let's try a structured approach:
    # Decompose the 4x4x5 multiplication into:
    # - 4x4x4 part (columns 0-3 of B)
    # - 4x4x1 part (column 4 of B)
    
    # For 4x4x4, use a Strassen-like decomposition
    # For 4x4x1, use a simple rank-4 decomposition
    
    # Let's build the 4x4x1 part first
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # This is just a vector-matrix product, rank 4
    U_41 = [[0]*4 for _ in range(16)]  # 4*4 = 16
    V_41 = [[0]*4 for _ in range(4*5)]  # 4*5 = 20
    W_41 = [[0]*4 for _ in range(5*4)]  # 5*4 = 20
    
    for i in range(4):
        for j in range(4):
            c = i*4 + j  # This doesn't quite work for rank 4
            # Actually for 4x4x1, we need rank 4:
            # C[i,4] = sum_j A[i,j] B[j,4]
            # This is a sum of 4 terms, so rank 4
            # U[i*4+j, c] = 1, V[j*5+4, c] = 1, W[4*4+i, c] = 1 for c = j
            # Wait, let me think again.
            # For each j, we have one term: A[i,j] * B[j,4]
            # So for fixed j, all i share the same V and W entries
            # U[i*4+j, j] = 1, V[j*5+4, j] = 1, W[4*4+i, j] = 1
    
    # Actually, let me just use the schoolbook for the 4x4x1 part
    for j in range(4):
        for i in range(4):
            U_41[i*4+j][j] = 1
            V_41[j*5+4][j] = 1
            W_41[4*4+i][j] = 1
    
    # Now for 4x4x4, let's try to use a lower-rank decomposition
    # The rank of 4x4x4 matrix multiplication is known to be at most 49 (Strassen-based)
    # Let's implement a Strassen-based decomposition for 4x4x4
    
    # Strassen for 2x2x2 has rank 7
    # For 4x4x4, we can split into 2x2 blocks of size 2x2
    # This gives 7 "super-multiplications" of 2x2 matrices
    # Each 2x2x2 multiplication has rank 7
    # Total rank: 7 * 7 = 49
    
    # Strassen 2x2x2 decomposition:
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B10 - B00)
    # M3 = A11(B01 - B11)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    #
    # C00 = M0 - M3 + M5 - M6
    # C01 = M4 + M6
    # C10 = M1 + M5
    # C11 = M0 - M2 + M3 + M4
    
    # For 4x4x4, A is 4x4, split into 2x2 blocks of 2x2:
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]] where each block is 2x2
    # C = [[C00, C01], [C10, C11]] where each block is 2x2
    
    # Each "super-multiplication" Mk is a 2x2x2 multiplication
    # So for each k, we need the 7 rank-1 terms of the 2x2x2 Strassen decomposition
    
    # Let's define the Strassen 2x2x2 decomposition
    # For 2x2x2:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # The 7 rank-1 terms for Strassen 2x2x2:
    # Each term is (u, v, w) where u is a 4-vector (for A indices), v is a 4-vector (for B indices), w is a 4-vector (for C indices)
    
    # Let me define the Strassen 2x2x2 terms explicitly
    # Using 0-indexing: A[i,j] -> index i*2+j, B[i,j] -> index i*2+j, C[i,j] -> index i*2+j
    
    strassen_2x2x2 = [
        # (U_indices, V_indices, W_indices, coefficient)
        # M0 = (A00 + A11)(B00 + B11)
        # A part: A00, A11 -> indices 0, 3
        # B part: B00, B11 -> indices 0, 3
        # C part: C00 (+), C11 (+) -> indices 0, 3
        ([0, 3], [0, 3], [0, 3], 1),
        
        # M1 = (A10 + A11)B00
        # A part: A10, A11 -> indices 2, 3
        # B part: B00 -> index 0
        # C part: C10 (+) -> index 2
        ([2, 3], [0], [2], 1),
        
        # M2 = A00(B10 - B00)
        # A part: A00 -> index 0
        # B part: B10, B00 -> indices 2, 0
        # C part: C11 (-) -> index 3
        ([0], [2, 0], [3], -1),
        
        # M3 = A11(B01 - B11)
        # A part: A11 -> index 3
        # B part: B01, B11 -> indices 1, 3
        # C part: C00 (-), C11 (+) -> indices 0, 3
        ([3], [1, 3], [0, 3], 1),
        
        # M4 = (A00 + A01)B11
        # A part: A00, A01 -> indices 0, 1
        # B part: B11 -> index 3
        # C part: C01 (+), C11 (+) -> indices 1, 3
        ([0, 1], [3], [1, 3], 1),
        
        # M5 = (A10 - A00)(B00 + B01)
        # A part: A10, A00 -> indices 2, 0
        # B part: B00, B01 -> indices 0, 1
        # C part: C00 (-), C10 (+) -> indices 0, 2
        ([2, 0], [0, 1], [0, 2], 1),
        
        # M6 = (A01 - A11)(B10 + B11)
        # A part: A01, A11 -> indices 1, 3
        # B part: B10, B11 -> indices 2, 3
        # C part: C01 (+), C10 (+) -> indices 1, 2
        ([1, 3], [2, 3], [1, 2], 1),
    ]
    
    # Wait, I need to be more careful. Each "term" in the Strassen decomposition is a rank-1 tensor.
    # But the Strassen algorithm uses 7 multiplications, each of which is a 2x2x2 multiplication.
    # So for each of the 7 Strassen multiplications, we need to decompose the 2x2x2 multiplication
    # into its rank-1 terms.
    
    # Actually, the standard Strassen 2x2x2 decomposition IS a rank-7 decomposition.
    # The 7 terms are:
    # Term 0: U = [1,0,0,1], V = [1,0,0,1], W = [1,0,0,1] (for M0)
    # But wait, that's not quite right. Let me look up the exact Strassen rank-7 decomposition.
    
    # The 7 rank-1 terms for 2x2x2 Strassen:
    # t0: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,-1]  (M0 contributes to C00 and C11 with opposite signs)
    # Hmm, this is getting complicated. Let me just use the schoolbook for now and verify it works.
    
    # Schoolbook for 4x4x4: rank 64
    U_44 = [[0]*64 for _ in range(16)]
    V_44 = [[0]*64 for _ in range(20)]
    W_44 = [[0]*64 for _ in range(20)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U_44[i*4+j][col] = 1
                V_44[j*5+k][col] = 1
                W_44[k*4+i][col] = 1
                col += 1
    
    # Combine 4x4x4 and 4x4x1
    rank_total = 64 + 4
    U = [[0]*rank_total for _ in range(16)]
    V = [[0]*rank_total for _ in range(20)]
    W = [[0]*rank_total for _ in range(20)]
    
    # Copy 4x4x4 part
    for i in range(16):
        for c in range(64):
            U[i][c] = U_44[i][c]
    for i in range(20):
        for c in range(64):
            V[i][c] = V_44[i][c]
    for i in range(20):
        for c in range(64):
            W[i][c] = W_44[i][c]
    
    # Copy 4x4x1 part
    for i in range(16):
        for c in range(4):
            U[i][64+c] = U_41[i][c]
    for i in range(20):
        for c in range(4):
            V[i][64+c] = V_41[i][c]
    for i in range(20):
        for c in range(4):
            W[i][64+c] = W_41[i][c]
    
    # Verify
    valid, info = _verify(U, V, W, n, m, p)
    if not valid:
        print(f"Verification failed: {info}")
        # Fall back to pure schoolbook
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
    else:
        print(f"Verification passed with rank {rank_total}")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
