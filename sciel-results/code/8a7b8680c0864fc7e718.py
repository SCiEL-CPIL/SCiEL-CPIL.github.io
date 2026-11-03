import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Strategy: Strassen for 2x2x2 blocks where i-block has size 2
    # i-blocks: {0,1} (size 2) and {2} (size 1)
    # j-blocks: {0,1} and {2,3}
    # k-blocks: {0,1}, {2,3}, {4,5}
    #
    # For i-block {0,1}: 3 k-blocks × 2 j-blocks = 6 Strassen calls, each rank 7 → 42
    # For i-block {2}: 3 k-blocks × 2 j-blocks = 6 schoolbook 1x2x2, each rank 4 → 24
    # Total: 42 + 24 = 66
    
    # But wait, that's the same as current. Let me think if I can do better.
    #
    # Alternative: Can I use a 1x2x2 rank-3 decomposition? No, rank of 1x2x2 is 4 (it's just outer products).
    # Actually, 1x2x2: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]. This is a sum of 2 rank-1 tensors in the (j,k) space for fixed i.
    # The tensor is sum over j of A[i,j] ⊗ B[j,k]. For fixed i, this is a 2x2 matrix multiplication viewed as a tensor.
    # Rank of 1x2x2 matmul tensor = 2 (since it's just A[i,0]*B[0,:] + A[i,1]*B[1,:], but these share the same i index).
    # Actually no: the tensor T[i,j,k] = delta(i,i0) * ... hmm, let me think again.
    
    # For the 1x2x2 subproblem: C[i0, k] = A[i0, j0]*B[j0, k] + A[i0, j0+1]*B[j0+1, k]
    # In tensor form: T[i,j,k] = 1 if i=i0, and the (j,k) part is the 2x2 matrix product.
    # The rank-1 decomposition: T = A_col ⊗ B_row ⊗ C_mat where... 
    # Actually, each term A[i0,j]*B[j,k] is rank 1 in the tensor (i,j,k) space:
    # T1[i,j,k] = delta(i,i0) * delta(j,j0) * B[j0,k] — but B[j0,k] varies with k, so this isn't a simple rank-1 tensor.
    # 
    # The correct view: the tensor for matmul(1,2,2) has T[i,j,k] = 1 if i=0, and it represents C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k].
    # As a tensor in (i,j,k) space with i fixed at 0: T[0,j,k] = 1 for the "identity" mapping.
    # The rank is 4: we need 4 rank-1 terms, one for each (j,k) pair.
    
    # So my calculation of 66 is correct for this partitioning.
    
    # Let me try a different partitioning to see if I can get below 66.
    # What if I use Strassen for the full 2x4x6 part (rows 0-1 of A)?
    # 2x4x6: split into j-blocks {0,1}, {2,3} and k-blocks {0,1}, {2,3}, {4,5}
    # That's 2×3 = 6 blocks of 2x2x2, each rank 7 → 42
    # Plus row 2: 1x4x6 = 6 blocks of 1x2x2... no, 1x4x6 = 24 terms
    # Total: 42 + 24 = 66. Same.
    
    # What about using a 3x2x2 Strassen-like? The rank of 3x2x2 matmul...
    # 3x2x2: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=0,1,2
    # This is rank 6 (schoolbook) or can it be done in less?
    # The tensor is sum_j sum_k A[:,j] ⊗ B[j,:] ⊗ ... 
    # I believe rank(3,2,2) = 6 since the 2x2 part has rank 4 and we have 3 rows... 
    # Actually, it's the sum over j of (A[:,j] ⊗ B[j,:]) which is 2 terms, each of which is a 3x2 matrix viewed as...
    # I think rank(3,2,2) = 6 is minimal.
    
    # Let me just go with the 66 decomposition and make sure it's correct.
    
    rank_total = 66
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    col = 0
    
    def strassen_block(i0, j0, k0):
        """Strassen for A[i0:i0+2, j0:j0+2] x B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
        nonlocal col
        ai, bi = i0, i0 + 1
        aj, bj = j0, j0 + 1
        bk, ck = k0, k0 + 1
        
        # m1 = (a+d)(e+h): (A[ai,aj]+A[bi,bj])*(B[aj,bk]+B[bj,ck])
        U[ai*m+aj][col] = 1; U[bi*m+bj][col] = 1
        V[aj*p+bk][col] = 1; V[bj*p+ck][col] = 1
        W[bk*n+ai][col] = 1; W[ck*n+bi][col] = 1
        col += 1
        
        # m2 = (c+d)e: (A[bi,aj]+A[bi,bj])*B[aj,bk]
        U[bi*m+aj][col] = 1; U[bi*m+bj][col] = 1
        V[aj*p+bk][col] = 1
        W[bk*n+bi][col] = 1; W[ck*n+bi][col] = -1
        col += 1
        
        # m3 = a(f-h): A[ai,aj]*(B[aj,ck]-B[bj,ck])
        U[ai*m+aj][col] = 1
        V[aj*p+ck][col] = 1; V[bj*p+ck][col] = -1
        W[ck*n+ai][col] = 1; W[ck*n+bi][col] = 1
        col += 1
        
        # m4 = d(g-e): A[bi,bj]*(B[bj,bk]-B[aj,bk])
        U[bi*m+bj][col] = 1
        V[bj*p+bk][col] = 1; V[aj*p+bk][col] = -1
        W[bk*n+ai][col] = 1; W[bk*n+bi][col] = 1
        col += 1
        
        # m5 = (a+b)h: (A[ai,aj]+A[ai,bj])*B[bj,ck]
        U[ai*m+aj][col] = 1; U[ai*m+bj][col] = 1
        V[bj*p+ck][col] = 1
        W[bk*n+ai][col] = -1; W[ck*n+ai][col] = 1
        col += 1
        
        # m6 = (c-a)(e+f): (A[bi,aj]-A[ai,aj])*(B[aj,bk]+B[aj,ck])
        U[bi*m+aj][col] = 1; U[ai*m+aj][col] = -1
        V[aj*p+bk][col] = 1; V[aj*p+ck][col] = 1
        W[ck*n+bi][col] = 1
        col += 1
        
        # m7 = (b-d)(g+h): (A[ai,bj]-A[bi,bj])*(B[bj,bk]+B[bj,ck])
        U[ai*m+bj][col] = 1; U[bi*m+bj][col] = -1
        V[bj*p+bk][col] = 1; V[bj*p+ck][col] = 1
        W[bk*n+ai][col] = 1
        col += 1
    
    def schoolbook_1x2x2(i0, j0, k0):
        """A[i0, j0:j0+2] x B[j0:j0+2, k0:k0+2] -> C[i0, k0:k0+2]"""
        nonlocal col
        for j in range(2):
            for k in range(2):
                U[i0*m+(j0+j)][col] = 1
                V[(j0+j)*p+(k0+k)][col] = 1
                W[(k0+k)*n+i0][col] = 1
                col += 1
    
    # Rows 0-1 of A, all j-blocks and k-blocks
    for j_base in [0, 2]:
        for k_base in [0, 2, 4]:
            strassen_block(0, j_base, k_base)
    
    # Row 2 of A, all j-blocks and k-blocks
    for j_base in [0, 2]:
        for k_base in [0, 2, 4]:
            schoolbook_1x2x2(2, j_base, k_base)
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    recon = np.zeros((n*m, m*p, p*n))
    for c in range(rank_total):
        recon += np.einsum('i,j,k->ijk', U_arr[:,c], V_arr[:,c], W_arr[:,c])
    
    if not np.allclose(recon, target, atol=1e-6):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col_fb] = 1
                    V_fb[j*p+k][col_fb] = 1
                    W_fb[k*n+i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}

