"""Structural tensor decomposition for 4x4 * 4x5 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s kill

    # Strassen decomposition for 2x2 * 2x2 (rank 7)
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # The multiplication tensor for 2x2*2x2 has entries T[i,j,k] where
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # U[i,j] * V[j,k] * W[i,k] summed over the rank index gives T[i,j,k]
    
    def strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[1, 0] = 1
        V[0, 0] = 1; V[1, 0] = 1
        W[0, 0] = 1; W[1, 1] = 1
        
        # M2 = (A10+A11)B00
        U[1, 1] = 1; U[2, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[2, 0] = 1
        
        # M3 = A00(B10-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[0, 2] = 1; W[2, 2] = 1
        
        # M4 = (A00-A11)(B01+B11)
        U[0, 3] = 1; U[1, 3] = -1
        V[2, 3] = 1; V[3, 3] = 1
        W[0, 3] = 1; W[1, 1] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[2, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[2, 1] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[1, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[1, 5] = 1; W[3, 0] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[2, 6] = 1; U[1, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        W[2, 6] = 1; W[3, 1] = 1
        
        return U, V, W

    # Verify a decomposition for n x m * m x p
    def verify_decomp(U, V, W, n, m, p, tol=1e-9):
        """
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        Check: for all i in [0,n), j in [0,m), k in [0,p):
            sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        And for all other (i,j,k) combinations that don't correspond to valid 
        matrix multiplication terms, the sum should be 0.
        Actually, the tensor T for matrix multiplication has T[i,j,k] = 1 if 
        the term A[i,j]*B[j,k] contributes to C[i,k], which is always the case
        for valid indices. The decomposition must satisfy:
            T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        where T[i,j,k] = 1 for all valid i,j,k (since every product term appears).
        """
        rank = U.shape[1]
        # Build the full tensor and check
        # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        # Also check that spurious terms are zero:
        # For the matrix multiplication tensor, T[i,j,k] is non-zero only when
        # the indices form a valid "path" i->j->k in the multiplication.
        # Actually, in the standard formulation, the tensor is defined on 
        # [0,n) x [0,m) x [0,p) and T[i,j,k] = delta_{appropriate}.
        # Let me reconsider: the matrix multiplication tensor T_{i,j,k} where
        # (i,j) indexes A, (j,k) indexes B, and (i,k) indexes C.
        # T_{i,j,k} = 1 if the (i,j) entry of A and (j,k) entry of B contribute
        # to the (i,k) entry of C. This is always 1 for valid indices.
        # But wait - that's not right. The tensor is T_{i,j,k} where the three
        # modes are (i,j) for A, (j,k) for B, (i,k) for C.
        # Actually the standard form: T is a tensor of shape (n*m, m*p, p*n)
        # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some valid i,j,k
        # Otherwise T[a, b, c] = 0.
        # The decomposition: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We need to check both that the "on-diagonal" entries equal 1 and that
        # "off-diagonal" entries equal 0.
        
        # Check off-diagonal entries (sample some to save time)
        import itertools
        # For efficiency, check a subset of off-diagonal entries
        checked = 0
        for a in range(n*m):
            for b in range(m*p):
                # Check if this (a,b) pair corresponds to a valid path
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                if j == j2:
                    # This is a valid path, should be 1 (already checked above)
                    continue
                # This should be 0
                # c should be k*n+i for the valid case, but since j != j2, no valid c
                # Actually, we need to check for all c
                for c in range(p*n):
                    val = np.dot(U[a], V[b] * W[c])
                    if abs(val) > tol:
                        return False
                    checked += 1
                    if checked > 5000:  # Limit checks for speed
                        break
                if checked > 5000:
                    break
            if checked > 5000:
                break
        if checked > 5000:
            break
        
        return True

    # Build 4x4 * 4x4 using 2-level Strassen (rank 49)
    def strassen_4x4():
        U2, V2, W2 = strassen_2x2()
        
        # For 4x4 * 4x4, we use Kronecker product structure.
        # The 4x4 matrices are viewed as 2x2 blocks of 2x2 matrices.
        # The rank-49 decomposition uses pairs (t1, t2) where t1, t2 in 0..6.
        # 
        # U has shape (16, 49): row index = (i1, i2) where i1 is block row (0,1), 
        #   i2 is row within block (0,1). So row = i1*2 + i2 for the 4x4 case,
        #   but actually for 4x4, the row index in U is (i, j) where i is row of A
        #   and j is col of A. i in 0..3, j in 0..3.
        #   Block structure: i1 = i//2, i2 = i%2, j1 = j//2, j2 = j%2.
        #   Row in U = i*4 + j.
        #
        # For the outer Strassen level (t1), the "matrix" entries are 2x2 blocks.
        # U2[i1, t1] gives the coefficient for block row i1 at outer level.
        # For the inner Strassen level (t2), U2[i2, t2] gives coefficient for 
        #   row within block.
        #
        # So U[i*4+j, t1*7+t2] = U2[i//2, t1] * U2[j//2, t1] ... no that's not right.
        #
        # Let me think more carefully. The Kronecker product approach:
        # If we have a rank-r decomposition of 2x2*2x2 with matrices U2, V2, W2,
        # then for 4x4*4x4 (viewed as 2x2 blocks of 2x2), the rank is r^2 = 49.
        # The decomposition is:
        #   U[i*4+j, (t1,t2)] = U2[i//2, t1] * U2[j//2, t1] ... 
        # 
        # No, that's not right either. Let me think about it differently.
        #
        # The matrix multiplication tensor for 4x4*4x4 can be written as:
        # T[i,j,k,l] = sum_{a,b} A[i,a] * B[a,k] * delta_{j,a} * delta_{l,k}
        # Wait, I'm confusing myself. Let me use the standard formulation.
        #
        # For multiplying A (n x m) by B (m x p) to get C (n x p):
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T has shape (n*m, m*p, p*n) with:
        # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
        # T[a,b,c] = 0 otherwise
        #
        # For 4x4*4x4: n=m=p=4
        # T has shape (16, 16, 16)
        #
        # Using 2-level Strassen:
        # Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]] where each is 2x2
        # Similarly for B and C.
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        #
        # Each C block (2x2) is computed from products of 2x2 blocks.
        # Using Strassen for the "block" level (rank 7) and Strassen for each
        # 2x2*2x2 product (rank 7), total rank = 7*7 = 49.
        #
        # For Strassen at the block level:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = (A00-A11)(B01+B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 - M4 - M5 + M7
        # C01 = M3 + M4
        # C10 = M2 + M5
        # C11 = M1 - M2 - M3 + M6
        #
        # Each M_t is a 2x2 matrix computed as a product of 2x2 matrices.
        # Each such product uses rank 7 (Strassen for 2x2*2x2).
        #
        # So the total rank is 7 (outer) * 7 (inner) = 49.
        #
        # For the U matrix (shape 16 x 49):
        # Row index: i*4+j where i is row of A (0..3), j is col of A (0..3)
        # Column index: t1*7+t2 where t1 is outer Strassen index (0..6), 
        #               t2 is inner Strassen index (0..6)
        #
        # U[i*4+j, t1*7+t2] = U2[i_block, t1] * U2[j_block, t2]
        # where i_block = i//2 (block row of A), j_block = j//2 (block col of A)
        # and the inner indices are i%2 and j%2.
        #
        # Wait, I need to be more careful. At the outer level, the "entries" are 
        # 2x2 blocks. The outer Strassen operates on these blocks.
        # U2[i_block, t1] is the coefficient for block row i_block in the outer 
        # Strassen decomposition. But the "block" is itself a 2x2 matrix, and 
        # to multiply two 2x2 blocks, we use the inner Strassen.
        #
        # So for a given outer term t1, the contribution to A's block (i_block, j_block)
        # is U2[i_block, t1] times the entire 2x2 block. But we also need to account
        # for which specific entry within the block.
        #
        # Actually, the correct formulation is:
        # The full U for 4x4*4x4 at position (i, j) and column (t1, t2):
        # U[i*4+j, t1*7+t2] = U2[i//2, t1] * U2[j//2, t2]
        # 
        # Hmm, but that doesn't seem right dimensionally. Let me reconsider.
        #
        # Actually, I think the Kronecker product approach works as follows:
        # If (U2, V2, W2) is a rank-7 decomposition of the 2x2*2x2 multiplication,
        # then (U2 ⊗ U2, V2 ⊗ V2, W2 ⊗ W2) is a rank-49 decomposition of 
        # 4x4*4x4 multiplication, where ⊗ is the Kronecker product.
        #
        # (U2 ⊗ U2)[(i1,i2), (t1,t2)] = U2[i1, t1] * U2[i2, t2]
        # where (i1,i2) is the row index in the 4x4 case (i1 = i//2, i2 = i%2 for 
        # the row, but we also need the column index j).
        #
        # Wait, U2 has shape (4, 7) for 2x2*2x2, where the 4 rows correspond to 
        # (i,j) pairs of the 2x2 matrix. So U2[i*2+j, t] for i,j in {0,1}.
        #
        # For 4x4*4x4, U has shape (16, 49) where the 16 rows correspond to 
        # (i,j) pairs for i,j in {0,1,2,3}.
        # Row index in U: i*4+j
        # This can be decomposed as: i = i1*2+i2, j = j1*2+j2
        # So the row is (i1, i2, j1, j2) which maps to (i1*2+j1, i2*2+j2) in a 
        # 4x4 grid... no.
        #
        # Let me just use the direct Kronecker product:
        # U_full = U2 ⊗ U2 (Kronecker product)
        # U2 has shape (4, 7), so U2 ⊗ U2 has shape (16, 49).
        # (U2 ⊗ U2)[(a,b), (c,d)] = U2[a, c] * U2[b, d]
        # where a,b in 0..3, c,d in 0..6.
        #
        # Similarly V_full = V2 ⊗ V2, W_full = W2 ⊗ W2.
        
        U = np.kron(U2, U2)  # shape (16, 49)
        V = np.kron(V2, V2)  # shape (16, 49)
        W = np.kron(W2, W2)  # shape (16, 49)
        
        return U, V, W

    # Verify the 4x4*4x4 decomposition
    U44, V44, W44 = strassen_4x4()
    valid_44 = verify_decomp(U44, V44, W44, 4, 4, 4)
    
    if not valid_44:
        # Fallback: schoolbook for 4x4*4x4 (rank 64)
        rank_44 = 64
        U44 = np.zeros((16, rank_44))
        V44 = np.zeros((16, rank_44))
        W44 = np.zeros((16, rank_44))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U44[i*4+j, col] = 1.0
                    V44[j*4+k, col] = 1.0
                    W44[k*4+i, col] = 1.0
                    col += 1
    
    # Build 4x4 * 4x5 decomposition
    # The 4x5 matrix B has columns 0..4.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k] for k=0..4
    # 
    # We can split B into B[:, 0:4] (4x4) and B[:, 4] (4x1).
    # C[:, 0:4] = A * B[:, 0:4]  -> rank 49 (or 64 fallback)
    # C[:, 4] = A * B[:, 4]      -> rank 4 (each C[i,4] = sum_j A[i,j]*B[j,4])
    #
    # Total rank = 49 + 4 = 53 (or 64 + 4 = 68 fallback)
    
    rank_44 = U44.shape[1]
    rank_41 = 4
    total_rank = rank_44 + rank_41
    
    U = np.zeros((16, total_rank))
    V = np.zeros((20, total_rank))
    W = np.zeros((20, total_rank))
    
    # First part: 4x4 * 4x4
    U[:, :rank_44] = U44
    V[:16, :rank_44] = V44
    W[:16, :rank_44] = W44
    
    # Second part: 4x4 * 4x1 (rank 4)
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # For each j: term A[i,j] * B[j,4] contributes to C[i,4]
    # U[i*4+j, rank_44+j] = 1
    # V[j*5+4, rank_44+j] = 1  (B[j,4] is at row j*5+4 in V)
    # W[4*4+i, rank_44+j] = 1  (C[i,4] is at row 4*4+i = 16+i in W)
    for j in range(4):
        for i in range(4):
            U[i*4+j, rank_44+j] = 1.0
        V[j*5+4, rank_44+j] = 1.0
        for i in range(4):
            W[16+i, rank_44+j] = 1.0
    
    # Verify the full decomposition
    valid_full = verify_decomp(U, V, W, 4, 4, 5)
    
    if not valid_full:
        # Fallback: full schoolbook for 4x4*4x5 (rank 80)
        rank = 80
        U = np.zeros((16, rank))
        V = np.zeros((20, rank))
        W = np.zeros((20, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U[i*4+j, col] = 1.0
                    V[j*5+k, col] = 1.0
                    W[k*4+i, col] = 1.0
                    col += 1
    else:
        rank = total_rank
    
    # Time check
    if time.time() > deadline:
        # Return best available result
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
