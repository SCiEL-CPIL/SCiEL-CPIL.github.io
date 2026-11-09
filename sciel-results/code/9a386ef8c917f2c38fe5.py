"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    def _build_schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.int32)
        V = np.zeros((m * p, rank), dtype=np.int32)
        W = np.zeros((p * n, rank), dtype=np.int32)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        """Verify that U, V, W form a valid tensor decomposition."""
        rank = U.shape[1]
        # Build the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # The target tensor has T[i,j,k] = 1 if i,j,k are "compatible" (j appears in both)
        # Actually for matrix mult: T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
        # contribute to (i,k) entry of C. So T[i,j,k] = 1 for all valid i,j,k.
        
        # Check: for each (i,k) pair, sum_j T[i,j,k] should give C[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor identity is: C[i,k] = sum_j A[i,j] * B[j,k]
        # In tensor form: C[i,k] = sum_{j,r} A[i,j] * B[j,k] * delta_{ir}
        # Our decomposition: C[i,k] = sum_r (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k]) * W[k*n+i,r]
        
        # Actually the standard form is:
        # C[i,k] = sum_r U[i,m,r] * V[m,k,r] * W[k,n,r] where U has shape (n,m,rank) etc.
        # But here U has shape (n*m, rank), so U[i*m+j, r] corresponds to the (i,j) coefficient.
        
        # The tensor T is defined by: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # This should equal 1 for all (i,j,k) where j is the shared index.
        
        # Wait, I need to be more careful. The matrix multiplication tensor T_{ijk} satisfies:
        # C_{ik} = sum_j A_{ij} B_{jk}
        # So T_{ijk} = 1 for all i,j,k (it's the all-ones tensor in the appropriate indices).
        
        # Let me verify by checking a random matrix multiplication.
        rng = np.random.RandomState(42)
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        C_computed = np.zeros((n, p))
        for r in range(rank):
            # U[:, r] gives coefficients for A
            # V[:, r] gives coefficients for B  
            # W[:, r] gives coefficients for C
            # The bilinear form: C += (U_col @ A) * (V_col @ B) * W_col
            # But U has shape (n*m, rank), so U[:, r] is a vector of length n*m
            # We need to reshape: U[:, r] -> (n, m)
            U_col = U[:, r].reshape(n, m)
            V_col = V[:, r].reshape(m, p)
            W_col = W[:, r]  # shape (p*n,)
            
            # The term is: sum_{i,j,k} U[i,j,r] * A[i,j] * V[j,k,r] * B[j,k] * W[k*n+i, r]
            # This equals: sum_r (sum_{i,j} U[i,j,r]*A[i,j]) * (sum_{j,k} V[j,k,r]*B[j,k]) * ... 
            # No, that's not right. Let me think again.
            
            # Actually, the standard tensor decomposition for matrix multiplication is:
            # C[i,k] = sum_r alpha_r(i) * beta_r(k) * gamma_r
            # where alpha_r(i) depends on A, beta_r(k) depends on B, gamma_r is a scalar.
            
            # In our formulation with U, V, W:
            # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
            
            # This doesn't look right either. Let me reconsider.
            
            # The correct interpretation:
            # U[i*m+j, r] is the coefficient for A[i,j] in the r-th term
            # V[j*p+k, r] is the coefficient for B[j,k] in the r-th term
            # W[k*n+i, r] is the coefficient for C[i,k] in the r-th term
            
            # So: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
            
            # But this is a trilinear form, not bilinear. Matrix multiplication is bilinear.
            
            # I think the issue is that the standard decomposition is:
            # C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k])
            # without the W factor, or W is absorbed.
            
            # Let me check the seed program's convention. In the seed:
            # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
            # So for schoolbook, each term r corresponds to a unique (i,j,k) triple.
            # C[i,k] = sum_r W[k*n+i,r] * U[i*m+j_r,r] * A[i,j_r] * V[j_r*p+k,r] * B[j_r,k]
            # For schoolbook, W[k*n+i,r] = 1 only when r corresponds to (i,j_r,k), and U,V are 1.
            # So C[i,k] = sum_j A[i,j] * B[j,k], which is correct.
            
            # So the formula is: C[i,k] = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k])
            
            # Let me implement this correctly.
            pass
        
        # Let me redo the verification properly
        C_computed = np.zeros((n, p))
        for r in range(rank):
            # Compute sum_j U[i*m+j, r] * A[i,j] for each i
            # U[:, r] has shape (n*m,), reshape to (n, m)
            U_col = U[:, r].reshape(n, m)
            V_col = V[:, r].reshape(m, p)
            
            # For each i: alpha_i = sum_j U_col[i,j] * A[i,j]
            # For each k: beta_k = sum_j V_col[j,k] * B[j,k]
            # C[i,k] += W[k*n+i, r] * alpha_i * beta_k
            
            alpha = np.array([np.dot(U_col[i, :], A[i, :]) for i in range(n)])  # shape (n,)
            beta = np.array([np.dot(V_col[:, k], B[:, k]) for k in range(p)])  # shape (p,)
            
            # W[:, r] has shape (p*n,), reshape to (p, n) then transpose to (n, p)
            W_col = W[:, r].reshape(p, n).T  # shape (n, p)
            
            C_computed += W_col * np.outer(alpha, beta)
        
        return np.allclose(C_computed, C_expected, atol=tol)
    
    def _build_strassen_2x2x2():
        """Build a rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
        # Strassen's 7 multiplications for 2x2 matrices
        # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        # C = A @ B
        
        # M1 = (a11 + a22)(b11 + b22)
        # M2 = (a21 + a22)b11
        # M3 = a11(b12 - b22)
        # M4 = a22(b21 - b11)
        # M5 = (a11 + a12)b22
        # M6 = (a21 - a11)(b11 + b12)
        # M7 = (a12 - a22)(b21 + b22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        rank = 7
        U = np.zeros((4, rank), dtype=np.int32)
        V = np.zeros((4, rank), dtype=np.int32)
        W = np.zeros((4, rank), dtype=np.int32)
        
        # Indexing: A[i,j] -> U[i*2+j, r], B[j,k] -> V[j*2+k, r], C[i,k] -> W[k*2+i, r]
        
        # M1 = (a11 + a22)(b11 + b22)
        U[0, 0] = 1; U[3, 0] = 1  # a11 + a22
        V[0, 0] = 1; V[3, 0] = 1  # b11 + b22
        W[0, 0] = 1; W[3, 0] = 1  # C11 += M1, C22 += M1 (W[k*2+i]: C11 is W[0*2+0]=W[0], C22 is W[1*2+1]=W[3])
        
        # M2 = (a21 + a22)b11
        U[2, 1] = 1; U[3, 1] = 1  # a21 + a22
        V[0, 1] = 1  # b11
        W[2, 1] = 1  # C21 += M2 (W[1*2+0]=W[2])
        W[3, 1] = -1  # C22 -= M2
        
        # M3 = a11(b12 - b22)
        U[0, 2] = 1  # a11
        V[1, 2] = 1; V[3, 2] = -1  # b12 - b22
        W[1, 2] = 1  # C12 += M3 (W[0*2+1]=W[1])
        W[3, 2] = 1  # C22 += M3
        
        # M4 = a22(b21 - b11)
        U[3, 3] = 1  # a22
        V[2, 3] = 1; V[0, 3] = -1  # b21 - b11
        W[2, 3] = 1  # C21 += M4
        W[0, 3] = 1  # C11 += M4
        
        # M5 = (a11 + a12)b22
        U[0, 4] = 1; U[1, 4] = 1  # a11 + a12
        V[3, 4] = 1  # b22
        W[1, 4] = 1  # C12 += M5
        W[0, 4] = -1  # C11 -= M5
        
        # M6 = (a21 - a11)(b11 + b12)
        U[2, 5] = 1; U[0, 5] = -1  # a21 - a11
        V[0, 5] = 1; V[1, 5] = 1  # b11 + b12
        W[3, 5] = 1  # C22 += M6
        
        # M7 = (a12 - a22)(b21 + b22)
        U[1, 6] = 1; U[3, 6] = -1  # a12 - a22
        V[2, 6] = 1; V[3, 6] = 1  # b21 + b22
        W[0, 6] = 1  # C11 += M7
        
        return U, V, W
    
    def _combine_decompositions(decomp_list):
        """Combine multiple decompositions by block diagonal stacking."""
        # Each decomp is (U, V, W) with shapes (n*m, r_i), (m*p, r_i), (p*n, r_i)
        total_rank = sum(d[0].shape[1] for d in decomp_list)
        n, m, p = decomp_list[0][0].shape[0], decomp_list[0][0].shape[0], decomp_list[0][1].shape[0]
        # Actually need to get n, m, p from the first decomposition
        U0, V0, W0 = decomp_list[0]
        nm = U0.shape[0]
        mp = V0.shape[0]
        pn = W0.shape[0]
        
        U = np.zeros((nm, total_rank), dtype=np.int32)
        V = np.zeros((mp, total_rank), dtype=np.int32)
        W = np.zeros((pn, total_rank), dtype=np.int32)
        
        col_offset = 0
        for U_i, V_i, W_i in decomp_list:
            r_i = U_i.shape[1]
            U[:, col_offset:col_offset+r_i] = U_i
            V[:, col_offset:col_offset+r_i] = V_i
            W[:, col_offset:col_offset+r_i] = W_i
            col_offset += r_i
        
        return U, V, W
    
    def _build_2x4x7_decomposition():
        """Build a decomposition for 2x4 @ 4x7 multiplication."""
        n, m, p = 2, 4, 7
        
        # Strategy: Split into blocks and use Strassen where possible
        # A is 2x4, B is 4x7
        # Split A into A1 (2x2), A2 (2x2)
        # Split B into B1 (2x7), B2 (2x7)
        # C = A1@B1 + A2@B2
        # Each A_i @ B_i is 2x2 @ 2x7
        
        # For 2x2 @ 2x7: Split B_i into three 2x2 blocks and one 2x1 block
        # Or use a different approach.
        
        # Let me try: for 2x2 @ 2xp, use Strassen on 2x2x2 blocks
        # 2x2 @ 2x7 = 2x2 @ (2x2 + 2x2 + 2x2 + 2x1)
        # This is messy. Let me use a simpler approach.
        
        # Alternative: use the fact that rank(M_{2,2,p}) <= p + ceil(p/2) for some constructions
        # For p=7: maybe rank 11?
        
        # Let me just use schoolbook for now: 2x2 @ 2x7 has rank 28
        # Two such blocks: total rank 56. Same as original.
        
        # To do better, I need a better 2x2 @ 2x7 decomposition.
        
        # Known result: rank(M_{2,2,2}) = 7 (Strassen)
        # For 2x2 @ 2x7, I can use 3 Strassen blocks (for 2x2 @ 2x6) + 1 schoolbook (for 2x2 @ 2x1)
        # 2x2 @ 2x6: split into three 2x2 @ 2x2 blocks, each rank 7, total 21
        # 2x2 @ 2x1: schoolbook rank 4
        # Total for 2x2 @ 2x7: 21 + 4 = 25
        # Two such: 50. Still better than 56!
        
        # Let me implement this.
        
        def _build_2x2x2x7():
            """Build 2x2 @ 2x7 decomposition using 3 Strassen + 1 schoolbook."""
            # 2x2 @ 2x7: A is 2x2, B is 2x7
            # Split B into B1 (2x2), B2 (2x2), B3 (2x2), B4 (2x1)
            # C = A@B1 + A@B2 + A@B3 + A@B4
            # A@B1, A@B2, A@B3: each 2x2 @ 2x2, use Strassen (rank 7 each)
            # A@B4: 2x2 @ 2x1, schoolbook (rank 4)
            
            decomps = []
            
            # Three Strassen blocks for the 2x2 @ 2x2 parts
            for block_idx in range(3):
                U_s, V_s, W_s = _build_strassen_2x2x2()
                # Need to embed this into the larger 2x2 @ 2x7 context
                # U has shape (2*2, 7) = (4, 7) for 2x2 @ 2x2
                # But in 2x2 @ 2x7, U should have shape (4, r) where r is the rank
                # The B matrix is 2x7, so V should have shape (2*7, r) = (14, r)
                # W should have shape (7*2, r) = (14, r)
                
                # For the block at columns [2*block_idx : 2*block_idx+2]:
                U_block = np.zeros((4, 7), dtype=np.int32)
                V_block = np.zeros((14, 7), dtype=np.int32)
                W_block = np.zeros((14, 7), dtype=np.int32)
                
                # Copy Strassen values
                U_block[:, :] = U_s
                # V: B[j,k] where j in [0,2), k in [2*block_idx, 2*block_idx+2)
                # V index is j*7 + k
                for j in range(2):
                    for k in range(2):
                        V_block[j*7 + (2*block_idx + k), :] = V_s[j*2 + k, :]
                # W: C[i,k] where i in [0,2), k in [2*block_idx, 2*block_idx+2)
                # W index is k*2 + i
                for i in range(2):
                    for k in range(2):
                        W_block[(2*block_idx + k)*2 + i, :] = W_s[k*2 + i, :]
                
                decomps.append((U_block, V_block, W_block))
            
            # One schoolbook block for 2x2 @ 2x1 (columns 6:7)
            U_sb = np.zeros((4, 4), dtype=np.int32)
            V_sb = np.zeros((14, 4), dtype=np.int32)
            W_sb = np.zeros((14, 4), dtype=np.int32)
            col = 0
            for i in range(2):
                for j in range(2):
                    for k in range(1):
                        U_sb[i*2 + j, col] = 1
                        V_sb[j*7 + (6 + k), col] = 1
                        W_sb[(6 + k)*2 + i, col] = 1
                        col += 1
            decomps.append((U_sb, V_sb, W_sb))
            
            # Combine
            return _combine_decompositions(decomps)
        
        # Build two 2x2 @ 2x7 decompositions
        decomp1 = _build_2x2x2x7()  # for A1 @ B1
        decomp2 = _build_2x2x2x7()  # for A2 @ B2
        
        # Now embed into the full 2x4 @ 4x7 context
        # A is 2x4: A1 = A[:, 0:2], A2 = A[:, 2:4]
        # B is 4x7: B1 = B[0:2, :], B2 = B[2:4, :]
        # C = A1@B1 + A2@B2
        
        # U has shape (2*4, r) = (8, r) for the full problem
        # For A1@B1: U indices are i*4 + j where i in [0,2), j in [0,2)
        # For A2@B2: U indices are i*4 + j where i in [0,2), j in [2,4)
        
        # V has shape (4*7, r) = (28, r)
        # For A1@B1: V indices are j*7 + k where j in [0,2), k in [0,7)
        # For A2@B2: V indices are j*7 + k where j in [2,4), k in [0,7)
        
        # W has shape (7*2, r) = (14, r)
        # Same for both: W indices are k*2 + i where i in [0,2), k in [0,7)
        
        r1 = decomp1[0].shape[1]
        r2 = decomp2[0].shape[1]
        total_rank = r1 + r2
        
        U = np.zeros((8, total_rank), dtype=np.int32)
        V = np.zeros((28, total_rank), dtype=np.int32)
        W = np.zeros((14, total_rank), dtype=np.int32)
        
        # Embed decomp1 (A1 @ B1)
        U1, V1, W1 = decomp1
        # U1 has shape (4, r1), corresponds to A1[i,j] with i in [0,2), j in [0,2)
        # In full U, A[i,j] is at index i*4 + j
        for i in range(2):
            for j in range(2):
                U[i*4 + j, :r1] = U1[i*2 + j, :]
        
        # V1 has shape (14, r1), corresponds to B1[j,k] with j in [0,2), k in [0,7)
        # In full V, B[j,k] is at index j*7 + k
        for j in range(2):
            for k in range(7):
                V[j*7 + k, :r1] = V1[j*7 + k, :]
        
        # W1 has shape (14, r1), corresponds to C[i,k] with i in [0,2), k in [0,7)
        # In full W, C[i,k] is at index k*2 + i
        W[:, :r1] = W1
        
        # Embed decomp2 (A2 @ B2)
        U2, V2, W2 = decomp2
        # U2 has shape (4, r2), corresponds to A2[i,j] with i in [0,2), j in [0,2)
        # In full U, A[i, 2+j] is at index i*4 + (2+j)
        for i in range(2):
            for j in range(2):
                U[i*4 + 2 + j, r1:r1+r2] = U2[i*2 + j, :]
        
        # V2 has shape (14, r2), corresponds to B2[j,k] with j in [0,2), k in [0,7)
        # In full V, B[2+j, k] is at index (2+j)*7 + k
        for j in range(2):
            for k in range(7):
                V[(2+j)*7 + k, r1:r1+r2] = V2[j*7 + k, :]
        
        # W2 has shape (14, r2), corresponds to C[i,k] with i in [0,2), k in [0,7)
        # In full W, C[i,k] is at index k*2 + i
        W[:, r1:r1+r2] = W2
        
        return U, V, W
    
    # Build the decomposition
    U, V, W = _build_2x4x7_decomposition()
    rank = U.shape[1]
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # If verification fails, fall back to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
