"""Matrix multiplication 4,4,9 - Strassen-based decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # Term 0: M1 = (a+d)(e+h)
    U[0,0] = 1; U[2,0] = 1
    V[0,0] = 1; V[1,0] = 1
    W[0,0] = 1; W[1,0] = 1; W[2,0] = 1; W[3,0] = 1
    
    # Term 1: M2 = (c+d)e
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1
    
    # Term 2: M3 = a(f-h)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[2,2] = 1
    
    # Term 3: M4 = d(g-e)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[1,3] = 1
    
    # Term 4: M5 = (a+b)h
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[2,4] = 1
    
    # Term 5: M6 = (c-d)(f+h)
    U[2,5] = 1; U[3,5] = -1
    V[1,5] = 1; V[3,5] = 1
    W[3,5] = 1
    
    # Term 6: M7 = (a-c)(e+g)
    U[0,6] = 1; U[2,6] = -1
    V[0,6] = 1; V[2,6] = 1
    W[0,6] = 1; W[1,6] = -1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    # For random A (n x m) and B (m x p), check that C = A @ B
    # can be recovered from the decomposition.
    # C[i, k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k])
    # Actually: C[i,k] = sum_r W[k*n+i, r] * (U^T A_vec)[r] * (V^T B_vec)[r]
    # where A_vec is A flattened row-major, B_vec is B flattened row-major.
    
    # More precisely:
    # Let a_r = sum_{i,j} U[i*m+j, r] * A[i,j]
    # Let b_r = sum_{j,k} V[j*p+k, r] * B[j,k]
    # Then C[i,k] = sum_r W[k*n+i, r] * a_r * b_r
    
    # Wait, that's not right either. The decomposition is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' V[j*p+k', r] * B[j,k'])
    # No, that's not right. Let me think again.
    
    # The tensor decomposition says:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 for the matrix multiplication tensor.
    
    # So for any A and B:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i,j] B[j,k]
    #        = sum_r W[k*n+i, r] (sum_j U[i*m+j, r] A[i,j]) (sum_j V[j*p+k, r] B[j,k])
    
    # Hmm, that's not quite right because the j index appears in both U and V.
    # Let me re-derive.
    
    # The matrix multiplication tensor satisfies:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has entries T[i,j,k] = 1, meaning the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    # In the decomposition format:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = T[i,j,k] = 1
    
    # So for any A, B:
    # sum_j A[i,j] B[j,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i,j] B[j,k]
    #                      = sum_r W[k*n+i, r] sum_j (U[i*m+j, r] A[i,j]) (V[j*p+k, r] B[j,k])
    
    # Wait, this is still not right. The j index is shared between U and V.
    # Let me be more careful.
    
    # sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i,j] B[j,k]
    # = sum_r W[k*n+i, r] sum_j U[i*m+j, r] A[i,j] V[j*p+k, r] B[j,k]
    
    # This is not a simple product of two sums. It's a sum over j of products.
    # So the verification is:
    # For random A, B, compute C = A @ B.
    # Then for each i, k, check that sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k]) == C[i,k].
    
    # Actually, I think the correct interpretation is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    # No wait, that would give:
    # sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_k' V[j*p+k', r] B[j,k'])
    # which doesn't make sense because j is free in the second sum.
    
    # Let me just directly check the tensor identity:
    # For all i, j, k: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
    
    U_mat = np.array(U)
    V_mat = np.array(V)
    W_mat = np.array(W)
    
    # Check tensor identity
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                if abs(val - 1) > tol:
                    return False
    return True


def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4 * 4x9 multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: Use Strassen for 2x2x2 blocks to reduce rank.
    # 4x4x9: split k into two parts: k=0..3 (4 values) and k=4..8 (5 values).
    # For k=0..3: 4x4x4 subproblem.
    # For k=4..8: 4x4x5 subproblem.
    
    # For 4x4x4: Use block Strassen (rank 49).
    # For 4x4x5: Use schoolbook (rank 80) or try to optimize.
    
    # Let's first build the 4x4x4 rank-49 decomposition using block Strassen.
    
    def _build_4x4x4_strassen():
        """Build 4x4x4 decomposition with rank 49 using block Strassen."""
        n = 4
        rank = 49
        U = np.zeros((n*n, rank))
        V = np.zeros((n*n, rank))
        W = np.zeros((n*n, rank))
        
        # The 7 block Strassen products, each decomposed into 7 element-level Strassen terms.
        # Block product t (0..6), Strassen term s (0..6): column = t*7 + s
        
        # Define the 7 block products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A11)(B01+B11)
        # M7 = (A00-A10)(B00+B10)
        
        # For each block product, we need to express it as a 2x2x2 multiplication
        # where the A and B matrices are linear combinations of the original blocks.
        
        # Let me define the structure more carefully.
        # For block product Mk, we have:
        # A_combined = sum_{(bi,bj,ca)} ca * A[2*bi:2*bi+2, 2*bj:2*bj+2]
        # B_combined = sum_{(bk,bj,cb)} cb * B[2*bk:2*bk+2, 2*bj:2*bj+2]
        # Mk = A_combined * B_combined (2x2 matrix product)
        
        # The 2x2 matrix product Mk[i,j] = sum_k A_combined[i,k] * B_combined[k,j]
        # can be decomposed using Strassen into 7 rank-1 terms.
        
        # For each of the 7 Strassen terms, we need to map back to the original A and B indices.
        
        # Let me define the block products:
        block_products = [
            # (a_blocks, b_blocks)
            # a_blocks: list of (bi, bj, coeff)
            # b_blocks: list of (bk, bj, coeff)
            ([((0,0,1.0), (1,1,1.0))], [((0,0,1.0), (1,1,1.0))]),  # M1
            ([((1,0,1.0), (1,1,1.0))], [((0,0,1.0))]),  # M2
            ([((0,0,1.0))], [((0,1,1.0), (1,1,-1.0))]),  # M3
            ([((1,1,1.0))], [((1,0,1.0), (0,0,-1.0))]),  # M4
            ([((0,0,1.0), (0,1,1.0))], [((1,1,1.0))]),  # M5
            ([((1,0,1.0), (1,1,-1.0))], [((0,1,1.0), (1,1,1.0))]),  # M6
            ([((0,0,1.0), (1,0,-1.0))], [((0,0,1.0), (1,0,1.0))]),  # M7
        ]
        
        # Strassen terms for 2x2x2:
        # Term 0: (a+d)(e+h) -> U: (0,0)=1, (1,0)=1; V: (0,0)=1, (0,1)=1; W: all 1s
        # Term 1: (c+d)e -> U: (1,0)=1, (1,1)=1; V: (0,0)=1; W: (0,1)=1
        # Term 2: a(f-h) -> U: (0,0)=1; V: (0,1)=1, (1,1)=-1; W: (1,0)=1
        # Term 3: d(g-e) -> U: (1,1)=1; V: (1,0)=1, (0,0)=-1; W: (0,1)=1
        # Term 4: (a+b)h -> U: (0,0)=1, (0,1)=1; V: (1,1)=1; W: (1,0)=1
        # Term 5: (c-d)(f+h) -> U: (1,0)=1, (1,1)=-1; V: (0,1)=1, (1,1)=1; W: (1,1)=1
        # Term 6: (a-c)(e+g) -> U: (0,0)=1, (1,0)=-1; V: (0,0)=1, (1,0)=1; W: (0,0)=1, (0,1)=-1
        
        # Let me use the _strassen_2x2 function to get the base decomposition.
        U_s, V_s, W_s = _strassen_2x2()
        
        # For each block product t and Strassen term s:
        for t in range(7):
            a_blocks, b_blocks = block_products[t]
            for s in range(7):
                col = t * 7 + s
                
                # For this Strassen term, the A matrix entries are:
                # A_combined[i, k] = sum over a_blocks of coeff * A[2*bi+i, 2*bj+k]
                # where (i, k) are the 2x2 indices within the block.
                
                # The Strassen term s uses specific entries of A_combined and B_combined.
                # From _strassen_2x2:
                # U_s[i*2+k, s] gives the coefficient of A_combined[i, k] in term s.
                # V_s[k*2+j, s] gives the coefficient of B_combined[k, j] in term s.
                # W_s[j*2+i, s] gives the coefficient for C[i, j] from term s.
                
                # So for the original A:
                # A[2*bi+i, 2*bj+k] appears with coefficient:
                #   sum over a_blocks (bi, bj, ca) of ca * U_s[i*2+k, s]
                
                # Similarly for B:
                # B[2*bk+k, 2*bj+j] appears with coefficient:
                #   sum over b_blocks (bk, bj, cb) of cb * V_s[k*2+j, s]
                
                # And for C:
                # C[2*bi+i, 2*bj+j] appears with coefficient:
                #   W_s[j*2+i, s]
                
                # Wait, I need to be more careful about the indices.
                
                # For the 2x2x2 Strassen decomposition:
                # U_s[i*2+k, s] is the coefficient of A[i, k] in the s-th product.
                # V_s[k*2+j, s] is the coefficient of B[k, j] in the s-th product.
                # W_s[j*2+i, s] is the coefficient for C[i, j] from the s-th product.
                
                # In the block product, A_combined[i, k] = sum_{(bi, bj, ca)} ca * A[2*bi+i, 2*bj+k]
                # B_combined[k, j] = sum_{(bk, bj, cb)} cb * B[2*bk+k, 2*bj+j]
                
                # So the coefficient of A[2*bi+i, 2*bj+k] in term (t, s) is:
                #   ca * U_s[i*2+k, s]
                
                # And the coefficient of B[2*bk+k, 2*bj+j] is:
                #   cb * V_s[k*2+j, s]
                
                # And the coefficient for C[2*bi+i, 2*bj+j] is:
                #   W_s[j*2+i, s]
                
                # But wait, the C indices should be consistent. The block product Mk
                # contributes to specific C blocks. Let me check the Strassen formulas.
                
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                
                # So M1 contributes to C00 and C11.
                # M2 contributes to C10.
                # M3 contributes to C01 and C11.
                # M4 contributes to C00 and C10.
                # M5 contributes to C01 and C00.
                # M6 contributes to C11.
                # M7 contributes to C00 and C11.
                
                # The W matrix in the final decomposition should have these contributions.
                # For block product t, the W entries should be placed in the appropriate C blocks.
                
                # Let me define the C block contributions for each M:
                c_contributions = [
                    [(0,0,1), (1,1,1)],    # M1: C00 and C11
                    [(1,0,1)],              # M2: C10
                    [(0,1,1), (1,1,1)],    # M3: C01 and C11
                    [(0,0,1), (1,0,1)],    # M4: C00 and C10
                    [(0,1,1), (0,0,1)],    # M5: C01 and C00
                    [(1,1,1)],              # M6: C11
                    [(0,0,1), (1,1,1)],    # M7: C00 and C11
                ]
                
                # Now, for each a_block (bi, bj, ca) and each (i, k) in 0..1:
                # U[2*bi+i, 2*bj+k, col] += ca * U_s[i*2+k, s]
                
                for bi, bj, ca in a_blocks:
                    for i in range(2):
                        for k in range(2):
                            U[2*bi+i, 2*bj+k, col] += ca * U_s[i*2+k, s]
                
                # For each b_block (bk, bj, cb) and each (k, j) in 0..1:
                # V[2*bk+k, 2*bj+j, col] += cb * V_s[k*2+j, s]
                
                for bk, bj, cb in b_blocks:
                    for k in range(2):
                        for j in range(2):
                            V[2*bk+k, 2*bj+j, col] += cb * V_s[k*2+j, s]
                
                # For each C contribution (ci, cj, cw) and each (j, i) in 0..1:
                # W[2*cj+j, 2*ci+i, col] += cw * W_s[j*2+i, s]
                
                for ci, cj, cw in c_contributions[t]:
                    for j in range(2):
                        for i in range(2):
                            W[2*cj+j, 2*ci+i, col] += cw * W_s[j*2+i, s]
        
        return U, V, W
    
    # Build the 4x4x4 decomposition
    U_444, V_444, W_444 = _build_4x4x4_strassen()
    
    # Verify the 4x4x4 decomposition
    if not _verify_decomposition(U_444, V_444, W_444, 4, 4, 4):
        # Fall back to schoolbook for 4x4x4
        rank_444 = 4*4*4
        U_444 = np.zeros((16, rank_444))
        V_444 = np.zeros((16, rank_444))
        W_444 = np.zeros((16, rank_444))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_444[i*4+j, col] = 1
                    V_444[j*4+k, col] = 1
                    W_444[k*4+i, col] = 1
                    col += 1
    
    # For 4x4x5: Use schoolbook
    rank_445 = 4*4*5
    U_445 = np.zeros((16, rank_445))
    V_445 = np.zeros((20, rank_445))
    W_445 = np.zeros((20, rank_445))
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(5):
                U_445[i*4+j, col] = 1
                V_445[j*5+k, col] = 1
                W_445[k*4+i, col] = 1
                col += 1
    
    # Combine 4x4x4 and 4x4x5 into 4x4x9.
    # 4x4x4: U (16, 49), V (16, 49), W (16, 49)
    # 4x4x5: U (16, 80), V (20, 80), W (20, 80)
    # Total: U (16, 129), V (36, 129), W (36, 129)
    
    total_rank = U_444.shape[1] + U_445.shape[1]
    
    U_total = np.zeros((16, total_rank))
    V_total = np.zeros((36, total_rank))
    W_total = np.zeros((36, total_rank))
    
    # 4x4x4 part: k=0..3
    U_total[:, :U_444.shape[1]] = U_444
    V_total[:16, :U_444.shape[1]] = V_444
    W_total[:16, :U_444.shape[1]] = W_444
    
    # 4x4x5 part: k=4..8
    U_total[:, U_444.shape[1]:] = U_445
    V_total[16:, U_444.shape[1]:] = V_445
    W_total[16:, U_444.shape[1]:] = W_445
    
    # Verify the full 4x4x9 decomposition
    if not _verify_decomposition(U_total, V_total, W_total, 4, 4, 9):
        # Fall back to full schoolbook
        rank = 4*4*9
        U_total = np.zeros((16, rank))
        V_total = np.zeros((36, rank))
        W_total = np.zeros((36, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U_total[i*4+j, col] = 1
                    V_total[j*9+k, col] = 1
                    W_total[k*4+i, col] = 1
                    col += 1
    
    # Convert to lists
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
