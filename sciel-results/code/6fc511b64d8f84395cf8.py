"""Tensor decomposition for 4x5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    U: (2*2, 7), V: (2*2, 7), W: (2*2, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1   # C00 += M1
    W[3,0] = 1   # C11 += M1
    
    # M2 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1   # C10 += M2
    
    # M3 = A00(B10-B11)
    U[0,2] = 1
    V[2,2] = 1; V[3,2] = -1
    W[1,2] = 1   # C01 += M3
    W[3,2] = -1  # C11 -= M3
    
    # M4 = A11(B01-B00)
    U[3,3] = 1
    V[1,3] = 1; V[0,3] = -1
    W[0,3] = -1  # C00 -= M4
    W[2,3] = 1   # C10 += M4
    
    # M5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[1,4] = 1   # C01 += M5
    
    # M6 = (A10-A00)(B00+B01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[0,5] = 1   # C00 += M6
    
    # M7 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[3,6] = 1   # C11 += M7
    
    return U, V, W

def _schoolbook(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _combine_blocks(U1, V1, W1, U2, V2, W2,
                    n1, m1, p1,
                    n2, m2, p2,
                    n_total, m_total, p_total):
    """Combine two block decompositions into one for the full matrix.
    
    Block 1 handles rows [0..n1), cols [0..m1) of A, rows [0..m1), cols [0..p1) of B
    Block 2 handles rows [n1..n_total), cols [m1..m_total) of A, rows [m1..m_total), cols [p1..p_total) of B
    
    Wait, this isn't quite right. Let me think about this more carefully.
    
    Actually, for the block decomposition to work, we need:
    - A is split into A_top (n1 x m) and A_bot (n2 x m) where n1+n2=n_total
    - B is split into B_left (m x p1) and B_right (m x p2) where p1+p2=p_total
    - C = A*B has blocks:
      C_top_left = A_top * B_left  (n1 x p1)
      C_top_right = A_top * B_right (n1 x p2)
      C_bot_left = A_bot * B_left  (n2 x p1)
      C_bot_right = A_bot * B_right (n2 x p2)
    
    So we need 4 sub-decompositions, not 2. Let me redo this.
    """
    pass

def _block_decompose_4x5x5():
    """Try to decompose 4x5x5 using block structure.
    
    Split A (4x5) into A_top (2x5) and A_bot (2x5)
    Split B (5x5) into B_left (5x2) and B_right (5x3)
    
    This gives 4 sub-problems:
    1. C_top_left = A_top * B_left: 2x5 * 5x2 -> 2x2 (use Strassen, rank 7)
    2. C_top_right = A_top * B_right: 2x5 * 5x3 -> 2x3 (schoolbook rank 30)
    3. C_bot_left = A_bot * B_left: 2x5 * 5x2 -> 2x2 (use Strassen, rank 7)
    4. C_bot_right = A_bot * B_right: 2x5 * 5x3 -> 2x3 (schoolbook rank 30)
    
    Total rank: 7 + 30 + 7 + 30 = 74
    
    But wait, for the 2x5 * 5x3 problem, we can also try to use block structure.
    Actually, 2x5 * 5x3 is just a standard matrix multiplication with dimensions 2,5,3.
    Schoolbook rank = 2*5*3 = 30.
    
    Can we do better for 2x5x3? Let's think...
    The tensor rank of [2,5,3] is at most 30. Known results suggest it might be lower.
    But for now, let's use schoolbook for these.
    
    Total: 7 + 30 + 7 + 30 = 74
    """
    n, m, p = 4, 5, 5
    n1, n2 = 2, 2  # Split rows of A
    p1, p2 = 2, 3  # Split cols of B
    
    # Sub-problem 1: 2x5 * 5x2 -> 2x2 (Strassen)
    U1, V1, W1 = _strassen_2x2()
    
    # Sub-problem 2: 2x5 * 5x3 -> 2x3 (schoolbook)
    U2, V2, W2 = _schoolbook(2, 5, 3)
    
    # Sub-problem 3: 2x5 * 5x2 -> 2x2 (Strassen)
    U3, V3, W3 = _strassen_2x2()
    
    # Sub-problem 4: 2x5 * 5x3 -> 2x3 (schoolbook)
    U4, V4, W4 = _schoolbook(2, 5, 3)
    
    total_rank = 7 + 30 + 7 + 30
    U_full = np.zeros((n * m, total_rank), dtype=float)
    V_full = np.zeros((m * p, total_rank), dtype=float)
    W_full = np.zeros((p * n, total_rank), dtype=float)
    
    col_offset = 0
    
    # Sub-problem 1: A_top (rows 0-1 of A) * B_left (cols 0-1 of B)
    # U1 has shape (2*5, 7) = (10, 7)
    # U1[i*5+j, c] corresponds to A_top[i,j] = A[i,j] for i in 0..1, j in 0..4
    # In full U: U_full[i*5+j, col] for i in 0..1, j in 0..4
    for c in range(7):
        for i in range(2):
            for j in range(5):
                U_full[i * 5 + j, col_offset + c] = U1[i * 5 + j, c]
                V_full[j * 5 + (0 + 0), col_offset + c] = V1[j * 5 + 0, c]  # B_left row j, col 0
                # Wait, V1 has shape (5*2, 7) = (10, 7)
                # V1[j*2+k, c] corresponds to B_left[j,k] = B[j,k] for j in 0..4, k in 0..1
                # In full V: V_full[j*5+k, col] for j in 0..4, k in 0..1
                # So V1[j*2+k, c] maps to V_full[j*5+k, col_offset+c]
        
        # Let me redo this more carefully
        pass
    
    # This is getting complicated. Let me use a cleaner approach.
    
    # Actually, let me just build the full matrices directly.
    
    U_full = np.zeros((20, total_rank), dtype=float)  # 4*5
    V_full = np.zeros((25, total_rank), dtype=float)  # 5*5
    W_full = np.zeros((20, total_rank), dtype=float)  # 5*4
    
    col_offset = 0
    
    # Sub-problem 1: 2x5 * 5x2 -> 2x2
    # A indices: i in 0..1, j in 0..4 -> U_full[i*5+j, col]
    # B indices: j in 0..4, k in 0..1 -> V_full[j*5+k, col]
    # C indices: k in 0..1, i in 0..1 -> W_full[k*4+i, col]
    for c in range(7):
        for i in range(2):
            for j in range(5):
                U_full[i * 5 + j, col_offset + c] = U1[i * 5 + j, c]
        for j in range(5):
            for k in range(2):
                V_full[j * 5 + k, col_offset + c] = V1[j * 2 + k, c]
        for k in range(2):
            for i in range(2):
                W_full[k * 4 + i, col_offset + c] = W1[k * 2 + i, c]
    col_offset += 7
    
    # Sub-problem 2: 2x5 * 5x3 -> 2x3
    # A indices: i in 0..1, j in 0..4 -> U_full[i*5+j, col]
    # B indices: j in 0..4, k in 2..4 -> V_full[j*5+k, col]
    # C indices: k in 2..4, i in 0..1 -> W_full[k*4+i, col]
    for c in range(30):
        for i in range(2):
            for j in range(5):
                U_full[i * 5 + j, col_offset + c] = U2[i * 5 + j, c]
        for j in range(5):
            for k in range(3):
                V_full[j * 5 + (2 + k), col_offset + c] = V2[j * 5 + k, c]
        for k in range(3):
            for i in range(2):
                W_full[(2 + k) * 4 + i, col_offset + c] = W2[k * 2 + i, c]
    col_offset += 30
    
    # Sub-problem 3: 2x5 * 5x2 -> 2x2
    # A indices: i in 2..3, j in 0..4 -> U_full[i*5+j, col]
    # B indices: j in 0..4, k in 0..1 -> V_full[j*5+k, col]
    # C indices: k in 0..1, i in 2..3 -> W_full[k*4+i, col]
    for c in range(7):
        for i in range(2):
            for j in range(5):
                U_full[(2 + i) * 5 + j, col_offset + c] = U3[i * 5 + j, c]
        for j in range(5):
            for k in range(2):
                V_full[j * 5 + k, col_offset + c] = V3[j * 2 + k, c]
        for k in range(2):
            for i in range(2):
                W_full[k * 4 + (2 + i), col_offset + c] = W3[k * 2 + i, c]
    col_offset += 7
    
    # Sub-problem 4: 2x5 * 5x3 -> 2x3
    # A indices: i in 2..3, j in 0..4 -> U_full[i*5+j, col]
    # B indices: j in 0..4, k in 2..4 -> V_full[j*5+k, col]
    # C indices: k in 2..4, i in 2..3 -> W_full[k*4+i, col]
    for c in range(30):
        for i in range(2):
            for j in range(5):
                U_full[(2 + i) * 5 + j, col_offset + c] = U4[i * 5 + j, c]
        for j in range(5):
            for k in range(3):
                V_full[j * 5 + (2 + k), col_offset + c] = V4[j * 5 + k, c]
        for k in range(3):
            for i in range(2):
                W_full[(2 + k) * 4 + (2 + i), col_offset + c] = W4[k * 2 + i, c]
    col_offset += 30
    
    assert col_offset == total_rank, f"Column offset {col_offset} != total rank {total_rank}"
    
    return U_full, V_full, W_full

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Check dimensions
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] = 1 if j==k (wait, no)
    # Actually, for matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = delta_{j,k}... no wait.
    # The tensor for matrix multiplication has T[i,j,k] = 1 if the indices match appropriately.
    # Actually, the matrix multiplication tensor is defined as:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k ⊗ delta_{j,k}... no.
    
    # Let me think again. The matrix multiplication tensor is:
    # M = sum_{i,j,k} M_{ijk} (e_i ⊗ e_j ⊗ e_k)
    # where M_{ijk} = 1 if the multiplication contributes, 0 otherwise.
    # For C = A*B, C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor has M_{ijk} = 1 for all valid i,j,k (since every term contributes).
    # Wait no, that's not right either.
    
    # Actually, the standard definition: the matrix multiplication tensor [n,m,p] is
    # the tensor T in R^{n x m x p} where T[i,j,k] = 1 if and only if the
    # (i,j) entry of A and (j,k) entry of B contribute to the (i,k) entry of C.
    # Since C[i,k] = sum_j A[i,j] B[j,k], every triple (i,j,k) with valid indices
    # contributes, so T[i,j,k] = 1 for all i,j,k.
    
    # Wait, that can't be right because then the tensor would just be all ones,
    # which has rank 1.
    
    # Let me reconsider. The tensor decomposition of matrix multiplication:
    # We want to compute C = A * B where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    # A tensor decomposition of rank r means we can write:
    # C[i,k] = sum_{c=0}^{r-1} U[i, c] * V[k, c] * ... 
    # No, that's not right either.
    
    # The correct formulation: the matrix multiplication tensor T in R^{n x m x p}
    # is defined by T[i,j,k] = delta_{jk}... no.
    
    # OK let me look at this from the perspective of the factorization.
    # We have U (n*m x r), V (m*p x r), W (p*n x r).
    # The decomposition says:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # where T[i,j,k] is the matrix multiplication tensor.
    
    # For matrix multiplication, T[i,j,k] = 1 if and only if... 
    # Actually, I think the tensor is T[i,j,k] = 1 for all i,j,k, and the
    # "multiplication" is encoded in how we contract.
    
    # Let me just verify by checking that for random A and B, the formula works.
    
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We claim: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_k' V[j'*p+k, c] B[j',k])
    # No, that's not right.
    
    # The correct formula for tensor decomposition of matrix multiplication:
    # Given U, V, W with the specified shapes, the decomposition is valid if:
    # For all i, j, k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{something}
    
    # I think the correct condition is:
    # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    # No, that's not right either.
    
    # Let me just test with the schoolbook decomposition to see what the condition should be.
    
    # Schoolbook: U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1 for col = i*m*p + j*p + k
    # So sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if there exists c such that
    # c corresponds to (i,j,k), i.e., c = i*m*p + j*p + k.
    # In that case, U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1, so the sum is 1.
    # For any other (i',j',k'), the sum would be 0 unless (i',j',k') = (i,j,k).
    # So the tensor T[i,j,k] = 1 for all i,j,k.
    
    # Wait, that means the matrix multiplication tensor is just the all-ones tensor?
    # That can't be right because the all-ones tensor has rank 1.
    
    # I'm confused. Let me re-read the problem statement.
    
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # So W is indexed by k*n+i, where k is the column of C and i is the row of C.
    
    # I think the correct interpretation is:
    # The tensor T has entries T[i,j,k] where i indexes rows of A/C, j indexes columns of A / rows of B,
    # and k indexes columns of B/C.
    # T[i,j,k] = 1 for all valid i,j,k (since every A[i,j]*B[j,k] term appears in C[i,k]).
    
    # But then the tensor rank would be 1, which is clearly wrong.
    
    # I think I'm misunderstanding the tensor decomposition. Let me reconsider.
    
    # In a tensor decomposition of matrix multiplication, we write:
    # C[i,k] = sum_{c=0}^{r-1} (sum_j U[i*m+j, c] * A[i,j]) * (sum_j' V[j'*p+k, c] * B[j',k]) * W[k*n+i, c]
    # No, that's not right either because of the double sum over j and j'.
    
    # I think the correct formula is:
    # C[i,k] = sum_c W[k*n+i, c] * (U^T A)[i, c] * (B V)[c, k]
    # where (U^T A)[i, c] = sum_j U[i*m+j, c] * A[i,j]... no, that doesn't make sense dimensionally.
    
    # OK, I think the standard formulation is:
    # Define matrices U_c, V_c, W_c for c=0..r-1 where:
    # U_c is n x m with U_c[i,j] = U[i*m+j, c]
    # V_c is m x p with V_c[j,k] = V[j*p+k, c]  
    # W_c is p x n with W_c[k,i] = W[k*n+i, c]
    
    # Then the decomposition says:
    # For all i, k: C[i,k] = sum_j A[i,j] B[j,k] = sum_c W_c[k,i] * (sum_j U_c[i,j] A[i,j]) * (sum_j' V_c[j',k] B[j',k])
    # No, that still doesn't work.
    
    # I think the correct statement is that the tensor T = sum_c U_c ⊗ V_c ⊗ W_c^T (or some permutation)
    # equals the matrix multiplication tensor.
    
    # Let me just verify numerically.
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    # Compute C using the decomposition
    C_comp = np.zeros((n, p))
    for c in range(rank):
        # U_c is n x m
        U_c = U[:n*m, c].reshape(n, m)
        # V_c is m x p
        V_c = V[:m*p, c].reshape(m, p)
        # W_c is p x n
        W_c = W[:p*n, c].reshape(p, n)
        
        # The contribution to C[i,k] from this term
        # I believe it's: sum_j U_c[i,j] * A[i,j] * sum_j' V_c[j',k] * B[j',k] * W_c[k,i]
        # But that's a product of two sums, which gives a double sum.
        
        # Actually, I think the formula is:
        # C[i,k] += W_c[k,i] * (U_c * A) [i] * (B * V_c)[k]
        # where (U_c * A)[i] = sum_j U_c[i,j] * A[i,j]... no, that's not a standard operation.
        
        # Let me try: the scalar product of U_c and A, times the scalar product of B and V_c, times W_c[k,i]
        # But U_c and A are both n x m, so their "scalar product" would be sum_{i,j} U_c[i,j] * A[i,j],
        # which is a single number, not indexed by i.
        
        # I'm really confused. Let me try a different approach.
        
        # Maybe the formula is:
        # C[i,k] += W_c[k,i] * (sum_j U_c[i,j] * A[i,j]) * (sum_j V_c[j,k] * B[j,k])
        # But that's still a product of two sums.
        
        # Or maybe:
        # C[i,k] += sum_j U_c[i,j] * V_c[j,k] * W_c[k,i] * A[i,j] * B[j,k]
        # No, that mixes A and B in a weird way.
        
        # I think the correct formula for a rank-1 term in the tensor decomposition is:
        # C[i,k] += U_c[i, *] dot A_row_i * V_c[*, k] dot B_col_k * W_c[k, i]
        # But that's not standard.
        
        # Let me try yet another interpretation. Maybe:
        # The rank-r decomposition means:
        # C = sum_c W_c^T * (U_c * A) * (B * V_c)
        # where U_c * A means element-wise product summed over j: (U_c * A)[i] = sum_j U_c[i,j] A[i,j]
        # and B * V_c means element-wise product summed over j: (B * V_c)[k] = sum_j B[j,k] V_c[j,k]
        # Then C[i,k] = sum_c W_c[k,i] * (sum_j U_c[i,j] A[i,j]) * (sum_j B[j,k] V_c[j,k])
        
        # Let's test this with schoolbook.
        # For schoolbook, U_c[i,j] = 1 if c corresponds to (i,j,k), else 0.
        # So (U_c * A)[i] = A[i,j_c] where j_c is the j associated with c.
        # Similarly (B * V_c)[k] = B[j_c, k] where j_c is the same.
        # And W_c[k,i] = 1.
        # So C[i,k] = sum_c A[i,j_c] * B[j_c,k] * 1 = sum_j A[i,j] B[j,k] = C_true[i,k]. ✓
        
        # Great, so the formula is:
        # C[i,k] = sum_c W_c[k,i] * (sum_j U_c[i,j] A[i,j]) * (sum_j B[j,k] V_c[j,k])
        
        u_a = np.sum(U_c * A, axis=1)  # shape (n,)
        b_v = np.sum(B * V_c, axis=0)  # shape (p,)
        C_comp += W_c.T * np.outer(u_a, b_v)
    
    if np.allclose(C_comp, C_true, atol=1e-10):
        return True
    return False

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Schoolbook as fallback
    def schoolbook():
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try block decomposition
    if time.time() < deadline:
        try:
            U_blk, V_blk, W_blk = _block_decompose_4x5x5()
            if _verify_decomposition(U_blk, V_blk, W_blk, n, m, p):
                rank_blk = U_blk.shape[1]
                if rank_blk < best_rank:
                    best_rank = rank_blk
                    best_result = {"U": U_blk.tolist(), "V": V_blk.tolist(), "W": W_blk.tolist()}
        except Exception:
            pass
    
    if time.time() > deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
