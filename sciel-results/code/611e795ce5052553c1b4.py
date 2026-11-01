import numpy as np
import time

def _verify_fast(U, V, W, n=2, m=4, p=8, trials=50, tol=1e-9):
    """
    Fast vectorized verification.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = U.shape[1]
    rng = np.random.RandomState(42)
    
    U_reshaped = U.reshape(n, m, rank)  # U[i, j, c]
    V_reshaped = V.reshape(m, p, rank)  # V[l, k, c]
    W_reshaped = W.reshape(p, n, rank)  # W[k, i, c]
    
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        factor_A = np.einsum('ij,ijc->ic', A, U_reshaped)
        factor_B = np.einsum('lk,lkc->kc', B, V_reshaped)
        
        C_computed = np.einsum('kic,ic,kc->ik', W_reshaped, factor_A, factor_B)
        
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_4x4x4():
    """
    Returns U, V, W for 4x4x4 matrix multiplication with rank 20 (Strassen recursion).
    U: (16, 20), V: (16, 20), W: (16, 20)
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    U4 = np.zeros((16, 20))
    V4 = np.zeros((16, 20))
    W4 = np.zeros((16, 20))
    
    # 4x4 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # Strassen terms for 4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map the 2x2 Strassen decomposition to the 4x4 blocks.
    # Each term is a product of two 2x2 matrices, resulting in a 2x2 matrix.
    # This 2x2 matrix is placed in the corresponding 2x2 block of the 4x4 result.
    
    # Block indices for 4x4 (flattened):
    # A11: rows 0-3, cols 0-3
    # A12: rows 0-3, cols 4-7
    # A21: rows 4-7, cols 0-3
    # A22: rows 4-7, cols 4-7
    
    # For each of the 7 terms, we add 2x2=4 sub-terms? No, the rank is additive.
    # R(4x4x4) <= 7 * R(2x2x2) = 7 * 7 = 49.
    # Wait, the prompt says "known rank-20 decomposition".
    # Is R(4x4x4) = 20?
    # The best known upper bound for R(4x4x4) is 49 (Strassen).
    # Lower bounds are around 40-50.
    # Rank 20 is likely incorrect for 4x4x4.
    # Let's check R(4x4x4).
    # Actually, maybe the inspiration meant R(4x4x4) is not 20, but something else?
    # Or maybe it's a specific algorithm?
    # No, standard Strassen gives 49.
    # However, the task says "apply the known rank-20 decomposition... to achieve total rank 20".
    # This implies the inspiration believes R(4x4x4) = 20 or that the embedded problem has rank 20.
    # If I use Strassen (49), the rank will be 49, which is < 56.
    # If the inspiration is wrong about 20, I still improve from 56 to 49.
    # Let's implement the standard Strassen 4x4x4 (rank 49).
    
    # Let's build the 4x4x4 tensor using 7 blocks of 2x2x2.
    # Term t (0..6):
    # U_term: combination of A blocks
    # V_term: combination of B blocks
    # W_term: combination of C blocks
    
    for t in range(7):
        base_col = t * 7
        for c in range(7):
            col_idx = base_col + c
            
            # U part
            # A11 is at (0,0), A12 at (0,1), A21 at (1,0), A22 at (1,1)
            # U2 has entries for a 2x2 matrix.
            # We need to map U2[i,j] to the 4x4 matrix.
            # The 4x4 matrix A is composed of 2x2 blocks.
            # A11 corresponds to A[0:2, 0:2].
            # A12 corresponds to A[0:2, 2:4].
            # A21 corresponds to A[2:4, 0:2].
            # A22 corresponds to A[2:4, 2:4].
            
            # U2[i,j] contributes to A_block[i//2, j//2][i%2, j%2] ?
            # No, U2 is for a 2x2 matrix.
            # If the term is (A11+A22), then U2[0,0] (which is A11[0,0]) adds to A[0,0] and A[2,2]?
            # Wait, Strassen 2x2x2 uses A11, A12, A21, A22 as scalars? No, as 2x2 matrices.
            # The U matrix in 2x2x2 decomposition has shape (4, 7).
            # U[i, j, c] where i,j are indices in 2x2.
            # So U2[i,j,c] is the coefficient for A[i,j] in the c-th term.
            
            # For 4x4x4, we have blocks A11, A12, A21, A22.
            # The coefficient for A11[i,j] in term t is U2[i,j,t].
            # So U4[row, col] where row corresponds to A11[i,j] is U2[i,j,t].
            
            # Mapping:
            # A11[i,j] -> U4[i*4+j, col] += U2[i,j,t]
            # A12[i,j] -> U4[i*4+2+j, col] += U2[i,j,t]  (Note: A12 is top-right, so cols 2,3)
            # A21[i,j] -> U4[2*4+i+j, col] += U2[i,j,t]  (Note: A21 is bottom-left, so rows 2,3)
            # A22[i,j] -> U4[2*4+2+i+j, col] += U2[i,j,t]
            
            # But which blocks are involved in term t?
            # M1: A11, A22. U2 has non-zeros at (0,0) and (1,1).
            # So for M1, U2[0,0,t]=1, U2[1,1,t]=1.
            # This means A11[0,0] and A22[1,1] are used?
            # No, U2[i,j,t] is the coefficient for the element A[i,j] of the 2x2 matrix.
            # In the 4x4 context, the "2x2 matrix" is actually a block.
            # So if U2[0,0,t]=1, it means the top-left element of the top-left block (A11[0,0]) is used.
            # And U2[1,1,t]=1 means the bottom-right element of the bottom-right block (A22[1,1]) is used.
            # This is correct for Strassen.
            
            for i in range(2):
                for j in range(2):
                    val = U2[i, j, t]
                    if val != 0:
                        # A11
                        U4[i*4+j, col_idx] += val
                        # A12
                        U4[i*4+2+j, col_idx] += val
                        # A21
                        U4[2*4+i+j, col_idx] += val
                        # A22
                        U4[2*4+2+i+j, col_idx] += val
            
            # V part (similar logic for B)
            for l in range(2):
                for k in range(2):
                    val = V2[l, k, t]
                    if val != 0:
                        # B11 (top-left)
                        V4[l*4+k, col_idx] += val
                        # B12 (top-right)
                        V4[l*4+2+k, col_idx] += val
                        # B21 (bottom-left)
                        V4[2*4+l+k, col_idx] += val
                        # B22 (bottom-right)
                        V4[2*4+2+l+k, col_idx] += val
            
            # W part (similar logic for C)
            for k in range(2):
                for i in range(2):
                    val = W2[k, i, t]
                    if val != 0:
                        # C11 (top-left)
                        W4[k*4+i, col_idx] += val
                        # C12 (top-right)
                        W4[k*4+2+i, col_idx] += val
                        # C21 (bottom-left)
                        W4[2*4+k+i, col_idx] += val
                        # C22 (bottom-right)
                        W4[2*4+2+k+i, col_idx] += val
                        
    return U4, V4, W4

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 2, 4, 8
    
    # Strategy: Embed 2x4x8 into 4x4x8?
    # The inspiration says "Embed ... into 4x4 by 4x4 ... rank 20".
    # This is likely a misconception about the rank of 4x4x4.
    # However, we can embed 2x4x8 into 4x4x8?
    # Or 4x4x4?
    # If we pad A (2x4) to 4x4 (adding 2 rows of 0).
    # Pad B (4x8) to 4x4? No, B is 4x8. We can't pad B to 4x4 without changing the problem.
    # We can pad B to 4x8 by adding 0 columns? No, it's already 4x8.
    # We can pad the output C (2x8) to 4x8.
    # So the problem is 4x4x8 multiplication, but we only care about the top 2 rows of C.
    # R(4x4x8) is not standard.
    
    # Alternative: Embed into 4x4x4?
    # If we pad A (2x4) to 4x4.
    # If we pad B (4x8) to 4x4? Impossible, B has 8 columns.
    # Unless we split B into two 4x4 matrices B1, B2.
    # Then C = A_padded * [B1, B2] = [C1, C2].
    # This is two 4x4x4 multiplications.
    # Rank = 2 * R(4x4x4) = 2 * 49 = 98. Worse.
    
    # Let's re-read the inspiration: "Embed the 2x4 by 4x8 multiplication into a 4x4 by 4x4 matrix multiplication... rank 20".
    # This is mathematically suspect. R(4x4x4) is ~49.
    # Maybe it means R(4x4x4) is 49, and the total rank is 49?
    # How can 2x4x8 be embedded in 4x4x4?
    # It can't, unless we lose information.
    
    # Let's try a different embedding.
    # 2x4x8 tensor.
    # Can we view it as a 4x4x4 tensor with some zeros?
    # No, dimensions don't match.
    
    # Let's stick to the proven 56 (Strassen on blocks) or 49 (Strassen on 4x4x4 if we can map it).
    # Wait, can we map 2x4x8 to 4x4x4?
    # No.
    
    # What if the inspiration meant R(4x4x4) = 49, and we can solve 2x4x8 with rank 49?
    # How?
    # 2x4x8 is a sub-tensor of 4x4x8?
    # If we have a rank-R decomposition of 4x4x8, we can restrict it to 2x4x8.
    # But we don't have a decomposition of 4x4x8.
    
    # Let's try to construct a rank-49 decomposition by padding 2x4x8 to 4x4x8?
    # No, 4x4x8 is not standard.
    
    # Let's try to construct a rank-56 decomposition (current best) but verify if we can do better with a different block structure.
    # Current: 4 blocks of 2x4x2. Rank 4*14 = 56.
    # Alternative: 2 blocks of 2x4x4?
    # R(2x4x4) <= R(2x2x4) + R(2x2x4)?
    # R(2x2x4) = R(2x2x2) + R(2x2x2) = 14.
    # So R(2x4x4) <= 14 + 14 = 28.
    # Total rank = 2 * 28 = 56. Same.
    
    # What about R(2x4x4) directly?
    # Is R(2x4x4) < 28?
    # R(2xn x 2) = 2n+2?
    # R(2x2x2)=7. 2*2+2=6? No, 7.
    # R(2x3x2)=11?
    # R(2x4x2)=14?
    # If R(2x4x2)=14, then R(2x4x4) = 28.
    
    # It seems 56 is a solid lower bound for this approach.
    
    # Let's implement the 56 rank solution as the baseline, and then try to find a rank-49 solution if possible.
    # Actually, the prompt says "current best of 56".
    # If I cannot find < 56, I should return 56.
    # But the inspiration suggests 20.
    # If I implement the 4x4x4 Strassen (49) and try to map it, it might fail.
    
    # Let's just implement the 56 rank solution robustly.
    
    U_total = np.zeros((n*m, 56))
    V_total = np.zeros((m*p, 56))
    W_total = np.zeros((p*n, 56))
    
    U_str, V_str, W_str = _strassen_2x2x2()
    
    for t in range(4):
        base_col = t * 14
        for c in range(7):
            col_idx = base_col + c
            
            # U for A1 (cols 0,1)
            for i in range(2):
                for j in range(2):
                    val = U_str[i*2+j, c]
                    if val != 0:
                        U_total[i*4+j, col_idx] += val
                        
            # V for B1_top (rows 0,1)
            for l in range(2):
                for k in range(2):
                    val = V_str[l*2+k, c]
                    if val != 0:
                        V_total[l*8 + (2*t + k), col_idx] += val
                        
            # W for C_sub
            for k in range(2):
                for i in range(2):
                    val = W_str[k*2+i, c]
                    if val != 0:
                        W_total[(2*t + k)*2 + i, col_idx] += val
                        
        base_col = t * 14 + 7
        for c in range(7):
            col_idx = base_col + c
            
            # U for A2 (cols 2,3)
            for i in range(2):
                for j in range(2):
                    val = U_str[i*2+j, c]
                    if val != 0:
                        U_total[i*4+(2+j), col_idx] += val
                        
            # V for B1_bot (rows 2,3)
            for l in range(2):
                for k in range(2):
                    val = V_str[l*2+k, c]
                    if val != 0:
                        V_total[(2+l)*8 + (2*t + k), col_idx] += val
                        
            # W for C_sub
            for k in range(2):
                for i in range(2):
                    val = W_str[k*2+i, c]
                    if val != 0:
                        W_total[(2*t + k)*2 + i, col_idx] += val
                        
    if not _verify_fast(U_total, V_total, W_total, n, m, p, trials=50):
        # Fallback to schoolbook
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
