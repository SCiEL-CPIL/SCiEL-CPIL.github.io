"""Matrix multiplication 4,4,8 - rank-98 Strassen-Kronecker decomposition."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_rank_7_factors():
    """
    Standard Strassen decomposition for 2x2 matrix multiplication (rank 7).
    
    U: shape (4, 7) - A matrix is 2x2, flattened row-major: A[0,0], A[0,1], A[1,0], A[1,1]
    V: shape (4, 7) - B matrix is 2x2, flattened row-major: B[0,0], B[0,1], B[1,0], B[1,1]
    W: shape (4, 7) - C matrix is 2x2, flattened row-major: C[0,0], C[0,1], C[1,0], C[1,1]
    
    C = A @ B where C[i,j] = sum_k A[i,k] * B[k,j]
    
    Strassen's 7 products:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B10 - B11)
    M4 = A11(B01 - B00)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3
    
    U = [[1, 0, 1, 0, 1, 0, 0],   # A00
         [0, 0, 0, 0, 1, 0, -1],  # A01
         [0, 1, 0, 0, 0, -1, 0],  # A10
         [1, 1, 0, 1, 0, 0, -1]]  # A11
    
    V = [[1, 1, 0, -1, 0, 1, 0],   # B00
         [0, 0, 0, 1, 0, 1, 0],   # B01
         [1, 0, 1, 0, 0, 0, 1],   # B10
         [1, 0, -1, 0, 1, 0, 1]]  # B11
    
    W = [[1, 0, 0, 1, -1, 0, 1],   # C00
         [0, 0, 1, 0, 1, 0, 0],   # C01
         [0, 1, 0, 1, 0, 0, 0],   # C10
         [1, -1, 1, 0, 0, 1, 0]]  # C11
    
    return np.array(U, dtype=int), np.array(V, dtype=int), np.array(W, dtype=int)

def _verify_strassen_2x2():
    """Verify the Strassen factors are correct."""
    U, V, W = _strassen_rank_7_factors()
    # Reconstruct: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # For 2x2 matrix mult: T[i,j,k] should be 1 if j==k (standard matmult A[i,k]*B[k,j] = C[i,j])
    # Actually, the tensor for 2x2 matmult: T[i,j,k] = delta_{j,k} where i=row of A/col of C, 
    # j=col of A/row of B, k=col of B/row of C... 
    
    # Let me just verify with random matrices
    np.random.seed(42)
    for _ in range(10):
        A = np.random.randint(-5, 6, size=(2, 2)).astype(float)
        B = np.random.randint(-5, 6, size=(2, 2)).astype(float)
        C_direct = A @ B
        
        # Compute using factorization: C_flat = (U^T * A_flat) @ (V^T * B_flat) ... 
        # Actually: for each r, P_r = sum_{i,j} U[i,j,r] * A[i,j] * (sum_{k,l} V[k,l,r] * B[k,l])
        # Then C[i,j] = sum_r W[i,j,r] * P_r
        # 
        # In our representation, U, V, W are (4,7) where 4 is flattened (2x2) index.
        # P_r = dot(A_flat, U[:,r]) * dot(B_flat, V[:,r])
        # C_flat[s] = sum_r W[s,r] * P_r
        #
        # Wait, that's not right either. Let me think more carefully.
        #
        # The matrix multiplication tensor T has T[i,j,k] = 1 if this term A[i,j]*B[j,k] appears in C[i,k].
        # Actually no: C[i,k] = sum_j A[i,j] * B[j,k]
        # So T[i,k,j] = delta... Let me use the convention from the task.
        #
        # The task says: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
        # For our 2x2 case: n=2, m=2, p=2
        # U[2,7], V[4,7], W[4,7]... wait that's for 2,2,2 which is rank 8 schoolbook.
        #
        # For 2x2 x 2x2 = 2x2: n=2, m=2, p=2
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        # C[i,k] = sum_{j,r} ... actually the decomposition is:
        # C[i,k] = sum_r W[k*n+i, r] * (sum_{j} ... )
        #
        # Actually the standard form: T = sum_r U_r ⊗ V_r ⊗ W_r
        # where (U_r) is an n×m matrix, (V_r) is an m×p matrix, (W_r) is a p×n matrix
        # and T[i,j,k] (where C[i,k] += A[i,j]*B[j,k]) = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
        #
        # Wait, let me re-read. The task says:
        # "U, V, W': factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
        # "W[k*n+i,r]"
        # 
        # So the tensor entry T[a,b,c] where a indexes A (n*m), b indexes B (m*p), c indexes C (p*n)
        # is: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # 
        # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
        # A[i,j] has flat index i*m+j in U
        # B[j,k] has flat index j*p+k in V
        # C[i,k] has flat index k*n+i in W
        # T[i*m+j, j*p+k, k*n+i] = 1 for valid indices, 0 otherwise.
        #
        # Let me verify my Strassen factors produce this.
        
        A_flat = A.flatten()  # [A00, A01, A10, A11]
        B_flat = B.flatten()  # [B00, B01, B10, B11]
        
        # Products: P_r = sum_{(i,j)} U[i*m+j, r] * A[i,j] * ... 
        # Actually in the tensor form: the r-th term contributes U[a,r]*V[b,r]*W[c,r] to T[a,b,c]
        # The computed C is: C_computed[k*n+i] = sum_{(i,j),(j,k)} T[i*m+j, j*p+k, k*n+i] * A[i,j] * B[j,k]
        # = sum_r [sum_{i,j} U[i*m+j,r] * A[i,j]] * [sum_{j,k} V[j*p+k,r] * B[j,k]] * W[k*n+i,r]
        # 
        # Hmm, that's not quite right. Let me think again.
        #
        # The way to use the factors:
        # C_computed = zeros
        # for r in range(rank):
        #     C_computed[k*n + i] += W[k*n+i, r] * (sum over appropriate A,B terms)
        #
        # Actually the standard interpretation:
        # For each rank component r, we form:
        #   A_r = U[:, r] reshaped to (n, m)
        #   B_r = V[:, r] reshaped to (m, p)  
        #   W_r = W[:, r] reshaped to (p, n)
        # And C = sum_r W_r ⊙ (A @ B_r) ... no.
        #
        # Let me just directly check: T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # This should equal 1 for all valid (i,j,k).
        
        U_arr, V_arr, W_arr = U, V, W
        n2, m2, p2 = 2, 2, 2
        ok = True
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    a_idx = i * m2 + j
                    b_idx = j * p2 + k
                    c_idx = k * n2 + i
                    t_val = sum(U_arr[a_idx, r] * V_arr[b_idx, r] * W_arr[c_idx, r] for r in range(7))
                    if t_val != 1:
                        ok = False
                        print(f"FAIL: i={i},j={j},k={k}: T={t_val}")
        if ok:
            pass  # Correct!
        else:
            raise ValueError("Strassen factors verification failed")

def construct():
    """
    Construct rank-98 decomposition for 4x4 @ 4x8 -> 4x8.
    
    The 4x4x8 tensor decomposes into two independent 4x4x4 tensors (for columns 0-3 and 4-7).
    Each 4x4x4 tensor can be handled by a rank-49 Strassen-Kronecker decomposition.
    Total rank = 49 + 49 = 98.
    
    For the 4x4x4 case (n=4, m=4, p=4):
    We use the Kronecker product approach with Strassen.
    
    The 4x4 matmul can be seen as a 2x2 block matmul where each block is 2x2.
    Using Strassen at the block level (rank 7) and direct multiplication within blocks,
    we get a decomposition.
    
    Actually, let me use a simpler correct approach: 
    Split the 4x4x4 into four 2x2x2 sub-problems is wrong.
    
    Better approach: Use the fact that 4x4x4 can be decomposed using Strassen applied
    hierarchically. But the known rank-49 is more complex.
    
    For reliability, let me construct a valid decomposition:
    - For 4x4x8: split into two 4x4x4 blocks.
    - For each 4x4x4: use a valid decomposition.
    
    A simple valid decomposition for 4x4x4 of rank 49 uses the Strassen hierarchy:
    View 4x4 as 2x2 blocks of 2x2. Block-level Strassen gives 7 block products.
    Each block product is a 2x2 matmul of rank 7.
    But this gives rank 7*7 = 49 for 4x4x4. 
    
    Let me verify: For 4x4 @ 4x4 = 4x4, view as 2x2 blocks.
    C_blocks = f(A_blocks, B_blocks) using Strassen with 7 block multiplies.
    Each block multiply is a 2x2 @ 2x2 = 2x2 which needs rank 7 (Strassen).
    Total rank = 7 * 7 = 49. 
    
    Let me construct this.
    """
    t0 = time.time()
    
    # Get 2x2 Strassen factors
    U2, V2, W2 = _strassen_rank_7_factors()
    
    # For 4x4x4: n=4, m=4, p=4
    # Block structure: 2x2 blocks of size 2x2
    # A is 4x4 -> 2x2 blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B is 4x4 -> 2x2 blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    # C is 4x4 -> 2x2 blocks: C00, C01, C10, C11
    
    # Strassen block products:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # For the rank decomposition of 4x4x4:
    # The block-level Strassen has rank 7, with factors (U_block, V_block, W_block) each of size 4x7
    # (4 being the 2x2 block indices flattened, 7 being the Strassen rank)
    # 
    # The sub-block multiplication has rank 7 (Strassen for 2x2).
    #
    # Combined: rank 49, with factors of size 16x49.
    
    # Block-level Strassen factors (for 2x2 block indices):
    # U_block[bi, s] where bi in {0,1,2,3} = {(0,0),(0,1),(1,0),(1,1)}, s in 0..6
    U_block = U2  # (4, 7)
    V_block = V2  # (4, 7)
    W_block = W2  # (4, 7)
    
    # Now for the full 4x4x4:
    # U4 has shape (16, 49)
    # V4 has shape (16, 49)
    # W4 has shape (16, 49)
    #
    # Index in U4: i*4+j where i,j in 0..3 (row, col of 4x4 A)
    # Index in V4: j*4+k where j,k in 0..3
    # Index in W4: k*4+i where k,i in 0..3
    #
    # The combined rank index is (s, r) where s is block-Strassen (0..6) and r is sub-block-Strassen (0..6)
    # Total 49.
    
    # For U4[i*4+j, s*7+r]:
    # Block (i_block, j_block) = (i//2, j//2), sub-block indices (i_sub, j_sub) = (i%2, j%2)
    # U4[i*4+j, s*7+r] = U_block[(i//2)*2 + (j//2), s] * U2[i%2, r]
    
    # For V4[j*4+k, s*7+r]:
    # Block (j_block, k_block) = (j//2, k//2), sub-block (j_sub, k_sub) = (j%2, k%2)
    # V4[j*4+k, s*7+r] = V_block[(j//2)*2 + (k//2), s] * V2[j%2, r]
    
    # For W4[k*4+i, s*7+r]:
    # Block (k_block, i_block) = (k//2, i//2), sub-block (k_sub, i_sub) = (k%2, i%2)
    # W4[k*4+i, s*7+r] = W_block[(k//2)*2 + (i//2), s] * W2[k%2, r]
    
    def make_4x4x4_factors():
        """Construct rank-49 factors for 4x4x4."""
        rank4 = 49
        U4 = [[0] * rank4 for _ in range(16)]
        V4 = [[0] * rank4 for _ in range(16)]
        W4 = [[0] * rank4 for _ in range(16)]
        
        for i in range(4):
            for j in range(4):
                ib, jb = i // 2, j // 2
                is_, js = i % 2, j % 2
                for s in range(7):
                    for r in range(7):
                        U4[i * 4 + j][s * 7 + r] = U_block[ib * 2 + jb][s] * U2[is_][r]
        
        for j in range(4):
            for k in range(4):
                jb, kb = j // 2, k // 2
                js, ks = j % 2, k % 2
                for s in range(7):
                    for r in range(7):
                        V4[j * 4 + k][s * 7 + r] = V_block[jb * 2 + kb][s] * V2[js][r]
        
        for k in range(4):
            for i in range(4):
                kb, ib = k // 2, i // 2
                ks, is_ = k % 2, i % 2
                for s in range(7):
                    for r in range(7):
                        W4[k * 4 + i][s * 7 + r] = W_block[kb * 2 + ib][s] * W2[ks][r]
        
        return U4, V4, W4
    
    # Verify the 4x4x4 factors
    U4, V4, W4 = make_4x4x4_factors()
    
    # Quick verification on random inputs
    np.random.seed(123)
    A_test = np.random.randint(-5, 6, size=(4, 4)).astype(float)
    B_test = np.random.randint(-5, 6, size=(4, 4)).astype(float)
    C_direct = A_test @ B_test
    
    # Compute using factors
    A_flat = A_test.flatten()  # 16 elements
    B_flat = B_test.flatten()  # 16 elements
    C_flat = np.zeros(16)
    for idx in range(16):
        C_flat[idx] = 0
    for rr in range(49):
        prod_val = 0.0
        for a_idx in range(16):
            if U4[a_idx][rr] != 0:
                prod_val += U4[a_idx][rr] * A_flat[a_idx]
                # This isn't how it works - it's a triple product
        # Actually: C[k*4+i] = sum_{j,r} T[j*4+k ... 
    
    # Let me use the tensor verification method
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] should equal delta for matmul
    
    ok_4x4 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a_idx = i * 4 + j
                b_idx = j * 4 + k
                c_idx = k * 4 + i
                t_val = sum(U4[a_idx][r] * V4[b_idx][r] * W4[c_idx][r] for r in range(49))
                expected = 1.0  # For matmul, T[i,j,k]=1 always for valid indices
                if abs(t_val - expected) > 1e-9:
                    ok_4x4 = False
                    print(f"4x4x4 FAIL: i={i},j={j},k={k}: T={t_val}")
    if not ok_4x4:
        raise ValueError("4x4x4 factor verification failed")
    
    # Now for 4x4x8: split into two 4x4x4 blocks
    # Block 1: columns 0-3 of B, columns 0-3 of C
    # Block 2: columns 4-7 of B, columns 4-7 of C
    
    n, m, p = 4, 4, 8
    total_rank = 98
    
    U_full = [[0] * total_rank for _ in range(n * m)]
    V_full = [[0] * total_rank for _ in range(m * p)]
    W_full = [[0] * total_rank for _ in range(p * n)]
    
    # Block 1: r in 0..48, uses columns 0-3 of B (indices 0-15 in V_full for 4x4)
    # B has 4*8=32 rows. For block 1, B rows are j*8+k where k in 0..3
    # C has 8*4=32 rows. For block 1, C rows are k*4+i where k in 0..3
    
    for a_idx in range(16):
        for r in range(49):
            U_full[a_idx][r] = U4[a_idx][r]
    
    for j in range(4):
        for k in range(4):
            v_idx_block1 = j * 8 + k  # In V_full (m*p = 32)
            v_idx_4x4 = j * 4 + k     # In V4 (16)
            for r in range(49):
                V_full[v_idx_block1][r] = V4[v_idx_4x4][r]
    
    for k in range(4):
        for i in range(4):
            w_idx_block1 = k * 4 + i  # In W_full (p*n = 32), for block 1
            w_idx_4x4 = k * 4 + i     # In W4 (16)
            for r in range(49):
                W_full[w_idx_block1][r] = W4[w_idx_4x4][r]
    
    # Block 2: r in 49..96, uses columns 4-7 of B, columns 4-7 of C
    for a_idx in range(16):
        for r in range(49):
            U_full[a_idx][49 + r] = U4[a_idx][r]
    
    for j in range(4):
        for k in range(4):
            v_idx_block2 = j * 8 + (k + 4)  # In V_full
            v_idx_4x4 = j * 4 + k           # In V4
            for r in range(49):
                V_full[v_idx_block2][49 + r] = V4[v_idx_4x4][r]
    
    for k in range(4):
        for i in range(4):
            w_idx_block2 = (k + 4) * 4 + i  # In W_full, for block 2
            w_idx_4x4 = k * 4 + i           # In W4
            for r in range(49):
                W_full[w_idx_block2][49 + r] = W4[w_idx_4x4][r]
    
    # Final verification with random matrices
    np.random.seed(456)
    A_rand = np.random.randint(-10, 11, size=(4, 4)).astype(float)
    B_rand = np.random.randint(-10, 11, size=(4, 8)).astype(float)
    C_direct = A_rand @ B_rand
    
    # Verify tensor: T[a,b,c] should give C when contracted with A and B
    # C[i,k] = sum_{j} A[i,j] * B[j,k]
    # In flat form: C[k*4+i] = sum_j A[i*4+j] * B[j*8+k]
    # T[i*4+j, j*8+k, k*4+i] should = 1
    
    ok_full = True
    for i in range(4):
        for j in range(4):
            for k in range(8):
                a_idx = i * 4 + j
                b_idx = j * 8 + k
                c_idx = k * 4 + i
                t_val = sum(U_full[a_idx][r] * V_full[b_idx][r] * W_full[c_idx][r] for r in range(total_rank))
                if abs(t_val - 1.0) > 1e-9:
                    ok_full = False
                    if t_val == 0:
                        pass  # expected 1
                    elif abs(t_val) > 1e-9:
                        print(f"Full FAIL: i={i},j={j},k={k}: T={t_val}")
    
    if not ok_full:
        raise ValueError("Full 4x4x8 factor verification failed")
    
    elapsed = time.time() - t0
    print(f"Constructed rank-{total_rank} decomposition in {elapsed:.2f}s")
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
