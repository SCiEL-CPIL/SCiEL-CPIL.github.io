"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    """Construct rank-56 decomposition for 4x4 * 4x8 matrix multiplication.
    
    Uses Kronecker product of:
    - 2x2x2 Strassen tensor (rank 7)
    - 2x2x4 tensor (rank 8)
    
    Total rank = 7 * 8 = 56
    """
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 120s limit
    
    # 2x2x2 Strassen decomposition (rank 7)
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # Strassen products:
    # m1 = (a-d)(e+h)
    # m2 = (a+b)e  
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+d)(h-f)  -- wait, let me use standard Strassen
    # Standard:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # For 2x2x2, indices: i in {0,1}, j in {0,1}, k in {0,1}
    # A[i,j], B[j,k], C[i,k]
    # U has shape (2*2, 7) = (4, 7): rows indexed by (i,j)
    # V has shape (2*2, 7) = (4, 7): rows indexed by (j,k)  
    # W has shape (2*2, 7) = (4, 7): rows indexed by (k,i) [trace convention W[k*n+i]]
    
    # Strassen factors for 2x2x2:
    # Product r: U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Let me define the 7 products:
    # p1 = (a+d)(e+h): U contributions: a->(0,0), d->(1,1); V: e->(0,0), h->(1,1)
    # p2 = (c+d)e: U: c->(1,0), d->(1,1); V: e->(0,0)
    # p3 = a(f-h): U: a->(0,0); V: f->(0,1), h->(1,1)
    # p4 = d(g-e): U: d->(1,1); V: g->(1,0), e->(0,0)
    # p5 = (a+b)h: U: a->(0,0), b->(0,1); V: h->(1,1)
    # p6 = (c-a)(e+f): U: c->(1,0), a->(0,0); V: e->(0,0), f->(0,1)
    # p7 = (b-d)(g+h): U: b->(0,1), d->(1,1); V: g->(1,0), h->(1,1)
    
    # W coefficients (for C[i,k] = sum over r of W[k,i,r] * p_r):
    # C[0,0] = p1 + p4 - p5 + p7
    # C[0,1] = p3 + p5
    # C[1,0] = p2 + p4
    # C[1,1] = p1 - p2 + p3 + p6
    
    # W indexed by [k*n + i] where n=2, so W[k*2 + i, r]
    # W[0*2+0, r] = W[0, r] for C[0,0]: [1, 0, 0, 1, -1, 0, 1]
    # W[1*2+0, r] = W[2, r] for C[1,0]: [0, 1, 0, 1, 0, 0, 0]
    # W[0*2+1, r] = W[1, r] for C[0,1]: [0, 0, 1, 0, 1, 0, 0]
    # W[1*2+1, r] = W[3, r] for C[1,1]: [1, -1, 1, 0, 0, 1, 0]
    
    # U for 2x2x2 (4 rows: (i,j) -> i*2+j, 7 cols):
    # p1: a(0,0)=1, d(1,1)=1
    # p2: c(1,0)=1, d(1,1)=1
    # p3: a(0,0)=1
    # p4: d(1,1)=1
    # p5: a(0,0)=1, b(0,1)=1
    # p6: c(1,0)=1, a(0,0)=-1
    # p7: b(0,1)=1, d(1,1)=-1
    U_222 = [
        [1, 0,  1, 0,  1, -1, 0],  # (0,0) = a
        [0, 0,  0, 0,  1,  0, 1],  # (0,1) = b
        [0, 1,  0, 0,  0,  1, 0],  # (1,0) = c
        [1, 1,  0, 1,  0,  0,-1],  # (1,1) = d
    ]
    
    # V for 2x2x2 (4 rows: (j,k) -> j*2+k, 7 cols):
    # p1: e(0,0)=1, h(1,1)=1
    # p2: e(0,0)=1
    # p3: f(0,1)=1, h(1,1)=-1
    # p4: g(1,0)=1, e(0,0)=-1
    # p5: h(1,1)=1
    # p6: e(0,0)=1, f(0,1)=1
    # p7: g(1,0)=1, h(1,1)=1
    V_222 = [
        [1, 1,  0,-1,  0, 1, 0],  # (0,0) = e
        [0, 0,  1, 0,  0, 1, 0],  # (0,1) = f
        [0, 0,  0, 1,  0, 0, 1],  # (1,0) = g
        [1, 0, -1, 0,  1, 0, 1],  # (1,1) = h
    ]
    
    # W for 2x2x2 (4 rows: (k,i) -> k*2+i, 7 cols):
    W_222 = [
        [ 1, 0, 0, 1, -1, 0, 1],  # (0,0) -> C[0,0]
        [ 0, 0, 1, 0,  1, 0, 0],  # (0,1) -> C[0,1]
        [ 0, 1, 0, 1,  0, 0, 0],  # (1,0) -> C[1,0]
        [ 1,-1, 1, 0,  0, 1, 0],  # (1,1) -> C[1,1]
    ]
    
    # 2x2x4 decomposition (rank 8)
    # A(2x2) * B(2x4) -> C(2x2)
    # B = [[b00,b01,b02,b03],[b10,b11,b12,b13]]
    # C[0,0] = a*b00 + b*b10
    # C[0,1] = a*b01 + b*b11
    # C[0,2] = a*b02 + b*b12
    # C[0,3] = a*b03 + b*b13
    # C[1,0] = c*b00 + d*b10
    # C[1,1] = c*b01 + d*b11
    # C[1,2] = c*b02 + d*b12
    # C[1,3] = c*b03 + d*b13
    # 
    # This is just 8 independent products (no sharing possible for minimum rank)
    # Actually, the rank is 8: each of the 4 A entries times each column of B
    # But we can group: for each column k of B, we compute A * B[:,k] (2x2 result)
    # That's 4 separate 2x2x2 multiplications... but they share A.
    # The tensor rank of 2x2x4 is 8 (it's a direct sum of 4 copies of the "diagonal" part)
    # 
    # Actually, let me think more carefully. The (2,2,4) tensor:
    # T[i,j,k] = A[i,j] * B[j,k]
    # This is a 2x2x4 tensor. Its rank: we can write it as sum of rank-1 tensors.
    # For each j in {0,1} and k in {0,1,2,3}: A[i,j] * B[j,k] = (e_i * delta_{j}) * (delta_j * e_k) ... 
    # Not quite. Let me use the standard: rank = number of terms.
    # T = sum_{j=0}^{1} sum_{k=0}^{3} (col_j(A)) ⊗ (row_j(B) at position k) ⊗ ...
    # 
    # Simpler: T[i,j,k] = sum over the "natural" decomposition:
    # For each (j,k) pair, we have A[i,j]*B[j,k]. The rank-1 term would be u_i * v_j * w_k.
    # If we fix j and k: u = e_i (depends on i), v = delta_j, w = delta_k. But u depends on i which is an index, not a free variable.
    # 
    # The correct rank-1 decomposition: T = sum_{j=0}^{1} sum_{k=0}^{3} (A_col_j) ⊗ (e_j ⊗ e_k) ⊗ ... no.
    # 
    # Let me just use the trivial rank-8: 
    # Product r = j*4 + k (for j in {0,1}, k in {0,1,2,3}):
    # U[i, j*4+k] = delta(i, ...) -- no, U[i,j,r] where i is row of A, j is col of A
    # U has shape (n*m, r) = (2*2, 8) = (4, 8): row index = i*2+j
    # V has shape (m*p, r) = (2*4, 8) = (8, 8): row index = j*4+k
    # W has shape (p*n, r) = (4*2, 8) = (8, 8): row index = k*2+i
    #
    # For product r = j*4+k:
    # U[i*2+j, r] = 1 if i is free... no.
    # The term is A[i,j] * B[j,k] contributing to C[i,k].
    # As a rank-1 tensor: (A[i,j]) * (B[j,k]) * (delta for C[i,k])
    # U[i*2+j, r] should pick out A[i,j]: so U[i*2+j, j*4+k] = 1
    # V[j*4+k, r] should pick out B[j,k]: so V[j*4+k, j*4+k] = 1
    # W[k*2+i, r] should be 1: W[k*2+i, j*4+k] = 1
    #
    # Let's verify: sum_r U[i*2+j, r] * V[j*4+k, r] * W[k*2+i, r]
    # = sum_r U[i*2+j, r] * V[j*4+k, r] * W[k*2+i, r]
    # Nonzero when r = j'*4+k' such that:
    #   U[i*2+j, j'*4+k'] = 1 => j = j'
    #   V[j*4+k, j'*4+k'] = 1 => j=j', k=k'
    #   W[k*2+i, j'*4+k'] = 1 => k=k', i=i' (always true for given i,k)
    # So r = j*4+k, and the product is 1*1*1 = 1. 
    # But we need A[i,j]*B[j,k], not 1. The U and V matrices should have entries that are the actual values... 
    # No! In tensor decomposition, U, V, W are FIXED coefficient matrices. The actual A and B values are plugged in as:
    # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * (sum_r U[i*2+j, r] V[j*4+k, r] W[k*2+i, r])
    # Wait no. The decomposition is:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]  (for the tensor itself)
    # And the bilinear form is: C[i,k] = sum_{j} A[i,j] B[j,k] T[i,j,k]... 
    # 
    # Actually the matrix multiplication tensor is: M[i,j,k] = delta (it's 1 for all valid i,j,k in the "right" sense)
    # No. The tensor for matrix multiplication A(m x n) * B(n x p) -> C(m x p) is:
    # T[i,j,k] where i indexes A rows, j indexes A cols / B rows, k indexes B cols
    # T[i,j,k] = 1 (it's the "multiplication" tensor, all ones in the appropriate sense)
    # And C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r where
    # C[i,k] = sum_r (sum_j A[i,j] u_r[i,j]) * (sum_j B[j,k] v_r[j,k]) * w_r[i,k]
    # 
    # So: U has shape (m*n, r), V has shape (n*p, r), W has shape (p*m, r) [with W[k*m+i, r] convention]
    # And: C[i,k] = sum_r (A . u_r)[i] * (v_r . B)[k] * W[k*m+i, r]
    # where (A . u_r)[i] = sum_j A[i,j] * U[i*n+j, r]
    # and (v_r . B)[k] = sum_j V[j*p+k, r] * B[j,k]
    
    # OK so for 2x2x4: m=2, n=2, p=4
    # U shape: (2*2, 8) = (4, 8)
    # V shape: (2*4, 8) = (8, 8)  
    # W shape: (4*2, 8) = (8, 8), W[k*2+i, r]
    #
    # Trivial rank-8: for each (j,k) in {0,1}x{0,1,2,3}, r = j*4+k:
    # U[i*2+j, r] = 1 for all i (so U[i*2+j, j*4+k] = 1)
    # V[j*4+k, r] = 1 (so V[j*4+k, j*4+k] = 1)
    # W[k*2+i, r] = 1 for all i (so W[k*2+i, j*4+k] = 1)
    
    U_224 = [[0]*8 for _ in range(4)]
    V_224 = [[0]*8 for _ in range(8)]
    W_224 = [[0]*8 for _ in range(8)]
    for j in range(2):
        for k in range(4):
            r = j * 4 + k
            for i in range(2):
                U_224[i*2 + j][r] = 1
                W_224[k*2 + i][r] = 1
            V_224[j*4 + k][r] = 1
    
    # Now Kronecker product for 4x4x8:
    # 4x4x8 = (2x2x4) ⊗ (2x2x2)
    # A(4x4) viewed as 2x2 blocks of 2x2: A = [[A00, A01],[A10, A11]]
    # B(4x8) viewed as 2x2 blocks of 2x4: B = [[B00, B01],[B10, B11]] where each Bij is 2x4
    # C(4x8) = [[C00, C01],[C10, C11]] where Cij = A00*Bij + A01*B(j+1, i)... 
    # 
    # More carefully: C[i,k] = sum_j A[i,j] B[j,k]
    # With i = i1*2+i2, j = j1*2+j2, k = k1*4+k2 (where i1,i2 in {0,1}, j1,j2 in {0,1}, k1 in {0,1}, k2 in {0,1,2,3})
    # C[i1*2+i2, k1*4+k2] = sum_{j1,j2} A[i1*2+i2, j1*2+j2] * B[j1*2+j2, k1*4+k2]
    # = sum_{j1,j2} A_block[i1,j1][i2,j2] * B_block[j1,k1][j2,k2]
    # = sum_{j1} (sum_{j2} A_block[i1,j1][i2,j2] * B_block[j1,k1][j2,k2])
    # = sum_{j1} (A_block[i1,j1] * B_block[j1,k1])[i2, k2]
    #
    # So C_block[i1,k1][i2,k2] = sum_{j1} (A_block[i1,j1] * B_block[j1,k1])[i2,k2]
    # This is: the 2x2x2 multiplication of the "block matrix" A_blocks (2x2 matrix of 2x2 blocks) 
    # with the "block matrix" B_blocks (2x2 matrix of 2x4 blocks), producing 2x2 blocks of 2x4.
    #
    # The outer 2x2x2 structure uses Strassen (rank 7), and each "scalar" multiplication is actually a 2x2x4 multiplication (rank 8).
    # Total rank = 7 * 8 = 56.
    #
    # Kronecker product construction:
    # For each Strassen product r1 (0..6) and each 2x2x4 product r2 (0..7):
    #   r = r1 * 8 + r2
    #   U_448[row, r] = U_222[row_outer, r1] * U_224[row_inner, r2]
    #   V_448[row, r] = V_222[row_outer, r1] * V_224[row_inner, r2]
    #   W_448[row, r] = W_222[row_outer, r1] * W_224[row_inner, r2]
    #
    # Indexing:
    # U_448: shape (4*4, 56) = (16, 56), row = (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2
    #   Outer index for 2x2x2: i_outer*2 + j_outer where i_outer=i1, j_outer=j1 -> row_outer = i1*2+j1
    #   Inner index for 2x2x4: i_inner*2 + j_inner where i_inner=i2, j_inner=j2 -> row_inner = i2*2+j2
    #   So U_448[i1*8+i2*4+j1*2+j2, r1*8+r2] = U_222[i1*2+j1, r1] * U_224[i2*2+j2, r2]
    #
    # V_448: shape (4*8, 56) = (32, 56), row = (j1*2+j2)*8 + (k1*4+k2) = j1*16+j2*8+k1*4+k2
    #   Outer: j_outer*2 + k_outer where j_outer=j1, k_outer=k1 -> row_outer = j1*2+k1
    #   Inner: j_inner*4 + k_inner where j_inner=j2, k_inner=k2 -> row_inner = j2*4+k2
    #   So V_448[j1*16+j2*8+k1*4+k2, r1*8+r2] = V_222[j1*2+k1, r1] * V_224[j2*4+k2, r2]
    #
    # W_448: shape (8*4, 56) = (32, 56), row = (k1*4+k2)*4 + (i1*2+i2) = k1*16+k2*4+i1*2+i2
    #   Outer: k_outer*2 + i_outer where k_outer=k1, i_outer=i1 -> row_outer = k1*2+i1
    #   Inner: k_inner*2 + i_inner where k_inner=k2, i_inner=i2 -> row_inner = k2*2+i2
    #   So W_448[k1*16+k2*4+i1*2+i2, r1*8+r2] = W_222[k1*2+i1, r1] * W_224[k2*2+i2, r2]
    
    rank = 56
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    
    for r1 in range(7):
        for r2 in range(8):
            r = r1 * 8 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            u_row = i1*8 + i2*4 + j1*2 + j2
                            u_outer = i1*2 + j1
                            u_inner = i2*2 + j2
                            val = U_222[u_outer][r1] * U_224[u_inner][r2]
                            if val != 0:
                                U[u_row][r] = val
            
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(4):
                            v_row = j1*16 + j2*8 + k1*4 + k2
                            v_outer = j1*2 + k1
                            v_inner = j2*4 + k2
                            val = V_222[v_outer][r1] * V_224[v_inner][r2]
                            if val != 0:
                                V[v_row][r] = val
            
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(4):
                        for i2 in range(2):
                            w_row = k1*16 + k2*4 + i1*2 + i2
                            w_outer = k1*2 + i1
                            w_inner = k2*2 + i2
                            val = W_222[w_outer][r1] * W_224[w_inner][r2]
                            if val != 0:
                                W[w_row][r] = val
    
    # Verify
    if time.time() < deadline - 20:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Test with random matrices
        np.random.seed(42)
        A_test = np.random.randn(4, 4)
        B_test = np.random.randn(4, 8)
        C_expected = A_test @ B_test
        
        # Compute via decomposition
        C_computed = np.zeros((4, 8))
        for r in range(rank):
            u_vec = U_np[:, r]  # shape (16,)
            v_vec = V_np[:, r]  # shape (32,)
            w_vec = W_np[:, r]  # shape (32,)
            
            # u_vec reshaped to (4,4): u_vec[i*4+j]
            U_mat = u_vec.reshape(4, 4)
            # v_vec reshaped to (4,8): v_vec[j*8+k]
            V_mat = v_vec.reshape(4, 8)
            # w_vec: w_vec[k*4+i]
            
            # (A * U_mat) has shape (4,4): [i,j] = sum_j' A[i,j'] * U_mat[j', j]... 
            # Actually: sum_j A[i,j] * U_mat[i*4+j]... 
            # The contraction: left = sum_j A[i,j] * U[i*4+j, r]
            # This is: left[i] = sum_j A[i,j] * U_flat[i*4+j]
            # If we reshape U_flat to (4,4) as U_mat[i,j] = U_flat[i*4+j], then left[i] = sum_j A[i,j]*U_mat[i,j]
            # That's element-wise product summed over j, which is not a matrix multiply.
            
            # Correct interpretation: 
            # left[i] = sum_j A[i,j] * U[i*4+j, r]
            # right[k] = sum_j V[j*8+k, r] * B[j,k]
            # C[i,k] += left[i] * right[k] * W[k*4+i, r]
            
            left = np.zeros(4)
            for i in range(4):
                for j in range(4):
                    left[i] += A_test[i, j] * U_np[i*4+j, r]
            
            right = np.zeros(8)
            for k in range(8):
                for j in range(4):
                    right[k] += V_np[j*8+k, r] * B_test[j, k]
            
            for i in range(4):
                for k in range(8):
                    C_computed[i, k] += left[i] * right[k] * W_np[k*4+i, r]
        
        max_err = np.max(np.abs(C_computed - C_expected))
        print(f"Verification: max error = {max_err}")
        if max_err > 1e-10:
            print("WARNING: Decomposition is invalid!")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
