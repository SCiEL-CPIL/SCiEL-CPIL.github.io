import numpy as np
import time
import sys
from fractions import Fraction
import scipy.optimize as so

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m  # 8
    dim_B = m * p  # 24
    dim_C = p * n  # 12
    
    # Build target tensor indices
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                w = k * n + i
                indices.append((a, b, w))
    
    idx_a = np.array([x[0] for x in indices], dtype=np.int32)
    idx_b = np.array([x[1] for x in indices], dtype=np.int32)
    idx_w = np.array([x[2] for x in indices], dtype=np.int32)
    num_terms = len(indices)  # 48
    
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    def verify_solution(U, V, W):
        """Verify that U, V, W exactly decompose the tensor"""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        
        hat_T = np.einsum('ar,br,wr->abw', U, V, W)
        
        # Check target entries
        vals = hat_T[idx_a, idx_b, idx_w]
        if not np.allclose(vals, 1.0, atol=1e-9):
            return False
        
        # Check non-target entries
        full_hat = hat_T.copy()
        full_hat[idx_a, idx_b, idx_w] = 0
        max_err = np.max(np.abs(full_hat))
        return max_err < 1e-9
    
    # WGA: Kronecker product of Strassen 2x2x2 (rank 7) and linear convolution 2x2x3 (rank 4)
    # Total rank = 7 * 4 = 28
    
    # Strassen 2x2x2 factor matrices
    # U_strassen: (4, 7), V_strassen: (4, 7), W_strassen: (4, 7)
    U_s = np.array([
        [1, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, -1]
    ], dtype=np.float64)
    
    V_s = np.array([
        [1, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, -1, 0, 1, 0, 0, 1]
    ], dtype=np.float64)
    
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=np.float64)
    
    # Linear convolution 2x2x3 factor matrices
    # For computing [a0, a1] * [b0, b1] -> [c0, c1, c2] where c0=a0*b0, c1=a0*b1+a1*b0, c2=a1*b1
    # Winograd: use rank 4 decomposition
    # U_conv: (2, 4), V_conv: (2, 4), W_conv: (3, 4)
    U_c = np.array([
        [1, 1, 0, 0],
        [1, -1, 1, 1]
    ], dtype=np.float64)
    
    V_c = np.array([
        [1, 0, 1, 1],
        [1, 1, -1, 1]
    ], dtype=np.float64)
    
    W_c = np.array([
        [1, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 1, 1]
    ], dtype=np.float64)
    
    # Verify the small blocks first
    # Check Strassen: C = A*B for 2x2 matrices
    # A = [[a0,a1],[a2,a3]], B = [[b0,b1],[b2,b3]], C = [[c0,c1],[c2,c3]]
    # Tensor: T[a_idx, b_idx, c_idx] = 1
    # a_idx: 0->a0, 1->a1, 2->a2, 3->a3
    # b_idx: 0->b0, 1->b1, 2->b2, 3->b3
    # c_idx: 0->c0, 1->c1, 2->c2, 3->c3
    
    # Check convolution: [a0,a1] * [b0,b1] -> [c0,c1,c2]
    # a_idx: 0->a0, 1->a1
    # b_idx: 0->b0, 1->b1
    # c_idx: 0->c0, 1->c1, 2->c2
    conv_U = np.einsum('ar,br,cr->abc', U_c, V_c, W_c)
    # Expected: c0=a0*b0, c1=a0*b1+a1*b0, c2=a1*b1
    expected_conv = np.zeros((2,2,3))
    expected_conv[0,0,0] = 1
    expected_conv[0,1,1] = 1
    expected_conv[1,0,1] = 1
    expected_conv[1,1,2] = 1
    if not np.allclose(conv_U, expected_conv, atol=1e-9):
        # Try alternative convolution factors
        # Standard Winograd for 2-point convolution
        U_c = np.array([
            [1, 1, 0, 0],
            [1, -1, 1, 1]
        ], dtype=np.float64)
        V_c = np.array([
            [1, 0, 1, 1],
            [1, 1, -1, 1]
        ], dtype=np.float64)
        W_c = np.array([
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 1, 1]
        ], dtype=np.float64)
        conv_U = np.einsum('ar,br,cr->abc', U_c, V_c, W_c)
        if not np.allclose(conv_U, expected_conv, atol=1e-9):
            # Use schoolbook for convolution (rank 3) as fallback
            U_c = np.array([
                [1, 0, 0],
                [0, 1, 1]
            ], dtype=np.float64)
            V_c = np.array([
                [1, 1, 0],
                [0, 1, 1]
            ], dtype=np.float64)
            W_c = np.array([
                [1, 0, 0],
                [1, 1, 0],
                [0, 0, 1]
            ], dtype=np.float64)
            # This gives rank 3, total rank 21
            conv_U = np.einsum('ar,br,cr->abc', U_c, V_c, W_c)
    
    # Now construct the full 2x4x6 tensor using Kronecker products
    # The 2x4x6 multiplication can be seen as:
    # A is 2x4: rows i=0,1; cols j=0,1,2,3
    # B is 4x6: rows j=0,1,2,3; cols k=0,1,2,3,4,5
    # C is 6x2: rows k=0,1,2,3,4,5; cols i=0,1
    # C[k,i] = sum_j A[i,j] * B[j,k]
    
    # Decomposition: j = 2*j_low + j_high where j_high in {0,1}, j_low in {0,1}
    # k = 3*k_low + k_high where... this doesn't decompose cleanly.
    
    # Alternative: think of it as two 2x2x3 convolutions
    # For each i (0 or 1):
    #   C[k, i] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # Split j into two groups: j in {0,1} and j in {2,3}
    # C[k, i] = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j-2] * B[j,k]
    
    # This is getting complex. Let me use a direct approach.
    # Build the 2x4x6 tensor and try to find a rank-28 decomposition.
    
    # Actually, let me construct the WGA decomposition directly.
    # The key insight: 2x4x6 = (2x2x2) ⊗ (1x2x3) in some sense,
    # but more precisely, we can view the 4 columns of A as two 2-column blocks,
    # and the 6 rows of C as three 2-row blocks.
    
    # Let me try a different decomposition:
    # A[i, j] where j = 2*j1 + j0, j1 in {0,1}, j0 in {0,1}
    # B[j, k] where j = 2*j1 + j0, k = 2*k1 + k0, k1 in {0,1,2}, k0 in {0,1}
    # C[k, i] where k = 2*k1 + k0
    
    # C[2*k1+k0, i] = sum_{j1=0}^{1} sum_{j0=0}^{1} A[i, 2*j1+j0] * B[2*j1+j0, 2*k1+k0]
    
    # This is a 2x2x2 (for j1) times something...
    
    # Let me just use the Kronecker product approach properly.
    # Tensor T[a,b,w] where a = i*m+j, b = j*p+k, w = k*n+i
    
    # For the Kronecker product to work, we need:
    # a = a1 * m1 + a0 where a1 in {0,1}, a0 in {0,1,2,3} -> a = 4*a1 + a0
    # But a = i*m + j = 2*i + j, so a1 = i, a0 = j
    # b = j*p + k = 4*j + k, so b1 = j (0..3), b0 = k (0..5)
    # w = k*n + i = 2*k + i, so w1 = k (0..5), w0 = i (0..1)
    
    # The issue is that j appears in both a and b, and k appears in both b and w.
    # This is not a simple Kronecker product structure.
    
    # Let me try a different approach: use the fact that 2x4x6 can be decomposed
    # as a sum of 2x2x3 tensors.
    
    # Actually, the cleanest WGA approach:
    # View the 2x4x6 multiplication as computing two 2x3 vectors from a 2x4 and 4x6.
    # For fixed i: C[:, i] = B^T * A[i, :]
    # This is a 6x1 = 6x4 * 4x1 multiplication, which is just a matrix-vector product.
    
    # Let me just directly construct a rank-28 solution using the Kronecker product
    # of Strassen (7) and a 2x2x3 convolution (4).
    
    # The 2x4x6 tensor can be seen as:
    # For each of the 2 rows of A, we compute a 6-vector output.
    # The 4 intermediate dimensions j=0,1,2,3 can be split into two 2-dim blocks.
    
    # Block 1: j=0,1 -> contributes to k=0,1,2 (first 3 rows of C)
    # Block 2: j=2,3 -> contributes to k=2,3,4,5 (last 4 rows of C)
    # But this overlaps at k=2.
    
    # This is getting too complex for a clean Kronecker product. Let me use a
    # different strategy: start from the known rank-42 solution and try to
    # optimize down to rank 28.
    
    # For now, let me provide the rank-42 fallback and try optimization.
    def get_fallback():
        U = [[0] * 42 for _ in range(n * m)]
        V = [[0] * 42 for _ in range(m * p)]
        W = [[0] * 42 for _ in range(p * n)]
        
        def add_strassen_block(a_idx, b_idx, w_idx, start_col):
            U[a_idx[0]][start_col + 0] += 1
            U[a_idx[3]][start_col + 0] += 1
            U[a_idx[2]][start_col + 1] += 1
            U[a_idx[3]][start_col + 1] += 1
            U[a_idx[0]][start_col + 2] += 1
            U[a_idx[3]][start_col + 3] += 1
            U[a_idx[0]][start_col + 4] += 1
            U[a_idx[1]][start_col + 4] += 1
            U[a_idx[2]][start_col + 5] += 1
            U[a_idx[0]][start_col + 5] -= 1
            U[a_idx[1]][start_col + 6] += 1
            U[a_idx[3]][start_col + 6] -= 1
            
            V[b_idx[0]][start_col + 0] += 1
            V[b_idx[3]][start_col + 0] += 1
            V[b_idx[0]][start_col + 1] += 1
            V[b_idx[1]][start_col + 2] += 1
            V[b_idx[3]][start_col + 2] -= 1
            V[b_idx[2]][start_col + 3] += 1
            V[b_idx[0]][start_col + 3] -= 1
            V[b_idx[3]][start_col + 4] += 1
            V[b_idx[0]][start_col + 5] += 1
            V[b_idx[1]][start_col + 5] += 1
            V[b_idx[2]][start_col + 6] += 1
            V[b_idx[3]][start_col + 6] += 1
            
            W[w_idx[0]][start_col + 0] += 1
            W[w_idx[0]][start_col + 3] += 1
            W[w_idx[0]][start_col + 4] -= 1
            W[w_idx[0]][start_col + 6] += 1
            
            W[w_idx[1]][start_col + 2] += 1
            W[w_idx[1]][start_col + 4] += 1
            
            W[w_idx[2]][start_col + 1] += 1
            W[w_idx[2]][start_col + 3] += 1
            
            W[w_idx[3]][start_col + 0] += 1
            W[w_idx[3]][start_col + 1] -= 1
            W[w_idx[3]][start_col + 2] += 1
            W[w_idx[3]][start_col + 5] += 1

        a_L = [0, 1, 4, 5]
        a_R = [2, 3, 6, 7]
        
        b_1_T = [0, 1, 6, 7]
        b_1_B = [12, 13, 18, 19]
        w_1 = [0, 2, 1, 3]
        add_strassen_block(a_L, b_1_T, w_1, 0)
        add_strassen_block(a_R, b_1_B, w_1, 7)
        
        b_2_T = [2, 3, 8, 9]
        b_2_B = [14, 15, 20, 21]
        w_2 = [4, 6, 5, 7]
        add_strassen_block(a_L, b_2_T, w_2, 14)
        add_strassen_block(a_R, b_2_B, w_2, 21)
        
        b_3_T = [4, 5, 10, 11]
        b_3_B = [16, 17, 22, 23]
        w_3 = [8, 10, 9, 11]
        add_strassen_block(a_L, b_3_T, w_3, 28)
        add_strassen_block(a_R, b_3_B, w_3, 35)
        
        return U, V, W
    
    U_fb, V_fb, W_fb = get_fallback()
    
    if not verify_solution(U_fb, V_fb, W_fb):
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    best_rank = 42
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try WGA: construct rank-28 using Kronecker products
    # 2x4x6 = (2x2) ⊗ (2x3) in the middle dimension
    # A[i, j] = sum over j1,j0: A_block[i, j1, j0]
    # B[j, k] = sum over j1,k1,k0: B_block[j1, j0, k1, k0]
    # C[k, i] = sum over k1,k0: C_block[k1, k0, i]
    
    # j = 2*j1 + j0, k = 3*k1 + k0 doesn't work cleanly.
    # j = 2*j1 + j0, k = 2*k1 + k0 with k1 in 0..2, k0 in 0..1
    # Then B[j, k] = B[2*j1+j0, 2*k1+k0]
    # C[2*k1+k0, i] = sum_{j1=0}^{1} sum_{j0=0}^{1} A[i, 2*j1+j0] * B[2*j1+j0, 2*k1+k0]
    
    # For fixed i, k1, k0:
    # C[2*k1+k0, i] = sum_{j1=0}^{1} sum_{j0=0}^{1} A[i, 2*j1+j0] * B[2*j1+j0, 2*k1+k0]
    
    # This is a 2x2 matrix multiply (over j1) for each (i, k1, k0) pair,
    # where the "matrices" are:
    # M1[j1, j0] = A[i, 2*j1+j0] (2x2)
    # M2[j1, k0'] where k0' depends on k1... this doesn't separate cleanly.
    
    # Let me try yet another approach: use the known rank bound.
    # For 2x4x6, the rank is at most 2*4*6/2 = 24 (by general bounds)
    # Actually, the trivial bound is 2*4*6 = 48, and Strassen gives 42.
    # The WGA should give 7*4 = 28 if the decomposition works.
    
    # Let me construct the 28-rank solution explicitly.
    # The key is to use the fact that the 2x4x6 tensor is a sum of
    # Kronecker products of smaller tensors.
    
    # Define the 2x2x3 linear convolution tensor T_conv:
    # T_conv[a, b, c] = 1 if c = a + b, else 0 (for a,b in {0,1}, c in {0,1,2})
    # This has rank 3 (schoolbook) or 4 (Winograd).
    
    # Define the 2x2x2 Strassen tensor T_str:
    # T_str[a, b, c] = 1 if a = b = c, else 0 (for a,b,c in {0,1,2,3})
    # Wait, that's not right. The 2x2x2 matrix multiplication tensor is:
    # T_str[a, b, c] where a = i*2+j, b = j*2+k, c = k*2+i
    # T_str[a, b, c] = 1 if the indices are consistent, else 0
    # This has rank 7 (Strassen).
    
    # The 2x4x6 tensor can be written as:
    # T_246[a, b, w] where a = i*4+j, b = j*6+k, w = k*2+i
    # = sum over j1 of T_str[i, 2*j1+j0, 2*j1+k0] * T_conv[j0, k0, ?]
    
    # This is getting too complex. Let me just use optimization to find
    # a rank-28 solution, starting from the rank-42 solution.
    
    def residual_full(theta, r):
        """Compute full residual sum of squares"""
        U = theta[:dim_A * r].reshape(dim_A, r)
        V = theta[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
        W = theta[dim_A * r + dim_B * r :].reshape(dim_C, r)
        
        hat_T = np.einsum('ar,br,wr->abw', U, V, W)
        
        # Residual on target indices
        res_target = hat_T[idx_a, idx_b, idx_w] - 1.0
        
        # Residual on non-target indices
        full_hat = hat_T.copy()
        full_hat[idx_a, idx_b, idx_w] = 0
        res_non_target = full_hat.flatten()
        
        return np.sum(res_target**2) + np.sum(res_non_target**2)

    def try_rank_optimization(r, num_restarts=5, init_from=None):
        """Try to find a rank-r decomposition using optimization on full residual"""
        for attempt in range(num_restarts):
            if time.time() > deadline:
                return None
            
            if init_from is not None:
                U0, V0, W0 = init_from
                # Truncate to rank r
                if r < U0.shape[1]:
                    U0 = U0[:, :r]
                    V0 = V0[:, :r]
                    W0 = W0[:, :r]
                elif r > U0.shape[1]:
                    U0 = np.hstack([U0, np.random.randn(U0.shape[0], r - U0.shape[1]) * 0.1])
                    V0 = np.hstack([V0, np.random.randn(V0.shape[0], r - V0.shape[1]) * 0.1])
                    W0 = np.hstack([W0, np.random.randn(W0.shape[0], r - W0.shape[1]) * 0.1])
            else:
                np.random.seed(r * 1000 + attempt)
                U0 = np.random.randn(dim_A, r) * 0.5
                V0 = np.random.randn(dim_B, r) * 0.5
                W0 = np.random.randn(dim_C, r) * 0.5
            
            theta0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                res = so.minimize(
                    lambda t: residual_full(t, r),
                    theta0,
                    method='L-BFGS-B',
                    options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                
                if res.fun < 1e-8:
                    U_sol = res.x[:dim_A * r].reshape(dim_A, r)
                    V_sol = res.x[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
                    W_sol = res.x[dim_A * r + dim_B * r :].reshape(dim_C, r)
                    
                    # Round to half-integers
                    U_int = np.round(U_sol * 2) / 2
                    V_int = np.round(V_sol * 2) / 2
                    W_int = np.round(W_sol * 2) / 2
                    
                    if verify_solution(U_int, V_int, W_int):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                    else:
                        # Try rounding to integers
                        U_int2 = np.round(U_sol)
                        V_int2 = np.round(V_sol)
                        W_int2 = np.round(W_sol)
                        if verify_solution(U_int2, V_int2, W_int2):
                            return {"U": U_int2.tolist(), "V": V_int2.tolist(), "W": W_int2.tolist()}
            except Exception:
                pass
        
        return None
    
    # Try ranks from 41 down to 24, starting from the rank-42 solution
    U_arr = np.array(U_fb, dtype=np.float64)
    V_arr = np.array(V_fb, dtype=np.float64)
    W_arr = np.array(W_fb, dtype=np.float64)
    
    for r in range(41, 23, -1):
        if time.time() > deadline:
            break
        
        # First try with initialization from the higher-rank solution
        res = try_rank_optimization(r, num_restarts=3, init_from=(U_arr, V_arr, W_arr))
        if res:
            best_rank = r
            best_result = res
            break
        
        # If that didn't work, try random initialization
        res = try_rank_optimization(r, num_restarts=5)
        if res:
            best_rank = r
            best_result = res
            break
    
    return best_result
