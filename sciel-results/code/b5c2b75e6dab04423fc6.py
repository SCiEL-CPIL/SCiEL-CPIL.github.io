"""Optimized matrix multiplication decomposition for 4x4 * 4x9."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_4x4_terms():
    """
    Return the 49-term Strassen decomposition for 4x4 * 4x4.
    Each term: (u_idx, v_idx, w_idx, sign)
    where U[i*4+j, col] = sign_u, V[j*4+k, col] = sign_v, W[k*4+i, col] = sign_w
    such that C[i,k] = sum_col U[i*4+j,col] * V[j*4+k,col] * W[k*4+i,col] * A[i,j] * B[j,k]
    
    Actually, let me define it properly.
    The tensor is T[i,j,k] = 1 for all i,j,k in [0,4).
    We want: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k.
    
    Strassen for 4x4*4x4:
    Split A into 2x2 blocks: A11, A12, A21, A22
    Split B into 2x2 blocks: B11, B12, B21, B22
    Split C into 2x2 blocks: C11, C12, C21, C22
    
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B21-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each Mt is a 2x2 matrix product. For 2x2*2x2, Strassen gives 7 terms.
    So total: 7 * 7 = 49 terms.
    
    Let me enumerate them.
    """
    terms = []  # Each: (i, j, k, sign_u, sign_v, sign_w)
    
    # 2x2 Strassen terms for a single 2x2*2x2 product
    # Input: a = [[a00,a01],[a10,a11]], b = [[b00,b01],[b10,b11]]
    # Output: c = a*b
    # 
    # s1 = a00+a11, s2 = a10+a11, s3 = a00-a10, s4 = a01-a11
    # s5 = b00+b11, s6 = b00-b11, s7 = b10+b11, s8 = b01-b11
    # 
    # m1 = s1*s5, m2 = s2*b00, m3 = a00*s6, m4 = a11*s7
    # m5 = s3*b11, m6 = s4*s8, m7 = s4*b10
    # 
    # c00 = m1+m4-m5+m7
    # c01 = m3+m5
    # c10 = m2+m4
    # c11 = m1-m2+m3+m6
    
    # For a 2x2*2x2 product with indices (i_off, j_off, k_off) for the block,
    # and scalar offsets (ii, jj, kk) within the block:
    # A[i_off+ii, j_off+jj] * B[j_off+jj, k_off+kk]
    
    def add_2x2_strassen(a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        """Add 7 Strassen terms for a 2x2*2x2 sub-product."""
        # a is 2x2 starting at (a_row_base, a_col_base)
        # b is 2x2 starting at (b_row_base, b_col_base)  
        # c is 2x2 starting at (c_row_base, c_col_base)
        
        # m1 = (a00+a11)*(b00+b11)
        # c00 += m1, c11 += m1
        for (ci, cj) in [(c_row_base, c_col_base), (c_row_base+1, c_col_base+1)]:
            for (ai, aj) in [(a_row_base, a_col_base), (a_row_base+1, a_col_base+1)]:
                for (bi, bj) in [(b_row_base, b_col_base), (b_row_base+1, b_col_base+1)]:
                    pass  # This isn't quite right, need to be more careful
        
        # Let me just enumerate all 49 terms explicitly.
        # This is getting complex. Let me use a different approach.
        pass
    
    # Actually, let me just directly construct the 49 terms by enumerating.
    # I'll use the known rank-49 decomposition.
    
    # Simpler: just return None and use schoolbook for 4x4*4x4 (rank 64)
    # Then total would be 64 + 80 = 144, same as schoolbook. Not helpful.
    
    # Let me try a different split: 4x4*4x9 = 4x4*4x4 + 4x4*4x5
    # Use schoolbook for both: 64 + 80 = 144. Same as direct schoolbook.
    
    # OK, the Strassen approach needs the actual 49-term decomposition.
    # Let me look up / derive it properly.
    
    # The 4x4*4x4 Strassen decomposition (49 terms):
    # I'll construct it by combining the 7 top-level Strassen products,
    # each of which is a 2x2*2x2 product decomposed with 7 inner Strassen terms.
    
    # Top-level:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # Where A11 = A[0:2,0:2], A12 = A[0:2,2:4], A21 = A[2:4,0:2], A22 = A[2:4,2:4]
    # Similarly for B and C.
    
    # Each Mt involves sums/differences of 2x2 blocks, so it's not a simple
    # 2x2*2x2 product. We need to expand.
    
    # This is getting very complex. Let me just use schoolbook and try to
    # find improvements through other means.
    
    return None

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 4, 4, 9
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36

    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 144
    U_fb = np.zeros((nm, rank_schoolbook))
    V_fb = np.zeros((mp, rank_schoolbook))
    W_fb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U = U_fb.copy()
    best_V = V_fb.copy()
    best_W = W_fb.copy()

    def tensor_residual(U, V, W):
        """Compute the residual tensor R[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1"""
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)

        UV = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)
        T = np.einsum('ijkr,kir->ijk', UV, W_3d)
        return T - 1.0

    def verify_decomposition(U, V, W, tol=1e-8):
        """Verify that the decomposition is exact."""
        T = tensor_residual(U, V, W)
        return np.max(np.abs(T)) < tol

    # Strategy: Split 4x4*4x9 into 4x4*4x4 + 4x4*4x5
    # Use schoolbook for both: 64 + 80 = 144 (same as direct)
    # But if we can find a better decomposition for 4x4*4x4, we save.
    
    # Let me try to find a rank < 64 decomposition for 4x4*4x4.
    # Known: Strassen gives 49.
    
    # For now, let me try to construct a 4x4*4x4 decomposition with rank < 64
    # using optimization, then combine with 4x4*4x5 schoolbook.
    
    def try_find_4x4_decomp(target_rank, max_steps=5000):
        """Try to find a rank-target decomposition for 4x4*4x4."""
        n4 = 4
        nm4 = 16
        mp4 = 16
        pn4 = 16
        
        for trial in range(5):
            if time.time() > deadline:
                return None
            
            if trial == 0:
                U4 = np.random.randn(nm4, target_rank) * 0.1
                V4 = np.random.randn(mp4, target_rank) * 0.1
                W4 = np.random.randn(pn4, target_rank) * 0.1
            else:
                scale = 0.1 ** (trial * 0.1)
                U4 = np.random.randn(nm4, target_rank) * scale
                V4 = np.random.randn(mp4, target_rank) * scale
                W4 = np.random.randn(pn4, target_rank) * scale
            
            # Adam optimizer
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            m_U = np.zeros_like(U4)
            v_U = np.zeros_like(U4)
            m_V = np.zeros_like(V4)
            v_V = np.zeros_like(V4)
            m_W = np.zeros_like(W4)
            v_W = np.zeros_like(W4)
            t = 0
            lr = 0.01
            
            for step in range(max_steps):
                if time.time() > deadline:
                    return None
                
                t += 1
                U4_3d = U4.reshape(n4, n4, target_rank)
                V4_3d = V4.reshape(n4, n4, target_rank)
                W4_3d = W4.reshape(n4, n4, target_rank)
                
                UV = np.einsum('ijr,jkr->ijkr', U4_3d, V4_3d)
                T = np.einsum('ijkr,kir->ijk', UV, W4_3d)
                err = T - 1.0
                
                grad_U = 2.0 * np.einsum('ijk,jkr,kir->ijr', err, V4_3d, W4_3d)
                grad_V = 2.0 * np.einsum('ijk,ijr,kir->jkr', err, U4_3d, W4_3d)
                grad_W = 2.0 * np.einsum('ijk,ijr,jkr->kir', err, U4_3d, V4_3d)
                
                m_U = beta1 * m_U + (1 - beta1) * grad_U
                v_U = beta2 * v_U + (1 - beta2) * grad_U**2
                U4 -= lr * (m_U / (1 - beta1**t)) / (np.sqrt(v_U / (1 - beta2**t)) + eps)
                
                m_V = beta1 * m_V + (1 - beta1) * grad_V
                v_V = beta2 * v_V + (1 - beta2) * grad_V**2
                V4 -= lr * (m_V / (1 - beta1**t)) / (np.sqrt(v_V / (1 - beta2**t)) + eps)
                
                m_W = beta1 * m_W + (1 - beta1) * grad_W
                v_W = beta2 * v_W + (1 - beta2) * grad_W**2
                W4 -= lr * (m_W / (1 - beta1**t)) / (np.sqrt(v_W / (1 - beta2**t)) + eps)
                
                if step % 500 == 0:
                    max_err = np.max(np.abs(T))
                    if max_err < 1e-8:
                        return U4, V4, W4
            
            # Final check
            U4_3d = U4.reshape(n4, n4, target_rank)
            V4_3d = V4.reshape(n4, n4, target_rank)
            W4_3d = W4.reshape(n4, n4, target_rank)
            UV = np.einsum('ijr,jkr->ijkr', U4_3d, V4_3d)
            T = np.einsum('ijkr,kir->ijk', UV, W4_3d)
            if np.max(np.abs(T - 1.0)) < 1e-8:
                return U4, V4, W4
        
        return None

    # Try to find a better 4x4*4x4 decomposition
    for target_rank in range(49, 65):
        if time.time() > deadline:
            break
        
        result = try_find_4x4_decomp(target_rank)
        if result is not None:
            U4, V4, W4 = result
            
            # Now combine with 4x4*4x5 schoolbook (rank 80)
            total_rank = target_rank + 80
            
            if total_rank < best_rank:
                U_new = np.zeros((nm, total_rank))
                V_new = np.zeros((mp, total_rank))
                W_new = np.zeros((pn, total_rank))
                
                # First 4x4*4x4 part (columns 0-3 of B)
                U_new[:, :target_rank] = U4
                V_new[:16, :target_rank] = V4  # j*p+k for j=0..3, k=0..3
                W_new[:16, :target_rank] = W4  # k*n+i for k=0..3, i=0..3
                
                # 4x4*4x5 schoolbook part (columns 4-8 of B)
                col = target_rank
                for i in range(4):
                    for j in range(4):
                        for k in range(4, 9):
                            U_new[i * 4 + j, col] = 1
                            V_new[j * 9 + k, col] = 1
                            W_new[k * 4 + i, col] = 1
                            col += 1
                
                if verify_decomposition(U_new, V_new, W_new):
                    best_rank = total_rank
                    best_U = U_new.copy()
                    best_V = V_new.copy()
                    best_W = W_new.copy()
                break
    
    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
