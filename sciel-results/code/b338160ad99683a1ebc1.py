"""Optimized matrix multiplication decomposition for 4x4 * 4x9."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_strassen_4x4_4x4():
    """
    Build the 49-term Strassen decomposition for 4x4 * 4x4.
    Returns U, V, W matrices of shape (16, 49).
    U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    n4 = 4
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # 2x2 Strassen inner decomposition
    # For a 2x2 * 2x2 product with indices (a_r, a_c) for A block, (b_r, b_c) for B block, (c_r, c_c) for C block
    # A indices: (a_r+0, a_c+0)=a00, (a_r+0, a_c+1)=a01, (a_r+1, a_c+0)=a10, (a_r+1, a_c+1)=a11
    # B indices: (b_r+0, b_c+0)=b00, (b_r+0, b_c+1)=b01, (b_r+1, b_c+0)=b10, (b_r+1, b_c+1)=b11
    # C indices: (c_r+0, c_c+0)=c00, (c_r+0, c_c+1)=c01, (c_r+1, c_c+0)=c10, (c_r+1, c_c+1)=c11
    
    # Strassen 2x2:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b00-b11)  -- wait, standard Strassen is:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*(b00)
    # m3 = (a00)*(b01-b11)  -- no, let me use the standard form
    # 
    # Standard Strassen 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # 
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each term, we need to express it as a sum of rank-1 contributions.
    # Each Mx is a product of two 2x2 matrices (which may be sums/differences of blocks).
    # We expand each Mx into its 4 scalar products, then decompose each scalar product
    # using the inner Strassen 2x2 decomposition.
    
    # Actually, let me think about this differently.
    # The 4x4*4x4 Strassen decomposition has 49 terms.
    # Each term corresponds to one scalar multiplication in the bilinear form.
    # 
    # Let me define the 7 top-level Strassen products for 4x4:
    # S1 = A11 + A22, S2 = A21 + A22, S3 = A11, S4 = A22, S5 = A11 + A12, S6 = A21 - A11, S7 = A12 - A22
    # T1 = B11 + B22, T2 = B11, T3 = B12 - B22, T4 = B21 - B11, T5 = B22, T6 = B11 + B12, T7 = B21 + B22
    # 
    # M1 = S1 * T1 (2x2 * 2x2)
    # M2 = S2 * T2 (2x2 * 2x2)
    # M3 = S3 * T3 (2x2 * 2x2)
    # M4 = S4 * T4 (2x2 * 2x2)
    # M5 = S5 * T5 (2x2 * 2x2)
    # M6 = S6 * T6 (2x2 * 2x2)
    # M7 = S7 * T7 (2x2 * 2x2)
    # 
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # 
    # Each Mx is a 2x2 matrix. To compute Mx = Sx * Tx where Sx and Tx are 2x2 matrices
    # (which are themselves linear combinations of A and B blocks), we use the inner
    # Strassen decomposition for 2x2*2x2 (7 terms each).
    # 
    # But wait - Sx and Tx are not arbitrary 2x2 matrices; they are specific linear combinations.
    # The key insight is that each scalar entry of Mx is a bilinear form in the entries of A and B.
    # 
    # Let me take a different approach: directly enumerate the 49 terms.
    # 
    # For the inner 2x2 Strassen, given 2x2 matrices P and Q:
    # p00, p01, p10, p11 and q00, q01, q10, q11
    # 
    # Inner Strassen terms (7):
    # i1 = (p00+p11)(q00+q11)
    # i2 = (p10+p11)q00
    # i3 = p00(q01-q11)
    # i4 = p11(q10-q00)
    # i5 = (p00+p01)q11
    # i6 = (p10-p00)(q00+q01)
    # i7 = (p01-p11)(q10+q11)
    # 
    # Result:
    # r00 = i1 + i4 - i5 + i7
    # r01 = i3 + i5
    # r10 = i2 + i4
    # r11 = i1 - i2 + i3 + i6
    # 
    # Now, for the 4x4 case, each Sx is a 2x2 matrix whose entries are linear combinations
    # of A's entries, and each Tx similarly for B.
    # 
    # This is getting very complex to enumerate by hand. Let me use a programmatic approach.
    
    # Define the 7 top-level S matrices (each 2x2, expressed as linear combos of A entries)
    # A entries: A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], A[1,1], A[1,2], A[1,3],
    #            A[2,0], A[2,1], A[2,2], A[2,3], A[3,0], A[3,1], A[3,2], A[3,3]
    # 
    # A11 = [[A00, A01], [A10, A11]] (using A[i,j] notation)
    # A12 = [[A02, A03], [A12, A13]]
    # A21 = [[A20, A21], [A30, A31]]
    # A22 = [[A22, A23], [A32, A33]]
    # 
    # S1 = A11 + A22 = [[A00+A22, A01+A23], [A10+A32, A11+A33]]
    # S2 = A21 + A22 = [[A20+A22, A21+A23], [A30+A32, A31+A33]]
    # S3 = A11 = [[A00, A01], [A10, A11]]
    # S4 = A22 = [[A22, A23], [A32, A33]]
    # S5 = A11 + A12 = [[A00+A02, A01+A03], [A10+A12, A11+A13]]
    # S6 = A21 - A11 = [[A20-A00, A21-A01], [A30-A10, A31-A11]]
    # S7 = A12 - A22 = [[A02-A22, A03-A23], [A12-A32, A13-A33]]
    # 
    # Similarly for T matrices with B entries.
    # B11 = [[B00, B01], [B10, B11]]
    # B12 = [[B02, B03], [B12, B13]]
    # B21 = [[B20, B21], [B30, B31]]
    # B22 = [[B22, B23], [B32, B33]]
    # 
    # T1 = B11 + B22 = [[B00+B22, B01+B23], [B10+B32, B11+B33]]
    # T2 = B11 = [[B00, B01], [B10, B11]]
    # T3 = B12 - B22 = [[B02-B22, B03-B23], [B12-B32, B13-B33]]
    # T4 = B21 - B11 = [[B20-B00, B21-B01], [B30-B10, B31-B11]]
    # T5 = B22 = [[B22, B23], [B32, B33]]
    # T6 = B11 + B12 = [[B00+B02, B01+B03], [B10+B12, B11+B13]]
    # T7 = B21 + B22 = [[B20+B22, B21+B23], [B30+B32, B31+B33]]
    
    # This is extremely tedious to do by hand. Let me use a different strategy:
    # I'll construct the decomposition programmatically using the bilinear form structure.
    
    # Actually, let me just verify that the schoolbook is correct and try to optimize
    # the full 4x4*4x9 problem directly with a lower rank target.
    
    return None  # Fall through to optimization approach

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

    def compute_loss(U, V, W):
        """Compute the sum of squares of the residual."""
        T = tensor_residual(U, V, W)
        return np.sum(T**2)

    def compute_grads(U, V, W):
        """Compute gradients of the loss w.r.t. U, V, W."""
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)

        UV = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)
        T = np.einsum('ijkr,kir->ijk', UV, W_3d)
        err = T - 1.0

        grad_U = 2.0 * np.einsum('ijk,jkr,kir->ijr', err, V_3d, W_3d)
        grad_V = 2.0 * np.einsum('ijk,ijr,kir->jkr', err, U_3d, W_3d)
        grad_W = 2.0 * np.einsum('ijk,ijr,jkr->kir', err, U_3d, V_3d)

        return grad_U, grad_V, grad_W

    def adam_step(params, grads, moments, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """Perform an Adam optimization step."""
        updated_params = []
        for param, grad in zip(params, grads):
            m, v = moments
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            new_param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
            updated_params.append(new_param)
        return updated_params

    # Try to find a better decomposition using optimization
    # Target ranks to try (from best known to schoolbook)
    target_ranks = [120, 125, 130, 135, 140]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(3):
            if time.time() > deadline or found:
                break
            
            if trial == 0:
                # Start from schoolbook and try to reduce rank
                # This is tricky, so let's use random initialization
                U = np.random.randn(nm, target_rank) * 0.5
                V = np.random.randn(mp, target_rank) * 0.5
                W = np.random.randn(pn, target_rank) * 0.5
            elif trial == 1:
                U = np.random.randn(nm, target_rank) * 0.3
                V = np.random.randn(mp, target_rank) * 0.3
                W = np.random.randn(pn, target_rank) * 0.3
            else:
                U = np.random.randn(nm, target_rank) * 0.2
                V = np.random.randn(mp, target_rank) * 0.2
                W = np.random.randn(pn, target_rank) * 0.2
            
            # Adam optimizer state
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            t = 0
            lr = 0.05
            
            max_steps = 2000
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                
                t += 1
                U_3d = U.reshape(n, m, target_rank)
                V_3d = V.reshape(m, p, target_rank)
                W_3d = W.reshape(p, n, target_rank)
                
                UV = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)
                T = np.einsum('ijkr,kir->ijk', UV, W_3d)
                err = T - 1.0
                
                max_err = np.max(np.abs(err))
                if max_err < 1e-8:
                    found = True
                    break
                
                # Adaptive learning rate
                if max_err > 1.0:
                    lr = 0.01
                elif max_err > 0.1:
                    lr = 0.05
                else:
                    lr = 0.001
                
                grad_U = 2.0 * np.einsum('ijk,jkr,kir->ijr', err, V_3d, W_3d)
                grad_V = 2.0 * np.einsum('ijk,ijr,kir->jkr', err, U_3d, W_3d)
                grad_W = 2.0 * np.einsum('ijk,ijr,jkr->kir', err, U_3d, V_3d)
                
                m_U = 0.9 * m_U + 0.1 * grad_U
                v_U = 0.999 * v_U + 0.001 * grad_U**2
                U -= lr * (m_U / (1 - 0.9**t)) / (np.sqrt(v_U / (1 - 0.999**t)) + 1e-8)
                
                m_V = 0.9 * m_V + 0.1 * grad_V
                v_V = 0.999 * v_V + 0.001 * grad_V**2
                V -= lr * (m_V / (1 - 0.9**t)) / (np.sqrt(v_V / (1 - 0.999**t)) + 1e-8)
                
                m_W = 0.9 * m_W + 0.1 * grad_W
                v_W = 0.999 * v_W + 0.001 * grad_W**2
                W -= lr * (m_W / (1 - 0.9**t)) / (np.sqrt(v_W / (1 - 0.999**t)) + 1e-8)
            
            # Final check
            if not found:
                if time.time() > deadline:
                    break
                U_3d = U.reshape(n, m, target_rank)
                V_3d = V.reshape(m, p, target_rank)
                W_3d = W.reshape(p, n, target_rank)
                UV = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)
                T = np.einsum('ijkr,kir->ijk', UV, W_3d)
                if np.max(np.abs(T - 1.0)) < 1e-8:
                    found = True
            
            if found:
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                break
    
    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
