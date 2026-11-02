"""Matrix multiplication 2x4 * 4x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 5
    
    def verify_decomp_fast(U, V, W, tol=1e-8):
        """Check if decomposition is valid using einsum."""
        rank = U.shape[1]
        U_idx = U.reshape(n, m, rank)
        V_idx = V.reshape(m, p, rank)
        W_idx = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_idx, V_idx, W_idx, optimize=True)
        if not np.allclose(T, np.ones((n, m, p)), atol=tol):
            return False
        return True

    def als_decompose_small(n_sub, m_sub, p_sub, target_rank, max_iters=1000, tol=1e-10, seed=42, time_limit=None):
        """
        ALS decomposition for multiplying n_sub x m_sub by m_sub x p_sub.
        Returns (U, V, W, valid) where U has shape (n_sub*m_sub, target_rank), etc.
        """
        rng = np.random.RandomState(seed)
        U = rng.randn(n_sub * m_sub, target_rank) * 0.1
        V = rng.randn(m_sub * p_sub, target_rank) * 0.1
        W = rng.randn(p_sub * n_sub, target_rank) * 0.1
        
        start = time.time()
        
        for iteration in range(max_iters):
            if time_limit is not None and (time.time() - start) > time_limit:
                break
                
            # Solve for U
            for i in range(n_sub):
                for j in range(m_sub):
                    M = np.zeros((p_sub, target_rank))
                    for k in range(p_sub):
                        M[k, :] = V[j * p_sub + k, :] * W[k * n_sub + i, :]
                    try:
                        U[i * m_sub + j, :] = np.linalg.lstsq(M, np.ones(p_sub), rcond=None)[0]
                    except:
                        pass
            
            # Solve for V
            for j in range(m_sub):
                for k in range(p_sub):
                    M = np.zeros((n_sub, target_rank))
                    for i in range(n_sub):
                        M[i, :] = U[i * m_sub + j, :] * W[k * n_sub + i, :]
                    try:
                        V[j * p_sub + k, :] = np.linalg.lstsq(M, np.ones(n_sub), rcond=None)[0]
                    except:
                        pass
            
            # Solve for W
            for k in range(p_sub):
                for i in range(n_sub):
                    M = np.zeros((m_sub, target_rank))
                    for j in range(m_sub):
                        M[j, :] = U[i * m_sub + j, :] * V[j * p_sub + k, :]
                    try:
                        W[k * n_sub + i, :] = np.linalg.lstsq(M, np.ones(m_sub), rcond=None)[0]
                    except:
                        pass
            
            if iteration % 10 == 0:
                if verify_decomp_small(U, V, W, n_sub, m_sub, p_sub, tol=1e-6):
                    return U, V, W, True
                    
        if verify_decomp_small(U, V, W, n_sub, m_sub, p_sub, tol=1e-8):
            return U, V, W, True
        return U, V, W, False

    def verify_decomp_small(U, V, W, n_sub, m_sub, p_sub, tol=1e-8):
        rank = U.shape[1]
        U_idx = U.reshape(n_sub, m_sub, rank)
        V_idx = V.reshape(m_sub, p_sub, rank)
        W_idx = W.reshape(p_sub, n_sub, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_idx, V_idx, W_idx, optimize=True)
        return np.allclose(T, np.ones((n_sub, m_sub, p_sub)), atol=tol)

    # --- Strassen Decomposition for 2x2x2 ---
    def strassen_2x2():
        # Standard Strassen matrices
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to construct U, V, W for 2x2 * 2x2 = 2x2
        # rank = 7
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: A11+A22 -> U rows 0 (i=0,j=0), 3 (i=1,j=1). B11+B22 -> V rows 0 (j=0,k=0), 3 (j=1,k=1). C11,C22 -> W rows 0 (k=0,i=0), 3 (k=1,i=1).
        # C11 += M1, C22 += M1
        # W[0,0]=1 (k=0,i=0 -> C11), W[3,0]=1 (k=1,i=1 -> C22)
        # U[0,0]=1 (A11), U[3,0]=1 (A22)
        # V[0,0]=1 (B11), V[3,0]=1 (B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2: (A21+A22)B11. C21 += M2.
        # A21 -> U[2], A22 -> U[3]. B11 -> V[0]. C21 -> W[2] (k=1, i=0).
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3: A11(B12-B22). C12 += M3.
        # A11 -> U[0]. B12 -> V[1], B22 -> V[3] (neg). C12 -> W[1] (k=0, i=1).
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4: A22(B21-B11). C22 += M4.
        # A22 -> U[3]. B21 -> V[2], B11 -> V[0] (neg). C22 -> W[3] (k=1, i=1).
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[3, 3] = 1
        
        # M5: (A11+A12)B22. C12 += M5.
        # A11 -> U[0], A12 -> U[1]. B22 -> V[3]. C12 -> W[1].
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1
        
        # M6: (A21-A11)(B11+B12). C21 += M6.
        # A21 -> U[2], A11 -> U[0] (neg). B11 -> V[0], B12 -> V[1]. C21 -> W[2].
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1
        
        # M7: (A12-A22)(B21+B22). C12 += M7? No, C12 and C22? 
        # Standard Strassen:
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # Wait, let me re-check the mapping.
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # Let's re-assign W signs based on this.
        # M1 (col 0): C11 (+1), C22 (+1). W[0,0]=1, W[3,0]=1. Correct.
        # M2 (col 1): C21 (+1). W[2,1]=1. Correct.
        # M3 (col 2): C12 (+1), C22 (+1). W[1,2]=1, W[3,2]=1. 
        #   My current W[1,2]=1. I need W[3,2]=1.
        # M4 (col 3): C11 (-1), C21 (+1), C22 (+1). 
        #   My current W[3,3]=1 (for C22). I need W[0,3]=-1 (C11), W[2,3]=1 (C21).
        # M5 (col 4): C11 (-1), C12 (+1). 
        #   My current W[1,4]=1 (C12). I need W[0,4]=-1 (C11).
        # M6 (col 5): C21 (+1). 
        #   My current W[2,5]=1. Correct.
        # M7 (col 6): C11 (-1), C22 (+1). 
        #   Need W[0,6]=-1, W[3,6]=1.
        
        # Let's rebuild U, V, W from scratch properly.
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Col 0: M1 = (A11+A22)(B11+B22)
        # A part: A11 (0), A22 (3). U[0,0]=1, U[3,0]=1.
        # B part: B11 (0), B22 (3). V[0,0]=1, V[3,0]=1.
        # C part: C11 (0) +1, C22 (3) +1. W[0,0]=1, W[3,0]=1.
        
        # Col 1: M2 = (A21+A22)B11
        # A part: A21 (2), A22 (3). U[2,1]=1, U[3,1]=1.
        # B part: B11 (0). V[0,1]=1.
        # C part: C21 (2) +1. W[2,1]=1.
        
        # Col 2: M3 = A11(B12-B22)
        # A part: A11 (0). U[0,2]=1.
        # B part: B12 (1) +1, B22 (3) -1. V[1,2]=1, V[3,2]=-1.
        # C part: C12 (1) +1, C22 (3) +1. W[1,2]=1, W[3,2]=1.
        
        # Col 3: M4 = A22(B21-B11)
        # A part: A22 (3). U[3,3]=1.
        # B part: B21 (2) +1, B11 (0) -1. V[2,3]=1, V[0,3]=-1.
        # C part: C11 (0) -1, C21 (2) +1, C22 (3) +1. W[0,3]=-1, W[2,3]=1, W[3,3]=1.
        
        # Col 4: M5 = (A11+A12)B22
        # A part: A11 (0) +1, A12 (1) +1. U[0,4]=1, U[1,4]=1.
        # B part: B22 (3). V[3,4]=1.
        # C part: C11 (0) -1, C12 (1) +1. W[0,4]=-1, W[1,4]=1.
        
        # Col 5: M6 = (A21-A11)(B11+B12)
        # A part: A21 (2) +1, A11 (0) -1. U[2,5]=1, U[0,5]=-1.
        # B part: B11 (0) +1, B12 (1) +1. V[0,5]=1, V[1,5]=1.
        # C part: C21 (2) +1. W[2,5]=1.
        
        # Col 6: M7 = (A12-A22)(B21+B22)
        # A part: A12 (1) +1, A22 (3) -1. U[1,6]=1, U[3,6]=-1.
        # B part: B21 (2) +1, B22 (3) +1. V[2,6]=1, V[3,6]=1.
        # C part: C11 (0) -1, C22 (3) +1. W[0,6]=-1, W[3,6]=1.
        
        # Let's verify
        if verify_decomp_small(U, V, W, 2, 2, 2, tol=1e-8):
            return U, V, W, 7
        else:
            # Fallback to schoolbook 8
            U = np.zeros((4, 8)); V = np.zeros((4, 8)); W = np.zeros((4, 8))
            col = 0
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        U[i*2+j, col] = 1
                        V[j*2+k, col] = 1
                        W[k*2+i, col] = 1
                        col += 1
            return U, V, W, 8

    # --- Try to find rank 9 for 2x2x3 ---
    def find_rank_9_2x2x3():
        # We want to decompose 2x2 * 2x3 = 2x3 with rank 9.
        # Schoolbook is 12.
        # We will use ALS.
        for seed in range(50):
            U, V, W, valid = als_decompose_small(2, 2, 3, 9, max_iters=2000, seed=seed, time_limit=5.0)
            if valid:
                return U, V, W, 9
        return None, None, None, 12

    # --- Combine ---
    # A (2x4) = [A1 A2] where A1, A2 are 2x2
    # B (4x5) = [B1; B2] where B1, B2 are 2x5
    # C = A1 B1 + A2 B2
    # B1 is 2x5. We can split B1 = [B11 B12] where B11 is 2x3, B12 is 2x2.
    # Then A1 B1 = A1 B11 + A1 B12.
    # A1 B11: 2x2 * 2x3. Rank 9?
    # A1 B12: 2x2 * 2x2. Rank 7 (Strassen).
    # A2 B2: 2x2 * 2x5. Split B2 = [B21 B22].
    # A2 B21: 2x2 * 2x3. Rank 9?
    # A2 B22: 2x2 * 2x2. Rank 7.
    
    # Total rank if we use distinct blocks: 9 + 7 + 9 + 7 = 32.
    # But can we share?
    # A1 B12 and A2 B22 are both 2x2x2. They produce C1 (2x5) and C2 (2x5).
    # Specifically:
    # C = A1 B11 + A1 B12 + A2 B21 + A2 B22
    # C is 2x5.
    # A1 B11 contributes to C[:, 0:3]
    # A1 B12 contributes to C[:, 3:5]
    # A2 B21 contributes to C[:, 0:3]
    # A2 B22 contributes to C[:, 3:5]
    
    # So for C[:, 0:3], we have A1 B11 + A2 B21.
    # For C[:, 3:5], we have A1 B12 + A2 B22.
    
    # Let's see if we can do better than 32.
    # Maybe a direct ALS on the whole 2x4x5 with rank 30, 31, 32?
    
    # First, let's just get a valid decomposition.
    # Strategy 1: Strassen for 2x2x2 parts, Schoolbook for 2x2x3 parts.
    # Rank: 8 (A1B11) + 7 (A1B12) + 8 (A2B21) + 7 (A2B22) = 30.
    # This is a solid baseline.
    
    # Let's build the 30-rank decomposition.
    U_total = np.zeros((8, 30))
    V_total = np.zeros((20, 30))
    W_total = np.zeros((10, 30))
    
    # Block 1: A1 B11 (2x2 * 2x3) -> Schoolbook 8
    # A1 rows in U: 0,1,4,5. B11 rows in V: 0..5. C11 cols in W: 0..2 (k=0..2, i=0..1 -> 0..5? No, W is k*n+i. k=0..2, i=0..1. 0,1, 2,3, 4,5).
    # Wait, W index is k*n+i. For 2x2x3, n=2, m=2, p=3.
    # W shape (p*n, rank) = (6, 8).
    # W[k*2+i].
    
    def get_schoolbook(n_sub, m_sub, p_sub, rank=None):
        if rank is None: rank = n_sub*m_sub*p_sub
        U = np.zeros((n_sub*m_sub, rank))
        V = np.zeros((m_sub*p_sub, rank))
        W = np.zeros((p_sub*n_sub, rank))
        col = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U[i*m_sub+j, col] = 1
                    V[j*p_sub+k, col] = 1
                    W[k*n_sub+i, col] = 1
                    col += 1
        return U, V, W, rank

    # Block 1: A1 (rows 0,1,4,5 in U_total), B11 (rows 0..5 in V_total), C part (cols 0..2 in W_total? No, W_total is global).
    # W_total index is k*n+i. Global n=2.
    # B11 is B[0:2, 0:3]. So j=0,1. k=0,1,2.
    # A1 is A[0:2, 0:2]. So i=0,1. j=0,1.
    # Contribution to C[0:2, 0:3].
    # W indices for k=0..2, i=0..1:
    # k=0: 0,1. k=1: 2,3. k=2: 4,5.
    
    U1, V1, W1, r1 = get_schoolbook(2, 2, 3)
    U_total[0, 0:r1] = U1[0] # i=0, j=0 -> 0
    U_total[1, 0:r1] = U1[1] # i=0, j=1 -> 1
    U_total[4, 0:r1] = U1[2] # i=1, j=0 -> 4? No. U1 index is i*m_sub+j.
    # U1 has shape (4, 8).
    # U1[0] is i=0, j=0. U1[1] is i=0, j=1. U1[2] is i=1, j=0. U1[3] is i=1, j=1.
    # U_total rows: i*4+j.
    # i=0, j=0 -> 0. i=0, j=1 -> 1. i=1, j=0 -> 4. i=1, j=1 -> 5.
    U_total[0, 0:r1] = U1[0]
    U_total[1, 0:r1] = U1[1]
    U_total[4, 0:r1] = U1[2]
    U_total[5, 0:r1] = U1[3]
    
    # V1 has shape (6, 8). V1 index j*p_sub+k. j=0,1. k=0,1,2.
    # V_total rows: j*5+k.
    # j=0, k=0 -> 0. j=0, k=1 -> 1. j=0, k=2 -> 2.
    # j=1, k=0 -> 5. j=1, k=1 -> 6. j=1, k=2 -> 7.
    V_total[0:3, 0:r1] = V1[0:3]
    V_total[5:8, 0:r1] = V1[3:6]
    
    # W1 has shape (6, 8). W1 index k*n_sub+i. k=0,1,2. i=0,1.
    # W_total rows: k*2+i.
    # k=0: 0,1. k=1: 2,3. k=2: 4,5.
    W_total[0:6, 0:r1] = W1
    
    # Block 2: A1 B12 (2x2 * 2x2) -> Strassen 7
    # A1 same rows. B12 is B[0:2, 3:5]. j=0,1. k=3,4.
    # V_total rows for B12:
    # j=0, k=3 -> 3. j=0, k=4 -> 4.
    # j=1, k=3 -> 8. j=1, k=4 -> 9.
    # C part is C[0:2, 3:5].
    # W_total rows for k=3,4.
    # k=3: 6,7. k=4: 8,9.
    
    U2, V2, W2, r2 = strassen_2x2()
    if r2 == 7:
        U_total[0, r1:r1+r2] = U2[0]
        U_total[1, r1:r1+r2] = U2[1]
        U_total[4, r1:r1+r2] = U2[2]
        U_total[5, r1:r1+r2] = U2[3]
        
        V_total[3:5, r1:r1+r2] = V2[0:2]
        V_total[8:10, r1:r1+r2] = V2[2:4]
        
        W_total[6:10, r1:r1+r2] = W2
        
    # Block 3: A2 B21 (2x2 * 2x3) -> Schoolbook 8
    # A2 is A[0:2, 2:4]. i=0,1. j=2,3.
    # U_total rows:
    # i=0, j=2 -> 2. i=0, j=3 -> 3.
    # i=1, j=2 -> 6. i=1, j=3 -> 7.
    # B21 is B[2:4, 0:3]. j=2,3. k=0,1,2.
    # V_total rows:
    # j=2, k=0 -> 10. j=2, k=1 -> 11. j=2, k=2 -> 12.
    # j=3, k=0 -> 15. j=3, k=1 -> 16. j=3, k=2 -> 17.
    # C part is C[0:2, 0:3].
    # W_total rows 0:6.
    
    U3, V3, W3, r3 = get_schoolbook(2, 2, 3)
    U_total[2, r1+r2:r1+r2+r3] = U3[0]
    U_total[3, r1+r2:r1+r2+r3] = U3[1]
    U_total[6, r1+r2:r1+r2+r3] = U3[2]
    U_total[7, r1+r2:r1+r2+r3] = U3[3]
    
    V_total[10:13, r1+r2:r1+r2+r3] = V3[0:3]
    V_total[15:18, r1+r2:r1+r2+r3] = V3[3:6]
    
    W_total[0:6, r1+r2:r1+r2+r3] += W3
    
    # Block 4: A2 B22 (2x2 * 2x2) -> Strassen 7
    # A2 same rows. B22 is B[2:4, 3:5]. j=2,3. k=3,4.
    # V_total rows:
    # j=2, k=3 -> 13. j=2, k=4 -> 14.
    # j=3, k=3 -> 18. j=3, k=4 -> 19.
    # C part is C[0:2, 3:5].
    # W_total rows 6:10.
    
    U4, V4, W4, r4 = strassen_2x2()
    if r4 == 7:
        U_total[2, r1+r2+r3:r1+r2+r3+r4] = U4[0]
        U_total[3, r1+r2+r3:r1+r2+r3+r4] = U4[1]
        U_total[6, r1+r2+r3:r1+r2+r3+r4] = U4[2]
        U_total[7, r1+r2+r3:r1+r2+r3+r4] = U4[3]
        
        V_total[13:15, r1+r2+r3:r1+r2+r3+r4] = V4[0:2]
        V_total[18:20, r1+r2+r3:r1+r2+r3+r4] = V4[2:4]
        
        W_total[6:10, r1+r2+r3:r1+r2+r3+r4] += W4
        
    current_rank = r1 + r2 + r3 + r4
    # 8 + 7 + 8 + 7 = 30
    
    # Try to improve by finding rank 9 for the 2x2x3 blocks?
    # No, 9 > 8. Schoolbook is better for 2x2x3 if we can't do better.
    # Can we do rank 7 for 2x2x3? Unlikely.
    # Can we share rank between blocks?
    
    # Let's try ALS on the whole thing with rank 29, 30, 31...
    # Actually, 30 is already a good improvement over 40.
    
    # Let's check if we can do rank 28 or 29.
    # Lower bound is 20.
    
    # Let's try a direct ALS for the whole tensor with rank 30, 31, etc.
    # If we find a valid rank 30 decomposition, it might be denser and more robust?
    # Or we can try to find a rank 29 decomposition.
    
    best_rank = current_rank
    best_U = U_total
    best_V = V_total
    best_W = W_total
    
    # Try to improve with global ALS
    for target_rank in [29, 30, 31]:
        if target_rank >= best_rank:
            continue
        found = False
        for seed in range(5):
            U, V, W, valid = als_decompose_small(2, 4, 5, target_rank, max_iters=1000, seed=seed, time_limit=10.0)
            if valid:
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                found = True
                break
    
    # Final check
    if verify_decomp_fast(best_U, best_V, best_W):
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fallback to schoolbook
        U_school = np.zeros((8, 40))
        V_school = np.zeros((20, 40))
        W_school = np.zeros((10, 40))
        col = 0
        for i in range(2):
            for j in range(4):
                for k in range(5):
                    U_school[i*4+j, col] = 1
                    V_school[j*5+k, col] = 1
                    W_school[k*2+i, col] = 1
                    col += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }

# EVOLVE-BLOCK-END
