"""Rewritten program for matrix multiplication 4x4x9 rank minimization."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Schoolbook rank
    rank_sb = n * m * p  # 4*4*9 = 144
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    def validate_fast(U, V, W, tol=1e-9):
        """Faster validation using precomputed dot products."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        r = U_arr.shape[1]
        if V_arr.shape[1] != r or W_arr.shape[1] != r:
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j,:], V_arr[j*p+k,:] * W_arr[k*n+i,:])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def build_2x2x2_strassen():
        """
        Build a 7-rank decomposition for 2x2 * 2x2 -> 2x2.
        Returns U, V, W each of shape (4, 7).
        Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        # Standard Strassen decomposition for 2x2x2
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(g-h)
        # M4 = d(f-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # C[0,0] = M1 - M5 + M4 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 + M3 - M2 - M6
        
        U2 = np.array([
            [1, 0, 1, 0, 1, 0, 0],  # (0,0)
            [0, 0, 0, 0, 1, 1, 0],  # (0,1)
            [0, 1, 0, 0, 0, -1, 0], # (1,0)
            [1, 1, 0, 1, 0, 0, 0],  # (1,1)
        ])
        V2 = np.array([
            [1, 1, 0, 0, 0, 1, 0],  # (0,0)
            [0, 0, 0, 1, 0, 1, 0],  # (0,1)
            [0, 0, 1, 0, 0, 0, 1],  # (1,0)
            [1, 0, -1, -1, 1, 0, 1], # (1,1)
        ])
        W2 = np.array([
            [1, 0, 1, 1, -1, 0, 1],  # (0,0)
            [0, 0, 0, 0, 1, 0, 0],   # (0,1)
            [0, 1, 0, 1, 0, 0, 0],   # (1,0)
            [1, -1, 1, 0, 0, 1, 1],  # (1,1)
        ])
        
        # Verify
        valid = True
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for r in range(7):
                        s += U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to schoolbook for 2x2x2
            U2 = np.zeros((4, 8))
            V2 = np.zeros((4, 8))
            W2 = np.zeros((4, 8))
            col = 0
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        U2[i*2+j, col] = 1
                        V2[j*2+k, col] = 1
                        W2[k*2+i, col] = 1
                        col += 1
            return U2, V2, W2, 8
        
        return U2, V2, W2, 7
    
    def build_4x4x4_strassen():
        """
        Build a 49-rank decomposition for 4x4 * 4x4 -> 4x4.
        Uses recursive Strassen.
        Returns U, V, W each of shape (16, 49).
        Indexing: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
        """
        U2, V2, W2, base_rank = build_2x2x2_strassen()
        
        # Block-level Strassen coefficients
        # A blocks: A00=[0:2,0:2], A01=[0:2,2:4], A10=[2:4,0:2], A11=[2:4,2:4]
        # B blocks: B00=[0:2,0:2], B01=[0:2,2:4], B10=[2:4,0:2], B11=[2:4,2:4]
        # C blocks: C00=[0:2,0:2], C01=[0:2,2:4], C10=[2:4,0:2], C11=[2:4,2:4]
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M5 + M4 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 + M3 - M2 - M6
        
        total_rank = 7 * base_rank
        U4 = np.zeros((16, total_rank))
        V4 = np.zeros((16, total_rank))
        W4 = np.zeros((16, total_rank))
        
        # Block-level left factor coefficients for each M_r
        block_left = np.zeros((7, 2, 2))
        block_left[0, 0, 0] = 1; block_left[0, 1, 1] = 1  # M1: A00+A11
        block_left[1, 1, 0] = 1; block_left[1, 1, 1] = 1  # M2: A10+A11
        block_left[2, 0, 0] = 1                              # M3: A00
        block_left[3, 1, 1] = 1                              # M4: A11
        block_left[4, 0, 0] = 1; block_left[4, 0, 1] = 1    # M5: A00+A01
        block_left[5, 1, 0] = 1; block_left[5, 0, 0] = -1   # M6: A10-A00
        block_left[6, 0, 1] = 1; block_left[6, 1, 1] = -1   # M7: A01-A11
        
        # Block-level right factor coefficients for each M_r
        block_right = np.zeros((7, 2, 2))
        block_right[0, 0, 0] = 1; block_right[0, 1, 1] = 1  # M1: B00+B11
        block_right[1, 0, 0] = 1                              # M2: B00
        block_right[2, 1, 0] = 1; block_right[2, 1, 1] = -1  # M3: B10-B11
        block_right[3, 0, 1] = 1; block_right[3, 0, 0] = -1  # M4: B01-B00
        block_right[4, 1, 1] = 1                              # M5: B11
        block_right[5, 0, 0] = 1; block_right[5, 0, 1] = 1   # M6: B00+B01
        block_right[6, 1, 0] = 1; block_right[6, 1, 1] = 1   # M7: B10+B11
        
        # Block-level output coefficients for each M_r
        block_output = np.zeros((7, 2, 2))
        block_output[0, 0, 0] = 1; block_output[0, 1, 1] = 1   # M1: C00:+1, C11:+1
        block_output[1, 1, 0] = 1; block_output[1, 1, 1] = -1  # M2: C10:+1, C11:-1
        block_output[2, 0, 1] = 1; block_output[2, 1, 1] = 1   # M3: C01:+1, C11:+1
        block_output[3, 0, 0] = 1; block_output[3, 1, 0] = 1   # M4: C00:+1, C10:+1
        block_output[4, 0, 0] = -1; block_output[4, 0, 1] = 1  # M5: C00:-1, C01:+1
        block_output[5, 1, 1] = -1                              # M6: C11:-1
        block_output[6, 0, 0] = 1; block_output[6, 1, 1] = 1   # M7: C00:+1, C11:+1
        
        # Now fill in U4, V4, W4
        for r in range(7):
            for s in range(base_rank):
                col = r * base_rank + s
                for i in range(4):
                    for j in range(4):
                        bi, bj = i // 2, j // 2
                        u_coeff = block_left[r, bi, bj]
                        if u_coeff != 0:
                            U4[i*4+j, col] = u_coeff * U2[(i%2)*2+(j%2), s]
                for j in range(4):
                    for k in range(4):
                        bi, bj = j // 2, k // 2
                        v_coeff = block_right[r, bi, bj]
                        if v_coeff != 0:
                            V4[j*4+k, col] = v_coeff * V2[(j%2)*2+(k%2), s]
                for k in range(4):
                    for i in range(4):
                        bi, bj = k // 2, i // 2
                        w_coeff = block_output[r, bi, bj]
                        if w_coeff != 0:
                            W4[k*4+i, col] = w_coeff * W2[(k%2)*2+(i%2), s]
        
        return U4, V4, W4, total_rank
    
    # Build the 4x4x4 Strassen decomposition
    U4, V4, W4, rank_4x4x4 = build_4x4x4_strassen()
    
    # Build the full 4x4x9 decomposition
    # Columns 0-3: first 4x4x4
    # Columns 4-7: second 4x4x4
    # Column 8: 4x4x1 schoolbook
    
    rank_4x4x1 = 16  # schoolbook for 4x4x1
    total_rank = 2 * rank_4x4x4 + rank_4x4x1
    
    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((36, total_rank))
    W_full = np.zeros((36, total_rank))
    
    # First 4x4x4 (columns 0-3)
    U_full[:, 0:rank_4x4x4] = U4
    for j in range(4):
        for k in range(4):
            V_full[j*9+k, 0:rank_4x4x4] = V4[j*4+k, 0:rank_4x4x4]
    for k in range(4):
        for i in range(4):
            W_full[k*4+i, 0:rank_4x4x4] = W4[k*4+i, 0:rank_4x4x4]
    
    # Second 4x4x4 (columns 4-7)
    offset2 = rank_4x4x4
    U_full[:, offset2:offset2+rank_4x4x4] = U4
    for j in range(4):
        for k in range(4):
            V_full[j*9+(k+4), offset2:offset2+rank_4x4x4] = V4[j*4+k, 0:rank_4x4x4]
    for k in range(4):
        for i in range(4):
            W_full[(k+4)*4+i, offset2:offset2+rank_4x4x4] = W4[k*4+i, 0:rank_4x4x4]
    
    # 4x4x1 schoolbook (column 8)
    offset3 = 2 * rank_4x4x4
    col = offset3
    for i in range(4):
        for j in range(4):
            U_full[i*4+j, col] = 1
            V_full[j*9+8, col] = 1
            W_full[8*4+i, col] = 1
            col += 1
    
    # Validate
    if validate_fast(U_full.tolist(), V_full.tolist(), W_full.tolist()):
        best_rank = total_rank
        best = {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Try to improve further if time permits using a more efficient validation
    if time.time() < deadline - 30:
        # Try to find a lower rank using random search with structured initialization
        np.random.seed(42)
        
        # Try ranks from best_rank-1 down to 1
        for target_rank in range(best_rank - 1, max(best_rank - 20, 0), -1):
            if time.time() > deadline - 10:
                break
            
            # Try a few random attempts at this rank
            for attempt in range(5):
                if time.time() > deadline - 10:
                    break
                
                # Initialize with structured patterns
                U_rand = np.zeros((16, target_rank))
                V_rand = np.zeros((36, target_rank))
                W_rand = np.zeros((36, target_rank))
                
                # Use a mix of random and structured initialization
                for r in range(target_rank):
                    if r % 3 == 0:
                        # Random pattern
                        U_rand[:, r] = np.random.randn(16)
                        V_rand[:, r] = np.random.randn(36)
                        W_rand[:, r] = np.random.randn(36)
                    elif r % 3 == 1:
                        # Sparse random
                        U_rand[np.random.choice(16, size=4, replace=False), r] = np.random.choice([-1, 1], size=4)
                        V_rand[np.random.choice(36, size=8, replace=False), r] = np.random.choice([-1, 1], size=8)
                        W_rand[np.random.choice(36, size=8, replace=False), r] = np.random.choice([-1, 1], size=8)
                    else:
                        # Block structured
                        block_size = 4
                        U_rand[:block_size, r] = np.random.choice([-1, 1], size=block_size)
                        V_rand[:block_size*4, r] = np.random.choice([-1, 1], size=block_size*4)
                        W_rand[:block_size*4, r] = np.random.choice([-1, 1], size=block_size*4)
                
                if validate_fast(U_rand, V_rand, W_rand):
                    best_rank = target_rank
                    best = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_rand.tolist()}
                    break
    
    return best
# EVOLVE-BLOCK-END
