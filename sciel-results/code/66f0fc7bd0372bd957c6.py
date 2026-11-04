import numpy as np

DIMENSIONS = [4, 4, 6]

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    
    Returns (U, V, W) where:
      U: (4, 7) - for A[i,j], row index = i*2+j
      V: (4, 7) - for B[j,k], row index = j*2+k
      W: (4, 7) - for C[i,k], row index = k*2+i  (trace convention: W[k*n+i] with n=2)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C[0,0] and C[1,1]
    # Wait, W index is k*2+i.
    # C[0,0] -> k=0, i=0 -> W[0]
    # C[1,1] -> k=1, i=1 -> W[3]
    # C[0,1] -> k=1, i=0 -> W[2]
    # C[1,0] -> k=0, i=1 -> W[1]
    
    # Let's re-verify W assignments for Strassen.
    # C = M1 - M2 + M3 + M4 ...? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1
    W[0, 0] = 1 # C00
    W[3, 0] = 1 # C11
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1 # C10
    W[3, 1] = -1 # C11
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1 # C01
    W[3, 2] = 1 # C11
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1 # C00
    W[1, 3] = 1 # C10
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C00
    W[2, 4] = 1 # C01
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C00
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    sU, sV, sW = _strassen_2x2()
    
    # Outer structure:
    # A is 2x2 blocks of 2x2.
    # B is 2x3 blocks of 2x2.
    # C is 2x3 blocks of 2x2.
    
    # Outer terms:
    # 7 Strassen terms for B columns 0,1 (2x2 block)
    # 4 Schoolbook terms for B column 2 (2x1 block)
    
    total_rank = 11 * 7
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    # Helper to add a term
    def add_term(ot_idx, t_idx, u_coeff, v_coeff, w_coeff, u_block, v_block, w_block):
        col = ot_idx * 7 + t_idx
        # u_block is (ai, bj), v_block is (bj, ck), w_block is (ai, ck)
        ai, bj = u_block
        bj_v, ck = v_block
        ai_w, ck_w = w_block
        
        # Verify consistency
        assert bj == bj_v
        assert ck == ck_w
        
        for ai2 in range(2):
            for bj2 in range(2):
                c_u = sU[ai2*2 + bj2, t_idx]
                if c_u != 0:
                    u_row = (2*ai + ai2) * 4 + (2*bj + bj2)
                    U[u_row, col] += u_coeff * c_u
                    
        for bj2 in range(2):
            for ck2 in range(2):
                c_v = sV[bj2*2 + ck2, t_idx]
                if c_v != 0:
                    v_row = (2*bj_v + bj2) * 6 + (2*ck + ck2)
                    V[v_row, col] += v_coeff * c_v
                    
        for ai2 in range(2):
            for ck2 in range(2):
                c_w = sW[ck2*2 + ai2, t_idx]
                if c_w != 0:
                    w_row = (2*ck_w + ck2) * 4 + (2*ai_w + ai2)
                    W[w_row, col] += w_coeff * c_w

    # Strassen outer terms
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # Outer U: (0,0):1, (1,1):1
    # Outer V: (0,0):1, (0,1):1
    # Outer W: (0,0):1, (1,1):1
    # Note: Strassen M1 contributes to C00 and C11.
    # In our block structure, C00 is block (0,0) and C11 is block (1,1) of the 2x2 result?
    # No, the Strassen decomposition is for the 2x2 block multiplication.
    # The "C00" in Strassen refers to the top-left 2x2 block of the result.
    # So for outer term M1, it affects C_block(0,0) and C_block(1,1)?
    # Wait. The Strassen algorithm computes a 2x2 matrix C from 2x2 matrices A and B.
    # C[0,0], C[0,1], C[1,0], C[1,1].
    # So M1 contributes to C[0,0] and C[1,1].
    # In the outer context, C[0,0] is the block at (0,0) of the 2x2 result (which is C columns 0,1).
    # So M1 contributes to C_block(0,0) and C_block(1,1) of the 2x2 sub-result.
    # But the outer result is 2x3.
    # The Strassen part handles the first 2 columns of B (2x2 matrix).
    # So the result of the Strassen part is a 2x2 matrix of blocks.
    # These blocks correspond to C(0,0), C(0,1), C(1,0), C(1,1) of the 2x2 sub-result.
    # Which maps to C_block(0,0), C_block(0,1), C_block(1,0), C_block(1,1) of the full C.
    
    # Let's define the outer Strassen coefficients for the 2x2 part.
    # sU, sV, sW are for the inner 2x2 multiplication.
    # We need the outer coefficients for the 2x2 block multiplication.
    # Let O_U, O_V, O_W be the outer Strassen matrices.
    
    O_U = np.zeros((4, 7), dtype=int) # 2x2 blocks of A
    O_V = np.zeros((4, 7), dtype=int) # 2x2 blocks of B (cols 0,1)
    O_W = np.zeros((4, 7), dtype=int) # 2x2 blocks of C (cols 0,1)
    
    # Copy Strassen structure to outer
    O_U[0, 0] = 1; O_U[3, 0] = 1
    O_V[0, 0] = 1; O_V[3, 0] = 1
    O_W[0, 0] = 1; O_W[3, 0] = 1
    
    O_U[2, 1] = 1; O_U[3, 1] = 1
    O_V[0, 1] = 1
    O_W[1, 1] = 1; O_W[3, 1] = -1
    
    O_U[0, 2] = 1
    O_V[1, 2] = 1; O_V[3, 2] = -1
    O_W[2, 2] = 1; O_W[3, 2] = 1
    
    O_U[3, 3] = 1
    O_V[2, 3] = 1; O_V[0, 3] = -1
    O_W[0, 3] = 1; O_W[1, 3] = 1
    
    O_U[0, 4] = 1; O_U[1, 4] = 1
    O_V[3, 4] = 1
    O_W[0, 4] = -1; O_W[2, 4] = 1
    
    O_U[2, 5] = 1; O_U[0, 5] = -1
    O_V[0, 5] = 1; O_V[1, 5] = 1
    O_W[3, 5] = 1
    
    O_U[1, 6] = 1; O_U[3, 6] = -1
    O_V[2, 6] = 1; O_V[3, 6] = 1
    O_W[0, 6] = 1
    
    # Iterate over outer Strassen terms
    for ot in range(7):
        for t in range(7):
            col = ot * 7 + t
            # Get outer coefficients
            u_outer = {}
            for ai in range(2):
                for bj in range(2):
                    c = O_U[ai*2+bj, ot]
                    if c != 0:
                        u_outer[(ai,bj)] = c
                        
            v_outer = {}
            for bj in range(2):
                for ck in range(2): # ck 0,1
                    c = O_V[bj*2+ck, ot]
                    if c != 0:
                        v_outer[(bj,ck)] = c
                        
            w_outer = {}
            for ai in range(2):
                for ck in range(2): # ck 0,1
                    c = O_W[ck*2+ai, ot]
                    if c != 0:
                        w_outer[(ai,ck)] = c
                        
            # Combine with inner
            for (ai, bj), c_u in u_outer.items():
                for (bj_v, ck), c_v in v_outer.items():
                    if bj != bj_v: continue
                    for (ai_w, ck_w), c_w in w_outer.items():
                        if ai != ai_w or ck != ck_w: continue
                        # Match found
                        add_term(ot, t, c_u, c_v, c_w, (ai,bj), (bj_v,ck), (ai_w,ck_w))

    # Schoolbook outer terms for column 2
    # 4 terms: (ai, bj) * (bj, 2) -> (ai, 2)
    sb_start = 7
    for ai in range(2):
        for bj in range(2):
            ot = sb_start + ai*2 + bj
            # U: (ai, bj) coeff 1
            # V: (bj, 2) coeff 1
            # W: (ai, 2) coeff 1
            for t in range(7):
                col = ot * 7 + t
                add_term(ot, t, 1, 1, 1, (ai,bj), (bj,2), (ai,2))

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition():
    n, m, p = DIMENSIONS
    result = construct()
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    rank = U.shape[1]
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        # U[:, r] is vector of length n*m
        # V[:, r] is vector of length m*p
        # W[:, r] is vector of length p*n
        # We need to compute sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j' V[j'*p+k, r] B[j',k])
        # This is not correct. The tensor decomposition is T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # So C[i,k] = sum_{j,r} A[i,j] B[j,k] T[i,j,k]
        # = sum_{j,r} A[i,j] B[j,k] U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # This can be computed as:
        # For each r:
        #   A_r = sum_j U[i*m+j, r] A[i,j] ? No.
        #   The term is A[i,j] * U[i*m+j, r].
        #   Let A_U = A * U^T? No.
        #   Let's just loop.
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += A[i,j] * B[j,k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_decomp[i,k] += s
                
    if not np.allclose(C_expected, C_decomp):
        print("Verification Failed")
        return False
    print(f"Verification Passed, Rank: {rank}")
    return True

if __name__ == "__main__":
    _verify_decomposition()
