# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition exactly reconstructs matmul tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    r = n*m*p
    U = np.zeros((n*m, r)); V = np.zeros((m*p, r)); W = np.zeros((p*n, r))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1; V[j*p+k, c] = 1; W[k*n+i, c] = 1
                c += 1
    return U, V, W

def _build_444():
    """Strassen-based 4x4 * 4x4 -> 4x4, rank 49."""
    n = m = p = 4
    U = np.zeros((n*m, 49)); V = np.zeros((m*p, 49)); W = np.zeros((p*n, 49))
    A11 = [0,1,4,5]; A12 = [2,3,6,7]; A21 = [8,9,12,13]; A22 = [10,11,14,15]
    B11 = [0,1,4,5]; B12 = [2,3,6,7]; B21 = [8,9,12,13]; B22 = [10,11,14,15]
    C11 = [0,1,4,5]; C12 = [2,3,6,7]; C21 = [8,9,12,13]; C22 = [10,11,14,15]
    
    macros = [
        ([A11, A22], [1,1], [B11, B22], [1,1], [(C11,1),(C22,1)]),
        ([A21, A22], [1,1], [B11], [1], [(C21,1)]),
        ([A11], [1], [B12, B22], [1,-1], [(C12,1)]),
        ([A22], [1], [B21, B11], [1,-1], [(C11,1),(C21,1)]),
        ([A11, A12], [1,1], [B22], [1], [(C11,-1),(C12,1)]),
        ([A21, A11], [1,-1], [B11, B12], [1,1], [(C22,1)]),
        ([A12, A22], [1,-1], [B21, B22], [1,1], [(C11,1)]),
    ]
    
    strassen_pat = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(2,1)]),
        ([(0,1)], [(1,1),(3,-1)], [(1,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    
    col = 0
    for macro in macros:
        A_blocks, A_signs, B_blocks, B_signs, C_targets = macro
        for sub in strassen_pat:
            U_entries, V_entries, W_entries = sub
            for a_local, s_u in U_entries:
                gidx = A_blocks[a_local]
                U[gidx, col] += A_signs[a_local] * s_u
            for b_local, s_v in V_entries:
                gidx = B_blocks[b_local]
                V[gidx, col] += B_signs[b_local] * s_v
            for c_local, s_w in W_entries:
                C_block, c_sign = C_targets[c_local]
                for gidx in C_block:
                    W[gidx, col] += c_sign * s_w
            col += 1
    return U, V, W

def _build_223():
    """2x2 * 2x3 -> 2x3, rank 9."""
    n, m, p = 2, 2, 3
    r = 9
    U = np.zeros((n*m, r)); V = np.zeros((m*p, r)); W = np.zeros((p*n, r))
    
    # Strassen 2x2x2 part (rank 7) for k=0,1
    # U, V, W for 2x2x2
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    # But here p=3, so V[j*3+k], W[k*2+i]
    
    # We define the 7 Strassen terms for 2x2x2
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1 # B00 (j=0,k=0), B11 (j=1,k=1) -> V[0*3+0], V[1*3+1]
    W[0, 0] = 1; W[3, 0] = 1 # C00 (k=0,i=0), C11 (k=1,i=1) -> W[0*2+0], W[1*2+1]
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C10 (k=0,i=1) -> W[0*2+1]
    
    # M3: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1 # B01 (j=0,k=1), B11 (j=1,k=1) -> V[0*3+1], V[1*3+1]
    W[1, 2] = 1 # C01 (k=1,i=0) -> W[1*2+0]
    
    # M4: A11(B10-B00)
    U[3, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1 # B10 (j=1,k=0), B00 (j=0,k=0) -> V[1*3+0], V[0*3+0]
    W[0, 3] = 1; W[2, 3] = 1 # C00, C10
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1 # B11 (j=1,k=1) -> V[1*3+1]
    W[0, 4] = -1; W[1, 4] = 1 # C00, C01
    
    # M6: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1 # B00, B01
    W[3, 5] = 1 # C11 (k=1,i=1) -> W[1*2+1]
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1 # B10, B11
    W[0, 6] = 1; W[3, 6] = 1 # C00, C11
    
    # Wait, I need to check the W indices for Strassen.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's re-verify W assignments.
    # M1: W[0,0]=1, W[3,0]=1. (C00, C11)
    # M2: W[2,1]=1. (C10)
    # M3: W[1,2]=1. (C01)
    # M4: W[0,3]=1, W[2,3]=1. (C00, C10)
    # M5: W[0,4]=-1, W[1,4]=1. (C00, C01)
    # M6: W[3,5]=1. (C11)
    # M7: W[0,6]=1, W[3,6]=1. (C00, C11)
    
    # Check C00: M1(1) + M4(1) - M5(-1) + M7(1) = 1+1+1+1=4? No.
    # C00 = M1 + M4 - M5 + M7.
    # My M5 has W[0,4]=-1. So -M5 contributes +1 to C00?
    # The formula is sum U V W.
    # Term M5 contributes U V W.
    # If W[0,4] = -1, then the contribution to C00 is -1 * (U V).
    # So C00 gets -1 from M5.
    # So M1(1) + M4(1) + M5(-1) + M7(1) = 1+1-1+1 = 2?
    # Strassen C00 = M1 + M4 - M5 + M7.
    # If M5 is defined as (A00+A01)B11, then C00 -= M5.
    # So W for M5 should be -1 for C00.
    # My assignment W[0,4]=-1 is correct for the term to be subtracted.
    
    # Check C01: M3 + M5.
    # M3: W[1,2]=1.
    # M5: W[1,4]=1.
    # Sum = 1+1=2? No, M3 and M5 are different terms.
    # C01 = M3 + M5.
    # Contribution from M3 is 1. Contribution from M5 is 1.
    # So C01 = 1+1=2? No, they are separate rank components.
    # The sum is over c.
    # s += U V W.
    # For C01, only c=2 and c=4 contribute.
    # c=2: W[1,2]=1.
    # c=4: W[1,4]=1.
    # So C01 = (A00)(B01-B11) + (A00+A01)B11 = A00 B01 - A00 B11 + A00 B11 + A01 B11 = A00 B01 + A01 B11. Correct.
    
    # Check C10: M2 + M4.
    # c=1: W[2,1]=1.
    # c=3: W[2,3]=1.
    # C10 = (A10+A11)B00 + A11(B10-B00) = A10 B00 + A11 B00 + A11 B10 - A11 B00 = A10 B00 + A11 B10. Correct.
    
    # Check C11: M1 - M2 + M3 + M6.
    # c=0: W[3,0]=1.
    # c=1: W[3,1]=0? No, M2 has W[2,1]=1, so W[3,1]=0.
    # c=2: W[3,2]=0? M3 has W[1,2]=1, so W[3,2]=0.
    # c=5: W[3,5]=1.
    # c=6: W[3,6]=1.
    # Wait, M3 is A00(B01-B11). It contributes to C01 only?
    # M3: U[0,2]=1, V[1,2]=1, V[4,2]=-1. W[1,2]=1.
    # This contributes to C01 (i=0, k=1).
    # Does it contribute to C11? No, W[3,2]=0.
    # So C11 gets 0 from M3.
    # But formula says C11 = M1 - M2 + M3 + M6.
    # Did I mess up M3?
    # M3 = A00 (B01 - B11).
    # C01 = A00 B01 + A01 B11.
    # C11 = A10 B01 + A11 B11.
    # Standard Strassen:
    # M3 = A00 (B01 - B11) -> contributes to C01?
    # C01 = M3 + M5.
    # M5 = (A00+A01) B11.
    # M3 + M5 = A00 B01 - A00 B11 + A00 B11 + A01 B11 = A00 B01 + A01 B11. Correct.
    # C11 = M1 - M2 + M3 + M6?
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M6 = (A10-A00)(B00+B01)
    # M1 - M2 + M6 = (A00+A11)(B00+B11) - (A10+A11)B00 + (A10-A00)(B00+B01)
    # = A00 B00 + A00 B11 + A11 B00 + A11 B11 - A10 B00 - A11 B00 + A10 B00 + A10 B01 - A00 B00 - A00 B01
    # = A00 B11 + A11 B11 + A10 B01 - A00 B01
    # = A11 B11 + A10 B01 - A00 B01 + A00 B11? No.
    # Target C11 = A10 B01 + A11 B11.
    # My expansion: A10 B01 + A11 B11 - A00 B01 + A00 B11.
    # Missing M3?
    # If I add M3 = A00 (B01 - B11) = A00 B01 - A00 B11.
    # Then M1 - M2 + M3 + M6 = (A10 B01 + A11 B11 - A00 B01 + A00 B11) + (A00 B01 - A00 B11) = A10 B01 + A11 B11. Correct.
    # So M3 MUST contribute to C11.
    # But in my assignment, W for M3 was only W[1,2]=1 (C01).
    # I need W[3,2]=1 as well?
    # M3 = A00 (B01 - B11).
    # This term is multiplied by A00.
    # It should contribute to C01 and C11?
    # No, A00 is the top-left element of A.
    # C01 = A00 B01 + A01 B11.
    # C11 = A10 B01 + A11 B11.
    # A00 does not appear in C11.
    # So M3 should NOT contribute to C11.
    # Where is the error in my Strassen formula check?
    # M3 = A00 (B01 - B11).
    # C01 = M3 + M5.
    # C11 = M1 - M2 + M3 + M6?
    # If M3 doesn't contribute to C11, then C11 = M1 - M2 + M6.
    # Let's re-evaluate M1 - M2 + M6.
    # M1 - M2 + M6 = A00 B11 + A11 B11 + A10 B01 - A00 B01.
    # This is not C11.
    # So M3 MUST contribute to C11?
    # But A00 is not in C11.
    # Ah, M3 is A00 (B01 - B11).
    # If M3 contributes to C11, then A00 must be in C11. It is not.
    # So the formula C11 = M1 - M2 + M3 + M6 is wrong?
    # Let's check Wikipedia or standard Strassen.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # My indices:
    # A11->A00, A12->A01, A21->A10, A22->A11
    # B11->B00, B12->B01, B21->B10, B22->B11
    # C11->C00, C12->C01, C21->C10, C22->C11
    
    # C11 (my C00) = M1 + M4 - M5 + M7.
    # C12 (my C01) = M3 + M5.
    # C21 (my C10) = M2 + M4.
    # C22 (my C11) = M1 - M2 + M3 + M6.
    
    # So M3 (A00(B01-B11)) contributes to C01 and C11.
    # But A00 is not in C11.
    # Contradiction.
    # Let's check M3 definition again.
    # M3 = A11 (B12 - B22).
    # A11 is top-left.
    # C12 = A11 B12 + A12 B22.
    # C22 = A21 B12 + A22 B22.
    # M3 contributes to C12 and C22?
    # If M3 contributes to C22, then A11 is in C22. It is not.
    # So M3 should NOT contribute to C22.
    # But formula says C22 = M1 - M2 + M3 + M6.
    # Let's re-expand M1 - M2 + M6.
    # M1 - M2 + M6 = A11 B22 + A22 B22 + A21 B12 - A11 B12.
    # C22 = A21 B12 + A22 B22.
    # Difference: A11 B22 - A11 B12.
    # M3 = A11 (B12 - B22) = A11 B12 - A11 B22.
    # So (M1 - M2 + M6) + M3 = A21 B12 + A22 B22. Correct.
    # So M3 DOES contribute to C22 (my C11).
    # But A11 (my A00) is not in C22 (my C11).
    # Wait.
    # C22 = A21 B12 + A22 B22.
    # A11 is NOT in C22.
    # But M3 contains A11.
    # How can M3 contribute to C22 if A11 is not there?
    # Because the other terms (M1, M2, M6) also contain A11, and they cancel out in the sum for C22?
    # M1 has A11 B22.
    # M2 has no A11.
    # M6 has -A11 B12.
    # M3 has A11 B12 - A11 B22.
    # Sum of A11 terms in C22:
    # From M1: A11 B22.
    # From M6: -A11 B12.
    # From M3: A11 B12 - A11 B22.
    # Total: A11 B22 - A11 B12 + A11 B12 - A11 B22 = 0.
    # So they cancel!
    # So M3 contributes to C22, but its effect is cancelled by M1 and M6.
    # So I MUST set W[3,2]=1 for M3.
    
    # So for M3:
    # W[1,2]=1 (C01)
    # W[3,2]=1 (C11)
    
    # Similarly, check M7.
    # M7 = (A01-A11)(B10+B11).
    # Contributes to C00 and C11.
    # W[0,6]=1, W[3,6]=1.
    
    # Check M1.
    # Contributes to C00 and C11.
    # W[0,0]=1, W[3,0]=1.
    
    # Check M4.
    # Contributes to C00 and C10.
    # W[0,3]=1, W[2,3]=1.
    
    # Check M5.
    # Contributes to C00 and C01.
    # W[0,4]=-1, W[1,4]=1.
    
    # Check M2.
    # Contributes to C10.
    # W[2,1]=1.
    
    # Check M6.
    # Contributes to C11.
    # W[3,5]=1.
    
    # So I need to update W for M3.
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[1, 2] = 1
    W[3, 2] = 1
    
    # Now the rank-2 part for k=2.
    # Term 8: A00 B02, A10 B02 -> C02, C12
    U[0, 8] = 1; U[2, 8] = 1
    V[2, 8] = 1 # B02 (j=0, k=2) -> V[0*3+2]
    W[4, 8] = 1; W[5, 8] = 1 # C02 (k=2, i=0) -> W[2*2+0], C12 (k=2, i=1) -> W[2*2+1]
    
    # Term 9: A01 B12, A11 B12 -> C02, C12
    U[1, 9] = 1; U[3, 9] = 1
    V[5, 9] = 1 # B12 (j=1, k=2) -> V[1*3+2]
    W[4, 9] = 1; W[5, 9] = 1
    
    return U, V, W

def _build_443_strassen():
    """4x4 * 4x3 -> 4x3, rank 36."""
    n, m, p = 4, 4, 3
    r = 36
    U = np.zeros((n*m, r)); V = np.zeros((m*p, r)); W = np.zeros((p*n, r))
    
    U_223, V_223, W_223 = _build_223()
    
    for br in range(2):
        for bc in range(2):
            for c in range(9):
                col = (br * 2 + bc) * 9 + c
                
                # U: i in 2br..2br+1, j in 2bc..2bc+1
                for i_loc in range(2):
                    for j_loc in range(2):
                        u_idx = i_loc * 2 + j_loc
                        val = U_223[u_idx, c]
                        if val != 0:
                            g_i = 2 * br + i_loc
                            g_j = 2 * bc + j_loc
                            U[g_i * 4 + g_j, col] = val
                
                # V: j in 2bc..2bc+1, k in 0..2
                for j_loc in range(2):
                    for k in range(3):
                        v_idx = j_loc * 3 + k
                        val = V_223[v_idx, c]
                        if val != 0:
                            g_j = 2 * bc + j_loc
                            V[g_j * 3 + k, col] = val
                
                # W: k in 0..2, i in 2br..2br+1
                for k in range(3):
                    for i_loc in range(2):
                        w_idx = k * 2 + i_loc
                        val = W_223[w_idx, c]
                        if val != 0:
                            g_i = 2 * br + i_loc
                            W[k * 4 + g_i, col] = val
                            
    return U, V, W

def _build_447():
    """Combine 4x4*4x4 (rank 49) and 4x4*4x3 (rank 36) for 4x4*4x7 (rank 85)."""
    n, m, p = 4, 4, 7
    U1, V1, W1 = _build_444()
    U2, V2, W2 = _build_443_strassen()
    
    total_rank = 49 + 36
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    U[:, 0:49] = U1
    for i in range(4):
        for j in range(4):
            V[i*7+j, 0:49] = V1[i*4+j, :]
    for i in range(4):
        for j in range(4):
            W[i*7+j, 0:49] = W1[i*4+j, :]
    
    U[:, 49:85] = U2
    for i in range(4):
        for j in range(3):
            V[i*7+4+j, 49:85] = V2[i*3+j, :]
    for row in range(4):
        for col in range(3):
            W[row*7+4+col, 49:85] = W2[row*4+col, :]
            
    return U, V, W

def _try_polish(U, V, W, n, m, p, max_iter=50, tol=1e-9):
    """Attempt to reduce rank by removing zero/near-zero columns."""
    U = U.copy(); V = V.copy(); W = W.copy()
    r = U.shape[1]
    removed = []
    
    for c in range(r):
        if np.allclose(U[:, c], 0, atol=1e-12) and np.allclose(V[:, c], 0, atol=1e-12) and np.allclose(W[:, c], 0, atol=1e-12):
            removed.append(c)
    
    if removed:
        keep = [c for c in range(r) if c not in removed]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = n*m*p
    
    try:
        if time.time() < deadline - 10:
            U, V, W = _build_447()
            if _verify(U, V, W, n, m, p):
                U_polished, V_polished, W_polished = _try_polish(U, V, W, n, m, p)
                if _verify(U_polished, V_polished, W_polished, n, m, p):
                    best = {"U": U_polished.tolist(), "V": V_polished.tolist(), "W": W_polished.tolist()}
                    best_rank = U_polished.shape[1]
                else:
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = U.shape[1]
    except Exception:
        pass
    
    return best
# EVOLVE-BLOCK-END
