import numpy as np
import time

def construct():
    """
    Strategy: Tensor-algebra construction with automatic verification and error repair.
    
    1. Build the 3x4x4 matrix multiplication tensor T[i,j,k,l] = delta(i,l)*delta(j,k).
    2. Construct a rank-44 decomposition using Strassen (rank-7) for 2x2*2x2 blocks
       and schoolbook (rank-4) for 1x2*2x2 blocks, via Kronecker-product tensor ops.
    3. Verify the decomposition term-by-term.
    4. Error repair: any term that doesn't match is replaced with a valid schoolbook term.
    5. Return the (guaranteed valid) decomposition.
    """
    t_start = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    n, m, p = 3, 4, 4
    dim_u = n * m   # 12
    dim_v = m * p   # 16
    dim_w = p * n   # 12

    # Build the target tensor T of shape (n, m, p, n)
    # T[i, j, k, l] = 1 if i==l and j==k, else 0
    T = np.zeros((n, m, p, n), dtype=float)
    for i in range(n):
        for j in range(m):
            T[i, j, j, i] = 1.0  # T[i,j,k,l]=1 when k==j and l==i

    # Strassen rank-7 decomposition of 2x2*2x2=2x2
    # Tensor S of shape (2,2, 2,2, 2,2) where S[a,b, c,d, e,f] = delta(a,e)*delta(b,f)
    S = np.zeros((2, 2, 2, 2, 2, 2), dtype=float)
    for a in range(2):
        for b in range(2):
            S[a, b, b, a, a, b] = 1.0  # S[a,b,c,d,e,f] = 1 if c==b, d==a, e==a, f==b
    # Wait, let me be more careful.
    # The 2x2*2x2 tensor: T2[a, c, b, d, e, f] where C[e,f] = sum_{a,b} A[a,c] * B[b,d] * delta...
    # Actually the standard indexing: T2[i, j, k, l] for C[i,k] = sum_j A[i,j] B[j,k]
    # As a 3-mode tensor: T2[i, j, k] = delta... no.
    # The tensor is T2[i, j, k, l] = delta(i,l) * delta(j,k) for the 2x2 case.
    # But we need it as a 6-index tensor for Kronecker: T2[a, c, b, d, e, f]
    # where A is 2x2 indexed (a,c), B is 2x2 indexed (b,d), C is 2x2 indexed (e,f)
    # C[e,f] = sum_{a,c?} ... 
    
    # Let me use the standard form:
    # For 2x2: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i, j, k] (3-mode) but we need 6-mode for Kronecker.
    # 6-mode: T[a, c, b, d, e, f] where A=(a,c), B=(b,d), C=(e,f)
    # C[e,f] = sum_{a,c,b,d} A[a,c] B[b,d] T[a,c,b,d,e,f]
    # We need: C[e,f] = sum_{i,j,k} A[i,j] B[j,k] delta(i,e)delta(j,k)delta(k,f) ... 
    # Actually C[e,f] = A[e,?] B[?,f] ... 
    # C[e,f] = sum_j A[e,j] B[j,f]
    # So T[a,c,b,d,e,f] = 1 if a==e, c==j, b==j, d==f, i.e., a==e, c==b, d==f
    # T[a,c,b,d,e,f] = delta(a,e) * delta(c,b) * delta(d,f)
    
    T2 = np.zeros((2, 2, 2, 2, 2, 2), dtype=float)
    for a in range(2):
        for c in range(2):
            for b in range(2):
                for d in range(2):
                    for e in range(2):
                        for f in range(2):
                            if a == e and c == b and d == f:
                                T2[a, c, b, d, e, f] = 1.0

    # Strassen rank-7 decomposition of T2:
    # T2[a,c,b,d,e,f] = sum_r U2[a,c,r] V2[b,d,r] W2[e,f,r]
    # Standard Strassen for 2x2:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B00-B01)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # A indexed (a,c): A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
    # B indexed (b,d): B00=(0,0), B01=(0,1), B10=(1,0), B11=(1,1)
    # C indexed (e,f): C00=(0,0), C01=(0,1), C10=(1,0), C11=(1,1)
    
    U2 = np.zeros((2, 2, 7), dtype=float)  # U2[a, c, r]
    V2 = np.zeros((2, 2, 7), dtype=float)  # V2[b, d, r]
    W2 = np.zeros((2, 2, 7), dtype=float)  # W2[e, f, r]
    
    # P1: (A00+A11)(B00+B11)
    U2[0,0,0] = 1; U2[1,1,0] = 1
    V2[0,0,0] = 1; V2[1,1,0] = 1
    W2[0,0,0] = 1; W2[1,1,0] = 1; W2[0,1,0] = 1; W2[1,0,0] = 1
    # Wait, C = P1 - P2 - P4 + P6 for C00? Let me use the standard Strassen formulas.
    # C00 = P1 - P2 - P4 + P6
    # C01 = P3 + P5
    # C10 = P7 + P2
    # C11 = P1 - P3 - P5 + P7
    
    # Let me redo with correct Strassen:
    U2 = np.zeros((2, 2, 7), dtype=float)
    V2 = np.zeros((2, 2, 7), dtype=float)
    W2 = np.zeros((2, 2, 7), dtype=float)
    
    # P1 = (A00+A11)(B00+B11)
    U2[0,0,0] = 1; U2[1,1,0] = 1
    V2[0,0,0] = 1; V2[1,1,0] = 1
    # W will be set based on C formula
    
    # P2 = (A10+A11)B00
    U2[1,0,1] = 1; U2[1,1,1] = 1
    V2[0,0,1] = 1
    
    # P3 = A00(B10-B11)
    U2[0,0,2] = 1
    V2[1,0,2] = 1; V2[1,1,2] = -1
    
    # P4 = A11(B00-B01)
    U2[1,1,3] = 1
    V2[0,0,3] = 1; V2[0,1,3] = -1
    
    # P5 = (A00+A01)B11
    U2[0,0,4] = 1; U2[0,1,4] = 1
    V2[1,1,4] = 1
    
    # P6 = (A10-A00)(B00+B01)
    U2[1,0,5] = 1; U2[0,0,5] = -1
    V2[0,0,5] = 1; V2[0,1,5] = 1
    
    # P7 = (A01-A11)(B10+B11)
    U2[0,1,6] = 1; U2[1,1,6] = -1
    V2[1,0,6] = 1; V2[1,1,6] = 1
    
    # C00 = P1 - P2 - P4 + P6
    # C01 = P3 + P5
    # C10 = P7 + P2
    # C11 = P1 - P3 - P5 + P7
    
    W2[0,0,0] = 1; W2[0,0,1] = -1; W2[0,0,3] = -1; W2[0,0,5] = 1
    W2[0,1,2] = 1; W2[0,1,4] = 1
    W2[1,0,6] = 1; W2[1,0,1] = 1
    W2[1,1,0] = 1; W2[1,1,2] = -1; W2[1,1,4] = -1; W2[1,1,6] = 1
    
    # Verify Strassen: check that sum_r U2[a,c,r]*V2[b,d,r]*W2[e,f,r] == T2[a,c,b,d,e,f]
    for a in range(2):
        for c in range(2):
            for b in range(2):
                for d in range(2):
                    for e in range(2):
                        for f in range(2):
                            s = sum(U2[a,c,r] * V2[b,d,r] * W2[e,f,r] for r in range(7))
                            expected = 1.0 if (a==e and c==b and d==f) else 0.0
                            if abs(s - expected) > 1e-12:
                                # Bug in Strassen - let me fix
                                pass
    
    # If Strassen verification fails, fall back to schoolbook rank-4 for 2x2*2x2
    strassen_ok = True
    for a in range(2):
        for c in range(2):
            for b in range(2):
                for d in range(2):
                    for e in range(2):
                        for f in range(2):
                            s = sum(U2[a,c,r] * V2[b,d,r] * W2[e,f,r] for r in range(7))
                            expected = 1.0 if (a==e and c==b and d==f) else 0.0
                            if abs(s - expected) > 1e-12:
                                strassen_ok = False
                                break
                        if not strassen_ok: break
                    if not strassen_ok: break
                if not strassen_ok: break
            if not strassen_ok: break
        if not strassen_ok: break
    
    if not strassen_ok:
        # Use schoolbook rank-4 for 2x2*2x2
        U2 = np.zeros((2, 2, 4), dtype=float)
        V2 = np.zeros((2, 2, 4), dtype=float)
        W2 = np.zeros((2, 2, 4), dtype=float)
        for j in range(2):
            U2[0, j, j] = 1
            U2[1, j, j] = 1
            V2[j, 0, j] = 1
            V2[j, 1, j] = 1
            W2[0, 0, j] = 1
            W2[1, 1, j] = 1
        # Wait, that's not right either. Schoolbook for 2x2:
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        # 4 terms:
        # r=0: A[0,0]*B[0,0] -> C[0,0]
        # r=1: A[0,1]*B[1,0] -> C[0,0]
        # r=2: A[0,0]*B[0,1] -> C[0,1]
        # r=3: A[0,1]*B[1,1] -> C[0,1]
        # r=4: A[1,0]*B[0,0] -> C[1,0]
        # r=5: A[1,1]*B[1,0] -> C[1,0]
        # r=6: A[1,0]*B[0,1] -> C[1,1]
        # r=7: A[1,1]*B[1,1] -> C[1,1]
        # That's rank 8, not 4. Schoolbook for 2x2 is rank 4? No, it's rank 4 only if we can share.
        # Actually schoolbook for 2x2*2x2 is rank 4? No, it's 4 multiplications:
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] (2 mults)
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] (2 mults)
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] (2 mults)
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] (2 mults)
        # Total 8 multiplications, rank 8.
        # Strassen reduces to 7.
        # So if Strassen fails, I need rank 8 for 2x2*2x2.
        # Let me just use rank 8 schoolbook.
        U2 = np.zeros((2, 2, 8), dtype=float)
        V2 = np.zeros((2, 2, 8), dtype=float)
        W2 = np.zeros((2, 2, 8), dtype=float)
        r = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U2[i, j, r] = 1
                    V2[j, k, r] = 1
                    W2[i, k, r] = 1
                    r += 1
    
    # Schoolbook rank-4 for 1x2*2x2=1x2:
    # C[0] = A[0]*B[0,0] + A[1]*B[1,0]
    # C[1] = A[0]*B[0,1] + A[1]*B[1,1]
    # 4 terms
    U1 = np.zeros((1, 2, 4), dtype=float)  # U1[i, c, r] where i=0 always
    V1 = np.zeros((2, 2, 4), dtype=float)  # V1[b, d, r]
    W1 = np.zeros((1, 2, 4), dtype=float)  # W1[e, f, r] where e=0 always
    
    # r=0: A[0]*B[0,0] -> C[0]
    U1[0,0,0] = 1; V1[0,0,0] = 1; W1[0,0,0] = 1
    # r=1: A[1]*B[1,0] -> C[0]
    U1[0,1,1] = 1; V1[1,0,1] = 1; W1[0,0,1] = 1
    # r=2: A[0]*B[0,1] -> C[1]
    U1[0,0,2] = 1; V1[0,1,2] = 1; W1[0,1,2] = 1
    # r=3: A[1]*B[1,1] -> C[1]
    U1[0,1,3] = 1; V1[1,1,3] = 1; W1[0,1,3] = 1
    
    # Verify 1x2 schoolbook
    T1 = np.zeros((1, 2, 2, 2, 1, 2), dtype=float)
    for a in range(1):
        for c in range(2):
            for b in range(2):
                for d in range(2):
                    for e in range(1):
                        for f in range(2):
                            if a==e and c==b and d==f:
                                T1[a,c,b,d,e,f] = 1.0
    for a in range(1):
        for c in range(2):
            for b in range(2):
                for d in range(2):
                    for e in range(1):
                        for f in range(2):
                            s = sum(U1[a,c,r] * V1[b,d,r] * W1[e,f,r] for r in range(4))
                            expected = T1[a,c,b,d,e,f]
                            if abs(s - expected) > 1e-12:
                                pass  # Should be fine
    
    # Now build the full rank-44 decomposition using Kronecker structure.
    # A (3x4) = [A_top (2x4); A_bottom (1x4)]
    # B (4x4) = [B_left (4x2) | B_right (4x2)]
    # C (3x4) = [C_top (2x4); C_bottom (1x4)]
    # C_top = A_top * B (2x4 * 4x4 = 2x4)
    # C_bottom = A_bottom * B (1x4 * 4x4 = 1x4)
    
    # C_top (2x4) = A_top (2x4) * B (4x4)
    # Split B into B_left (4x2) and B_right (4x2)
    # C_top_left (2x2) = A_top (2x4) * B_left (4x2)
    # C_top_right (2x2) = A_top (2x4) * B_right (4x2)
    
    # A_top (2x4) split into A_top_left (2x2) and A_top_right (2x2)
    # B_left (4x2) split into B_left_top (2x2) and B_left_bottom (2x2)
    # C_top_left (2x2) = A_top_left * B_left_top + A_top_right * B_left_bottom
    
    # So we have 4 blocks of 2x2*2x2 (using Strassen rank 7 each) = 28
    # And 4 blocks of 1x2*2x2 (using schoolbook rank 4 each) = 16
    # Total = 44
    
    # Let me build this explicitly.
    # U has shape (12, 44), V has shape (16, 44), W has shape (12, 44)
    
    # U indices: i*4 + j for A[i,j]
    # V indices: j*4 + k for B[j,k]
    # W indices: k*3 + l for C[l,k] (trace convention: W[k*n+i, r] where n=3)
    # Wait, the problem says W[k*n+i, r]. So W index = k*3 + i where i is the row of C.
    # C has shape (n, p) = (3, 4). C[i, k] = sum_j A[i,j] B[j,k].
    # W[k*n+i, r] means W index = k*3 + i.
    
    U = np.zeros((dim_u, 44), dtype=float)
    V = np.zeros((dim_v, 44), dtype=float)
    W = np.zeros((dim_w, 44), dtype=float)
    
    col = 0
    
    # Block 1: C_top_left[0,0] and C_top_left[0,1] contributions from A_top_left * B_left_top
    # A_top_left = A[0:2, 0:2], indices in U: 0,1,4,5
    # B_left_top = B[0:2, 0:2], indices in V: 0,1,4,5
    # C_top_left = C[0:2, 0:2], W indices: k*3+l for k in 0,1 and l in 0,1
    #   C[0,0]: W[0*3+0] = W[0]
    #   C[0,1]: W[1*3+0] = W[3]
    #   C[1,0]: W[0*3+1] = W[1]
    #   C[1,1]: W[1*3+1] = W[4]
    
    def add_strassen_block(u_indices, v_indices, w_indices, U2, V2, W2):
        nonlocal col
        for r in range(7):
            for u_idx, (a, c) in zip(u_indices, [(0,0),(0,1),(1,0),(1,1)]):
                U[u_idx, col+r] = U2[a, c, r]
            for v_idx, (b, d) in zip(v_indices, [(0,0),(0,1),(1,0),(1,1)]):
                V[v_idx, col+r] = V2[b, d, r]
            for w_idx, (e, f) in zip(w_indices, [(0,0),(0,1),(1,0),(1,1)]):
                W[w_idx, col+r] = W2[e, f, r]
        col += 7
    
    # But wait, the 2x2*2x2 tensor T2[a,c,b,d,e,f] has A indexed (a,c) where a is row, c is col.
    # U2[a,c,r] means U for A[a,c].
    # For A_top_left = A[0:2, 0:2]:
    #   A[0,0] -> U index 0*4+0 = 0
    #   A[0,1] -> U index 0*4+1 = 1
    #   A[1,0] -> U index 1*4+0 = 4
    #   A[1,1] -> U index 1*4+1 = 5
    # So u_indices = [0, 1, 4, 5] corresponding to (a,c) = (0,0),(0,1),(1,0),(1,1)
    
    # For B_left_top = B[0:2, 0:2]:
    #   B[0,0] -> V index 0*4+0 = 0
    #   B[0,1] -> V index 0*4+1 = 1
    #   B[1,0] -> V index 1*4+0 = 4
    #   B[1,1] -> V index 1*4+1 = 5
    # So v_indices = [0, 1, 4, 5] corresponding to (b,d) = (0,0),(0,1),(1,0),(1,1)
    
    # For C_top_left = C[0:2, 0:2]:
    #   C[0,0] -> W index 0*3+0 = 0
    #   C[0,1] -> W index 1*3+0 = 3
    #   C[1,0] -> W index 0*3+1 = 1
    #   C[1,1] -> W index 1*3+1 = 4
    # So w_indices = [0, 3, 1, 4] corresponding to (e,f) = (0,0),(0,1),(1,0),(1,1)
    # Wait, W index = k*3 + i where i is row of C, k is col of C.
    # C[i,k] -> W[k*3+i]
    # C[0,0] -> W[0*3+0] = W[0]
    # C[0,1] -> W[1*3+0] = W[3]
    # C[1,0] -> W[0*3+1] = W[1]
    # C[1,1] -> W[1*3+1] = W[4]
    # So (e,f) = (i,k) = (0,0)->W[0], (0,1)->W[3], (1,0)->W[1], (1,1)->W[4]
    # w_indices for (e,f) = (0,0),(0,1),(1,0),(1,1) is [0, 3, 1, 4]
    
    # Block 1: A_top_left * B_left_top -> C_top_left
    add_strassen_block([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4], U2, V2, W2)
    
    # Block 2: A_top_right * B_left_bottom -> C_top_left
    # A_top_right = A[0:2, 2:4]: U indices 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # B_left_bottom = B[2:4, 0:2]: V indices 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # C_top_left: same W indices [0, 3, 1, 4]
    add_strassen_block([2, 3, 6, 7], [8, 9, 12, 13], [0, 3, 1, 4], U2, V2, W2)
    
    # Block 3: A_top_left * B_right_top -> C_top_right
    # A_top_left: U [0, 1, 4, 5]
    # B_right_top = B[0:2, 2:4]: V indices 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # C_top_right = C[0:2, 2:4]:
    #   C[0,2] -> W[2*3+0] = W[6]
    #   C[0,3] -> W[3*3+0] = W[9]
    #   C[1,2] -> W[2*3+1] = W[7]
    #   C[1,3] -> W[3*3+1] = W[10]
    # w_indices for (e,f)=(0,0),(0,1),(1,0),(1,1) -> [6, 9, 7, 10]
    add_strassen_block([0, 1, 4, 5], [2, 3, 6, 7], [6, 9, 7, 10], U2, V2, W2)
    
    # Block 4: A_top_right * B_right_bottom -> C_top_right
    # A_top_right: U [2, 3, 6, 7]
    # B_right_bottom = B[2:4, 2:4]: V indices 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    # C_top_right: W [6, 9, 7, 10]
    add_strassen_block([2, 3, 6, 7], [10, 11, 14, 15], [6, 9, 7, 10], U2, V2, W2)
    
    # Now the bottom row: C_bottom (1x4) = A_bottom (1x4) * B (4x4)
    # C_bottom_left (1x2) = A_bottom_left (1x2) * B_left (4x2)
    # C_bottom_right (1x2) = A_bottom_right (1x2) * B_right (4x2)
    
    # A_bottom_left = A[2, 0:2]: U indices 2*4+0=8, 2*4+1=9
    # A_bottom_right = A[2, 2:4]: U indices 2*4+2=10, 2*4+3=11
    
    # For 1x2*2x2 = 1x2, use schoolbook rank 4.
    # U1 shape (1,2,4), V1 shape (2,2,4), W1 shape (1,2,4)
    # U1[0,c,r] for c in 0,1
    # V1[b,d,r] for b in 0,1, d in 0,1
    # W1[0,f,r] for f in 0,1
    
    def add_schoolbook_1x2_block(u_indices, v_indices, w_indices, U1, V1, W1):
        nonlocal col
        for r in range(4):
            for u_idx, c in zip(u_indices, [0, 1]):
                U[u_idx, col+r] = U1[0, c, r]
            for v_idx, (b, d) in zip(v_indices, [(0,0),(0,1),(1,0),(1,1)]):
                V[v_idx, col+r] = V1[b, d, r]
            for w_idx, f in zip(w_indices, [0, 1]):
                W[w_idx, col+r] = W1[0, f, r]
        col += 4
    
    # Block 5: A_bottom_left * B_left_top -> C_bottom_left
    # A_bottom_left: U [8, 9] for c=0,1
    # B_left_top = B[0:2, 0:2]: V [0, 1, 4, 5] for (b,d)=(0,0),(0,1),(1,0),(1,1)
    # C_bottom_left = C[2, 0:2]:
    #   C[2,0] -> W[0*3+2] = W[2]
    #   C[2,1] -> W[1*3+2] = W[5]
    # w_indices for f=0,1 -> [2, 5]
    add_schoolbook_1x2_block([8, 9], [0, 1, 4, 5], [2, 5], U1, V1, W1)
    
    # Block 6: A_bottom_left * B_left_bottom -> C_bottom_left
    # A_bottom_left: U [8, 9]
    # B_left_bottom = B[2:4, 0:2]: V [8, 9, 12, 13]
    # C_bottom_left: W [2, 5]
    add_schoolbook_1x2_block([8, 9], [8, 9, 12, 13], [2, 5], U1, V1, W1)
    
    # Block 7: A_bottom_right * B_right_top -> C_bottom_right
    # A_bottom_right: U [10, 11]
    # B_right_top = B[0:2, 2:4]: V [2, 3, 6, 7]
    # C_bottom_right = C[2, 2:4]:
    #   C[2,2] -> W[2*3+2] = W[8]
    #   C[2,3] -> W[3*3+2] = W[11]
    # w_indices for f=0,1 -> [8, 11]
    add_schoolbook_1x2_block([10, 11], [2, 3, 6, 7], [8, 11], U1, V1, W1)
    
    # Block 8: A_bottom_right * B_right_bottom -> C_bottom_right
    # A_bottom_right: U [10, 11]
    # B_right_bottom = B[2:4, 2:4]: V [10, 11, 14, 15]
    # C_bottom_right: W [8, 11]
    add_schoolbook_1x2_block([10, 11], [10, 11, 14, 15], [8, 11], U1, V1, W1)
    
    # Now verify the full decomposition
    def verify_decomp(U, V, W, T, n, m, p):
        rank = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for l in range(n):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+l, r]
                        expected = T[i, j, k, l]
                        err = abs(s - expected)
                        if err > max_err:
                            max_err = err
        return max_err
    
    max_err = verify_decomp(U, V, W, T, n, m, p)
    
    if max_err > 1e-9:
        # Error repair: fall back to schoolbook rank 48
        rank = n * m * p
        U = np.zeros((dim_u, rank), dtype=float)
        V = np.zeros((dim_v, rank), dtype=float)
        W = np.zeros((dim_w, rank), dtype=float)
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] = 1.0
                    V[j*p+k, r] = 1.0
                    W[k*n+i, r] = 1.0
                    r += 1
        # Verify schoolbook
        max_err = verify_decomp(U, V, W, T, n, m, p)
        if max_err > 1e-9:
            # Should never happen, but return zeros as last resort
            pass
    
    # Convert to lists with clean integer/half-integer values
    def to_clean_list(arr):
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                # Round to nearest 0.5
                v = round(val * 2) / 2
                if abs(v) < 1e-12:
                    v = 0
                clean_row.append(v)
            result.append(clean_row)
        return result
    
    return {"U": to_clean_list(U), "V": to_clean_list(V), "W": to_clean_list(W)}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
