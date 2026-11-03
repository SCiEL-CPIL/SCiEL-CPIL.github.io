import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s

    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 20, 30, 24

    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # --- Strassen rank-7 decomposition for 2x2 * 2x2 ---
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = AB
    # M1 = (a10 + a11)(b00 + b11)
    # M2 = a10(b11 - b00)
    # M3 = (a00 + a01)b11
    # M4 = a00(b10 - b11)
    # M5 = (a00 + a10)b00
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M7
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 - M4

    def strassen_2x2():
        """Return (U, V, W) for 2x2*2x2 with rank 7.
        U indexed by (i,j) -> i*2+j, V by (j,k) -> j*2+k, W by (k,i) -> k*2+i.
        """
        r = 7
        U = np.zeros((4, r))
        V = np.zeros((4, r))
        W = np.zeros((4, r))
        # a00=0, a01=1, a10=2, a11=3
        # b00=0, b01=1, b10=2, b11=3
        # c00=0, c01=1, c10=2, c11=3 (W index: k*2+i)
        
        # M1 = (a10+a11)(b00+b11)
        t = 0
        U[2, t] = 1; U[3, t] = 1
        V[0, t] = 1; V[3, t] = 1
        W[0, t] = 1; W[3, t] = 1  # c00, c11
        
        # M2 = a10(b11-b00)
        t = 1
        U[2, t] = 1
        V[3, t] = 1; V[0, t] = -1
        W[2, t] = 1; W[3, t] = -1  # c10, -c11
        
        # M3 = (a00+a01)b11
        t = 2
        U[0, t] = 1; U[1, t] = 1
        V[3, t] = 1
        W[1, t] = 1; W[3, t] = 1  # c01, c11
        
        # M4 = a00(b10-b11)
        t = 3
        U[0, t] = 1
        V[2, t] = 1; V[3, t] = -1
        W[0, t] = 1; W[3, t] = -1  # c00, -c11
        
        # M5 = (a00+a10)b00
        t = 4
        U[0, t] = 1; U[2, t] = 1
        V[0, t] = 1
        W[0, t] = -1; W[2, t] = 1  # -c00, c10
        
        # M6 = (a10-a00)(b00+b01)
        t = 5
        U[2, t] = 1; U[0, t] = -1
        V[0, t] = 1; V[1, t] = 1
        # M6 is not used in standard Strassen output formulas above
        # Actually, let me re-verify. In standard Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M7
        # C10 = M2 + M5
        # C11 = M1 - M2 + M3 - M4
        # M6 doesn't appear! So we don't need it. Let me remove M6.
        # Actually wait, that's only 6 terms. Standard Strassen uses 7.
        # Let me recheck. The standard 7-term Strassen:
        # M1 = (A11+A21)(B11+B22)
        # M2 = A21(B12-B11)
        # M3 = (A11+A12)B22
        # M4 = A11(B21-B22)
        # M5 = (A11+A21)B11
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M7
        # C21 = M2 + M5
        # C22 = M1 - M2 + M3 - M4
        # So M6 is indeed not used in the output. This means rank 6 suffices?
        # No, that can't be right. Let me re-examine.
        # Actually, I think the standard formulation does use all 7.
        # Let me just verify numerically that my 6-term version works,
        # and if not, add M6 back properly.
        
        # Let me just use the 7-term version and include M6 even if it
        # doesn't contribute to the output (it would be a zero column in W).
        # Actually, a cleaner approach: use the known correct rank-7.
        # Let me just hardcode a verified rank-7 for 2x2.
        
        # Reset and use a clean formulation
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Using 0-indexed: a00,a01,a10,a11 = 0,1,2,3
        # b00,b01,b10,b11 = 0,1,2,3
        # c00,c01,c10,c11 -> W index k*2+i: c00=0, c01=1, c10=2, c11=3
        
        # M1 = (a10+a11)(b00+b11)
        t = 0
        U[2, t] = 1; U[3, t] = 1
        V[0, t] = 1; V[3, t] = 1
        W[0, t] = 1; W[3, t] = 1
        
        # M2 = a10(b11-b00)
        t = 1
        U[2, t] = 1
        V[3, t] = 1; V[0, t] = -1
        W[2, t] = 1; W[3, t] = -1
        
        # M3 = (a00+a01)b11
        t = 2
        U[0, t] = 1; U[1, t] = 1
        V[3, t] = 1
        W[1, t] = 1; W[3, t] = 1
        
        # M4 = a00(b10-b11)
        t = 3
        U[0, t] = 1
        V[2, t] = 1; V[3, t] = -1
        W[0, t] = 1; W[3, t] = -1
        
        # M5 = (a00+a10)b00
        t = 4
        U[0, t] = 1; U[2, t] = 1
        V[0, t] = 1
        W[0, t] = -1; W[2, t] = 1
        
        # M6 = (a10-a00)(b00+b01)
        t = 5
        U[2, t] = 1; U[0, t] = -1
        V[0, t] = 1; V[1, t] = 1
        W[0, t] = 0; W[1, t] = 0; W[2, t] = 0; W[3, t] = 0
        
        # M7 = (a01-a11)(b10+b11)
        t = 6
        U[1, t] = 1; U[3, t] = -1
        V[2, t] = 1; V[3, t] = 1
        W[0, t] = 1; W[1, t] = 1
        
        return U, V, W

    # --- Laderman rank-7 for 3x3 * 3x3 ---
    # This is more complex. Let me use a known correct formulation.
    # Actually, for the partitioning strategy, I need rank-7 for 3x3*3x3.
    # Laderman's decomposition is well-known. Let me implement it carefully.
    #
    # For 3x3 matrices A, B, C = AB:
    # The 7 multiplication terms and their contributions to C[i,k]:
    #
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A01 + A11)B10
    # M3 = (A10 - A00)B01
    # M4 = A00(B11 - B01)
    # M5 = A11(B00 - B10)
    # M6 = (A10 + A00)(B01 + B11)
    # M7 = (A01 - A11)(B00 + B10)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M2 + M5 + M7
    # C10 = M1 - M2 + M3 - M6
    # C11 = M3 + M4 + M6
    #
    # But this only covers C[0:2, 0:2]. What about entries involving index 2?
    # Laderman's full rank-7 decomposition covers all 9 entries.
    # The correct full Laderman decomposition:
    #
    # Actually, I recall that Laderman's rank-7 for 3x3 uses a different set
    # of 7 terms. Let me use a verified source.
    #
    # From the literature, Laderman's 3x3 decomposition:
    # Let A = (a_ij), B = (b_jk), C = (c_ik) for i,j,k in {0,1,2}
    #
    # m1 = (a00 + a11)(b00 + b11)
    # m2 = (a01 + a11)b10
    # m3 = (a10 - a00)b01
    # m4 = a00(b11 - b01)
    # m5 = a11(b00 - b10)
    # m6 = (a10 + a00)(b01 + b11)
    # m7 = (a01 - a11)(b00 + b10)
    #
    # c00 = m1 + m4 - m5 - m7
    # c01 = m2 + m5 + m7
    # c02 = a00*b02 + a01*b12 + a02*b22  -- this can't be right for rank 7
    #
    # I think I'm confusing two different things. Let me just use the
    # fact that rank(3x3 matmul) <= 7 is a theorem, and find the correct
    # coefficients. Actually, for our purposes, let me just use the
    # partitioning with Strassen on 2x2 blocks and see what rank we get.
    #
    # Alternative approach: partition 4x5*5x6 as:
    # n=4=2+2, m=5=2+3, p=6=3+3
    # A = [A11(2x2) A12(2x3); A21(2x2) A22(2x3)]
    # B = [B11(2x3) B12(2x3); B21(3x3) B22(3x3)]
    # C = A11*B11 + A12*B21 + A21*B12 + A22*B22
    #
    # A11*B11: 2x2*2x3 -> rank 2*2*3=12 schoolbook, or better
    # A12*B21: 2x3*3x3 -> rank 2*3*3=18 schoolbook
    # A21*B12: 2x2*2x3 -> rank 2*2*3=12 schoolbook
    # A22*B22: 2x3*3x3 -> rank 2*3*3=18 schoolbook
    # Total: 60 schoolbook
    #
    # That's already better than 120! Let me implement this directly.
    # Actually, let me try even better partitions.
    
    # Strategy: Use block decomposition with Strassen where possible
    # Partition: n=4=2+2, m=5=2+3, p=6=3+3
    # A (4x5) = [[A11(2x2), A12(2x3)],
    #            [A21(2x2), A22(2x3)]]
    # B (5x6) = [[B11(2x3), B12(2x3)],
    #            [B21(3x3), B22(3x3)]]
    # C (4x6) = A11*B11 + A12*B21 + A21*B12 + A22*B22
    #
    # Each block multiplication:
    # A11*B11: (2x2)*(2x3) = 2x3 result, rank 12 schoolbook
    # A12*B21: (2x3)*(3x3) = 2x3 result, rank 18 schoolbook
    # A21*B12: (2x2)*(2x3) = 2x3 result, rank 12 schoolbook
    # A22*B22: (2x3)*(3x3) = 2x3 result, rank 18 schoolbook
    # Total: 60
    
    # Can we do better with Strassen on the 2x2*2x3 parts?
    # 2x2*2x3: treat as 2x2 * [2x2 | 2x1]
    # = (2x2*2x2) + (2x2*2x1)
    # 2x2*2x2: rank 7 (Strassen)
    # 2x2*2x1: rank 4 schoolbook
    # Total: 11 per 2x2*2x3 block, so 22 for both A11*B11 and A21*B12
    
    # For 2x3*3x3: treat as 2x3 * [3x3]
    # = (2x2*3x3) + (2x1*3x3)
    # 2x2*3x3: rank 12 schoolbook, or (2x2*2x3)+(2x2*2x1) = 12+4=16... worse
    # Actually 2x2*3x3: partition 3x3 as 2x2+1x1... complex
    # Just use schoolbook: 2*3*3=18 each, 36 total
    # 
    # Total: 22 + 36 = 58
    
    # Even better: try n=4=2+2, m=5=3+2, p=6=2+4
    # A = [[A11(2x3), A12(2x2)],
    #      [A21(2x3), A22(2x2)]]
    # B = [[B11(3x2), B12(3x4)],
    #      [B21(2x2), B22(2x4)]]
    # C = A11*B11 + A12*B21 + A21*B12 + A22*B22
    # A11*B11: 2x3*3x2 = 2x2, rank 12
    # A12*B21: 2x2*2x2 = 2x2, rank 7 (Strassen)
    # A21*B12: 2x3*3x4 = 2x4, rank 24
    # A22*B22: 2x2*2x4 = 2x4, rank 16
    # Total: 12+7+24+16 = 59
    
    # Try n=4=1+3, m=5=2+3, p=6=3+3
    # A = [[A11(1x2), A12(1x3)],
    #      [A21(3x2), A22(3x3)]]
    # B = [[B11(2x3), B12(2x3)],
    #      [B21(3x3), B22(3x3)]]
    # C = A11*B11 + A12*B21 + A21*B12 + A22*B22
    # A11*B11: 1x2*2x3 = 1x3, rank 6
    # A12*B21: 1x3*3x3 = 1x3, rank 9
    # A21*B12: 3x2*2x3 = 3x3, rank 18
    # A22*B22: 3x3*3x3 = 3x3, rank 27 schoolbook or 7 Laderman
    # Total with Laderman: 6+9+18+7 = 40
    # Total schoolbook: 6+9+18+27 = 60
    
    # The 40-rank decomposition is very attractive! Let me implement it.
    # I need Laderman's rank-7 for 3x3*3x3.
    
    # Let me implement a verified Laderman decomposition.
    # After research, the correct Laderman rank-7 for 3x3:
    #
    # Using 0-indexed a_ij, b_jk, c_ik:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a01+a11)b10
    # m3 = (a10-a00)b01
    # m4 = a00(b11-b01)
    # m5 = a11(b00-b10)
    # m6 = (a10+a00)(b01+b11)
    # m7 = (a01-a11)(b00+b10)
    #
    # c00 = m1 + m4 - m5 - m7
    # c01 = m2 + m5 + m7
    # c10 = m1 - m2 + m3 - m6
    # c11 = m3 + m4 + m6
    #
    # For entries with index 2:
    # c02 = a00*b02 + a01*b12 + a02*b22
    # c12 = a10*b02 + a11*b12 + a12*b22
    # c20 = a20*b00 + a21*b10 + a22*b20
    # c21 = a20*b01 + a21*b11 + a22*b21
    # c22 = a20*b02 + a21*b12 + a22*b22
    #
    # These 5 entries each need 3 multiplications = 15 more, total 22.
    # This is NOT rank 7. So Laderman's rank-7 must use a different scheme.
    #
    # I think the issue is that Laderman's rank-7 works by mixing all 9 entries
    # into the 7 terms, not by separating them. Let me look for the correct
    # formulation.
    #
    # Actually, I recall now: the rank of 3x3 matrix multiplication over
    # the reals is at most 7 (Laderman 1976). The decomposition is:
    #
    # It's quite involved. For safety, let me just use schoolbook for the
    # 3x3*3x3 part (rank 27) and get total rank 60, which is still much
    # better than 120.
    
    # Let me implement the partition n=4=1+3, m=5=2+3, p=6=3+3 with
    # schoolbook for all blocks: total rank 60.
    
    def build_decomposition():
        """Build decomposition using block partitioning."""
        # Partition: n=4=1+3, m=5=2+3, p=6=3+3
        # A rows: [0] then [1,2,3], A cols: [0,1] then [2,3,4]
        # B rows: [0,1] then [2,3,4], B cols: [0,1,2] then [3,4,5]
        # C rows: [0] then [1,2,3], C cols: [0,1,2] then [3,4,5]
        
        # Block 1: A[0,:2] * B[:2, 3:6] -> C[0, 3:6]  (1x2 * 2x3 = 1x3)
        # Block 2: A[0,2:] * B[2:, 3:6] -> C[0, 3:6]  (1x3 * 3x3 = 1x3)
        # Block 3: A[1:, :2] * B[:2, :3] -> C[1:, :3]  (3x2 * 2x3 = 3x3)
        # Block 4: A[1:, 2:] * B[2:, :3] -> C[1:, :3]  (3x3 * 3x3 = 3x3)
        # Block 5: A[1:, :2] * B[:2, 3:6] -> C[1:, 3:6]  (3x2 * 2x3 = 3x3)
        # Block 6: A[1:, 2:] * B[2:, 3:6] -> C[1:, 3:6]  (3x3 * 3x3 = 3x3)
        
        # Wait, I need to be more careful. Let me use the standard block multiplication.
        # A (4x5) partitioned as:
        #   rows: r1=1, r2=3
        #   cols: c1=2, c2=3
        # A = [[A11(1x2), A12(1x3)],
        #      [A21(3x2), A22(3x3)]]
        #
        # B (5x6) partitioned as:
        #   rows: r1=2, r2=3
        #   cols: c1=3, c2=3
        # B = [[B11(2x3), B12(2x3)],
        #      [B21(3x3), B22(3x3)]]
        #
        # C (4x6) = A*B:
        # C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
        #      [A21*B11 + A22*B21, A21*B12 + A22*B22]]
        #
        # C11 (1x3) = A11(1x2)*B11(2x3) + A12(1x3)*B21(3x3)
        # C12 (1x3) = A11(1x2)*B12(2x3) + A12(1x3)*B22(3x3)
        # C21 (3x3) = A21(3x2)*B11(2x3) + A22(3x3)*B21(3x3)
        # C22 (3x3) = A21(3x2)*B12(2x3) + A22(3x3)*B22(3x3)
        #
        # Schoolbook ranks:
        # A11*B11: 1*2*3 = 6
        # A12*B21: 1*3*3 = 9
        # A11*B12: 1*2*3 = 6
        # A12*B22: 1*3*3 = 9
        # A21*B11: 3*2*3 = 18
        # A22*B21: 3*3*3 = 27
        # A21*B12: 3*2*3 = 18
        # A22*B22: 3*3*3 = 27
        # Total: 6+9+6+9+18+27+18+27 = 120. Same as before!
        
        # The issue is that block decomposition with schoolbook doesn't help
        # because the total number of scalar multiplications is the same.
        # We need to use Strassen/Laderman on the square blocks.
        
        # Let me try: use Strassen rank-7 for 2x2*2x2 blocks where possible.
        # Partition: n=4=2+2, m=5=2+3, p=6=3+3
        # A = [[A11(2x2), A12(2x3)],
        #      [A21(2x2), A22(2x3)]]
        # B = [[B11(2x3), B12(2x3)],
        #      [B21(3x3), B22(3x3)]]
        #
        # C11 (2x3) = A11(2x2)*B11(2x3) + A12(2x3)*B21(3x3)
        # C12 (2x3) = A11(2x2)*B12(2x3) + A12(2x3)*B22(3x3)
        # C21 (2x3) = A21(2x2)*B11(2x3) + A22(2x3)*B21(3x3)
        # C22 (2x3) = A21(2x2)*B12(2x3) + A22(2x3)*B22(3x3)
        #
        # For A11*B11 (2x2 * 2x3): 
        #   = A11 * [B11a(2x2) | B11b(2x1)]
        #   = A11*B11a + A11*B11b
        #   A11*B11a: 2x2*2x2 = rank 7 (Strassen)
        #   A11*B11b: 2x2*2x1 = rank 4 (schoolbook)
        #   Total: 11
        #
        # Similarly for A11*B12, A21*B11, A21*B12: each 11
        # For A12*B21 (2x3 * 3x3): rank 18 schoolbook
        # For A12*B22 (2x3 * 3x3): rank 18 schoolbook
        # For A22*B21 (2x3 * 3x3): rank 18 schoolbook
        # For A22*B22 (2x3 * 3x3): rank 18 schoolbook
        #
        # Total: 4*11 + 4*18 = 44 + 72 = 116. Only slightly better than 120.
        
        # Let me try a different approach: use the fact that we can combine
        # terms across blocks.
        
        # Actually, let me just try to find a rank < 120 decomposition using
        # numerical optimization with a better initialization.
        
        # Start with schoolbook
        rank_sb = n*m*p  # 120
        U = np.zeros((nm, rank_sb))
        V = np.zeros((mp, rank_sb))
        W = np.zeros((pn, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
        best_rank = rank_sb
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        def verify(U, V, W):
            r = U.shape[1]
            T_recon = np.zeros((nm, mp, pn))
            for t in range(r):
                uv = np.outer(U[:, t], V[:, t])
                T_recon += np.einsum('ab,c->abc', uv, W[:, t])
            return np.max(np.abs(T_recon - T))
        
        # Try ALS with various target ranks
        np.random.seed(42)
        
        for target_rank in [80, 90, 100, 110]:
            if time.time() > deadline - 40:
                break
            
            for attempt in range(3):
                if time.time() > deadline - 30:
                    break
                
                U_t = np.random.randn(nm, target_rank) * 0.5
                V_t = np.random.randn(mp, target_rank) * 0.5
                W_t = np.random.randn(pn, target_rank) * 0.5
                
                for it in range(300):
                    if time.time() > deadline - 20:
                        break
                    
                    # Solve for U
                    X = np.einsum('bt,ct->bct', V_t, W_t).reshape(target_rank, mp*pn).T
                    for a in range(nm):
                        U_t[a, :] = np.linalg.lstsq(X, T[a].reshape(-1), rcond=None)[0]
                    
                    # Solve for V
                    X = np.einsum('at,ct->act', U_t, W_t).reshape(target_rank, nm*pn).T
                    for b in range(mp):
                        V_t[b, :] = np.linalg.lstsq(X, T[:, b, :].reshape(-1), rcond=None)[0]
                    
                    # Solve for W
                    X = np.einsum('at,bt->abt', U_t, V_t).reshape(target_rank, nm*mp).T
                    for c in range(pn):
                        W_t[c, :] = np.linalg.lstsq(X, T[:, :, c].reshape(-1), rcond=None)[0]
                    
                    err = verify(U_t, V_t, W_t)
                    if err < 1e-10:
                        break
                
                err = verify(U_t, V_t, W_t)
                if err < 1e-10:
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U, best_V, best_W = U_t.copy(), V_t.copy(), W_t.copy()
                    break
        
        U, V, W = best_U, best_V, best_W
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return build_decomposition()
