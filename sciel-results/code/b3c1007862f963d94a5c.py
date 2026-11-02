import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Strassen rank-7 decomposition for 2x2 x 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    def strassen_2x2():
        """Return (U, V, W) for 2x2 x 2x2 Strassen, each shape (4,7)"""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Index: U[row, col] where row = i*2+j for A[i,j], col = strassen step
        # V[row, col] where row = i*2+j for B[i,j], col = strassen step
        # W[row, col] where row = i*2+j for C[i,j], col = strassen step
        
        # Step 0: M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1  # A11
        U[3, 0] = 1  # A22
        V[0, 0] = 1  # B11
        V[3, 0] = 1  # B22
        W[0, 0] = 1  # C11
        W[3, 0] = 1  # C22
        
        # Step 1: M2 = (A21+A22)B11
        U[2, 1] = 1  # A21
        U[3, 1] = 1  # A22
        V[0, 1] = 1  # B11
        W[2, 1] = 1  # C21
        W[3, 1] = -1  # C22
        
        # Step 2: M3 = A11(B21-B22)
        U[0, 2] = 1  # A11
        V[2, 2] = 1  # B21
        V[3, 2] = -1  # B22
        W[1, 2] = 1  # C12
        W[3, 2] = 1  # C22
        
        # Step 3: M4 = A22(B12-B11)
        U[3, 3] = 1  # A22
        V[1, 3] = 1  # B12
        V[0, 3] = -1  # B11
        W[0, 3] = 1  # C11
        W[2, 3] = 1  # C21
        
        # Step 4: M5 = (A11+A12)B22
        U[0, 4] = 1  # A11
        U[1, 4] = 1  # A12
        V[3, 4] = 1  # B22
        W[0, 4] = -1  # C11
        W[1, 4] = 1  # C12
        
        # Step 5: M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1  # A21
        U[0, 5] = -1  # A11
        V[0, 5] = 1  # B11
        V[1, 5] = 1  # B12
        W[3, 5] = 1  # C22
        
        # Step 6: M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1  # A12
        U[3, 6] = -1  # A22
        V[2, 6] = 1  # B21
        V[3, 6] = 1  # B22
        W[0, 6] = 1  # C11
        
        return U, V, W
    
    # Build schoolbook as fallback
    def build_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Verify a decomposition
    def verify(U, V, W, tol=1e-10):
        R = U.shape[1]
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(R):
                        val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    if abs(val - 1.0) > tol:
                        return False
        # Check zero entries (sample)
        rng = np.random.default_rng(42)
        for _ in range(500):
            i = int(rng.integers(n))
            j = int(rng.integers(m))
            jp = int(rng.integers(m))
            k = int(rng.integers(p))
            kp = int(rng.integers(p))
            ip = int(rng.integers(n))
            if i == ip and j == jp and k == kp:
                continue
            val = 0.0
            for c in range(R):
                val += U[i*m + j, c] * V[jp*p + k, c] * W[kp*n + ip, c]
            if abs(val) > tol:
                return False
        return True
    
    # Try block-partitioning with Strassen
    # Split: A (3x4) into A_left (3x2) cols 0-1, A_right (3x2) cols 2-3
    #         B (4x6) into B_top (2x6) rows 0-1, B_bot (2x6) rows 2-3
    # C = A_left * B_top + A_right * B_bot
    # Each sub-product is 3x2 x 2x6. We need to decompose these.
    # For 3x2 x 2x6: we can pad to 3x3 x 3x3 and use a known decomposition,
    # or use schoolbook (rank 36 each, total 72 - no gain).
    # Alternative: split differently.
    
    # Better approach: Use the rank-68 construction from cell findings.
    # Split m=4 into 2+2, use Strassen on 2x2 blocks, handle remaining with schoolbook.
    # Actually, let's try: split n=3 into 2+1, m=4 into 2+2, p=6 into 3+3.
    # This creates 8 sub-problems of varying sizes.
    
    # Simplest valid improvement: Use Strassen on the 2x2 x 2x2 core and schoolbook for edges.
    # Split n: rows 0-1 (2 rows) + row 2 (1 row)
    # Split m: cols 0-1 (2 cols) + cols 2-3 (2 cols)  
    # Split p: cols 0-2 (3 cols) + cols 3-5 (3 cols)
    
    # Sub-problems:
    # (2x2) x (2x3) -> 2x3: use Strassen-like (pad B to 2x2, rank 7, but loses col 2)
    # This gets complicated. Let's use a known rank-60 or rank-68 construction.
    
    # For now, implement a structured rank-68 using 2x2 Strassen blocks:
    # Partition the 4 middle dimension into two 2s.
    # A = [A1 | A2] where A1, A2 are 3x2
    # B = [B1; B2] where B1, B2 are 2x6
    # C = A1*B1 + A2*B2
    # For A1*B1 (3x2 x 2x6): decompose using rank-7 Strassen on 2x2 x 2x2 sub-blocks
    # Split the 3 rows into 2+1, and 6 cols into 3+3.
    # A1 = [A1_top (2x2); A1_bot (1x2)]
    # B1 = [B1_left (2x3) | B1_right (2x3)]
    # A1*B1 = A1_top*B1_left + A1_top*B1_right + A1_bot*B1_left + A1_bot*B1_right
    # Each 2x2 x 2x3: pad to 2x2 x 2x2 (zero col), Strassen rank 7
    # Each 1x2 x 2x3: schoolbook rank 6
    # Total per 3x2 x 2x6: 4*7 + 4*6 = 28+24 = 52... that's worse.
    
    # Let me just use the direct schoolbook but with a small optimization:
    # Actually, the simplest improvement that's guaranteed valid:
    # Use rank-7 Strassen for the 2x2 x 2x2 part and schoolbook for the rest.
    
    # Partition:
    # n=3: rows {0,1} and {2}
    # m=4: cols {0,1} and {2,3}
    # p=6: cols {0,1} and {2,3,4,5}
    
    # Sub-problems:
    # 1. A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (2x2 x 2x2): Strassen rank 7
    # 2. A[0:2,0:2] * B[0:2,2:6] -> C[0:2,2:6] (2x2 x 2x4): schoolbook 16
    # 3. A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2] (2x2 x 2x2): Strassen rank 7
    # 4. A[0:2,2:4] * B[2:4,2:6] -> C[0:2,2:6] (2x2 x 2x4): schoolbook 16
    # 5. A[2:3,0:2] * B[0:2,0:2] -> C[2:3,0:2] (1x2 x 2x2): schoolbook 4
    # 6. A[2:3,0:2] * B[0:2,2:6] -> C[2:3,2:6] (1x2 x 2x4): schoolbook 8
    # 7. A[2:3,2:4] * B[2:4,0:2] -> C[2:3,0:2] (1x2 x 2x2): schoolbook 4
    # 8. A[2:3,2:4] * B[2:4,2:6] -> C[2:3,2:6] (1x2 x 2x4): schoolbook 8
    # Total: 7+16+7+16+4+8+4+8 = 70. Slightly better than 72.
    
    # Let's try a different split to get more Strassen blocks:
    # n=3: {0,1} and {2}
    # m=4: {0,1} and {2,3}
    # p=6: {0,1}, {2,3}, {4,5}
    
    # Sub-problems (8x3=24... too many). Let me just build it.
    
    def build_structured():
        # Split points
        n_splits = [(0,2), (2,3)]  # rows 0-1, row 2
        m_splits = [(0,2), (2,4)]  # cols 0-1, cols 2-3
        p_splits = [(0,2), (2,4), (4,6)]  # cols 0-1, 2-3, 4-5
        
        U_parts = []
        V_parts = []
        W_parts = []
        
        for (ni, nj) in n_splits:
            for (mi, mj) in m_splits:
                for (pi, pj) in p_splits:
                    sn = nj - ni
                    sm = mj - mi
                    sp = pj - pi
                    
                    if sn == 2 and sm == 2 and sp == 2:
                        # Use Strassen
                        sU, sV, sW = strassen_2x2()
                        U_parts.append((ni, mi, pi, sn, sm, sp, sU, sV, sW))
                    else:
                        # Schoolbook for this sub-block
                        rank_sub = sn * sm * sp
                        U_sub = np.zeros((sn*sm, rank_sub))
                        V_sub = np.zeros((sm*sp, rank_sub))
                        W_sub = np.zeros((sp*sn, rank_sub))
                        col = 0
                        for li in range(sn):
                            for lj in range(sm):
                                for lk in range(sp):
                                    U_sub[li*sm+lj, col] = 1
                                    V_sub[lj*sp+lk, col] = 1
                                    W_sub[lk*sn+li, col] = 1
                                    col += 1
                        U_parts.append((ni, mi, pi, sn, sm, sp, U_sub, V_sub, W_sub))
        
        # Assemble
        total_rank = sum(part[6].shape[1] for part in U_parts)
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        col = 0
        for (ni, mi, pi, sn, sm, sp, U_sub, V_sub, W_sub) in U_parts:
            R_sub = U_sub.shape[1]
            # Map U_sub
            for row_local in range(sn * sm):
                li = row_local // sm
                lj = row_local % sm
                gi = ni + li
                gj = mi + lj
                U[gi*m + gj, col:col+R_sub] = U_sub[row_local, :]
            # Map V_sub
            for row_local in range(sm * sp):
                lj = row_local // sp
                lk = row_local % sp
                gi = mi + lj
                gj = pi + lk
                V[gi*p + gj, col:col+R_sub] = V_sub[row_local, :]
            # Map W_sub
            for row_local in range(sp * sn):
                lk = row_local // sn
                li = row_local % sn
                gi = pi + lk
                gj = ni + li
                W[gi*n + gj, col:col+R_sub] = W_sub[row_local, :]
            col += R_sub
        
        return U, V, W
    
    # Try the structured approach
    try:
        if time.time() < deadline:
            U, V, W = build_structured()
            if verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W = build_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
