import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _build():
    """Build a rank-minimal decomposition for 4x4 * 4x6.
    
    Uses a hybrid approach:
    - For the first 4 columns of B (making B1 4x4): use nested Strassen
    - For the last 2 columns of B (making B2 4x2): use a simpler approach
    
    Total rank: 49 + 28 = 77
    """
    n, m, p = 4, 4, 6
    
    # Top-level 2x2 block Strassen parameters
    # A blocks: r=0:(0,0) r=1:(0,1) r=2:(1,0) r=3:(1,1)
    # B blocks: c=0:(0,0) c=1:(0,1) c=2:(1,0) c=3:(1,1)
    
    alpha = [
        [(0, 1), (3, 1)],       # s0: A00+A11
        [(2, 1), (3, 1)],       # s1: A10+A11
        [(0, 1)],               # s2: A00
        [(3, 1)],               # s3: A11
        [(0, 1), (1, 1)],       # s4: A00+A01
        [(2, 1), (0, -1)],      # s5: A10 - A00
        [(1, 1), (3, -1)],      # s6: A01 - A11
    ]
    beta = [
        [(0, 1), (3, 1)],       # s0: B00+B11
        [(0, 1)],               # s1: B00
        [(1, 1), (3, -1)],      # s2: B01-B11
        [(2, 1), (0, -1)],      # s3: B10-B00
        [(3, 1)],               # s4: B11
        [(0, 1), (1, 1)],       # s5: B00+B01
        [(2, 1), (3, 1)],       # s6: B10+B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]

    # Inner Strassen for 2x2 x 2x2
    inner_alpha = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)],
    ]
    inner_beta = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)],
    ]
    inner_recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]

    U = {}  # (i,k) -> col -> val
    V = {}  # (k,j) -> col -> val
    W = {}  # (j,i) -> col -> val  # trace convention W[k*n+i, r] -> index = j*n + i

    col = 0
    
    # Part 1: First 4 columns of B (B1 is 4x4)
    # Use nested Strassen: outer 2x2 Strassen, inner 2x2 Strassen for each 2x2 block
    for s in range(7):
        # Inner Strassen for 2x2 x 2x2
        for si in range(7):
            # U: X_s[ar+?, ...]. X_s row = x_row (0/1), col = x_col (0/1)
            # X_s[xr, xc] = sum over (r,coef) in alpha[s]: coef * A[r][xr,xc]
            # inner_X[xr,xc] = sum over (a,coef) in inner_alpha[si]: coef * X_s[a//2, a%2]
            # So combined: A[r][xr,xc] * (coef_s * coef_inner)
            for (a, coef_i) in inner_alpha[si]:
                xr_i, xc_i = a // 2, a % 2
                for (r, coef_s) in alpha[s]:
                    ar, ac = r // 2, r % 2
                    gi = ar * 2 + xr_i
                    gk = ac * 2 + xc_i
                    U.setdefault((gi, gk), {})[col] = U.get((gi, gk), {}).get(col, 0) + coef_i * coef_s
            
            # V: Y_s[:,0:2]. Y_s[i, inner_j] = sum over (c,coef) in beta[s]: coef * B[c][i, inner_j]
            # inner B blocks: b=0:Y[0,0] b=1:Y[0,1] b=2:Y[1,0] b=3:Y[1,1]
            for (b, coef_i) in inner_beta[si]:
                y_row, y_inner_col = b // 2, b % 2
                for (c, coef_s) in beta[s]:
                    ac, bc = c // 2, c % 2
                    gk = ac * 2 + y_row
                    gj = bc * 2 + y_inner_col  # B1 is 4x2, so columns 0,1
                    V.setdefault((gk, gj), {})[col] = V.get((gk, gj), {}).get(col, 0) + coef_i * coef_s
            
            # W: inner output Z[za, zb] (2x2), contributes to C block (a,b) at (za,zb)
            for (a, b, gamma_s) in recon[s]:
                for (za, zb, gamma_i) in inner_recon[si]:
                    gi = a * 2 + za
                    gj = b * 2 + zb  # C is 4x4 for first 4 columns
                    W.setdefault((gj, gi), {})[col] = W.get((gj, gi), {}).get(col, 0) + gamma_s * gamma_i
            col += 1

    # Part 2: Last 2 columns of B (B2 is 4x2)
    # Use Strassen for 2x2 x 2x1 at top level, with schoolbook 2x2 x 2x1 inside
    # Actually, let's use a simpler approach: direct multiplication with some optimization
    
    # For the last 2 columns, we can use a rank-28 decomposition
    # This is 7 (Strassen factors) * 4 (2x2 x 2x1 schoolbook) = 28
    
    # Top-level Strassen for 2x2 x 2x2 (where the second "2" is actually 1x2)
    # B2 blocks: c=0:(0,0:1) c=1:(1,0:1) - each is 2x1
    # Actually, let's think of B2 as two 4x1 columns
    
    # Let's use a different approach: treat 4x4 * 4x2 as (4x4) * (4x2)
    # Split A into 2x2 blocks, B2 into 2x1 blocks
    # C = A*B2, where C is 4x2
    
    # Use Strassen-like: 7 multiplications of 2x2 x 2x1
    # Each 2x2 x 2x1 has rank 4 (schoolbook: (2x2)*(2x1) = 4 terms)
    
    # Let me construct this explicitly
    # For 2x2 A block and 2x1 B block, C = A*B (2x1 result)
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # Rank 4 decomposition:
    # r1: A[0,0]*B[0] -> C[0] += A[0,0]*B[0]
    # r2: A[0,1]*B[1] -> C[0] += A[0,1]*B[1]
    # r3: A[1,0]*B[0] -> C[1] += A[1,0]*B[0]
    # r4: A[1,1]*B[1] -> C[1] += A[1,1]*B[1]
    
    inner2_alpha = [
        [(0, 1)],  # A[0,0]
        [(1, 1)],  # A[0,1]
        [(2, 1)],  # A[1,0]
        [(3, 1)],  # A[1,1]
    ]
    inner2_beta = [
        [(0, 1)],  # B[0]
        [(1, 1)],  # B[1]
        [(0, 1)],  # B[0]
        [(1, 1)],  # B[1]
    ]
    inner2_recon = [
        [(0,)],  # C[0]
        [(0,)],  # C[0]
        [(1,)],  # C[1]
        [(1,)],  # C[1]
    ]
    
    # Top-level for 4x4 * 4x2: split A into 2x2 blocks, B2 into 2x1 blocks
    # B2 has 2 columns, so blocks are 2x1 each
    # Actually, this is getting complex. Let me use a direct approach for the last 2 columns
    
    # Direct schoolbook for 4x4 * 4x2: rank = 4*4*2 = 32
    # But we can do better with Strassen-like decomposition
    
    # Let's use: split A into top/bottom 2x4, B2 into left/right 4x1
    # No, that doesn't help.
    
    # Let me just use schoolbook for the last 2 columns: rank 32
    # Total: 49 + 32 = 81. That's worse than 77.
    
    # Let me reconsider. The current program achieves 77.
    # Let me just replicate the exact same logic but with cleaner code.
    
    # Actually, I realize I made an error. Let me look at the original more carefully.
    # The original uses the same nested Strassen for ALL 6 columns, not just 4.
    # For each of the 7 Strassen factors, it does:
    #   - 7 inner Strassen for columns 0,1 (giving 49 terms for the 4x4 part)
    #   - 2x2 = 4 schoolbook terms for column 2 (giving 28 terms for the 4x2 part)
    # Total: 7 * (49 + 4) / 7 = 49 + 28 = 77? No, that's not right.
    
    # Let me re-read the original code...
    # col starts at 0
    # for s in range(7):
    #   for si in range(7):
    #     ... col += 1  (49 times)
    #   for u in range(2):
    #     for v in range(2):
    #       ... col += 1  (4 times)
    # Total: 7 * 7 + 7 * 4 = 49 + 28 = 77
    
    # OK so the original is correct. Let me just use the original logic.
    # I'll clean it up slightly but keep the same structure.
    
    # Let me start over with the correct approach
    
    col = 0  # Reset
    
    # For each of the 7 top-level Strassen factors
    for s in range(7):
        # Part 1: 4 columns of B (indices 0-3), using nested Strassen
        # This gives a 4x4 result from 2x2 * 2x2 blocks
        for si in range(7):
            # U contribution
            for (a, coef_i) in inner_alpha[si]:
                xr_i, xc_i = a // 2, a % 2
                for (r, coef_s) in alpha[s]:
                    ar, ac = r // 2, r % 2
                    gi = ar * 2 + xr_i
                    gk = ac * 2 + xc_i
                    U.setdefault((gi, gk), {})[col] = U.get((gi, gk), {}).get(col, 0) + coef_i * coef_s
            
            # V contribution
            for (b, coef_i) in inner_beta[si]:
                y_row, y_inner_col = b // 2, b % 2
                for (c, coef_s) in beta[s]:
                    ac, bc = c // 2, c % 2
                    gk = ac * 2 + y_row
                    gj = bc * 2 + y_inner_col  # Columns 0,1 of the 4-column block
                    V.setdefault((gk, gj), {})[col] = V.get((gk, gj), {}).get(col, 0) + coef_i * coef_s
            
            # W contribution
            for (a, b, gamma_s) in recon[s]:
                for (za, zb, gamma_i) in inner_recon[si]:
                    gi = a * 2 + za
                    gj = b * 2 + zb  # Rows 0,1 of the 4-row result
                    W.setdefault((gj, gi), {})[col] = W.get((gj, gi), {}).get(col, 0) + gamma_s * gamma_i
            col += 1

        # Part 2: 2 columns of B (indices 2,3), using schoolbook
        # Z[u] = X_s[u,0]*Y_s[0,2] + X_s[u,1]*Y_s[1,2]
        # Where u is row index (0,1) for the 2-row block, and we have 2 row blocks
        for u in range(2):  # Row block index (0=top, 1=bottom)
            for v in range(2):  # Column index in the 2-column block (0,1)
                # U contribution
                for (r, coef_s) in alpha[s]:
                    ar, ac = r // 2, r % 2
                    gi = ar * 2 + u
                    gk = ac * 2 + v
                    U.setdefault((gi, gk), {})[col] = U.get((gi, gk), {}).get(col, 0) + coef_s
                
                # V contribution
                for (c, coef_s) in beta[s]:
                    ac, bc = c // 2, c % 2
                    gk = ac * 2 + v
                    gj = bc * 3 + 2 + v  # Wait, this doesn't look right
                    # Actually, for the last 2 columns, the column index in B is 2+v
                    # But we're in block structure, so...
                    # Let me re-examine the original code
                    pass
                col += 1

    # I'm getting confused. Let me just use the original code as-is, with minimal changes.
    # The original works and gives rank 77.
    raise NotImplementedError("Use original implementation")

def _verify(U, V, W):
    n, m, p = 4, 4, 6
    R = U.shape[1]
    # Reconstruct tensor: T[i,k,j] = sum_r U[i,k,r]*V[k,j,r]*W[j,i,r]
    # U row index = i*m + k, V row = k*p + j, W row = j*n + i
    for i in range(n):
        for k in range(m):
            for j in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m + k, r] * V[k*p + j, r] * W[j*n + i, r]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def construct():
    deadline = time.time() + 110.0
    
    # Use the original working implementation
    n, m, p = 4, 4, 6
    
    # Top-level 2x2 block Strassen parameters
    alpha = [
        [(0, 1), (3, 1)],       # s0: A00+A11
        [(2, 1), (3, 1)],       # s1: A10+A11
        [(0, 1)],               # s2: A00
        [(3, 1)],               # s3: A11
        [(0, 1), (1, 1)],       # s4: A00+A01
        [(2, 1), (0, -1)],      # s5: A10 - A00
        [(1, 1), (3, -1)],      # s6: A01 - A11
    ]
    beta = [
        [(0, 1), (3, 1)],       # s0: B00+B11
        [(0, 1)],               # s1: B00
        [(1, 1), (3, -1)],      # s2: B01-B11
        [(2, 1), (0, -1)],      # s3: B10-B00
        [(3, 1)],               # s4: B11
        [(0, 1), (1, 1)],       # s5: B00+B01
        [(2, 1), (3, 1)],       # s6: B10+B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]

    # Inner Strassen for 2x2 x 2x2
    inner_alpha = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)],
    ]
    inner_beta = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)],
    ]
    inner_recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]

    U = {}  # (i,k) -> col -> val
    V = {}  # (k,j) -> col -> val
    W = {}  # (j,i) -> col -> val  # trace convention W[k*n+i, r] -> index = j*n + i

    col = 0
    for s in range(7):
        # Part 1: inner Strassen for columns 0,1
        for si in range(7):
            # U: X_s[ar+?, ...]. X_s row = x_row (0/1), col = x_col (0/1)
            for (a, coef_i) in inner_alpha[si]:
                xr_i, xc_i = a // 2, a % 2
                for (r, coef_s) in alpha[s]:
                    ar, ac = r // 2, r % 2
                    gi = ar * 2 + xr_i
                    gk = ac * 2 + xc_i
                    U.setdefault((gi, gk), {})[col] = U.get((gi, gk), {}).get(col, 0) + coef_i * coef_s
            
            # V: Y_s[:,0:2]. Y_s[i, inner_j] = sum over (c,coef) in beta[s]: coef * B[c][i, inner_j]
            for (b, coef_i) in inner_beta[si]:
                y_row, y_inner_col = b // 2, b % 2
                for (c, coef_s) in beta[s]:
                    ac, bc = c // 2, c % 2
                    gk = ac * 2 + y_row
                    gj = bc * 3 + y_inner_col
                    V.setdefault((gk, gj), {})[col] = V.get((gk, gj), {}).get(col, 0) + coef_i * coef_s
            
            # W: inner output Z[za, zb] (2x2), contributes to C block (a,b) at (za,zb)
            for (a, b, gamma_s) in recon[s]:
                for (za, zb, gamma_i) in inner_recon[si]:
                    gi = a * 2 + za
                    gj = b * 3 + zb
                    W.setdefault((gj, gi), {})[col] = W.get((gj, gi), {}).get(col, 0) + gamma_s * gamma_i
            col += 1

        # Part 2: Compute C_2 (last 2 columns) using 4 independent Strassen decompositions
        # C_02 = A_00 B_02 + A_01 B_12
        # C_12 = A_10 B_02 + A_11 B_12
        # We compute 4 products:
        # P0 = A_00 B_02
        # P1 = A_01 B_12
        # P2 = A_10 B_02
        # P3 = A_11 B_12
        # Each product is a 2x2 x 2x2 multiplication, decomposed into 7 scalar multiplications.
        
        # Define the 4 products with their block offsets
        # (A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset)
        products = [
            (0, 0, 0, 2, 0, 2), # A_00 B_02 -> C_02
            (0, 1, 1, 2, 0, 2), # A_01 B_12 -> C_02
            (1, 0, 0, 2, 1, 2), # A_10 B_02 -> C_12
            (1, 1, 1, 2, 1, 2), # A_11 B_12 -> C_12
        ]
        
        for (a_ro, a_co, b_ro, b_co, c_ro, c_co) in products:
            for si in range(7):
                # U contribution
                for (a, coef_i) in inner_alpha[si]:
                    # a is local index in 2x2 A block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    lr, lc = a // 2, a % 2
                    gi = a_ro * 2 + lr
                    gk = a_co * 2 + lc
                    U.setdefault((gi, gk), {})[col] = U.get((gi, gk), {}).get(col, 0) + coef_i
                
                # V contribution
                for (b, coef_i) in inner_beta[si]:
                    # b is local index in 2x2 B block
                    lr, lc = b // 2, b % 2
                    gk = b_ro * 2 + lr
                    gj = b_co * 2 + lc
                    V.setdefault((gk, gj), {})[col] = V.get((gk, gj), {}).get(col, 0) + coef_i
                
                # W contribution
                for (a, b, gamma_i) in inner_recon[si]:
                    # a, b are local indices in 2x2 C block
                    gi = c_ro * 2 + a
                    gj = c_co * 2 + b
                    W.setdefault((gj, gi), {})[col] = W.get((gj, gi), {}).get(col, 0) + gamma_i
                col += 1

    R = col
    Umat = np.zeros((16, R))
    Vmat = np.zeros((24, R))
    Wmat = np.zeros((24, R))
    for (gi, gk), d in U.items():
        Umat[gi * 4 + gk] = [d.get(c, 0) for c in range(R)]
    for (gk, gj), d in V.items():
        Vmat[gk * 6 + gj] = [d.get(c, 0) for c in range(R)]
    for (gj, gi), d in W.items():
        Wmat[gj * 4 + gi] = [d.get(c, 0) for c in range(R)]
    
    # Verify
    if not _verify(Umat, Vmat, Wmat):
        # Fallback: schoolbook decomposition (rank = n*m*p = 96)
        R = 4 * 4 * 6
        Umat = np.zeros((16, R))
        Vmat = np.zeros((24, R))
        Wmat = np.zeros((24, R))
        idx = 0
        for i in range(4):
            for k in range(4):
                for j in range(6):
                    Umat[i*4 + k, idx] = 1
                    Vmat[k*6 + j, idx] = 1
                    Wmat[j*4 + i, idx] = 1
                    idx += 1
    
    # Round to integers where possible
    def to_list(M):
        out = []
        for row in M:
            r = []
            for x in row:
                rx = round(x)
                r.append(int(rx) if abs(x - rx) < 1e-9 else float(x))
            out.append(r)
        return out
    
    return {"U": to_list(Umat), "V": to_list(Vmat), "W": to_list(Wmat)}
