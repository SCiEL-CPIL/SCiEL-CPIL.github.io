"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x4 matrix by a 4x7 matrix.
    
    Uses a block decomposition strategy:
    - Partition A (3x4) into A_top (2x4) and A_bot (1x4).
    - Partition B (4x7) into four column blocks: B1 (4x2), B2 (4x2), B3 (4x2), B4 (4x1).
    - Use Strassen (rank 7) for 2x2x2 blocks.
    - Use Winograd (rank 4) for 2x2x1 blocks.
    - Use Standard (rank 2) for 1x2x2 and 1x2x1 blocks.
    
    Total Rank:
    - 3 blocks of (2x4 x 4x2) -> 2 * (2x2 x 2x2) Strassen -> 2 * 7 = 14 each -> 42
    - 1 block of (2x4 x 4x1) -> 2 * (2x2 x 2x1) Winograd -> 2 * 4 = 8
    - 3 blocks of (1x4 x 4x2) -> 2 * (1x2 x 2x2) Standard -> 2 * 2 = 4 each -> 12
    - 1 block of (1x4 x 4x1) -> 2 * (1x2 x 2x1) Standard -> 2 * 2 = 4
    Total Rank = 42 + 8 + 12 + 4 = 66.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    total_rank = 66
    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_rank1(a_idx, b_idx, c_idx, u_val, v_val, w_val):
        nonlocal col
        U[a_idx, col] = u_val
        V[b_idx, col] = v_val
        W[c_idx, col] = w_val
        col += 1

    # Helper to add Strassen 2x2x2
    # A indices: i in {0,1}, j in {j0, j0+1}
    # B indices: j in {j0, j0+1}, k in {k0, k0+1}
    # C indices: i in {0,1}, k in {k0, k0+1}
    def add_strassen(i0, j0, k0):
        nonlocal col
        # A elements:
        # a11 = U[i0*m + j0, r], a12 = U[i0*m + j0+1, r]
        # a21 = U[(i0+1)*m + j0, r], a22 = U[(i0+1)*m + j0+1, r]
        # B elements:
        # b11 = V[j0*p + k0, r], b12 = V[j0*p + k0+1, r]
        # b21 = V[(j0+1)*p + k0, r], b22 = V[(j0+1)*p + k0+1, r]
        # C elements:
        # c11 = W[k0*n + i0, r], c12 = W[(k0+1)*n + i0, r]
        # c21 = W[k0*n + i0+1, r], c22 = W[(k0+1)*n + i0+1, r]
        
        # M1 = (a11+a22)(b11+b22)
        add_rank1(i0*m+j0, j0*p+k0, k0*n+i0, 1, 1, 1)
        add_rank1((i0+1)*m+j0+1, (j0+1)*p+k0+1, k0*n+i0, 1, 1, 1)
        add_rank1(i0*m+j0, (j0+1)*p+k0+1, k0*n+i0, 1, 1, 1)
        add_rank1((i0+1)*m+j0+1, j0*p+k0, k0*n+i0, 1, 1, 1)
        # Wait, Strassen is:
        # M1 = (a11+a22)(b11+b22)
        # U: a11 gets 1, a22 gets 1. V: b11 gets 1, b22 gets 1. W: c11 gets 1, c22 gets -1? No.
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        
        # Let's map the 7 multiplications to the columns.
        # r=0: M1
        # U: a11=1, a22=1. V: b11=1, b22=1. W: c11=1, c22=1.
        # But we need to add contributions to multiple W indices.
        # The decomposition is:
        # T = sum_r U_r * V_r * W_r
        # So for each r, we set U, V, W vectors.
        
        # M1 = (a11+a22)(b11+b22)
        # U: a11=1, a22=1
        # V: b11=1, b22=1
        # W: c11=1, c22=1
        U[i0*m+j0, col] = 1; U[(i0+1)*m+j0+1, col] = 1
        V[j0*p+k0, col] = 1; V[(j0+1)*p+k0+1, col] = 1
        W[k0*n+i0, col] = 1; W[(k0+1)*n+i0+1, col] = 1
        col += 1
        
        # M2 = (a21+a22)b11
        # U: a21=1, a22=1
        # V: b11=1
        # W: c21=1, c22=-1
        U[(i0+1)*m+j0, col] = 1; U[(i0+1)*m+j0+1, col] = 1
        V[j0*p+k0, col] = 1
        W[k0*n+i0+1, col] = 1; W[(k0+1)*n+i0+1, col] = -1
        col += 1
        
        # M3 = a11(b12-b22)
        # U: a11=1
        # V: b12=1, b22=-1
        # W: c12=1, c22=1
        U[i0*m+j0, col] = 1
        V[j0*p+k0+1, col] = 1; V[(j0+1)*p+k0+1, col] = -1
        W[(k0+1)*n+i0, col] = 1; W[(k0+1)*n+i0+1, col] = 1
        col += 1
        
        # M4 = a22(b21-b11)
        # U: a22=1
        # V: b21=1, b11=-1
        # W: c11=-1, c21=1
        U[(i0+1)*m+j0+1, col] = 1
        V[(j0+1)*p+k0, col] = 1; V[j0*p+k0, col] = -1
        W[k0*n+i0, col] = -1; W[k0*n+i0+1, col] = 1
        col += 1
        
        # M5 = (a11+a12)b22
        # U: a11=1, a12=1
        # V: b22=1
        # W: c11=-1, c12=1
        U[i0*m+j0, col] = 1; U[i0*m+j0+1, col] = 1
        V[(j0+1)*p+k0+1, col] = 1
        W[k0*n+i0, col] = -1; W[(k0+1)*n+i0, col] = 1
        col += 1
        
        # M6 = (a21-a11)(b11+b12)
        # U: a21=1, a11=-1
        # V: b11=1, b12=1
        # W: c22=1
        U[(i0+1)*m+j0, col] = 1; U[i0*m+j0, col] = -1
        V[j0*p+k0, col] = 1; V[j0*p+k0+1, col] = 1
        W[(k0+1)*n+i0+1, col] = 1
        col += 1
        
        # M7 = (a12-a22)(b21+b22)
        # U: a12=1, a22=-1
        # V: b21=1, b22=1
        # W: c11=1
        U[i0*m+j0+1, col] = 1; U[(i0+1)*m+j0+1, col] = -1
        V[(j0+1)*p+k0, col] = 1; V[(j0+1)*p+k0+1, col] = 1
        W[k0*n+i0, col] = 1
        col += 1

    # Helper to add Winograd 2x2x1
    # A: i in {0,1}, j in {j0, j0+1}
    # B: j in {j0, j0+1}, k=k0
    # C: i in {0,1}, k=k0
    def add_winograd_2x2x1(i0, j0, k0):
        nonlocal col
        # M1 = (a11+a21)b1
        # U: a11=1, a21=1. V: b1=1. W: c1=1, c2=0? No.
        # c1 = M1 + M2 - M3
        # c2 = M3 + M4
        # M1 = (a11+a21)b1
        # M2 = (a12+a22)b2
        # M3 = a11*b1
        # M4 = a22*b2
        
        # r=0: M1
        U[i0*m+j0, col] = 1; U[(i0+1)*m+j0, col] = 1
        V[j0*p+k0, col] = 1
        W[k0*n+i0, col] = 1
        col += 1
        
        # r=1: M2
        U[i0*m+j0+1, col] = 1; U[(i0+1)*m+j0+1, col] = 1
        V[(j0+1)*p+k0, col] = 1
        W[k0*n+i0, col] = 1
        col += 1
        
        # r=2: M3
        U[i0*m+j0, col] = 1
        V[j0*p+k0, col] = 1
        W[k0*n+i0, col] = -1; W[k0*n+i0+1, col] = 1
        col += 1
        
        # r=3: M4
        U[(i0+1)*m+j0+1, col] = 1
        V[(j0+1)*p+k0, col] = 1
        W[k0*n+i0+1, col] = 1
        col += 1

    # Helper to add Standard 1x2x2
    # A: i=i0, j in {j0, j0+1}
    # B: j in {j0, j0+1}, k in {k0, k0+1}
    # C: i=i0, k in {k0, k0+1}
    def add_std_1x2x2(i0, j0, k0):
        nonlocal col
        # c1 = a1*b11 + a2*b21
        # c2 = a1*b12 + a2*b22
        # r=0: a1*b11 -> c1
        U[i0*m+j0, col] = 1
        V[j0*p+k0, col] = 1
        W[k0*n+i0, col] = 1
        col += 1
        # r=1: a2*b21 -> c1
        U[i0*m+j0+1, col] = 1
        V[(j0+1)*p+k0, col] = 1
        W[k0*n+i0, col] = 1
        col += 1
        # r=2: a1*b12 -> c2
        U[i0*m+j0, col] = 1
        V[j0*p+k0+1, col] = 1
        W[(k0+1)*n+i0, col] = 1
        col += 1
        # r=3: a2*b22 -> c2
        U[i0*m+j0+1, col] = 1
        V[(j0+1)*p+k0+1, col] = 1
        W[(k0+1)*n+i0, col] = 1
        col += 1

    # Helper to add Standard 1x2x1
    # A: i=i0, j in {j0, j0+1}
    # B: j in {j0, j0+1}, k=k0
    # C: i=i0, k=k0
    def add_std_1x2x1(i0, j0, k0):
        nonlocal col
        # c1 = a1*b1 + a2*b2
        # r=0: a1*b1
        U[i0*m+j0, col] = 1
        V[j0*p+k0, col] = 1
        W[k0*n+i0, col] = 1
        col += 1
        # r=1: a2*b2
        U[i0*m+j0+1, col] = 1
        V[(j0+1)*p+k0, col] = 1
        W[k0*n+i0, col] = 1
        col += 1

    # 1. A_top (i=0,1) * B1 (k=0,1) -> C_top (i=0,1, k=0,1)
    # Split j into 0,1 and 2,3
    add_strassen(0, 0, 0)
    add_strassen(0, 2, 0)
    
    # 2. A_top (i=0,1) * B2 (k=2,3) -> C_top (i=0,1, k=2,3)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 2)
    
    # 3. A_top (i=0,1) * B3 (k=4,5) -> C_top (i=0,1, k=4,5)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    # 4. A_top (i=0,1) * B4 (k=6) -> C_top (i=0,1, k=6)
    # Split j into 0,1 and 2,3
    add_winograd_2x2x1(0, 0, 6)
    add_winograd_2x2x1(0, 2, 6)
    
    # 5. A_bot (i=2) * B1 (k=0,1) -> C_bot (i=2, k=0,1)
    # Split j into 0,1 and 2,3
    add_std_1x2x2(2, 0, 0)
    add_std_1x2x2(2, 2, 0)
    
    # 6. A_bot (i=2) * B2 (k=2,3) -> C_bot (i=2, k=2,3)
    add_std_1x2x2(2, 0, 2)
    add_std_1x2x2(2, 2, 2)
    
    # 7. A_bot (i=2) * B3 (k=4,5) -> C_bot (i=2, k=4,5)
    add_std_1x2x2(2, 0, 4)
    add_std_1x2x2(2, 2, 4)
    
    # 8. A_bot (i=2) * B4 (k=6) -> C_bot (i=2, k=6)
    add_std_1x2x1(2, 0, 6)
    add_std_1x2x1(2, 2, 6)
    
    # Verify
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.max(np.abs(reconstructed - T))
    
    if error < 1e-10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if construction failed
        rank_sb = n * m * p
        U_sb = np.zeros((nm, rank_sb))
        V_sb = np.zeros((mp, rank_sb))
        W_sb = np.zeros((pn, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_sb[a, col] = 1.0
                    V_sb[b, col] = 1.0
                    W_sb[c, col] = 1.0
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
