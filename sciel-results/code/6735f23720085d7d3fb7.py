"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import math

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using numpy-style triple product check.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k.
    """
    rank = len(U[0]) if U else 0
    # Build tensor check: for each (i,j,k), sum over c
    # We'll do it with nested loops but early-exit on first failure
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            for k in range(p):
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                s = 0.0
                for c in range(rank):
                    uc = u_row[c]
                    if uc == 0:
                        continue
                    vc = v_row[c]
                    if vc == 0:
                        continue
                    s += uc * vc * w_row[c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen (rank 7).
    U rows: i*2+j, V rows: j*2+k, W rows: k*2+i
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # c=0: M1 = (A00+A11)(B00+B11) -> C00,C01,C10,C11
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[1][0]=1; W[2][0]=1; W[3][0]=1
    
    # c=1: M2 = (A10+A11)B00 -> C10
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[2][1]=1
    
    # c=2: M3 = A00(B01-B11) -> C01
    U[0][2]=1
    V[1][2]=1; V[3][2]=-1
    W[1][2]=1
    
    # c=3: M4 = A11(B10-B00) -> C00,C10
    U[3][3]=1
    V[2][3]=1; V[0][3]=-1
    W[0][3]=1; W[2][3]=1
    
    # c=4: M5 = (A00+A01)B11 -> C00(-),C01(+)
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[0][4]=-1; W[1][4]=1
    
    # c=5: M6 = (A10-A00)(B00+B01) -> C11
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1
    
    # c=6: M7 = (A01-A11)(B10+B11) -> C00(-),C11(+)
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1
    W[0][6]=-1; W[3][6]=1
    
    return U, V, W

def _build_tensor_product(U1, V1, W1, U2, V2, W2, n1, m1, p1):
    """
    Tensor product of two decompositions.
    (U1,V1,W1) for (n1,m1,p1), (U2,V2,W2) for (n2,m2,p2)
    Result for (n1*n2, m1*m2, p1*p2)
    """
    n2 = len(U2) // m1 if m1 else len(U2) // p1  # derive n2 from U2 shape
    # Actually: U2 shape is (n2*m2, r2). We need to pass n2,m2,p2.
    # Let's re-derive: caller should pass dims.
    # For now, assume square: n2=m2=p2.
    # Better: pass explicitly.
    pass

def _build_strassen_strassen_schoolbook():
    """
    Build 6x6x6 decomposition as:
    - Partition 6x6 into 3x3 blocks of 2x2 (for A, B, C)
    - Use Strassen (rank 7) for 2x2 block multiplication
    - Use Strassen (rank 7) for the 3x3 super-matrix? No, 3x3 isn't 2x2.
    
    Alternative: Partition 6x6 into 2x3 blocks? 
    Let's try: partition into blocks of size 2 in first dim, 3 in second?
    That's not standard for matrix mult.
    
    Better approach: 
    6 = 2 * 3. View 6x6 as 2x2 blocks of 3x3 matrices.
    Then super-matrix is 2x2, use Strassen (rank 7) for super.
    Block multiplication is 3x3 x 3x3, use schoolbook (rank 27).
    Total rank = 7 * 27 = 189.
    
    Or: 6 = 3 * 2. View as 3x3 blocks of 2x2.
    Super-matrix is 3x3, use schoolbook (rank 27).
    Block multiplication is 2x2 x 2x2, use Strassen (rank 7).
    Total rank = 27 * 7 = 189.
    
    Both give 189. Let's use the second: 3x3 blocks of 2x2.
    """
    n, m, p = 6, 6, 6
    
    # Strassen for 2x2
    Us, Vs, Ws = _strassen_2x2()
    rs = 7
    
    # Schoolbook for 3x3
    rb = 27
    Ub = [[0]*rb for _ in range(9)]
    Vb = [[0]*rb for _ in range(9)]
    Wb = [[0]*rb for _ in range(9)]
    col = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                Ub[I*3+J][col] = 1
                Vb[J*3+K][col] = 1
                Wb[K*3+I][col] = 1
                col += 1
    
    r_full = rs * rb
    U_full = [[0] * r_full for _ in range(n * m)]
    V_full = [[0] * r_full for _ in range(m * p)]
    W_full = [[0] * r_full for _ in range(p * n)]
    
    for I in range(3):
        for J in range(3):
            for K in range(3):
                row_b_U = I * 3 + J
                row_b_V = J * 3 + K
                row_b_W = K * 3 + I
                
                for cb in range(rb):
                    ub = Ub[row_b_U][cb]
                    vb = Vb[row_b_V][cb]
                    wb = Wb[row_b_W][cb]
                    if ub == 0 and vb == 0 and wb == 0:
                        continue
                    
                    for ip in range(2):
                        for jp in range(2):
                            i = 3 * I + ip
                            j = 3 * J + jp
                            row_s_U = ip * 2 + jp
                            
                            for cp in range(rs):
                                us = Us[row_s_U][cp]
                                if us == 0:
                                    continue
                                col_full = cp * rb + cb
                                U_full[i * m + j][col_full] += us * ub
                            
                            for kp in range(2):
                                k = 3 * K + kp
                                row_s_V = jp * 2 + kp
                                
                                for cp in range(rs):
                                    vs = Vs[row_s_V][cp]
                                    if vs == 0:
                                        continue
                                    col_full = cp * rb + cb
                                    V_full[j * p + k][col_full] += vs * vb
                                    
                                row_s_W = kp * 2 + ip
                                for cp in range(rs):
                                    ws = Ws[row_s_W][cp]
                                    if ws == 0:
                                        continue
                                    col_full = cp * rb + cb
                                    W_full[k * n + i][col_full] += ws * wb
    
    return U_full, V_full, W_full

def _column_norm(U, V, W, c):
    """Compute the Frobenius norm of column c across all three matrices."""
    s = 0.0
    for row in range(len(U)):
        v = U[row][c]
        s += v * v
    for row in range(len(V)):
        v = V[row][c]
        s += v * v
    for row in range(len(W)):
        v = W[row][c]
        s += v * v
    return s

def _error_tensor(U, V, W, n, m, p):
    """Compute the error tensor: E[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1"""
    rank = len(U[0]) if U else 0
    E = [[[0.0]*p for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            u_row = U[i*m+j]
            for k in range(p):
                v_row = V[j*p+k]
                w_row = W[k*n+i]
                s = 0.0
                for c in range(rank):
                    uc = u_row[c]
                    if uc == 0: continue
                    vc = v_row[c]
                    if vc == 0: continue
                    s += uc * vc * w_row[c]
                E[i][j][k] = s - 1.0
    return E

def _try_remove_column(U, V, W, n, m, p, c_remove):
    """
    Try removing column c_remove. Compute the resulting error.
    Then try to repair by adjusting other columns.
    Returns (success, new_U, new_V, new_W)
    """
    rank = len(U[0])
    
    # Compute error after removal
    # E[i,j,k] = -U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # We need to find delta adjustments to other columns to cancel this error.
    # This is a linear system: for each (i,j,k), sum_{c != c_remove} delta_c * (U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]) = E[i,j,k]
    # This is nonlinear in deltas. Instead, use a greedy repair:
    # For each (i,j,k) with error, find a column c that can help and adjust.
    
    # Simpler approach: just check if removing the column and setting it to zero
    # still gives valid decomposition. If not, try to find a minimal repair.
    
    # Let's compute the error magnitude first
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                uc = U[i*m+j][c_remove]
                vc = V[j*p+k][c_remove]
                wc = W[k*n+i][c_remove]
                err = uc * vc * wc
                if abs(err) > max_err:
                    max_err = abs(err)
    
    if max_err < 1e-15:
        # Column is already zero, just remove it
        new_rank = rank - 1
        new_U = [[U[i][c] for c in range(rank) if c != c_remove] for i in range(len(U))]
        new_V = [[V[i][c] for c in range(rank) if c != c_remove] for i in range(len(V))]
        new_W = [[W[i][c] for c in range(rank) if c != c_remove] for i in range(len(W))]
        return True, new_U, new_V, new_W
    
    # Try repair: for each error site, find a column that can absorb it
    # This is complex. Let's use a simpler heuristic:
    # If the column has small norm, try removing and see if we can fix with nearby columns.
    
    # For now, return failure unless error is tiny
    if max_err < 1e-12:
        new_rank = rank - 1
        new_U = [[U[i][c] for c in range(rank) if c != c_remove] for i in range(len(U))]
        new_V = [[V[i][c] for c in range(rank) if c != c_remove] for i in range(len(V))]
        new_W = [[W[i][c] for c in range(rank) if c != c_remove] for i in range(len(W))]
        return True, new_U, new_V, new_W
    
    return False, None, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build initial rank-189 decomposition
    U, V, W = _build_strassen_strassen_schoolbook()
    
    # Verify initial
    if not _verify_fast(U, V, W, n, m, p):
        # Fallback to schoolbook
        return _schoolbook(n, m, p)
    
    current_rank = len(U[0])
    best_rank = current_rank
    best_result = {"U": U, "V": V, "W": W}
    
    # Iterative column removal with repair
    # Strategy: repeatedly find the column with smallest norm, try to remove it.
    # If removal creates error, try to repair by adjusting a small number of other columns.
    
    max_iterations = 500
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        if len(U) == 0 or len(U[0]) == 0:
            break
        
        rank = len(U[0])
        
        # Find column with smallest norm
        min_norm = float('inf')
        min_col = -1
        for c in range(rank):
            norm = _column_norm(U, V, W, c)
            if norm < min_norm:
                min_norm = norm
                min_col = c
        
        if min_col < 0 or min_norm > 1e-10:
            break
        
        # Try to remove this column
        success, new_U, new_V, new_W = _try_remove_column(U, V, W, n, m, p, min_col)
        
        if success:
            # Verify the new decomposition
            if _verify_fast(new_U, new_V, new_W, n, m, p):
                U, V, W = new_U, new_V, new_W
                current_rank = len(U[0])
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_result = {"U": [row[:] for row in U], 
                                   "V": [row[:] for row in V], 
                                   "W": [row[:] for row in W]}
            else:
                # Repair failed verification, revert
                break
        else:
            # Can't remove this column, mark it as non-removable
            # For simplicity, stop if we can't remove the smallest
            break
    
    # Return best result found
    if time.time() > deadline:
        return best_result
    
    return best_result

# EVOLVE-BLOCK-END
