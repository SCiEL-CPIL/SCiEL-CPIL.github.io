"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    # For each (i,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all j
    # Actually the tensor condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # But that's not right either. The correct condition is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # With decomposition: C[i,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k]) * W[k*n+i,r]
    # This must equal sum_j A[i,j]*B[j,k] for all A,B.
    # So: sum_r U[i*m+j,r] * V[j'*p+k,r] * W[k*n+i,r] = delta_{j,j'} for all i,j,j',k
    # Wait no. Let me re-derive.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Decomposition: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k])
    # = sum_r W[k*n+i,r] * sum_j sum_j' U[i*m+j,r] V[j'*p+k,r] A[i,j] B[j',k]
    # For this to equal sum_j A[i,j] B[j,k] for all A,B:
    # sum_r W[k*n+i,r] * U[i*m+j,r] * V[j*p+k,r] = 1 for all i,j,k
    # (and the cross terms must vanish, but since A and B are independent, we need the coefficient of each A[i,j]*B[j',k] to match)
    # Actually: coefficient of A[i,j]*B[j',k] in the decomposition is sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k*n+i,r]
    # This must equal delta_{j,j'} (1 if j=j', 0 otherwise)
    
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += Ua[i*m+j, r] * Va[j2*p+k, r] * Wa[k*n+i, r]
                    expected = 1.0 if j == j2 else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True


def _build_decomposition():
    """
    Block decomposition for 3x3 @ 3x7 using Strassen on 2x2 blocks.
    
    Strategy:
    - Split A into A_top (2x3) and A_bottom (1x3)
    - Split B into B_top (2x7) and B_bottom (1x7)
    - C = A_top @ B_top + A_bottom @ B_bottom
    
    For A_top @ B_top (2x3 @ 3x7):
      Split into columns of A_top: A_top[:,0:2] (2x2) and A_top[:,2] (2x1)
      C_top = A_top[:,:2] @ B_top[:2,:] + A_top[:,2] @ B_top[2,:]
      
      A_top[:,:2] @ B_top[:2,:]: 2x2 @ 2x7
        Split B_top[:2,:] into 2x2 blocks: cols 0-1, 2-3, 4-5, and col 6
        Use Strassen for each 2x2 @ 2x2: 3 * 7 = 21
        Schoolbook for 2x2 @ 2x1: 4
        Subtotal: 25
        
      A_top[:,2] @ B_top[2,:]: 2x1 @ 1x7, schoolbook rank 7
      
    For A_bottom @ B_bottom (1x3 @ 3x7): schoolbook rank 3
    
    Total: 25 + 7 + 3 = 35
    
    But wait, I need to be more careful. Let me use a cleaner approach.
    """
    n, m, p = 3, 3, 7
    
    U = []
    V = []
    W = []
    
    def make_U(indices):
        u = [0.0] * (n * m)
        for idx, val in indices:
            u[idx] = val
        return u
    
    def make_V(indices):
        v = [0.0] * (m * p)
        for idx, val in indices:
            v[idx] = val
        return v
    
    def make_W(indices):
        w = [0.0] * (p * n)
        for idx, val in indices:
            w[idx] = val
        return w
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    def strassen_2x2(a_rows, a_cols, b_rows, b_cols, c_row_idx, c_col_idx):
        """
        Strassen for 2x2 @ 2x2 subproblem.
        a_rows: [r0, r1] - rows of A in the full matrix
        a_cols: [c0, c1] - cols of A
        b_rows: [r0, r1] - rows of B
        b_cols: [c0, c1] - cols of B
        c_row_idx: [r0, r1] - which C rows (i indices) this contributes to
        c_col_idx: [c0, c1] - which C cols (k indices) this contributes to
        
        A submatrix: A[a_rows[0], a_cols[0:2]], A[a_rows[1], a_cols[0:2]]
        B submatrix: B[b_rows[0:2], b_cols[0:2]]
        C contribution: C[c_row_idx, c_col_idx] += A_sub @ B_sub
        """
        ar0, ar1 = a_rows
        ac0, ac1 = a_cols
        br0, br1 = b_rows
        bc0, bc1 = b_cols
        cr0, cr1 = c_row_idx
        cc0, cc1 = c_col_idx
        
        # U indices for A elements
        ua11 = u_idx(ar0, ac0)
        ua12 = u_idx(ar0, ac1)
        ua21 = u_idx(ar1, ac0)
        ua22 = u_idx(ar1, ac1)
        
        # V indices for B elements
        vb11 = v_idx(br0, bc0)
        vb12 = v_idx(br0, bc1)
        vb21 = v_idx(br1, bc0)
        vb22 = v_idx(br1, bc1)
        
        # W indices for C elements
        wc11 = w_idx(cc0, cr0)
        wc12 = w_idx(cc1, cr0)
        wc21 = w_idx(cc0, cr1)
        wc22 = w_idx(cc1, cr1)
        
        # M1 = (a11+a22)(b11+b22) -> contributes to C11 and C22
        U.append(make_U([(ua11, 1), (ua22, 1)]))
        V.append(make_V([(vb11, 1), (vb22, 1)]))
        W.append(make_W([(wc11, 1), (wc22, 1)]))
        
        # M2 = (a21+a22)b11 -> contributes to C21 and -C22
        U.append(make_U([(ua21, 1), (ua22, 1)]))
        V.append(make_V([(vb11, 1)]))
        W.append(make_W([(wc21, 1), (wc22, -1)]))
        
        # M3 = a11(b21-b22) -> contributes to C12 and C22
        U.append(make_U([(ua11, 1)]))
        V.append(make_V([(vb21, 1), (vb22, -1)]))
        W.append(make_W([(wc12, 1), (wc22, 1)]))
        
        # M4 = a22(b12-b11) -> contributes to C11 and C21
        U.append(make_U([(ua22, 1)]))
        V.append(make_V([(vb12, 1), (vb11, -1)]))
        W.append(make_W([(wc11, 1), (wc21, 1)]))
        
        # M5 = (a11+a12)b22 -> contributes to -C11 and C12
        U.append(make_U([(ua11, 1), (ua12, 1)]))
        V.append(make_V([(vb22, 1)]))
        W.append(make_W([(wc11, -1), (wc12, 1)]))
        
        # M6 = (a21-a11)(b11+b12) -> contributes to C22
        U.append(make_U([(ua21, 1), (ua11, -1)]))
        V.append(make_V([(vb11, 1), (vb12, 1)]))
        W.append(make_W([(wc22, 1)]))
        
        # M7 = (a12-a22)(b21+b22) -> contributes to C11 and C22
        U.append(make_U([(ua12, 1), (ua22, -1)]))
        V.append(make_V([(vb21, 1), (vb22, 1)]))
        W.append(make_W([(wc11, 1), (wc22, 1)]))
    
    # Part 1: A_top (rows 0,1) @ B_top (rows 0,1,2)
    # = A_top[:,0:2] @ B_top[0:2,:] + A_top[:,2] @ B_top[2,:]
    
    # A_top[:,0:2] @ B_top[0:2,:]: 2x2 @ 2x7
    # Split B_top[0:2,:] into cols [0,1], [2,3], [4,5], [6]
    # Strassen for 2x2 @ 2x2 on cols [0,1], [2,3], [4,5]
    strassen_2x2(a_rows=[0,1], a_cols=[0,1], b_rows=[0,1], b_cols=[0,1], c_row_idx=[0,1], c_col_idx=[0,1])
    strassen_2x2(a_rows=[0,1], a_cols=[0,1], b_rows=[0,1], b_cols=[2,3], c_row_idx=[0,1], c_col_idx=[2,3])
    strassen_2x2(a_rows=[0,1], a_cols=[0,1], b_rows=[0,1], b_cols=[4,5], c_row_idx=[0,1], c_col_idx=[4,5])
    
    # Schoolbook for 2x2 @ 2x1 on col [6]
    for i in [0, 1]:
        for j in [0, 1]:
            U.append(make_U([(u_idx(i, j), 1)]))
            V.append(make_V([(v_idx(j, 6), 1)]))
            W.append(make_W([(w_idx(6, i), 1)]))
    
    # A_top[:,2] @ B_top[2,:]: 2x1 @ 1x7, schoolbook rank 7
    for i in [0, 1]:
        for k in range(7):
            U.append(make_U([(u_idx(i, 2), 1)]))
            V.append(make_V([(v_idx(2, k), 1)]))
            W.append(make_W([(w_idx(k, i), 1)]))
    
    # Part 2: A_bottom (row 2) @ B_bottom (row 2)
    # 1x3 @ 3x7: schoolbook rank 3
    for k in range(7):
        U.append(make_U([(u_idx(2, 2), 1)]))
        V.append(make_V([(v_idx(2, k), 1)]))
        W.append(make_W([(w_idx(k, 2), 1)]))
    
    # Wait, I'm missing A_bottom @ B_top and A_top @ B_bottom!
    # C = A_top @ B_top + A_bottom @ B_bottom is WRONG.
    # C = A @ B = sum_j A[:,j] @ B[j,:]
    # = A[:,0] @ B[0,:] + A[:,1] @ B[1,:] + A[:,2] @ B[2,:]
    
    # Let me restart with the correct decomposition.
    U = []
    V = []
    W = []
    
    # C = A[:,0] @ B[0,:] + A[:,1] @ B[1,:] + A[:,2] @ B[2,:]
    # Each term is (3x1) @ (1x7) = 3x7, rank 7 each in schoolbook.
    # Total schoolbook: 21. But we can do better by grouping.
    
    # Group: (A[:,0:2] @ B[0:2,:]) + (A[:,2] @ B[2,:])
    # A[:,0:2] @ B[0:2,:]: (3x2) @ (2x7)
    #   = A[0:2,0:2] @ B[0:2,:2] + A[0:2,0:2] @ B[0:2,2:4] + A[0:2,0:2] @ B[0:2,4:6] 
    #     + A[0:2,0:2] @ B[0:2,6:7] + A[2,0:2] @ B[0:2,:]
    #   A[0:2,0:2] @ B[0:2,0:2]: 2x2 @ 2x2, Strassen rank 7
    #   A[0:2,0:2] @ B[0:2,2:4]: 2x2 @ 2x2, Strassen rank 7
    #   A[0:2,0:2] @ B[0:2,4:6]: 2x2 @ 2x2, Strassen rank 7
    #   A[0:2,0:2] @ B[0:2,6:7]: 2x2 @ 2x1, schoolbook rank 4
    #   A[2,0:2] @ B[0:2,0:7]: 1x2 @ 2x7, schoolbook rank 14
    #   Subtotal: 7+7+7+4+14 = 39
    
    # A[:,2] @ B[2,:]: 3x1 @ 1x7, schoolbook rank 7
    
    # Total: 39 + 7 = 46
    
    # Hmm, that's not great. Let me think differently.
    
    # Actually, the cleanest approach:
    # C = A @ B where A is 3x3, B is 3x7.
    # Use Strassen on the 2x2 block of A (rows 0,1; cols 0,1) and corresponding B blocks.
    
    # Let me just implement a clean version:
    # Split A into:
    #   A1 = A[0:2, 0:2] (2x2)
    #   A2 = A[0:2, 2] (2x1)
    #   A3 = A[2, 0:2] (1x2)
    #   A4 = A[2, 2] (1x1)
    # Split B into:
    #   B1 = B[0:2, :] (2x7)
    #   B2 = B[2, :] (1x7)
    # C = A1 @ B1[0:2,:] + A2 @ B2 + A3 @ B1 + A4 @ B2
    # Wait, that's not right either. Let me be very explicit.
    
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # Group by i:
    # For i=0,1: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # For i=2: C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k]
    
    # For i=0,1: use Strassen on the 2x2 part (A[0:2,0:2] @ B[0:2,:])
    #   But B[0:2,:] is 2x7, so split into 2x2 blocks.
    # C[0:2,k] = (A[0:2,0:2] @ B[0:2,:])[:,k] + A[0:2,2]*B[2,k]
    # A[0:2,0:2] @ B[0:2,:]: 2x2 @ 2x7
    #   Split B[0:2,:] into B[0:2,0:2], B[0:2,2:4], B[0:2,4:6], B[0:2,6:7]
    #   Strassen for first three (2x2 @ 2x2 each): 3*7 = 21
    #   Schoolbook for last (2x2 @ 2x1): 4
    #   Subtotal: 25
    # A[0:2,2]*B[2,k]: 2x1 @ 1x7, schoolbook rank 7
    # Subtotal for i=0,1: 25 + 7 = 32
    
    # For i=2: C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k]
    # Schoolbook rank 7 (or can we do better? 1x3 @ 3x7 is just sum of 3 rank-1 terms)
    # Actually 1x3 @ 3x7: C[2,k] = sum_j A[2,j]*B[j,k]. This is inherently rank 3 as a tensor slice.
    # But in the full tensor, each (j,k) pair gives a rank-1 term. So rank 7.
    
    # Total: 32 + 7 = 39
    
    # Let me implement this.
    U = []
    V = []
    W = []
    
    # Part 1: A[0:2,0:2] @ B[0:2,:] for C[0:2,:]
    # Strassen for 2x2 @ 2x2 on B cols [0,1]
    strassen_2x2(a_rows=[0,1], a_cols=[0,1], b_rows=[0,1], b_cols=[0,1], c_row_idx=[0,1], c_col_idx=[0,1])
    # Strassen for 2x2 @ 2x2 on B cols [2,3]
    strassen_2x2(a_rows=[0,1], a_cols=[0,1], b_rows=[0,1], b_cols=[2,3], c_row_idx=[0,1], c_col_idx=[2,3])
    # Strassen for 2x2 @ 2x2 on B cols [4,5]
    strassen_2x2(a_rows=[0,1], a_cols=[0,1], b_rows=[0,1], b_cols=[4,5], c_row_idx=[0,1], c_col_idx=[4,5])
    # Schoolbook for 2x2 @ 2x1 on B col [6]
    for i in [0, 1]:
        for j in [0, 1]:
            U.append(make_U([(u_idx(i, j), 1)]))
            V.append(make_V([(v_idx(j, 6), 1)]))
            W.append(make_W([(w_idx(6, i), 1)]))
    
    # Part 2: A[0:2,2] * B[2,:] for C[0:2,:]
    for i in [0, 1]:
        for k in range(7):
            U.append(make_U([(u_idx(i, 2), 1)]))
            V.append(make_V([(v_idx(2, k), 1)]))
            W.append(make_W([(w_idx(k, i), 1)]))
    
    # Part 3: A[2,:] * B[:,k] for C[2,:]
    for j in range(3):
        for k in range(7):
            U.append(make_U([(u_idx(2, j), 1)]))
            V.append(make_V([(v_idx(j, k), 1)]))
            W.append(make_W([(w_idx(k, 2), 1)]))
    
    return U, V, W


def construct():
    start = time.time()
    deadline = start + 115  # safety margin
    
    n, m, p = DIMENSIONS
    
    try:
        U, V, W = _build_decomposition()
        rank = len(U)
        
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
