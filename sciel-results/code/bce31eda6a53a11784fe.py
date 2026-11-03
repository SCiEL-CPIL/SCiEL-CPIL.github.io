import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    Condition: For all i, k:
    (A @ B)[i, k] = sum_r (sum_{j} U[i*m+j, r] * A[i, j]) * (sum_{j'} V[j'*p+k, r] * B[j', k]) * W[k*n+i, r]
    Wait, the standard definition is:
    C[i, k] = sum_j A[i, j] B[j, k]
    
    In tensor decomposition form:
    T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r] ... No.
    
    The bilinear form is:
    C = sum_r (u_r(A)) (v_r(B)) w_r
    where u_r(A) is a scalar, v_r(B) is a scalar, and w_r is a matrix (or vector of matrix entries).
    
    Specifically:
    C[i, k] = sum_r (sum_{a,b} U[a*m+b, r] A[a,b]) * (sum_{c,d} V[c*p+d, r] B[c,d]) * W[k*n+i, r]
    
    Let's check indices:
    U index: a*m + b. a in [0, n), b in [0, m).
    V index: c*p + d. c in [0, m), d in [0, p).
    W index: k*n + i. k in [0, p), i in [0, n).
    
    So C[i, k] = sum_r alpha_r * beta_r * W[k*n+i, r]
    where alpha_r = sum_{a,b} U[a*m+b, r] A[a,b]
          beta_r = sum_{c,d} V[c*p+d, r] B[c,d]
    
    This must equal sum_{j} A[i, j] B[j, k].
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    for _ in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        
        # alpha_r = U^T vec(A)
        # vec(A) is (n*m,). U is (n*m, rank).
        # U.T is (rank, n*m).
        alpha = U.T @ A.flatten()
        beta = V.T @ B.flatten()
        
        # C_comp[i, k] = sum_r alpha[r] * beta[r] * W[k*n + i, r]
        # Let's compute this efficiently.
        # prod = alpha * beta (rank,)
        # C_comp_flat[i, k] = dot(prod, W[k*n+i, :])
        # We can reshape W to (p, n, rank)
        W_reshaped = W.reshape(p, n, -1)
        # prod is (rank,)
        # We want C_comp[i, k] = sum_r W_reshaped[k, i, r] * prod[r]
        # This is equivalent to einsum('k,i,r,r->i,k', W_reshaped, prod)
        C_comp = np.einsum('k,i,r,r->i,k', W_reshaped, prod)
        
        if not np.allclose(C_comp, C_true, atol=tol):
            return False
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    n=2, m=2, p=2. Rank = 7.
    
    Indices:
    U: (4, 7) -> [i*m+j, r]
    V: (4, 7) -> [j*p+k, r]
    W: (4, 7) -> [k*n+i, r]
    
    Strassen formulas:
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Let's construct U, V, W.
    U[i*m+j, r] corresponds to coefficient of A[i,j] in linear form r.
    V[j*p+k, r] corresponds to coefficient of B[j,k] in linear form r.
    W[k*n+i, r] corresponds to coefficient of C[i,k] in linear form r.
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # r=0: M1 = (A11 + A22) * (B11 + B22)
    # A11 is index 0*2+0=0, A22 is 1*2+1=3
    # B11 is index 0*2+0=0, B22 is 1*2+1=3
    # C11 (0,0) gets +1, C22 (1,1) gets +1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0*2+0, 0] = 1  # C11 (k=0, i=0)
    W[1*2+1, 0] = 1  # C22 (k=1, i=1)
    
    # r=1: M2 = (A21 + A22) * B11
    # A21 is 1*2+0=2, A22 is 3
    # B11 is 0
    # C21 (1,0) gets +1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[0*2+1, 1] = 1  # C21 (k=0, i=1)
    
    # r=2: M3 = A11 * (B12 - B22)
    # A11 is 0
    # B12 is 1, B22 is 3
    # C12 (0,1) gets +1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1*2+0, 2] = 1  # C12 (k=1, i=0)
    
    # r=3: M4 = A22 * (B21 - B11)
    # A22 is 3
    # B21 is 2, B11 is 0
    # C11 (0,0) gets +1, C21 (1,0) gets +1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0*2+0, 3] = 1  # C11
    W[0*2+1, 3] = 1  # C21
    
    # r=4: M5 = (A11 + A12) * B22
    # A11 is 0, A12 is 1
    # B22 is 3
    # C11 (0,0) gets -1, C12 (0,1) gets +1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0*2+0, 4] = -1 # C11
    W[1*2+0, 4] = 1  # C12
    
    # r=5: M6 = (A21 - A11) * (B11 + B12)
    # A21 is 2, A11 is 0
    # B11 is 0, B12 is 1
    # C22 (1,1) gets +1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[1*2+1, 5] = 1  # C22
    
    # r=6: M7 = (A12 - A22) * (B21 + B22)
    # A12 is 1, A22 is 3
    # B21 is 2, B22 is 3
    # C11 (0,0) gets +1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0*2+0, 6] = 1  # C11
    
    return U, V, W, rank

def _schoolbook_block(n, m, p):
    """
    Returns U, V, W for schoolbook multiplication of n x m by m x p.
    Rank = n*m*p.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank

def _combine_decompositions(blocks):
    """
    Combines a list of decompositions.
    Each block is a tuple (U, V, W, row_offsets_A, row_offsets_B, col_offsets_C, row_offsets_C)
    Actually, it's easier to just concatenate if we handle the index mapping.
    
    Let's define the structure more clearly.
    We are decomposing C = A @ B.
    A is (n, m), B is (m, p), C is (n, p).
    
    We split A into horizontal strips? No, usually we split A into vertical strips and B into horizontal strips?
    Or we can split A into blocks A1, A2 and B into blocks B1, B2 such that C = A1 B1 + A2 B2.
    
    Let's use a general block multiplication scheme.
    A = [A_top; A_bottom] (n1+m1 = n)
    B = [B_left B_right] (p1+p2 = p)
    This doesn't align dimensions for multiplication directly unless we split the middle dimension m.
    
    Correct block multiplication:
    Split m into m1 + m2.
    A = [A1 A2] where A1 is (n, m1), A2 is (n, m2).
    B = [B1; B2] where B1 is (m1, p), B2 is (m2, p).
    C = A1 B1 + A2 B2.
    
    So we can decompose the multiplication into two independent multiplications:
    C1 = A1 B1 (n x m1 by m1 x p -> n x p)
    C2 = A2 B2 (n x m2 by m2 x p -> n x p)
    
    The total rank is rank(C1) + rank(C2).
    The U, V, W matrices can be constructed by placing the decompositions of C1 and C2 side-by-side.
    
    U_total:
    Rows correspond to A. A is (n, m).
    For C1, U1 has rows (n, m1). These correspond to the first m1 columns of A.
    For C2, U2 has rows (n, m2). These correspond to the last m2 columns of A.
    So U_total is a block matrix:
    [ U1  0 ]
    [ 0   U2 ]
    Wait, U1 is (n*m1, r1), U2 is (n*m2, r2).
    The full U is (n*m, r1+r2).
    The first n*m1 rows of U_total correspond to A1.
    The last n*m2 rows of U_total correspond to A2.
    
    Similarly for V:
    V1 is (m1*p, r1), V2 is (m2*p, r2).
    V_total is (m*p, r1+r2).
    First m1*p rows for B1.
    Last m2*p rows for B2.
    
    Similarly for W:
    W1 is (p*n, r1), W2 is (p*n, r2).
    W_total is (p*n, r1+r2).
    Columns are just concatenated.
    
    Let's verify this logic.
    C = C1 + C2.
    C1 = sum_{r in r1} alpha1_r beta1_r W1_r
    C2 = sum_{r in r2} alpha2_r beta2_r W2_r
    alpha1_r depends on A1. alpha2_r depends on A2.
    beta1_r depends on B1. beta2_r depends on B2.
    
    In the combined U:
    For r in r1, U_total[:, r] has non-zeros only in the A1 part (first n*m1 rows).
    For r in r2, U_total[:, r] has non-zeros only in the A2 part (last n*m2 rows).
    
    This works perfectly.
    """
    pass # Placeholder, logic implemented in construct

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Strategy: Split m=4 into m1=2, m2=2.
    # A = [A1 A2], A1 is 3x2, A2 is 3x2.
    # B = [B1; B2], B1 is 2x7, B2 is 2x7.
    # C = A1 B1 + A2 B2.
    
    # Decomposition 1: C1 = A1 B1
    # A1 is 3x2, B1 is 2x7.
    # We can further split A1 into 2x2 and 1x2 blocks?
    # Let's split n=3 into n1=2, n2=1 for A1?
    # A1 = [A1_top; A1_bot] where A1_top is 2x2, A1_bot is 1x2.
    # B1 is 2x7.
    # C1 = A1_top B1_top_part? No, B1 is not split vertically.
    # C1 = A1 @ B1.
    # We can use Strassen for the 2x2 part of A1 if we split B1?
    # B1 is 2x7. We can split p=7 into p1=2, p2=5? Or p1=3, p2=4?
    # Let's stick to a simpler first step:
    # Split m=4 into 2+2.
    # Rank 1: 3x2 by 2x7. Schoolbook rank = 3*2*7 = 42.
    # Rank 2: 3x2 by 2x7. Schoolbook rank = 3*2*7 = 42.
    # Total = 84. No gain.
    
    # We need to apply Strassen to 2x2x2 blocks.
    # To get 2x2x2, we need n=2, m=2, p=2.
    
    # Let's split n=3 into 2+1.
    # Let's split m=4 into 2+2.
    # Let's split p=7 into 2+5.
    
    # Block 1: A[0:2, 0:2] (2x2) * B[0:2, 0:2] (2x2) -> C[0:2, 0:2] (2x2)
    # Use Strassen. Rank 7.
    
    # Block 2: A[0:2, 0:2] (2x2) * B[0:2, 2:7] (2x5) -> C[0:2, 2:7] (2x5)
    # Use Schoolbook. Rank 2*2*5 = 20.
    
    # Block 3: A[0:2, 2:4] (2x2) * B[2:4, 0:2] (2x2) -> C[0:2, 0:2] (2x2)
    # Use Strassen. Rank 7.
    
    # Block 4: A[0:2, 2:4] (2x2) * B[2:4, 2:7] (2x5) -> C[0:2, 2:7] (2x5)
    # Use Schoolbook. Rank 2*2*5 = 20.
    
    # Block 5: A[2:3, 0:2] (1x2) * B[0:2, 0:2] (2x2) -> C[2:3, 0:2] (1x2)
    # Use Schoolbook. Rank 1*2*2 = 4.
    
    # Block 6: A[2:3, 0:2] (1x2) * B[0:2, 2:7] (2x5) -> C[2:3, 2:7] (1x5)
    # Use Schoolbook. Rank 1*2*5 = 10.
    
    # Block 7: A[2:3, 2:4] (1x2) * B[2:4, 0:2] (2x2) -> C[2:3, 0:2] (1x2)
    # Use Schoolbook. Rank 1*2*2 = 4.
    
    # Block 8: A[2:3, 2:4] (1x2) * B[2:4, 2:7] (2x5) -> C[2:3, 2:7] (1x5)
    # Use Schoolbook. Rank 1*2*5 = 10.
    
    # Total Rank = 7 + 20 + 7 + 20 + 4 + 10 + 4 + 10 = 82.
    # Schoolbook is 84. Improvement of 2.
    
    # Can we do better?
    # Maybe split p=7 into 3+4?
    # 2x2x3 schoolbook rank = 12.
    # 2x2x4 schoolbook rank = 16.
    # Strassen for 2x2x2 is 7.
    
    # Let's try to maximize Strassen usage.
    # Strassen requires square 2x2 blocks.
    # We have n=3, m=4, p=7.
    # Max number of 2x2x2 blocks we can fit?
    # n: 1 block of 2, 1 leftover 1.
    # m: 2 blocks of 2.
    # p: 3 blocks of 2, 1 leftover 1.
    
    # Let's arrange the blocks.
    # i in {0,1,2}, j in {0,1,2,3}, k in {0..6}
    
    # We will define a list of operations.
    # Each operation is (i_start, i_end, j_start, j_end, k_start, k_end, method)
    # method: 'strassen' or 'schoolbook'
    
    ops = []
    
    # i=0,1 (2 rows)
    # j=0,1 (2 cols)
    # k=0,1 (2 cols) -> Strassen
    ops.append((0, 2, 0, 2, 0, 2, 'strassen'))
    
    # i=0,1
    # j=0,1
    # k=2,3 (2 cols) -> Strassen
    ops.append((0, 2, 0, 2, 2, 4, 'strassen'))
    
    # i=0,1
    # j=0,1
    # k=4,5 (2 cols) -> Strassen
    ops.append((0, 2, 0, 2, 4, 6, 'strassen'))
    
    # i=0,1
    # j=0,1
    # k=6 (1 col) -> Schoolbook (2x2x1) rank 4
    ops.append((0, 2, 0, 2, 6, 7, 'schoolbook'))
    
    # i=0,1
    # j=2,3 (2 cols)
    # k=0,1 (2 cols) -> Strassen
    ops.append((0, 2, 2, 4, 0, 2, 'strassen'))
    
    # i=0,1
    # j=2,3
    # k=2,3 (2 cols) -> Strassen
    ops.append((0, 2, 2, 4, 2, 4, 'strassen'))
    
    # i=0,1
    # j=2,3
    # k=4,5 (2 cols) -> Strassen
    ops.append((0, 2, 2, 4, 4, 6, 'strassen'))
    
    # i=0,1
    # j=2,3
    # k=6 (1 col) -> Schoolbook (2x2x1) rank 4
    ops.append((0, 2, 2, 4, 6, 7, 'schoolbook'))
    
    # i=2 (1 row)
    # j=0,1 (2 cols)
    # k=0,1 (2 cols) -> Schoolbook (1x2x2) rank 4
    ops.append((2, 3, 0, 2, 0, 2, 'schoolbook'))
    
    # i=2
    # j=0,1
    # k=2,3 (2 cols) -> Schoolbook (1x2x2) rank 4
    ops.append((2, 3, 0, 2, 2, 4, 'schoolbook'))
    
    # i=2
    # j=0,1
    # k=4,5 (2 cols) -> Schoolbook (1x2x2) rank 4
    ops.append((2, 3, 0, 2, 4, 6, 'schoolbook'))
    
    # i=2
    # j=0,1
    # k=6 (1 col) -> Schoolbook (1x2x1) rank 2
    ops.append((2, 3, 0, 2, 6, 7, 'schoolbook'))
    
    # i=2
    # j=2,3 (2 cols)
    # k=0,1 (2 cols) -> Schoolbook (1x2x2) rank 4
    ops.append((2, 3, 2, 4, 0, 2, 'schoolbook'))
    
    # i=2
    # j=2,3
    # k=2,3 (2 cols) -> Schoolbook (1x2x2) rank 4
    ops.append((2, 3, 2, 4, 2, 4, 'schoolbook'))
    
    # i=2
    # j=2,3
    # k=4,5 (2 cols) -> Schoolbook (1x2x2) rank 4
    ops.append((2, 3, 2, 4, 4, 6, 'schoolbook'))
    
    # i=2
    # j=2,3
    # k=6 (1 col) -> Schoolbook (1x2x1) rank 2
    ops.append((2, 3, 2, 4, 6, 7, 'schoolbook'))
    
    # Let's sum the ranks:
    # 6 Strassen blocks: 6 * 7 = 42
    # 2 (2x2x1) blocks: 2 * 4 = 8
    # 6 (1x2x2) blocks: 6 * 4 = 24
    # 2 (1x2x1) blocks: 2 * 2 = 4
    # Total = 42 + 8 + 24 + 4 = 78.
    
    # This is a significant improvement over 84.
    
    # Now, we need to construct the global U, V, W.
    # Global U: (3*4, TotalRank) = (12, 78)
    # Global V: (4*7, TotalRank) = (28, 78)
    # Global W: (7*3, TotalRank) = (21, 78)
    
    total_rank = sum(
        7 if op[6] == 'strassen' else (op[1]-op[0])*(op[3]-op[2])*(op[5]-op[4])
        for op in ops
    )
    
    U_global = np.zeros((n*m, total_rank))
    V_global = np.zeros((m*p, total_rank))
    W_global = np.zeros((p*n, total_rank))
    
    col_offset = 0
    
    for op in ops:
        i0, i1, j0, j1, k0, k1, method = op
        ni = i1 - i0
        mj = j1 - j0
        pk = k1 - k0
        
        if method == 'strassen':
            if ni != 2 or mj != 2 or pk != 2:
                raise ValueError("Strassen requires 2x2x2")
            U_blk, V_blk, W_blk, r_blk = _strassen_2x2x2()
        else:
            U_blk, V_blk, W_blk, r_blk = _schoolbook_block(ni, mj, pk)
        
        # Place U_blk into U_global
        # U_blk rows correspond to A[i0:i1, j0:j1]
        # Global A row index for A[i, j] is i*m + j
        # We need to map local (li, lj) to global (gi, gj)
        # gi = i0 + li, gj = j0 + lj
        # Global row index = gi * m + gj
        
        for li in range(ni):
            for lj in range(mj):
                gi = i0 + li
                gj = j0 + lj
                global_row = gi * m + gj
                U_global[global_row, col_offset:col_offset+r_blk] = U_blk[li*mj+lj, :]
        
        # Place V_blk into V_global
        # V_blk rows correspond to B[j0:j1, k0:k1]
        # Local (lj, lk) -> Global (gj, gk)
        # gj = j0 + lj, gk = k0 + lk
        # Global row index = gj * p + gk
        
        for lj in range(mj):
            for lk in range(pk):
                gj = j0 + lj
                gk = k0 + lk
                global_row = gj * p + gk
                V_global[global_row, col_offset:col_offset+r_blk] = V_blk[lj*pk+lk, :]
        
        # Place W_blk into W_global
        # W_blk rows correspond to C[i0:i1, k0:k1]
        # Local (lk, li) -> Global (gk, gi)
        # gk = k0 + lk, gi = i0 + li
        # Global row index = gk * n + gi
        
        for lk in range(pk):
            for li in range(ni):
                gk = k0 + lk
                gi = i0 + li
                global_row = gk * n + gi
                W_global[global_row, col_offset:col_offset+r_blk] = W_blk[lk*ni+li, :]
        
        col_offset += r_blk
    
    # Verify
    if not _verify(U_global, V_global, W_global, n, m, p):
        # Fallback to schoolbook if something went wrong
        U_sb, V_sb, W_sb, _ = _schoolbook(n, m, p)
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
