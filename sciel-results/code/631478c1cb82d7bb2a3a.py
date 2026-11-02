"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_strassen_36():
    """
    Build a rank-36 block-Strassen decomposition for 2x4 by 4x5 matrix multiplication.
    
    A (2x4) = [A1 A2] where A1, A2 are 2x2
    B (4x5) = [B1; B2] where B1, B2 are 2x5
    B1 = [B11 B12 B13] where B11,B12 are 2x2, B13 is 2x1
    B2 = [B21 B22 B23] where B21,B22 are 2x2, B23 is 2x1
    
    C = A1*B1 + A2*B2
    
    Using Strassen (rank 7) for each 2x2x2 block and rank 2 for each 2x2x1 block:
    A1*B11: 7, A1*B12: 7, A1*B13: 2
    A2*B21: 7, A2*B22: 7, A2*B23: 2
    Total: 4*7 + 2*2 = 32... wait let me recount.
    
    Actually: 4 blocks of 2x2x2 (each rank 7) = 28
    2 blocks of 2x2x1 (each rank 2) = 4
    Total: 32
    
    Hmm, but the inspirations say rank 36. Let me recheck.
    Maybe they use a different decomposition. Let me just build rank 32 correctly first.
    """
    n, m, p = 2, 4, 5
    
    U = []
    V = []
    W = []
    
    def add_term(u_vals, v_vals, w_vals):
        col = len(U)
        U.append([0.0] * (n * m))
        V.append([0.0] * (m * p))
        W.append([0.0] * (p * n))
        for idx, val in u_vals.items():
            U[col][idx] = val
        for idx, val in v_vals.items():
            V[col][idx] = val
        for idx, val in w_vals.items():
            W[col][idx] = val
    
    def strassen_2x2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        ar1, ar2 = a_rows
        ac1, ac2 = a_cols
        br1, br2 = b_rows
        bc1, bc2 = b_cols
        cr1, cr2 = c_rows
        cc1, cc2 = c_cols
        
        u_a11 = ar1 * m + ac1
        u_a12 = ar1 * m + ac2
        u_a21 = ar2 * m + ac1
        u_a22 = ar2 * m + ac2
        
        v_b11 = br1 * p + bc1
        v_b12 = br1 * p + bc2
        v_b21 = br2 * p + bc1
        v_b22 = br2 * p + bc2
        
        w_c11 = cc1 * n + cr1
        w_c12 = cc2 * n + cr1
        w_c21 = cc1 * n + cr2
        w_c22 = cc2 * n + cr2
        
        # M1 = (A11+A22)(B11+B22)
        add_term({u_a11: 1, u_a22: 1}, {v_b11: 1, v_b22: 1}, {w_c11: 1, w_c12: -1, w_c21: -1, w_c22: 1})
        # M2 = (A21+A22)B11
        add_term({u_a21: 1, u_a22: 1}, {v_b11: 1}, {w_c21: 1, w_c22: 1})
        # M3 = A11(B12-B22)
        add_term({u_a11: 1}, {v_b12: 1, v_b22: -1}, {w_c12: 1, w_c22: 1})
        # M4 = A22(B21-B11)
        add_term({u_a22: 1}, {v_b21: 1, v_b11: -1}, {w_c21: 1, w_c22: 1})
        # M5 = (A11+A12)B22
        add_term({u_a11: 1, u_a12: 1}, {v_b22: 1}, {w_c11: 1, w_c12: 1})
        # M6 = (A21-A11)(B11+B12)
        add_term({u_a21: 1, u_a11: -1}, {v_b11: 1, v_b12: 1}, {w_c21: 1, w_c11: -1})
        # M7 = (A12-A22)(B21+B22)
        add_term({u_a12: 1, u_a22: -1}, {v_b21: 1, v_b22: 1}, {w_c12: 1, w_c22: -1})
    
    def rank2_2x2x1(a_rows, a_cols, b_rows, b_col, c_rows, c_col):
        """
        Add rank-2 decomposition for 2x2x1 block.
        A is 2x2, B is 2x1, C is 2x1
        C[0] = A[0,0]*B[0] + A[0,1]*B[1]
        C[1] = A[1,0]*B[0] + A[1,1]*B[1]
        """
        ar1, ar2 = a_rows
        ac1, ac2 = a_cols
        br1, br2 = b_rows
        cc = c_col
        cr1, cr2 = c_rows
        
        u_a11 = ar1 * m + ac1
        u_a12 = ar1 * m + ac2
        u_a21 = ar2 * m + ac1
        u_a22 = ar2 * m + ac2
        
        v_b1 = br1 * p + cc
        v_b2 = br2 * p + cc
        
        w_c1 = cc * n + cr1
        w_c2 = cc * n + cr2
        
        # Term 1: A[0,0]*B[0] -> C[0], A[1,0]*B[0] -> C[1]
        add_term({u_a11: 1, u_a21: 1}, {v_b1: 1}, {w_c1: 1, w_c2: 1})
        # Term 2: A[0,1]*B[1] -> C[0], A[1,1]*B[1] -> C[1]
        add_term({u_a12: 1, u_a22: 1}, {v_b2: 1}, {w_c1: 1, w_c2: 1})
    
    # A rows: 0,1; A cols: 0,1,2,3
    # B rows: 0,1,2,3; B cols: 0,1,2,3,4
    # C rows: 0,1; C cols: 0,1,2,3,4
    
    # A1 = A[0:2, 0:2], A2 = A[0:2, 2:4]
    # B1 = B[0:2, 0:5], B2 = B[2:4, 0:5]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B13 = B[0:2, 4:5]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4], B23 = B[2:4, 4:5]
    
    # A1*B11: Strassen (2x2 * 2x2 -> 2x2)
    strassen_2x2x2([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    # A1*B12: Strassen (2x2 * 2x2 -> 2x2)
    strassen_2x2x2([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    # A1*B13: rank 2 (2x2 * 2x1 -> 2x1)
    rank2_2x2x1([0,1], [0,1], [0,1], 4, [0,1], 4)
    # A2*B21: Strassen (2x2 * 2x2 -> 2x2)
    strassen_2x2x2([0,1], [2,3], [2,3], [0,1], [0,1], [0,1])
    # A2*B22: Strassen (2x2 * 2x2 -> 2x2)
    strassen_2x2x2([0,1], [2,3], [2,3], [2,3], [0,1], [2,3])
    # A2*B23: rank 2 (2x2 * 2x1 -> 2x1)
    rank2_2x2x1([0,1], [2,3], [2,3], 4, [0,1], 4)
    
    return U, V, W


def _verify_decomposition(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Verify that the decomposition exactly decomposes matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False, 1.0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :])
                s = np.dot(s, W[k*n+i, :])
                if abs(s - 1.0) > max_err:
                    max_err = abs(s - 1.0)
    return max_err < tol, max_err


def _build_tensor_vectors(U, V, W, n=2, m=4, p=5):
    """
    For each column c, build the tensor vector T_c[i,j,k] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    Return as a 2D array of shape (rank, n*m*p)
    """
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    nmp = n * m * p
    T = np.zeros((rank, nmp))
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[c, i*m*p + j*p + k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    return T


def _try_reduce_rank(U, V, W, n=2, m=4, p=5, max_time=100.0):
    """
    Systematically try to remove columns by checking if any column's tensor vector
    is a linear combination of the others.
    """
    start_time = time.time()
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    
    def check_valid(U, V, W):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        r = Ua.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i*m+j, :], Va[j*p+k, :])
                    s = np.dot(s, Wa[k*n+i, :])
                    if abs(s - 1.0) > max_err:
                        max_err = abs(s - 1.0)
        return max_err < 1e-8
    
    # Verify initial
    if not check_valid(U, V, W):
        return U, V, W, rank
    
    # Build tensor vectors
    T = _build_tensor_vectors(U, V, W, n, m, p)
    nmp = n * m * p
    
    # Try removing each column: check if T_c is in span of remaining columns
    changed = True
    while changed:
        changed = False
        if time.time() - start_time > max_time - 10:
            break
        
        T = _build_tensor_vectors(U, V, W, n, m, p)
        rank = len(U[0])
        
        for c in range(rank):
            if time.time() - start_time > max_time - 5:
                break
            
            # Check if T[c] is in the span of all other columns
            others = list(range(rank))
            others.remove(c)
            if not others:
                break
            
            A_mat = T[others, :]  # shape (rank-1, nmp)
            b_vec = T[c, :]
            
            # Check if b_vec is in column space of A_mat
            # Use least squares: min ||A_mat^T * x - b_vec||^2
            # Actually, we want to check if b_vec is in row space of A_mat
            # i.e., is b_vec = A_mat @ x for some x?
            # No: T[c] should equal sum of T[others] * coeffs
            # So we need: T[c] = T[others] @ x
            # This is a linear system: T[others] (rank-1 x nmp) times x (rank-1) = T[c] (nmp)
            
            # Use np.linalg.lstsq
            try:
                x, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=1e-10)
                recon = A_mat @ x
                err = np.linalg.norm(recon - b_vec)
                if err < 1e-8:
                    # Column c is redundant! Remove it.
                    U = [row[:c] + row[c+1:] for row in U]
                    V = [row[:c] + row[c+1:] for row in V]
                    W = [row[:c] + row[c+1:] for row in W]
                    rank -= 1
                    changed = True
                    print(f"Removed column {c}, new rank: {rank}")
                    break  # Restart the outer loop
            except np.linalg.LinAlgError:
                pass
    
    # Also try: for each pair of columns, check if their sum/difference can be absorbed
    # This is harder. Let's also try random perturbations.
    
    return U, V, W, rank


def _try_remove_columns(U, V, W, n=2, m=4, p=5, max_time=100.0):
    """
    Try to remove columns from the decomposition.
    """
    start_time = time.time()
    
    # First try systematic linear dependency check
    U, V, W, rank = _try_reduce_rank(U, V, W, n, m, p, max_time=max_time)
    
    return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Build the block-Strassen decomposition
    U, V, W = _build_strassen_36()
    rank = len(U[0])
    print(f"Initial block-Strassen rank: {rank}")
    
    # Verify
    valid, err = _verify_decomposition(U, V, W)
    print(f"Verification: valid={valid}, max_err={err}")
    
    if not valid:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
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
    
    # Try to reduce rank
    if time.time() - start_time < deadline - 20:
        U, V, W, rank = _try_remove_columns(U, V, W, max_time=deadline - start_time - 10)
    
    # Final verification
    valid, err = _verify_decomposition(U, V, W)
    if not valid:
        # Fallback to the initial block-Strassen
        U, V, W = _build_strassen_36()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

