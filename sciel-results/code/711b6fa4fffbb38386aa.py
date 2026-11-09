"""Matrix multiplication 2x5 x 5x5 - rank-minimal tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
n, m, p = DIMENSIONS
NM, MP, PN = n*m, m*p, p*n  # 10, 25, 10

def _build_tensor():
    """T[a,b,c] where a=(i,j), b=(j',k), c=(k',i'); T = delta_{j,j'}*delta_{k,k'}."""
    T = np.zeros((NM, MP, PN), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            a = i*m + j
            for jp in range(m):   # j'
                for k in range(p):
                    b = jp*p + k
                    for kp in range(p):  # k'
                        for ip in range(n):  # i'
                            c = kp*n + ip
                            if jp == j and kp == k:
                                T[a, b, c] = 1.0
    return T

TENSOR = _build_tensor()

def _verify_fast(U, V, W, tol=1e-9):
    """Fast vectorized verification: sum_r U[a,r]V[b,r]W[c,r] == T[a,b,c]."""
    R = U.shape[1]
    # Compute UV[a,b,r] = U[a,r]*V[b,r]
    UV = np.einsum('ar,br->abr', U, V)
    # Result[a,b,c] = sum_r UV[a,b,r]*W[c,r]
    result = np.einsum('abr,cr->abc', UV, W)
    return np.allclose(result, TENSOR, atol=tol)

def _verify_exact(U, V, W):
    """Exact verification using Fractions (for half-integer values)."""
    R = U.shape[1]
    Uq = [[Fraction(int(round(2*x)), 2) for x in row] for row in U]
    Vq = [[Fraction(int(round(2*x)), 2) for x in row] for row in V]
    Wq = [[Fraction(int(round(2*x)), 2) for x in row] for row in W]
    for a in range(NM):
        for b in range(MP):
            prod = [Uq[a][r] * Vq[b][r] for r in range(R)]
            for c in range(PN):
                val = sum(prod[r] * Wq[c][r] for r in range(R))
                if val != Fraction(int(TENSOR[a, b, c]), 1):
                    return False
    return True

def _strassen_block(a, b):
    """
    Compute 2x2 matrix product C = A * B using Strassen's algorithm.
    Returns the 7 products (M1..M7) needed for the rank-7 decomposition.
    A and B are 2x2 numpy arrays.
    """
    A11, A12 = a[0,0], a[0,1]
    A21, A22 = a[1,0], a[1,1]
    B11, B12 = b[0,0], b[0,1]
    B21, B22 = b[1,0], b[1,1]
    
    # Strassen's 7 products
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    return [M1, M2, M3, M4, M5, M6, M7]

def _construct_strassen_hybrid():
    """
    Construct a rank-46 decomposition using Strassen's algorithm on 2x2 blocks.
    
    The 2x5 x 5x5 multiplication can be viewed as:
    - Splitting the 5x5 matrix B into blocks: 
      B = [[B11, B12, B13, B14, B15],
           [B21, B22, B23, B24, B25]]
      where B11 is 1x1, B12-B15 are 1x1, B21 is 1x1, B22-B25 are 1x1.
      
    Actually, let's think of it differently. We have A (2x5) and B (5x5).
    C = A * B is 2x5.
    
    Let's split A into columns: A = [A_col1, A_col2, A_col3, A_col4, A_col5]
    where each A_col is 2x1.
    
    Let's split B into rows: B = [B_row1; B_row2; B_row3; B_row4; B_row5]
    where each B_row is 1x5.
    
    This doesn't directly help with Strassen.
    
    Better approach: View the computation as two separate 1x5 x 5x5 multiplications
    (one for each row of A), but that's just schoolbook.
    
    The key insight from the cell finding: "Using Strassen's algorithm for 2x2x2 sub-blocks
    reduces the rank from 50 to 46 by replacing 8 scalar multiplications with 7 for each
    of the four 2x2x2 blocks."
    
    How do we get four 2x2x2 blocks? 
    If we split the 5x5 matrix B into:
    - A 2x2 block B11
    - Remaining parts
    
    Let's try: Split B into [[B11, B12], [B21, B22]] where:
    - B11 is 2x2 (rows 0-1, cols 0-1)
    - B12 is 2x3 (rows 0-1, cols 2-4)
    - B21 is 3x2 (rows 2-4, cols 0-1)
    - B22 is 3x3 (rows 2-4, cols 2-4)
    
    And A is 2x5, split into [A1, A2] where A1 is 2x2, A2 is 2x3.
    
    Then C = A1*B11 + A1*B12 + A2*B21 + A2*B22.
    
    The term A1*B11 is a 2x2 x 2x2 multiplication, which can be done with rank 7 (Strassen).
    The other terms are:
    - A1*B12: 2x2 x 2x3 = 12 scalar mults (rank 12)
    - A2*B21: 2x3 x 3x2 = 12 scalar mults (rank 12)
    - A2*B22: 2x3 x 3x3 = 18 scalar mults (rank 18)
    
    Total rank: 7 + 12 + 12 + 18 = 49. Not 46.
    
    Let's try a different split. What if we split into four 2x2 blocks?
    B is 5x5, so we can't cleanly split into four 2x2 blocks.
    
    Alternative: Maybe the "four 2x2x2 blocks" refers to something else.
    Perhaps we're splitting the 5x5 matrix into overlapping or structured blocks.
    
    Actually, let's reconsider. The 2x5 x 5x5 multiplication has 50 scalar multiplications in schoolbook.
    If we can find 4 independent 2x2x2 sub-problems, each can be done with rank 7 instead of 8,
    saving 4 multiplications total, giving rank 46.
    
    How to find 4 independent 2x2x2 sub-problems?
    
    Consider the matrix B. Let's partition its rows and columns:
    Rows: {0,1}, {2,3}, {4}
    Cols: {0,1}, {2,3}, {4}
    
    This gives 9 blocks. The 2x2 blocks are at positions (0,0), (0,1), (1,0), (1,1) if we
    group rows {0,1},{2,3} and cols {0,1},{2,3}.
    
    Let:
    - B11 = B[0:2, 0:2]  (2x2)
    - B12 = B[0:2, 2:4]  (2x2)
    - B21 = B[2:4, 0:2]  (2x2)
    - B22 = B[2:4, 2:4]  (2x2)
    - B13 = B[0:2, 4]    (2x1)
    - B23 = B[2:4, 4]    (2x1)
    - B31 = B[4, 0:2]    (1x2)
    - B32 = B[4, 2:4]    (1x2)
    - B33 = B[4, 4]      (1x1)
    
    A is 2x5, split into:
    - A1 = A[:, 0:2]  (2x2)
    - A2 = A[:, 2:4]  (2x2)
    - A3 = A[:, 4]    (2x1)
    
    C = A1*B11 + A1*B12 + A1*B13 + A2*B21 + A2*B22 + A2*B23 + A3*B31 + A3*B32 + A3*B33
    
    The four 2x2x2 multiplications are:
    1. A1*B11: rank 7
    2. A1*B12: rank 7
    3. A2*B21: rank 7
    4. A2*B22: rank 7
    
    The remaining terms:
    - A1*B13: 2x2 x 2x1 = 4 scalar mults (rank 4)
    - A2*B23: 2x2 x 2x1 = 4 scalar mults (rank 4)
    - A3*B31: 2x1 x 1x2 = 4 scalar mults (rank 4)
    - A3*B32: 2x1 x 1x2 = 4 scalar mults (rank 4)
    - A3*B33: 2x1 x 1x1 = 2 scalar mults (rank 2)
    
    Total rank: 7*4 + 4 + 4 + 4 + 4 + 2 = 28 + 18 = 46.
    
    This matches the expected rank of 46!
    """
    rank = 46
    U = np.zeros((NM, rank), dtype=np.float64)
    V = np.zeros((MP, rank), dtype=np.float64)
    W = np.zeros((PN, rank), dtype=np.float64)
    
    col = 0
    
    # Helper to add a Strassen decomposition for a 2x2 x 2x2 multiplication
    def add_strassen(A_idx, B_idx, C_idx):
        """
        A_idx: list of 4 indices for A (i,j) pairs, in order [(0,0),(0,1),(1,0),(1,1)]
        B_idx: list of 4 indices for B (j,k) pairs, in order [(0,0),(0,1),(1,0),(1,1)]
        C_idx: list of 4 indices for C (k,i) pairs, in order [(0,0),(0,1),(1,0),(1,1)]
        
        For each of the 7 Strassen products, we need to determine which entries
        of U, V, W are involved.
        """
        nonlocal col
        
        # Strassen products:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each product, we need to set up U, V, W such that:
        # sum_r U[a,r] V[b,r] W[c,r] = C[i,k]
        
        # The indices are:
        # A entries: (i,j) -> U index = i*m + j
        # B entries: (j,k) -> V index = j*p + k
        # C entries: (k,i) -> W index = k*n + i
        
        # For the first 2x2 block (A1*B11):
        # A1 has indices:
        #   A1[0,0] -> i=0, j=0 -> U[0*5+0] = U[0]
        #   A1[0,1] -> i=0, j=1 -> U[0*5+1] = U[1]
        #   A1[1,0] -> i=1, j=0 -> U[1*5+0] = U[5]
        #   A1[1,1] -> i=1, j=1 -> U[1*5+1] = U[6]
        
        # B11 has indices:
        #   B11[0,0] -> j=0, k=0 -> V[0*5+0] = V[0]
        #   B11[0,1] -> j=0, k=1 -> V[0*5+1] = V[1]
        #   B11[1,0] -> j=1, k=0 -> V[1*5+0] = V[5]
        #   B11[1,1] -> j=1, k=1 -> V[1*5+1] = V[6]
        
        # C11 (result of A1*B11) has indices:
        #   C[0,0] -> k=0, i=0 -> W[0*2+0] = W[0]
        #   C[0,1] -> k=0, i=1 -> W[0*2+1] = W[1]
        #   C[1,0] -> k=1, i=0 -> W[1*2+0] = W[2]
        #   C[1,1] -> k=1, i=1 -> W[1*2+1] = W[3]
        
        # We'll create a generic function that takes the actual global indices
        
        # Let's define the local to global index mapping for each block
        
        # For block (row_group, col_group) where row_group is 0 or 1 (for rows 0-1 or 2-3)
        # and col_group is 0 or 1 (for cols 0-1 or 2-3)
        
        pass
    
    # Let's implement this more directly for each of the four blocks
    
    # Block 1: A1 * B11 (rows 0-1 of B, cols 0-1 of B)
    # A1: columns 0-1 of A
    # A1[0,0] = A[0,0], A1[0,1] = A[0,1], A1[1,0] = A[1,0], A1[1,1] = A[1,1]
    # B11: rows 0-1, cols 0-1 of B
    # B11[0,0] = B[0,0], B11[0,1] = B[0,1], B11[1,0] = B[1,0], B11[1,1] = B[1,1]
    # C11: result[0:2, 0:2]
    # C11[0,0] = C[0,0], C11[0,1] = C[0,1], C11[1,0] = C[1,0], C11[1,1] = C[1,1]
    
    def add_strassen_block(a_rows, a_cols, b_rows, b_cols):
        """
        a_rows: list of 2 row indices for A
        a_cols: list of 2 col indices for A
        b_rows: list of 2 row indices for B
        b_cols: list of 2 col indices for B
        
        The result C has entries at C[a_rows[i], b_cols[k]] for i in 0,1 and k in 0,1.
        """
        nonlocal col
        
        # A entries
        A00_idx = a_rows[0]*m + a_cols[0]  # U index
        A01_idx = a_rows[0]*m + a_cols[1]
        A10_idx = a_rows[1]*m + a_cols[0]
        A11_idx = a_rows[1]*m + a_cols[1]
        
        # B entries
        B00_idx = b_rows[0]*p + b_cols[0]  # V index
        B01_idx = b_rows[0]*p + b_cols[1]
        B10_idx = b_rows[1]*p + b_cols[0]
        B11_idx = b_rows[1]*p + b_cols[1]
        
        # C entries
        C00_idx = b_cols[0]*n + a_rows[0]  # W index: k*n + i
        C01_idx = b_cols[1]*n + a_rows[0]
        C10_idx = b_cols[0]*n + a_rows[1]
        C11_idx = b_cols[1]*n + a_rows[1]
        
        # Strassen products
        # M1 = (A00 + A11) * (B00 + B11)
        U[A00_idx, col] += 1
        U[A11_idx, col] += 1
        V[B00_idx, col] += 1
        V[B11_idx, col] += 1
        W[C00_idx, col] += 1
        W[C11_idx, col] += 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[A10_idx, col] += 1
        U[A11_idx, col] += 1
        V[B00_idx, col] += 1
        W[C10_idx, col] += 1
        col += 1
        
        # M3 = A00 * (B01 - B11)
        U[A00_idx, col] += 1
        V[B01_idx, col] += 1
        V[B11_idx, col] -= 1
        W[C01_idx, col] += 1
        col += 1
        
        # M4 = A11 * (B10 - B00)
        U[A11_idx, col] += 1
        V[B10_idx, col] += 1
        V[B00_idx, col] -= 1
        W[C00_idx, col] += 1
        W[C11_idx, col] += 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[A00_idx, col] += 1
        U[A01_idx, col] += 1
        V[B11_idx, col] += 1
        W[C00_idx, col] -= 1
        W[C11_idx, col] -= 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[A10_idx, col] += 1
        U[A00_idx, col] -= 1
        V[B00_idx, col] += 1
        V[B01_idx, col] += 1
        W[C11_idx, col] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[A01_idx, col] += 1
        U[A11_idx, col] -= 1
        V[B10_idx, col] += 1
        V[B11_idx, col] += 1
        W[C00_idx, col] += 1
        W[C11_idx, col] += 1
        col += 1
    
    # Block 1: A[:, 0:2] * B[0:2, 0:2]
    add_strassen_block([0, 1], [0, 1], [0, 1], [0, 1])
    
    # Block 2: A[:, 0:2] * B[0:2, 2:4]
    add_strassen_block([0, 1], [0, 1], [0, 1], [2, 3])
    
    # Block 3: A[:, 2:4] * B[2:4, 0:2]
    add_strassen_block([0, 1], [2, 3], [2, 3], [0, 1])
    
    # Block 4: A[:, 2:4] * B[2:4, 2:4]
    add_strassen_block([0, 1], [2, 3], [2, 3], [2, 3])
    
    # Now the remaining terms:
    # A[:, 0:2] * B[0:2, 4]  (2x2 x 2x1)
    # A[:, 2:4] * B[2:4, 4]  (2x2 x 2x1)
    # A[:, 4] * B[4, 0:2]    (2x1 x 1x2)
    # A[:, 4] * B[4, 2:4]    (2x1 x 1x2)
    # A[:, 4] * B[4, 4]      (2x1 x 1x1)
    
    # A[:, 0:2] * B[0:2, 4]
    # This is: sum_{j=0}^{1} A[i,j] * B[j,4] for i=0,1
    # 4 scalar multiplications
    for i in [0, 1]:
        for j in [0, 1]:
            U[i*m + j, col] = 1
            V[j*p + 4, col] = 1
            W[4*n + i, col] = 1
            col += 1
    
    # A[:, 2:4] * B[2:4, 4]
    # This is: sum_{j=2}^{3} A[i,j] * B[j,4] for i=0,1
    for i in [0, 1]:
        for j in [2, 3]:
            U[i*m + j, col] = 1
            V[j*p + 4, col] = 1
            W[4*n + i, col] = 1
            col += 1
    
    # A[:, 4] * B[4, 0:2]
    # This is: A[i,4] * B[4,k] for i=0,1, k=0,1
    for i in [0, 1]:
        for k in [0, 1]:
            U[i*m + 4, col] = 1
            V[4*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
    
    # A[:, 4] * B[4, 2:4]
    for i in [0, 1]:
        for k in [2, 3]:
            U[i*m + 4, col] = 1
            V[4*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
    
    # A[:, 4] * B[4, 4]
    for i in [0, 1]:
        U[i*m + 4, col] = 1
        V[4*p + 4, col] = 1
        W[4*n + i, col] = 1
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return U, V, W

def _als(R, iters=200, seed=0, tol=1e-8, deadline=None):
    """Alternating least squares for rank-R decomposition with time guard."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((NM, R))
    V = rng.standard_normal((MP, R))
    W = rng.standard_normal((PN, R))
    U /= (np.linalg.norm(U, axis=0, keepdims=True) + 1e-12)
    V /= (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
    W /= (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    
    for it in range(iters):
        if deadline is not None and time.time() > deadline:
            break
        # Update U
        M = np.einsum('br,cr->bcr', V, W).reshape(MP*PN, R)
        t = TENSOR.reshape(NM, MP*PN)
        U_new = np.zeros((NM, R))
        for a in range(NM):
            sol, *_ = np.linalg.lstsq(M, t[a], rcond=None)
            U_new[a] = sol
        U = U_new
        # Update V
        M2 = np.einsum('ar,cr->acr', U, W).reshape(NM*PN, R)
        V_new = np.zeros((MP, R))
        for b in range(MP):
            t2b = TENSOR[:, b, :].reshape(NM*PN)
            sol, *_ = np.linalg.lstsq(M2, t2b, rcond=None)
            V_new[b] = sol
        V = V_new
        # Update W
        Mc = np.einsum('ar,br->abr', U, V).reshape(NM*MP, R)
        W_new = np.zeros((PN, R))
        for c in range(PN):
            tc = TENSOR[:, :, c].reshape(NM*MP)
            sol, *_ = np.linalg.lstsq(Mc, tc, rcond=None)
            W_new[c] = sol
        W = W_new
    return U, V, W

def _local_search(U, V, W, steps=50, lr=0.1, deadline=None):
    """Local search: try small perturbations and accept if residual decreases."""
    R = U.shape[1]
    def residual(U, V, W):
        UV = np.einsum('ar,br->abr', U, V)
        result = np.einsum('abr,cr->abc', UV, W)
        return np.sum((result - TENSOR)**2)
    
    cur_res = residual(U, V, W)
    rng = np.random.default_rng(42)
    for step in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        # Random perturbation
        dU = rng.standard_normal((NM, R)) * lr
        dV = rng.standard_normal((MP, R)) * lr
        dW = rng.standard_normal((PN, R)) * lr
        new_res = residual(U + dU, V + dV, W + dW)
        if new_res < cur_res:
            U, V, W = U + dU, V + dV, W + dW
            cur_res = new_res
    return U, V, W

def _round_half(U, V, W):
    """Round to nearest half-integer."""
    return (2*U).round()/2.0, (2*V).round()/2.0, (2*W).round()/2.0

def _to_list(U):
    """Convert numpy array to list of lists with clean half-integers."""
    def clean(x):
        hx = round(2*x)
        if abs(hx - round(hx)) < 1e-6:
            return int(hx)/2
        return float(x)
    return [[clean(x) for x in row] for row in U]

def construct():
    t0 = time.time()
    DEADLINE = 140.0
    SAFETY = 10.0
    deadline = t0 + DEADLINE
    
    # Schoolbook fallback (rank 50)
    def schoolbook():
        rank = n*m*p
        U = [[0]*rank for _ in range(NM)]
        V = [[0]*rank for _ in range(MP)]
        W = [[0]*rank for _ in range(PN)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    U0, V0, W0, r0 = schoolbook()
    best = (U0, V0, W0)
    best_rank = r0
    
    # Try Strassen hybrid (rank 46) - this should work
    try:
        U46, V46, W46 = _construct_strassen_hybrid()
        if _verify_fast(U46, V46, W46, tol=1e-9):
            if _verify_exact(U46, V46, W46):
                best = (U46, V46, W46)
                best_rank = 46
    except Exception:
        pass
    
    U, V, W = best
    Uf = _to_list(U)
    Vf = _to_list(V)
    Wf = _to_list(W)
    return {"U": Uf, "V": Vf, "W": Wf}
