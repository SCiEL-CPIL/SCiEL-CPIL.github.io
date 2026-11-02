import numpy as np
import time

def _build_block_rank28(n, m, p):
    """
    Build the known rank-28 block decomposition for 2x3x5 matrix multiplication.
    Decomposition:
    - j=0: 2x1x5 schoolbook (rank 10)
    - j=1,2: 2x2x5 block
      - k=0,1: 2x2x2 Strassen (rank 7)
      - k=2: 2x2x1 schoolbook (rank 4)
      - k=3,4: 2x2x2 Strassen (rank 7)
    Total rank: 10 + 7 + 4 + 7 = 28.
    """
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    # Block 1: j=0, k=0..4 (schoolbook 2x1x5)
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1

    # Strassen 2x2x2 helper
    # Maps local indices (i', j', k') to global (i, j, k)
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1: (A00+A11)(B00+B11) -> C00, C11
        U[(i_off+0)*m + (j_off+0), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[(k_off+0)*n + (i_off+0), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1
        
        # M2: (A10+A11)B00 -> C10, -C11
        U[(i_off+1)*m + (j_off+0), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        W[(k_off+0)*n + (i_off+1), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = -1.0
        col += 1
        
        # M3: A00(B01-B11) -> C01, C11
        U[(i_off+0)*m + (j_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = -1.0
        W[(k_off+1)*n + (i_off+0), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1
        
        # M4: A11(B10-B00) -> C00, C10
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = -1.0
        W[(k_off+0)*n + (i_off+0), col] = 1.0
        W[(k_off+0)*n + (i_off+1), col] = 1.0
        col += 1
        
        # M5: (A00+A01)B11 -> -C00, C01
        U[(i_off+0)*m + (j_off+0), col] = 1.0
        U[(i_off+0)*m + (j_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[(k_off+0)*n + (i_off+0), col] = -1.0
        W[(k_off+1)*n + (i_off+0), col] = 1.0
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11
        U[(i_off+1)*m + (j_off+0), col] = 1.0
        U[(i_off+0)*m + (j_off+0), col] = -1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+1), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00
        U[(i_off+0)*m + (j_off+1), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = -1.0
        V[(j_off+1)*p + (k_off+0), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[(k_off+0)*n + (i_off+0), col] = 1.0
        col += 1

    # Block 2a: j=1,2, k=0,1 (Strassen 2x2x2)
    add_strassen(i_off=0, j_off=1, k_off=0)
    
    # Block 2b: j=1,2, k=2 (schoolbook 2x2x1)
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j, col] = 1.0
            V[j * p + 2, col] = 1.0
            W[2 * n + i, col] = 1.0
            col += 1
            
    # Block 2c: j=1,2, k=3,4 (Strassen 2x2x2)
    add_strassen(i_off=0, j_off=1, k_off=3)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W

def _verify(U, V, W, n, m, p, rank, atol=1e-10):
    """Verify decomposition on several random matrices."""
    rng = np.random.RandomState(42)
    for _ in range(5):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C = A @ B
        C_d = np.zeros((n, p))
        for t in range(rank):
            u_t = U[:, t].reshape(n, m)
            v_t = V[:, t].reshape(m, p)
            w_t = W[:, t].reshape(p, n)
            # Term: (A * u_t).sum() * (B * v_t).sum() * w_t.T
            C_d += (A * u_t).sum() * (B * v_t).sum() * w_t.T
        if not np.allclose(C, C_d, atol=atol):
            return False
    return True

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x5 matrix.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit

    n, m, p = 2, 3, 5  # dimensions

    # Construct the rank-28 block decomposition
    U, V, W = _build_block_rank28(n, m, p)
    
    # Verify the decomposition
    if not _verify(U, V, W, n, m, p, 28):
        # Fallback to schoolbook rank-30 if verification fails (should not happen)
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb), dtype=float)
        V_sb = np.zeros((m * p, rank_sb), dtype=float)
        W_sb = np.zeros((p * n, rank_sb), dtype=float)
        col_sb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col_sb] = 1.0
                    V_sb[j * p + k, col_sb] = 1.0
                    W_sb[k * n + i, col_sb] = 1.0
                    col_sb += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
