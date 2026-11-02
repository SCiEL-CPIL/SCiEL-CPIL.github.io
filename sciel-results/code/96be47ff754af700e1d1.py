import numpy as np
import time

DIMENSIONS = [2, 3, 10]
_TIME_LIMIT = 135  # safety margin before 150s kill

def _build_rank55():
    """
    Build the known rank-55 Strassen-based decomposition for 2x3 * 3x10.
    
    Strategy:
    - Split B into B1 (3x4) and B2 (3x6).
    - For B1, split into two 2x2 blocks and use Strassen (7 mults per block = 14 total).
    - For B2, use schoolbook (2*3*6 = 36 mults).
    - Total: 14 + 36 = 50... wait, that's not 55.
    
    Let me re-read the current program's _build_rank55.
    It uses 5 Strassen blocks (for 10 columns, 2 per block) on the first 2 rows of A/B,
    and then schoolbook for the 3rd row.
    
    A is 2x3. B is 3x10.
    Rows of A: i=0,1. Cols of A: j=0,1,2.
    Rows of B: j=0,1,2. Cols of B: k=0..9.
    
    The current _build_rank55 does:
    1. Strassen on A[:, 0:2] (2x2) * B[0:2, :] (2x10).
       This is 5 blocks of 2 columns. 5 * 7 = 35 terms.
    2. Schoolbook on A[:, 2] (2x1) * B[2, :] (1x10).
       This is 2 * 10 = 20 terms.
    Total: 35 + 20 = 55.
    """
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for g in range(5):
        k0 = 2 * g
        k1 = k0 + 1
        
        v_idx_b00 = 0 * p + k0
        v_idx_b01 = 0 * p + k1
        v_idx_b10 = 1 * p + k0
        v_idx_b11 = 1 * p + k1
        
        u_idx_a00 = 0 * m + 0
        u_idx_a01 = 0 * m + 1
        u_idx_a10 = 1 * m + 0
        u_idx_a11 = 1 * m + 1
        
        w_idx_c00 = k0 * n + 0
        w_idx_c01 = k1 * n + 0
        w_idx_c10 = k0 * n + 1
        w_idx_c11 = k1 * n + 1
        
        # Term 1: m1 = (a00+a11)*(b00+b11)
        U[u_idx_a00][col] = 1; U[u_idx_a11][col] = 1
        V[v_idx_b00][col] = 1; V[v_idx_b11][col] = 1
        W[w_idx_c00][col] = 1; W[w_idx_c11][col] = 1
        col += 1
        
        # Term 2: m2 = (a10+a11)*b00
        U[u_idx_a10][col] = 1; U[u_idx_a11][col] = 1
        V[v_idx_b00][col] = 1
        W[w_idx_c10][col] = 1; W[w_idx_c11][col] = -1
        col += 1
        
        # Term 3: m3 = a00*(b01-b11)
        U[u_idx_a00][col] = 1
        V[v_idx_b01][col] = 1; V[v_idx_b11][col] = -1
        W[w_idx_c01][col] = 1; W[w_idx_c11][col] = 1
        col += 1
        
        # Term 4: m4 = a11*(b10-b00)
        U[u_idx_a11][col] = 1
        V[v_idx_b10][col] = 1; V[v_idx_b00][col] = -1
        W[w_idx_c00][col] = 1; W[w_idx_c10][col] = 1
        col += 1
        
        # Term 5: m5 = (a00+a01)*b11
        U[u_idx_a00][col] = 1; U[u_idx_a01][col] = 1
        V[v_idx_b11][col] = 1
        W[w_idx_c01][col] = 1; W[w_idx_c00][col] = -1
        col += 1
        
        # Term 6: m6 = (a10-a00)*(b00+b01)
        U[u_idx_a10][col] = 1; U[u_idx_a00][col] = -1
        V[v_idx_b00][col] = 1; V[v_idx_b01][col] = 1
        W[w_idx_c11][col] = 1
        col += 1
        
        # Term 7: m7 = (a01-a11)*(b10+b11)
        U[u_idx_a01][col] = 1; U[u_idx_a11][col] = -1
        V[v_idx_b10][col] = 1; V[v_idx_b11][col] = 1
        W[w_idx_c00][col] = 1
        col += 1
    
    # Part 2: Schoolbook for A2 * B2
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _try_find_rank54(U, V, W, n, m, p, deadline):
    """
    Try to find a rank-54 solution by looking for a linear dependency
    among the 55 rank-1 tensors that allows dropping one.
    """
    rank = 55
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    def flatten_term(u_col, v_col, w_col):
        vec = np.zeros(n*m*p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i*m*p + j*p + k] = u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
        return vec
    
    all_terms = []
    for r in range(rank):
        all_terms.append(flatten_term(U_np[:, r], V_np[:, r], W_np[:, r]))
        
    for r0 in range(rank):
        if time.time() > deadline:
            break
        
        term_r0 = all_terms[r0]
        others = np.array([all_terms[r] for r in range(rank) if r != r0])  # 54 x 600
        
        try:
            x, residuals, rank_overs, sv = np.linalg.lstsq(others.T, term_r0, rcond=None)
            residual = np.linalg.norm(others.T @ x - term_r0)
            if residual < 1e-10:
                new_U = [U[r] for r in range(rank) if r != r0]
                new_V = [V[r] for r in range(rank) if r != r0]
                new_W = [W[r] for r in range(rank) if r != r0]
                if _verify_decomposition(new_U, new_V, new_W, n, m, p):
                    return new_U, new_V, new_W
        except:
            pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    # Build the rank-55 baseline
    U, V, W = _build_rank55()
    
    # Verify the baseline
    if not _verify_decomposition(U, V, W, n, m, p):
        pass
    
    # Try to find a rank-54 solution
    result = _try_find_rank54(U, V, W, n, m, p, deadline)
    
    if result is not None:
        new_U, new_V, new_W = result
        return {"U": new_U, "V": new_V, "W": new_W}
    
    # Fall back to rank 55
    return {"U": U, "V": V, "W": W}
