import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try rank 22 construction (Strassen for 2x2*2x4 + outer product for 2x1*1x4)
    result = _try_rank_22()
    if result is not None:
        if _verify(result, n, m, p):
            return result
    
    # Try rank 23
    if time.time() < deadline:
        result = _try_rank_23()
        if result is not None:
            if _verify(result, n, m, p):
                return result
    
    # Fall back to schoolbook rank 24
    return _schoolbook()

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(decomp, n, m, p, trials=100):
    """Verify that the decomposition exactly decomposes matrix multiplication."""
    U = np.array(decomp["U"], dtype=np.float64)
    V = np.array(decomp["V"], dtype=np.float64)
    W = np.array(decomp["W"], dtype=np.float64)
    rank = U.shape[1]
    
    for _ in range(trials):
        # Random integer matrices
        A = np.random.randint(-5, 6, size=(n, m))
        B = np.random.randint(-5, 6, size=(m, p))
        C_expected = A @ B  # shape (n, p)
        
        # Compute using tensor decomposition
        # u = U^T A_flat, v = V^T B_flat, w = W^T C_flat
        A_flat = A.flatten()  # shape (n*m,), index i*m+j
        B_flat = B.flatten()  # shape (m*p,), index j*p+k
        C_flat = np.zeros(n * p)
        
        for r in range(rank):
            u_r = U[:, r] @ A_flat  # scalar
            v_r = V[:, r] @ B_flat  # scalar
            w_r = W[:, r]
            C_flat += u_r * v_r * w_r
        
        # C_flat has index k*n+i for C[i,k]
        # So C_flat[k*n + i] = C[i, k]
        C_reconstructed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                C_reconstructed[i, k] = C_flat[k * n + i]
        
        if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
            return False
    
    return True

def _try_rank_22():
    """
    Construct rank-22 decomposition using correct Strassen formulas.
    Splits A(2x3) into A1(2x2)+A2(2x1), B(3x4) into B1(2x4)+B2(1x4).
    A1*B1 uses two 2x2*2x2 Strassen blocks (14 mults), A2*B2 is outer product (8 mults).
    """
    n, m, p = DIMENSIONS
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(u_a, u_b, u_c, u_d, v_e, v_f, v_g, v_h, w_00, w_01, w_10, w_11, col):
        """Add Strassen 2x2*2x2 block. Formulas verified by _verify."""
        # P1 = (a+b)(e+h)
        U[u_a][col] += 1; U[u_b][col] += 1
        V[v_e][col] += 1; V[v_h][col] += 1
        W[w_00][col] += 1; W[w_11][col] += 1
        
        # P2 = (c+d)(e)
        U[u_c][col+1] += 1; U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        W[w_10][col+1] += 1; W[w_11][col+1] += -1
        
        # P3 = (a-c)(g+h)
        U[u_a][col+2] += 1; U[u_c][col+2] += -1
        V[v_g][col+2] += 1; V[v_h][col+2] += 1
        W[w_01][col+2] += 1; W[w_11][col+2] += 1
        
        # P4 = (d-b)(f+h)
        U[u_d][col+3] += 1; U[u_b][col+3] += -1
        V[v_f][col+3] += 1; V[v_h][col+3] += 1
        W[w_00][col+3] += 1; W[w_11][col+3] += 1
        
        # P5 = (a+d)(e+f)
        U[u_a][col+4] += 1; U[u_d][col+4] += 1
        V[v_e][col+4] += 1; V[v_f][col+4] += 1
        W[w_00][col+4] += -1; W[w_01][col+4] += 1
        
        # P6 = (b+d)(g-e)
        U[u_b][col+5] += 1; U[u_d][col+5] += 1
        V[v_g][col+5] += 1; V[v_e][col+5] += -1
        W[w_10][col+5] += 1; W[w_11][col+5] += 1
        
        # P7 = (c+a)(h-f)
        U[u_c][col+6] += 1; U[u_a][col+6] += 1
        V[v_h][col+6] += 1; V[v_f][col+6] += -1
        W[w_00][col+6] += 1; W[w_11][col+6] += 1
    
    # Block 1: cols 0-6
    add_strassen(0, 1, 3, 4, 0, 1, 4, 5, 0, 2, 1, 3, 0)
    # Block 2: cols 7-13
    add_strassen(0, 1, 3, 4, 2, 3, 6, 7, 4, 6, 5, 7, 7)
    
    # Outer product: cols 14-21
    for k in range(4):
        col = 14 + k
        U[2][col] = 1
        U[5][col] = 1
        V[8+k][col] = 1
        W[k*2][col] = 1
        W[k*2+1][col] = 1
    
    return {"U": U, "V": V, "W": W}

def _try_rank_23():
    """
    Try rank-23: use Strassen for one 2x2*2x2 block (7 mults) and
    schoolbook for the other 2x2*2x2 block (8 mults) plus outer product (8 mults) = 23.
    """
    n, m, p = DIMENSIONS
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(u_a, u_b, u_c, u_d, v_e, v_f, v_g, v_h, w_00, w_01, w_10, w_11, col):
        U[u_a][col] += 1; U[u_b][col] += 1
        V[v_e][col] += 1; V[v_h][col] += 1
        W[w_00][col] += 1; W[w_11][col] += 1
        U[u_c][col+1] += 1; U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        W[w_10][col+1] += 1; W[w_11][col+1] += -1
        U[u_a][col+2] += 1; U[u_c][col+2] += -1
        V[v_g][col+2] += 1; V[v_h][col+2] += 1
        W[w_01][col+2] += 1; W[w_11][col+2] += 1
        U[u_d][col+3] += 1; U[u_b][col+3] += -1
        V[v_f][col+3] += 1; V[v_h][col+3] += 1
        W[w_00][col+3] += 1; W[w_11][col+3] += 1
        U[u_a][col+4] += 1; U[u_d][col+4] += 1
        V[v_e][col+4] += 1; V[v_f][col+4] += 1
        W[w_00][col+4] += -1; W[w_01][col+4] += 1
        U[u_b][col+5] += 1; U[u_d][col+5] += 1
        V[v_g][col+5] += 1; V[v_e][col+5] += -1
        W[w_10][col+5] += 1; W[w_11][col+5] += 1
        U[u_c][col+6] += 1; U[u_a][col+6] += 1
        V[v_h][col+6] += 1; V[v_f][col+6] += -1
        W[w_00][col+6] += 1; W[w_11][col+6] += 1
    
    def add_schoolbook_2x2(u_a, u_b, u_c, u_d, v_e, v_f, v_g, v_h, w_00, w_01, w_10, w_11, col):
        # ae
        U[u_a][col] += 1; V[v_e][col] += 1; W[w_00][col] += 1
        # bg
        U[u_b][col+1] += 1; V[v_g][col+1] += 1; W[w_00][col+1] += 1
        # af
        U[u_a][col+2] += 1; V[v_f][col+2] += 1; W[w_01][col+2] += 1
        # bh
        U[u_b][col+3] += 1; V[v_h][col+3] += 1; W[w_01][col+3] += 1
        # ce
        U[u_c][col+4] += 1; V[v_e][col+4] += 1; W[w_10][col+4] += 1
        # dg
        U[u_d][col+5] += 1; V[v_g][col+5] += 1; W[w_10][col+5] += 1
        # cf
        U[u_c][col+6] += 1; V[v_f][col+6] += 1; W[w_11][col+6] += 1
        # dh
        U[u_d][col+7] += 1; V[v_h][col+7] += 1; W[w_11][col+7] += 1
    
    # Block 1: Strassen for cols 0-6
    add_strassen(0, 1, 3, 4, 0, 1, 4, 5, 0, 2, 1, 3, 0)
    # Block 2: Schoolbook for cols 7-14
    add_schoolbook_2x2(0, 1, 3, 4, 2, 3, 6, 7, 4, 6, 5, 7, 7)
    
    # Outer product: cols 15-22
    for k in range(4):
        col = 15 + k
        U[2][col] = 1
        U[5][col] = 1
        V[8+k][col] = 1
        W[k*2][col] = 1
        W[k*2+1][col] = 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
