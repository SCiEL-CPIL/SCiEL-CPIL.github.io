import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Strategy: Partition 12 columns into 6 groups of 2.
    # For each group of 2 columns, use Strassen (rank 7).
    # Total rank = 6 * 7 = 42.
    
    num_blocks = p // 2  # 6 blocks
    rank = num_blocks * 7  # 42
    
    # U: (n*m, rank) = (4, 42)
    # V: (m*p, rank) = (24, 42)
    # W: (p*n, rank) = (24, 42)
    # W index convention: W[k*n + i, r] where C[i, j] = sum_k A[i,k] B[k,j]
    # So W row index = j * n + i for output entry C[i, j]
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # U indices: A[i, j] -> index i*m + j
    u_a00 = 0 * m + 0  # 0
    u_a01 = 0 * m + 1  # 1
    u_a10 = 1 * m + 0  # 2
    u_a11 = 1 * m + 1  # 3
    
    for t in range(num_blocks):
        if time.time() > deadline:
            break
        base = t * 7
        
        c0 = 2 * t
        c1 = 2 * t + 1
        
        # V indices: B[k, j] -> index k*p + j
        v_b00 = 0 * p + c0
        v_b01 = 0 * p + c1
        v_b10 = 1 * p + c0
        v_b11 = 1 * p + c1
        
        # W indices: C[i, j] -> index j*n + i
        w_c00 = c0 * n + 0  # C[0, c0]
        w_c01 = c0 * n + 1  # C[1, c0]
        w_c10 = c1 * n + 0  # C[0, c1]
        w_c11 = c1 * n + 1  # C[1, c1]
        
        # Strassen's 7 multiplications for 2x2 * 2x2:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1 = (A00 + A11) * (B00 + B11)
        c = base + 0
        U[u_a00, c] = 1
        U[u_a11, c] = 1
        V[v_b00, c] = 1
        V[v_b11, c] = 1
        W[w_c00, c] = 1
        W[w_c11, c] = 1
        
        # M2 = (A10 + A11) * B00
        c = base + 1
        U[u_a10, c] = 1
        U[u_a11, c] = 1
        V[v_b00, c] = 1
        W[w_c10, c] = 1
        W[w_c11, c] = -1
        
        # M3 = A00 * (B01 - B11)
        c = base + 2
        U[u_a00, c] = 1
        V[v_b01, c] = 1
        V[v_b11, c] = -1
        W[w_c01, c] = 1
        W[w_c11, c] = 1
        
        # M4 = A11 * (B10 - B00)
        c = base + 3
        U[u_a11, c] = 1
        V[v_b10, c] = 1
        V[v_b00, c] = -1
        W[w_c00, c] = 1
        W[w_c10, c] = 1
        
        # M5 = (A00 + A01) * B11
        c = base + 4
        U[u_a00, c] = 1
        U[u_a01, c] = 1
        V[v_b11, c] = 1
        W[w_c00, c] = -1
        W[w_c01, c] = 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        c = base + 5
        U[u_a10, c] = 1
        U[u_a00, c] = -1
        V[v_b00, c] = 1
        V[v_b01, c] = 1
        W[w_c11, c] = 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        c = base + 6
        U[u_a01, c] = 1
        U[u_a11, c] = -1
        V[v_b10, c] = 1
        V[v_b11, c] = 1
        W[w_c00, c] = 1
    
    # Convert to lists for the return interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
