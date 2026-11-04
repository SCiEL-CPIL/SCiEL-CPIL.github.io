"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2x2_factors():
    """
    Return (U_local, V_local, W_local) for 2x2x2 Strassen decomposition.
    
    U_local: shape (4, 7) for (i,a) in {0,0:0, 0,1:1, 1,0:2, 1,1:3}
    V_local: shape (4, 7) for (b,k) in {0,0:0, 0,1:1, 1,0:2, 1,1:3}
    W_local: shape (4, 7) for (k,i) in {0,0:0, 0,1:1, 1,0:2, 1,1:3} i.e. k*2+i
    
    The constraint: for all i,a,b,k in {0,1}:
      sum_c W_local[k*2+i, c] * U_local[i*2+a, c] * V_local[b*2+k, c] = delta[a,b]
    """
    # Strassen 7 products:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)(B00)
    # M2 = (A00)(B01-B11)
    # M3 = (A11)(B10-B00)
    # M4 = (A00+A01)(B11)
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 - M3 + M5
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M2 + M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # c=0: M0 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1; W[3, 0] = 1  # C00, C11
    
    # c=1: M1 = (A10+A11)(B00)
    U[2, 1] = 1; U[3, 1] = 1  # A10, A11
    V[0, 1] = 1               # B00
    W[2, 1] = 1               # C10
    
    # c=2: M2 = (A00)(B01-B11)
    U[0, 2] = 1               # A00
    V[1, 2] = 1; V[3, 2] = -1 # B01, -B11
    W[1, 2] = 1; W[3, 2] = -1 # C01, -C11
    
    # c=3: M3 = (A11)(B10-B00)
    U[3, 3] = 1               # A11
    V[2, 3] = 1; V[0, 3] = -1 # B10, -B00
    W[2, 3] = 1; W[0, 3] = -1 # C10, -C00
    
    # c=4: M4 = (A00+A01)(B11)
    U[0, 4] = 1; U[1, 4] = 1  # A00, A01
    V[3, 4] = 1               # B11
    W[1, 4] = 1               # C01
    
    # c=5: M5 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1 # A10, -A00
    V[0, 5] = 1; V[1, 5] = 1  # B00, B01
    W[0, 5] = 1               # C00
    
    # c=6: M6 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1 # A01, -A11
    V[2, 6] = 1; V[3, 6] = 1  # B10, B11
    W[3, 6] = 1               # C11
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the factor matrices correctly decompose matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For all i in [0,n), a in [0,m), b in [0,m), k in [0,p):
    #   sum_c W[k*n+i, c] * U[i*m+a, c] * V[b*p+k, c] = delta[a,b]
    
    # Build the tensor T[i,a,b,k] = sum_c W[k*n+i,c] * U[i*m+a,c] * V[b*p+k,c]
    # This should equal delta[a,b] for all i,k
    
    # Efficient check using einsum
    # T[i,a,b,k] = sum_c W[k*n+i, c] * U[i*m+a, c] * V[b*p+k, c]
    
    # Reshape for einsum:
    # U has shape (n*m, rank), index (i,a) -> row i*m+a
    # V has shape (m*p, rank), index (b,k) -> row b*p+k  
    # W has shape (p*n, rank), index (k,i) -> row k*n+i
    
    rank = U.shape[1]
    
    # Check a subset if too large, but for n=3,m=5,p=7 it's manageable
    # Total checks: n*m*m*p = 3*5*5*7 = 525, each with rank multiplications
    
    max_err = 0.0
    for i in range(n):
        for a in range(m):
            for b in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += W[k * n + i][c] * U[i * m + a][c] * V[b * p + k][c]
                    expected = 1.0 if a == b else 0.0
                    err = abs(s - expected)
                    if err > max_err:
                        max_err = err
                    if err > tol:
                        return False, max_err
    return True, max_err


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication
    using block Strassen where possible.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Block partitions
    I_blocks = [(0, 2), (2, 1)]       # I0: indices 0,1 (size 2); I1: index 2 (size 1)
    J_blocks = [(0, 2), (2, 2), (4, 1)]  # J0: 0,1; J1: 2,3; J2: 4
    K_blocks = [(0, 2), (2, 2), (4, 2), (6, 1)]  # K0: 0,1; K1: 2,3; K2: 4,5; K3: 6
    
    # Precompute Strassen factors
    U_str, V_str, W_str = _strassen_2x2x2_factors()
    
    # Calculate total rank
    total_rank = 0
    block_info = []
    
    for (i_start, i_size) in I_blocks:
        for (j_start, j_size) in J_blocks:
            for (k_start, k_size) in K_blocks:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    rank_contrib = 7  # Strassen
                    method = 'strassen'
                else:
                    rank_contrib = i_size * j_size * k_size  # Schoolbook
                    method = 'schoolbook'
                total_rank += rank_contrib
                block_info.append((i_start, i_size, j_start, j_size, k_start, k_size, rank_contrib, method))
    
    # Initialize factor matrices
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    for (i_start, i_size, j_start, j_size, k_start, k_size, rank_contrib, method) in block_info:
        if method == 'strassen':
            # Embed Strassen factors
            for local_i in range(2):
                for local_a in range(2):
                    global_i = i_start + local_i
                    global_a = j_start + local_a
                    global_u_row = global_i * m + global_a
                    for c in range(7):
                        U[global_u_row, col_offset + c] = U_str[local_i * 2 + local_a, c]
            
            for local_b in range(2):
                for local_k in range(2):
                    global_b = j_start + local_b
                    global_k = k_start + local_k
                    global_v_row = global_b * p + global_k
                    for c in range(7):
                        V[global_v_row, col_offset + c] = V_str[local_b * 2 + local_k, c]
            
            for local_k in range(2):
                for local_i in range(2):
                    global_k = k_start + local_k
                    global_i = i_start + local_i
                    global_w_row = global_k * n + global_i
                    for c in range(7):
                        W[global_w_row, col_offset + c] = W_str[local_k * 2 + local_i, c]
            
            col_offset += 7
        
        else:
            # Schoolbook for this block
            for local_i in range(i_size):
                for local_a in range(j_size):
                    for local_k in range(k_size):
                        global_i = i_start + local_i
                        global_a = j_start + local_a
                        global_k = k_start + local_k
                        
                        global_u_row = global_i * m + global_a
                        global_v_row = global_a * p + global_k
                        global_w_row = global_k * n + global_i
                        
                        U[global_u_row, col_offset] = 1
                        V[global_v_row, col_offset] = 1
                        W[global_w_row, col_offset] = 1
                        col_offset += 1
    
    # Verify the decomposition
    valid, max_err = _verify_decomposition(U, V, W, n, m, p)
    
    if not valid:
        # Fallback to pure schoolbook
        rank = n * m * p
        U_f = np.zeros((n * m, rank))
        V_f = np.zeros((m * p, rank))
        W_f = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, col] = 1
                    V_f[j * p + k, col] = 1
                    W_f[k * n + i, col] = 1
                    col += 1
        U, V, W = U_f, V_f, W_f
    
    # Convert to lists of lists with integer/half-integer entries
    U_list = [[int(round(val)) if abs(val - round(val)) < 1e-9 else (2*int(round(val*2)))/2 for val in row] for row in U]
    V_list = [[int(round(val)) if abs(val - round(val)) < 1e-9 else (2*int(round(val*2)))/2 for val in row] for row in V]
    W_list = [[int(round(val)) if abs(val - round(val)) < 1e-9 else (2*int(round(val*2)))/2 for val in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
