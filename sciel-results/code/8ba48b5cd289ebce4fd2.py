"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Constructs a tensor decomposition for matrix multiplication C = A @ B
    where A is n x m and B is m x p.
    
    Decomposition form:
    C_k_i = sum_l W_{k,i,l} * ( sum_{i',j'} U_{i',j',l} A_{i',j'} ) * ( sum_{j'',k'} V_{j'',k',l} B_{j'',k'} )
    
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)  <-- Index is k*n + i
    """
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    
    # Deadline guard
    start_time = time.time()
    timeout = 100.0  # Keep well within 120s limit
    
    rank = 36
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def u_idx(i, j): return i*m + j
    def v_idx(j, k): return j*p + k
    def w_idx(k, i): return k*n + i
    
    col = 0
    
    def add_strassen_2x2x2(ai0, ai1, aj0, aj1, bj0, bj1, bk0, bk1):
        nonlocal col
        a00 = u_idx(ai0, aj0)
        a01 = u_idx(ai0, aj1)
        a10 = u_idx(ai1, aj0)
        a11 = u_idx(ai1, aj1)
        
        b00 = v_idx(bj0, bk0)
        b01 = v_idx(bj0, bk1)
        b10 = v_idx(bj1, bk0)
        b11 = v_idx(bj1, bk1)
        
        c00 = w_idx(bk0, ai0)
        c01 = w_idx(bk1, ai0)
        c10 = w_idx(bk0, ai1)
        c11 = w_idx(bk1, ai1)
        
        # M1
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        # M2
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1
        
        # M3
        U[a00][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        
        # M4
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        
        # M5
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        
        # M6
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7
        U[a01][col] += 1; U[a11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1
        col += 1
        
    def add_vec_mat_2x2x1(ai0, ai1, aj0, aj1, bj0, bj1, bk):
        nonlocal col
        a00 = u_idx(ai0, aj0)
        a01 = u_idx(ai0, aj1)
        a10 = u_idx(ai1, aj0)
        a11 = u_idx(ai1, aj1)
        
        b0 = v_idx(bj0, bk)
        b1 = v_idx(bj1, bk)
        
        c0 = w_idx(bk, ai0)
        c1 = w_idx(bk, ai1)
        
        # M1: A00 B0 -> C0
        U[a00][col] = 1; V[b0][col] = 1; W[c0][col] += 1; col += 1
        # M2: A01 B1 -> C0
        U[a01][col] = 1; V[b1][col] = 1; W[c0][col] += 1; col += 1
        # M3: A10 B0 -> C1
        U[a10][col] = 1; V[b0][col] = 1; W[c1][col] += 1; col += 1
        # M4: A11 B1 -> C1
        U[a11][col] = 1; V[b1][col] = 1; W[c1][col] += 1; col += 1
        
    # Block 1: A_L (rows 0,1 cols 0,1) @ B_11 (rows 0,1 cols 0,1) -> C_11 (rows 0,1 cols 0,1)
    add_strassen_2x2x2(0, 1, 0, 1, 0, 1, 0, 1)
    
    # Block 2: A_L @ B_12 (rows 0,1 cols 2,3) -> C_12 (rows 2,3 cols 0,1)
    add_strassen_2x2x2(0, 1, 0, 1, 0, 1, 2, 3)
    
    # Block 3: A_L @ B_13 (rows 0,1 col 4) -> C_13 (row 4 cols 0,1)
    add_vec_mat_2x2x1(0, 1, 0, 1, 0, 1, 4)
    
    # Block 4: A_R (rows 0,1 cols 2,3) @ B_21 (rows 2,3 cols 0,1) -> C_21 (rows 0,1 cols 0,1)
    add_strassen_2x2x2(0, 1, 2, 3, 2, 3, 0, 1)
    
    # Block 5: A_R @ B_22 (rows 2,3 cols 2,3) -> C_22 (rows 2,3 cols 0,1)
    add_strassen_2x2x2(0, 1, 2, 3, 2, 3, 2, 3)
    
    # Block 6: A_R @ B_23 (rows 2,3 col 4) -> C_23 (row 4 cols 0,1)
    add_vec_mat_2x2x1(0, 1, 2, 3, 2, 3, 4)
    
    # Convert to NumPy arrays for compatibility
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    return {"U": U_np, "V": V_np, "W": W_np}

# EVOLVE-BLOCK-END
