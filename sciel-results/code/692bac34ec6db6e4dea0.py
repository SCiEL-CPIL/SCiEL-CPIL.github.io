"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-reduced exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x7 matrix using a block decomposition strategy.
    
    Strategy:
    Split A (2x3) into A1 (2x2) and A2 (2x1).
    Split B (3x7) into B1 (2x7) and B2 (1x7).
    Compute C = A1*B1 + A2*B2.
    
    Split columns of B1 and B2 into blocks of size 2, 2, 2, 1.
    For 2-column blocks:
      - A1*B1_block (2x2 by 2x2) uses Strassen (rank 7).
      - A2*B2_block (2x1 by 1x2) uses schoolbook (rank 4).
      Total rank per 2-col block: 11.
    For 1-column block:
      - A1*B1_block (2x2 by 2x1) uses schoolbook (rank 4).
      - A2*B2_block (2x1 by 1x1) uses schoolbook (rank 2).
      Total rank per 1-col block: 6.
      
    Total rank: 3 * 11 + 6 = 39.
    """
    n, m, p = DIMENSIONS
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)] # 6 x 39
    V = [[0] * rank for _ in range(m * p)] # 21 x 39
    W = [[0] * rank for _ in range(p * n)] # 14 x 39
    
    col_idx = 0
    
    # Helper to add a rank-1 term
    def add_term(u_indices, v_indices, w_indices, coeff=1):
        nonlocal col_idx
        for u in u_indices:
            U[u][col_idx] += coeff
        for v in v_indices:
            V[v][col_idx] += coeff
        for w in w_indices:
            W[w][col_idx] += coeff
        col_idx += 1

    # Indices for A (2x3)
    # A[i][j] -> index i*3 + j
    # A1: i in {0,1}, j in {0,1}
    # A2: i in {0,1}, j = 2
    
    # Indices for B (3x7)
    # B[j][k] -> index j*7 + k
    # B1: j in {0,1}, k in {0..6}
    # B2: j = 2, k in {0..6}
    
    # Indices for C (2x7)
    # C[i][k] -> index i*7 + k
    
    # Define Strassen terms for 2x2 by 2x2
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 - M4 + M5 + M6
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M2 + M3 + M7
    
    def add_strassen_block(k_start):
        # k_start is 0, 2, 4
        # B1 columns are k_start, k_start+1
        # B2 columns are k_start, k_start+1
        
        # A1 indices
        a_idx = 0*3 + 0
        b_idx = 0*3 + 1
        c_idx = 1*3 + 0
        d_idx = 1*3 + 1
        
        # A2 indices
        p_idx = 0*3 + 2
        q_idx = 1*3 + 2
        
        # B1 indices for this block
        e_idx = 0*7 + k_start
        f_idx = 0*7 + (k_start + 1)
        g_idx = 1*7 + k_start
        h_idx = 1*7 + (k_start + 1)
        
        # B2 indices for this block
        r_idx = 2*7 + k_start
        s_idx = 2*7 + (k_start + 1)
        
        # C indices for this block
        c11_idx = 0*7 + k_start
        c12_idx = 0*7 + (k_start + 1)
        c21_idx = 1*7 + k_start
        c22_idx = 1*7 + (k_start + 1)
        
        # --- Strassen Part (Rank 7) ---
        
        # M1 = (a+d)(e+h)
        add_term([a_idx, d_idx], [e_idx, h_idx], [])
        # We need to accumulate W contributions separately or handle in a way that W is sum of products.
        # The standard decomposition is C = sum M_k * W_k.
        # Here M_k is the scalar product. W_k is the matrix added to C.
        # My add_term function adds coeff to U, V, and W.
        # This implies the term is (sum U * A) * (sum V * B) * (W * C).
        # Wait, the tensor decomposition is C = sum_k (sum_i U_{i,k} A_i) (sum_j V_{j,k} B_j) W_k.
        # W_k is a vector in the output space.
        # So for each term, we specify which A entries are multiplied, which B entries, and which C entries receive the result (with what coefficient).
        
        # Let's refine add_term to take a list of (index, coeff) for W.
        
        col_idx -= 1 # rollback for previous dummy call if any, but I used [] for W.
        # Actually, let's rewrite add_term to be more flexible.
        
    # Redefine add_term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        """
        u_coeffs: dict {idx: coeff}
        v_coeffs: dict {idx: coeff}
        w_coeffs: dict {idx: coeff}
        """
        nonlocal col_idx
        for idx, c in u_coeffs.items():
            U[idx][col_idx] += c
        for idx, c in v_coeffs.items():
            V[idx][col_idx] += c
        for idx, c in w_coeffs.items():
            W[idx][col_idx] += c
        col_idx += 1

    # Re-implement Strassen block addition
    def add_strassen_block(k_start):
        a_idx = 0
        b_idx = 1
        c_idx = 3
        d_idx = 4
        
        p_idx = 2
        q_idx = 5
        
        e_idx = 0*7 + k_start
        f_idx = 0*7 + (k_start + 1)
        g_idx = 1*7 + k_start
        h_idx = 1*7 + (k_start + 1)
        
        r_idx = 2*7 + k_start
        s_idx = 2*7 + (k_start + 1)
        
        c11_idx = 0*7 + k_start
        c12_idx = 0*7 + (k_start + 1)
        c21_idx = 1*7 + k_start
        c22_idx = 1*7 + (k_start + 1)
        
        # M1 = (a+d)(e+h)
        # Contributes to C11 (+1), C22 (+1)
        add_term({a_idx: 1, d_idx: 1}, {e_idx: 1, h_idx: 1}, {c11_idx: 1, c22_idx: 1})
        
        # M2 = (c+d)e
        # Contributes to C21 (+1), C22 (-1)
        add_term({c_idx: 1, d_idx: 1}, {e_idx: 1}, {c21_idx: 1, c22_idx: -1})
        
        # M3 = a(f-h)
        # Contributes to C12 (+1), C22 (+1)
        add_term({a_idx: 1}, {f_idx: 1, h_idx: -1}, {c12_idx: 1, c22_idx: 1})
        
        # M4 = d(g-e)
        # Contributes to C11 (-1), C12 (+1)
        add_term({d_idx: 1}, {g_idx: 1, e_idx: -1}, {c11_idx: -1, c12_idx: 1})
        
        # M5 = (a+b)h
        # Contributes to C11 (+1), C21 (+1)
        add_term({a_idx: 1, b_idx: 1}, {h_idx: 1}, {c11_idx: 1, c21_idx: 1})
        
        # M6 = (c-a)(e+f)
        # Contributes to C11 (+1)
        add_term({c_idx: 1, a_idx: -1}, {e_idx: 1, f_idx: 1}, {c11_idx: 1})
        
        # M7 = (b-d)(g+h)
        # Contributes to C22 (+1)
        add_term({b_idx: 1, d_idx: -1}, {g_idx: 1, h_idx: 1}, {c22_idx: 1})
        
        # --- Outer Product Part (Rank 4) ---
        # A2 = [p, q]^T, B2 = [r, s]
        # C = [[pr, ps], [qr, qs]]
        
        # M8 = pr -> C11
        add_term({p_idx: 1}, {r_idx: 1}, {c11_idx: 1})
        
        # M9 = ps -> C12
        add_term({p_idx: 1}, {s_idx: 1}, {c12_idx: 1})
        
        # M10 = qr -> C21
        add_term({q_idx: 1}, {r_idx: 1}, {c21_idx: 1})
        
        # M11 = qs -> C22
        add_term({q_idx: 1}, {s_idx: 1}, {c22_idx: 1})

    # Add 3 blocks of size 2
    for k_start in [0, 2, 4]:
        add_strassen_block(k_start)
        
    # Add 1 block of size 1 (k=6)
    k_start = 6
    a_idx = 0
    b_idx = 1
    c_idx = 3
    d_idx = 4
    p_idx = 2
    q_idx = 5
    
    e_idx = 0*7 + k_start
    g_idx = 1*7 + k_start
    r_idx = 2*7 + k_start
    
    c1_idx = 0*7 + k_start
    c2_idx = 1*7 + k_start
    
    # A1 (2x2) * B1_col (2x1)
    # A = [[a,b],[c,d]], v = [e, g]^T (Note: B1 row 0 is e, row 1 is g? No, B1 is 2x7. 
    # In previous block, e was B[0, k], f was B[0, k+1]. g was B[1, k], h was B[1, k+1].
    # So for single col k, B1 entries are B[0,k] and B[1,k].
    # Let's call them x and y.
    x_idx = 0*7 + k_start
    y_idx = 1*7 + k_start
    
    # M1 = ax -> C1
    add_term({a_idx: 1}, {x_idx: 1}, {c1_idx: 1})
    # M2 = by -> C1
    add_term({b_idx: 1}, {y_idx: 1}, {c1_idx: 1})
    # M3 = cx -> C2
    add_term({c_idx: 1}, {x_idx: 1}, {c2_idx: 1})
    # M4 = dy -> C2
    add_term({d_idx: 1}, {y_idx: 1}, {c2_idx: 1})
    
    # A2 (2x1) * B2_col (1x1)
    # u = [p, q]^T, v = r
    # M5 = pr -> C1
    add_term({p_idx: 1}, {r_idx: 1}, {c1_idx: 1})
    # M6 = qr -> C2
    add_term({q_idx: 1}, {r_idx: 1}, {c2_idx: 1})

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
