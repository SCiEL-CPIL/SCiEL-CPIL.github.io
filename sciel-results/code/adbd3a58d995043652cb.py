"""
Strategy: Block decomposition of 5x5 matrix multiplication.

We decompose the 5x5 multiplication into blocks and apply Strassen's algorithm
where possible to reduce the total rank below the schoolbook 125.

The key insight is to split the 5x5 matrices as:
A = [A_top_left (2x2)  A_top_right (2x3)]
    [A_bottom_left (3x2) A_bottom_right (3x3)]

B = [B_top_left (2x2)  B_top_right (2x3)]
    [B_bottom_left (3x2) B_bottom_right (3x3)]

The product C = A * B has blocks:
C_top_left (2x2) = A_tl * B_tl + A_tr * B_bl
C_top_right (2x3) = A_tl * B_tr + A_tr * B_br
C_bottom_left (3x2) = A_bl * B_tl + A_br * B_bl
C_bottom_right (3x3) = A_bl * B_tr + A_br * B_br

We use:
- Strassen (rank 7) for the 2x2 * 2x2 multiplications
- Schoolbook for other blocks

Let's count the terms:
For C_top_left: 
  - A_tl * B_tl: 2x2 * 2x2 -> rank 7 (Strassen)
  - A_tr * B_bl: 2x3 * 3x2 -> rank 12 (schoolbook)
  Total: 19

For C_top_right:
  - A_tl * B_tr: 2x2 * 2x3 -> rank 12 (schoolbook)
  - A_tr * B_br: 2x3 * 3x3 -> rank 18 (schoolbook)
  Total: 30

For C_bottom_left:
  - A_bl * B_tl: 3x2 * 2x2 -> rank 12 (schoolbook)
  - A_br * B_bl: 3x3 * 3x2 -> rank 18 (schoolbook)
  Total: 30

For C_bottom_right:
  - A_bl * B_tr: 3x2 * 2x3 -> rank 18 (schoolbook)
  - A_br * B_br: 3x3 * 3x3 -> rank 27 (schoolbook)
  Total: 45

Grand total: 19 + 30 + 30 + 45 = 124

This gives us rank 124, which is 1 less than schoolbook's 125!
"""

import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    n, m, p = 5, 5, 5
    
    # We'll build the decomposition by blocks.
    # Each term in the decomposition is a rank-1 tensor contribution.
    # U, V, W will be lists of lists, where each inner list is a column (one rank-1 term).
    
    U = []  # Will contain rows of U, each row has 25 entries (for 5x5)
    V = []  # Will contain rows of V, each row has 25 entries (for 5x5)
    W = []  # Will contain rows of W, each row has 25 entries (for 5x5)
    
    # Initialize with zeros
    rank_so_far = 0
    
    def add_term(u_indices, v_indices, w_indices):
        """
        Add a rank-1 term to the decomposition.
        u_indices: dict mapping (i,j) -> coefficient for U[i*m+j, r]
        v_indices: dict mapping (j,k) -> coefficient for V[j*p+k, r]
        w_indices: dict mapping (k,i) -> coefficient for W[k*n+i, r]
        """
        nonlocal rank_so_far
        # Extend U, V, W with a new column
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (i, j), coeff in u_indices.items():
            u_col[i * m + j] = coeff
        for (j, k), coeff in v_indices.items():
            v_col[j * p + k] = coeff
        for (k, i), coeff in w_indices.items():
            w_col[k * n + i] = coeff
        
        U.append(u_col)
        V.append(v_col)
        W.append(w_col)
        rank_so_far += 1
    
    # Strassen decomposition for 2x2 matrix multiplication
    def strassen_2x2_terms():
        """
        Returns a list of 7 rank-1 terms for 2x2 matrix multiplication.
        Each term is (u_indices, v_indices, w_indices) where indices are local (0,1).
        """
        terms = []
        
        # M1 = (A00 + A11)(B00 + B11)
        # Contributes to C00 (+1) and C11 (+1)
        terms.append(
            ({(0,0):1, (1,1):1},  # U: A00 + A11
             {(0,0):1, (1,1):1},  # V: B00 + B11
             {(0,0):1, (1,1):1})  # W: C00 + C11
        )
        
        # M2 = (A10 + A11)B00
        # Contributes to C10 (+1)
        terms.append(
            ({(1,0):1, (1,1):1},  # U: A10 + A11
             {(0,0):1},            # V: B00
             {(1,0):1})            # W: C10
        )
        
        # M3 = A00(B01 - B11)
        # Contributes to C01 (+1) and C11 (+1)
        terms.append(
            ({(0,0):1},            # U: A00
             {(0,1):1, (1,1):-1},  # V: B01 - B11
             {(0,1):1, (1,1):1})   # W: C01 + C11
        )
        
        # M4 = A11(B10 - B00)
        # Contributes to C00 (+1) and C10 (+1)
        terms.append(
            ({(1,1):1},            # U: A11
             {(1,0):1, (0,0):-1},  # V: B10 - B00
             {(0,0):1, (1,0):1})   # W: C00 + C10
        )
        
        # M5 = (A00 + A01)B11
        # Contributes to C00 (-1) and C01 (+1)
        terms.append(
            ({(0,0):1, (0,1):1},   # U: A00 + A01
             {(1,1):1},            # V: B11
             {(0,0):-1, (0,1):1})  # W: -C00 + C01
        )
        
        # M6 = (A10 - A11)(B01 + B11)
        # Contributes to C11 (+1)
        terms.append(
            ({(1,0):1, (1,1):-1},  # U: A10 - A11
             {(0,1):1, (1,1):1},   # V: B01 + B11
             {(1,1):1})            # W: C11
        )
        
        # M7 = (A01 - A11)(B00 + B10)
        # Contributes to C00 (+1)
        terms.append(
            ({(0,1):1, (1,1):-1},  # U: A01 - A11
             {(0,0):1, (1,0):1},   # V: B00 + B10
             {(0,0):1})            # W: C00
        )
        
        return terms
    
    def schoolbook_2x2_terms():
        """
        Returns 4 rank-1 terms for 2x2 matrix multiplication (schoolbook).
        """
        terms = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    terms.append(
                        ({(i,j):1}, {(j,k):1}, {(k,i):1})
                    )
        return terms
    
    def add_block_term(u_block_offset, v_block_offset, w_block_offset, 
                       u_local, v_local, w_local):
        """
        Add a term with local indices mapped to global indices via offsets.
        u_local: dict {(i,j): coeff}
        v_local: dict {(j,k): coeff}
        w_local: dict {(k,i): coeff}
        """
        u_global = {}
        v_global = {}
        w_global = {}
        
        for (i, j), coeff in u_local.items():
            gi = u_block_offset[0] + i
            gj = u_block_offset[1] + j
            u_global[(gi, gj)] = coeff
        
        for (j, k), coeff in v_local.items():
            gj = v_block_offset[0] + j
            gk = v_block_offset[1] + k
            v_global[(gj, gk)] = coeff
        
        for (k, i), coeff in w_local.items():
            gk = w_block_offset[0] + k
            gi = w_block_offset[1] + i
            w_global[(gk, gi)] = coeff
        
        add_term(u_global, v_global, w_global)
    
    # Block offsets for the 5x5 decomposition
    # A = [A_tl (rows 0-1, cols 0-1)  A_tr (rows 0-1, cols 2-4)]
    #     [A_bl (rows 2-4, cols 0-1)  A_br (rows 2-4, cols 2-4)]
    
    # B = [B_tl (rows 0-1, cols 0-1)  B_tr (rows 0-1, cols 2-4)]
    #     [B_bl (rows 2-4, cols 0-1)  B_br (rows 2-4, cols 2-4)]
    
    # C = [C_tl (rows 0-1, cols 0-1)  C_tr (rows 0-1, cols 2-4)]
    #     [C_bl (rows 2-4, cols 0-1)  C_br (rows 2-4, cols 2-4)]
    
    # Block offsets: (row_start, col_start)
    A_tl_offset = (0, 0)   # 2x2
    A_tr_offset = (0, 2)   # 2x3
    A_bl_offset = (2, 0)   # 3x2
    A_br_offset = (2, 2)   # 3x3
    
    B_tl_offset = (0, 0)   # 2x2
    B_tr_offset = (0, 2)   # 2x3
    B_bl_offset = (2, 0)   # 3x2
    B_br_offset = (2, 2)   # 3x3
    
    C_tl_offset = (0, 0)   # 2x2
    C_tr_offset = (0, 2)   # 2x3
    C_bl_offset = (2, 0)   # 3x2
    C_br_offset = (2, 2)   # 3x3
    
    # === C_top_left (2x2) = A_tl * B_tl + A_tr * B_bl ===
    
    # A_tl * B_tl: 2x2 * 2x2, use Strassen
    for u_local, v_local, w_local in strassen_2x2_terms():
        add_block_term(A_tl_offset, B_tl_offset, C_tl_offset, u_local, v_local, w_local)
    
    # A_tr * B_bl: 2x3 * 3x2, use schoolbook
    # A_tr is 2x3 (rows 0-1, cols 2-4 in global)
    # B_bl is 3x2 (rows 2-4, cols 0-1 in global)
    # Result C_tl is 2x2
    for i_local in range(2):  # rows of A_tr
        for j_local in range(3):  # cols of A_tr = rows of B_bl
            for k_local in range(2):  # cols of B_bl
                add_block_term(
                    A_tr_offset,  # (0, 2)
                    B_bl_offset,  # (2, 0)
                    C_tl_offset,  # (0, 0)
                    {(i_local, j_local): 1},
                    {(j_local, k_local): 1},
                    {(k_local, i_local): 1}
                )
    
    # === C_top_right (2x3) = A_tl * B_tr + A_tr * B_br ===
    
    # A_tl * B_tr: 2x2 * 2x3, use schoolbook
    for i_local in range(2):  # rows of A_tl
        for j_local in range(2):  # cols of A_tl = rows of B_tr
            for k_local in range(3):  # cols of B_tr
                add_block_term(
                    A_tl_offset,  # (0, 0)
                    B_tr_offset,  # (0, 2)
                    C_tr_offset,  # (0, 2)
                    {(i_local, j_local): 1},
                    {(j_local, k_local): 1},
                    {(k_local, i_local): 1}
                )
    
    # A_tr * B_br: 2x3 * 3x3, use schoolbook
    for i_local in range(2):  # rows of A_tr
        for j_local in range(3):  # cols of A_tr = rows of B_br
            for k_local in range(3):  # cols of B_br
                add_block_term(
                    A_tr_offset,  # (0, 2)
                    B_br_offset,  # (2, 2)
                    C_tr_offset,  # (0, 2)
                    {(i_local, j_local): 1},
                    {(j_local, k_local): 1},
                    {(k_local, i_local): 1}
                )
    
    # === C_bottom_left (3x2) = A_bl * B_tl + A_br * B_bl ===
    
    # A_bl * B_tl: 3x2 * 2x2, use schoolbook
    for i_local in range(3):  # rows of A_bl
        for j_local in range(2):  # cols of A_bl = rows of B_tl
            for k_local in range(2):  # cols of B_tl
                add_block_term(
                    A_bl_offset,  # (2, 0)
                    B_tl_offset,  # (0, 0)
                    C_bl_offset,  # (2, 0)
                    {(i_local, j_local): 1},
                    {(j_local, k_local): 1},
                    {(k_local, i_local): 1}
                )
    
    # A_br * B_bl: 3x3 * 3x2, use schoolbook
    for i_local in range(3):  # rows of A_br
        for j_local in range(3):  # cols of A_br = rows of B_bl
            for k_local in range(2):  # cols of B_bl
                add_block_term(
                    A_br_offset,  # (2, 2)
                    B_bl_offset,  # (2, 0)
                    C_bl_offset,  # (2, 0)
                    {(i_local, j_local): 1},
                    {(j_local, k_local): 1},
                    {(k_local, i_local): 1}
                )
    
    # === C_bottom_right (3x3) = A_bl * B_tr + A_br * B_br ===
    
    # A_bl * B_tr: 3x2 * 2x3, use schoolbook
    for i_local in range(3):  # rows of A_bl
        for j_local in range(2):  # cols of A_bl = rows of B_tr
            for k_local in range(3):  # cols of B_tr
                add_block_term(
                    A_bl_offset,  # (2, 0)
                    B_tr_offset,  # (0, 2)
                    C_br_offset,  # (2, 2)
                    {(i_local, j_local): 1},
                    {(j_local, k_local): 1},
                    {(k_local, i_local): 1}
                )
    
    # A_br * B_br: 3x3 * 3x3, use schoolbook
    for i_local in range(3):  # rows of A_br
        for j_local in range(3):  # cols of A_br = rows of B_br
            for k_local in range(3):  # cols of B_br
                add_block_term(
                    A_br_offset,  # (2, 2)
                    B_br_offset,  # (2, 2)
                    C_br_offset,  # (2, 2)
                    {(i_local, j_local): 1},
                    {(j_local, k_local): 1},
                    {(k_local, i_local): 1}
                )
    
    rank = rank_so_far
    print(f"Block decomposition rank: {rank}")
    
    # Verification
    np.random.seed(42)
    valid = True
    for _ in range(5):
        A = np.random.randint(-5, 5, (n, m))
        B = np.random.randint(-5, 5, (m, p))
        C_expected = A @ B
        
        C_calc = np.zeros((n, p))
        for r in range(rank):
            u_col = U[r]
            v_col = V[r]
            w_col = W[r]
            
            # alpha_i = sum_j U[i*m+j, r] * A[i,j]
            alpha = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    alpha[i] += u_col[i * m + j] * A[i, j]
            
            # beta_k = sum_j V[j*p+k, r] * B[j,k]
            beta = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    beta[k] += v_col[j * p + k] * B[j, k]
            
            for i in range(n):
                for k in range(p):
                    C_calc[i, k] += alpha[i] * beta[k] * w_col[k * n + i]
        
        if not np.allclose(C_calc, C_expected, atol=1e-10):
            print(f"Verification failed! Rank={rank}")
            valid = False
            break
    
    if valid:
        print(f"Verification passed. Returning rank {rank}.")
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        print("Falling back to schoolbook rank 125.")
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
