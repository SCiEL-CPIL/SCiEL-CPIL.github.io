"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    # 1. Build Schoolbook Decomposition (Rank 40) as guaranteed fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # 2. Strassen-based decomposition (Rank 36)
    # Structure: 
    # A (2x4) = [A1, A2] where A1, A2 are 2x2
    # B (4x5) = [B1; B2] where B1, B2 are 2x5
    # C = A1 B1 + A2 B2
    #
    # B1 (2x5) split into B1_01 (2x2), B1_23 (2x2), B1_4 (2x1)
    # B2 (2x5) split into B2_01 (2x2), B2_23 (2x2), B2_4 (2x1)
    #
    # Cost:
    # A1 B1_01: 2x2 * 2x2 -> Rank 7 (Strassen)
    # A1 B1_23: 2x2 * 2x2 -> Rank 7 (Strassen)
    # A1 B1_4:  2x2 * 2x1 -> Rank 4 (Schoolbook)
    # A2 B2_01: 2x2 * 2x2 -> Rank 7 (Strassen)
    # A2 B2_23: 2x2 * 2x2 -> Rank 7 (Strassen)
    # A2 B2_4:  2x2 * 2x1 -> Rank 4 (Schoolbook)
    # Total Rank = 7+7+4+7+7+4 = 36
    
    total_rank = 36
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

    def add_2x2x2_block(a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off, col_off):
        """
        Adds a Strassen decomposition for a 2x2 * 2x2 block.
        Indices:
        A: i in [0,1], j in [0,1] -> global i=a_i_off+i, j=a_j_off+j
        B: j in [0,1], k in [0,1] -> global j=b_j_off+j, k=b_k_off+k
        C: i in [0,1], k in [0,1] -> global i=c_i_off+i, k=c_k_off+k
        """
        # Local indices for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Strassen Formulas:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Patterns: (u_vec, v_vec, w_vec)
        # u_vec corresponds to A elements [A00, A01, A10, A11]
        # v_vec corresponds to B elements [B00, B01, B10, B11]
        # w_vec corresponds to C elements [C00, C01, C10, C11]
        
        patterns = [
            # M1
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            # M2
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            # M3
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            # M4
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            # M5
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            # M6
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            # M7
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]

        for u_pat, v_pat, w_pat in patterns:
            # Fill U
            for idx, val in enumerate(u_pat):
                if val != 0:
                    li = idx // 2
                    lj = idx % 2
                    gi = a_i_off + li
                    gj = a_j_off + lj
                    U[gi * m + gj, col_off] = val
            
            # Fill V
            for idx, val in enumerate(v_pat):
                if val != 0:
                    lj = idx // 2
                    lk = idx % 2
                    gj = b_j_off + lj
                    gk = b_k_off + lk
                    V[gj * p + gk, col_off] = val
            
            # Fill W
            for idx, val in enumerate(w_pat):
                if val != 0:
                    li = idx // 2
                    lk = idx % 2
                    gi = c_i_off + li
                    gk = c_k_off + lk
                    W[gk * n + gi, col_off] = val
            
            col_off += 1
        
        return col_off

    def add_2x2x1_block(a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off, col_off):
        """
        Adds a schoolbook decomposition for a 2x2 * 2x1 block.
        Indices:
        A: i in [0,1], j in [0,1] -> global i=a_i_off+i, j=a_j_off+j
        B: j in [0,1], k=0 -> global j=b_j_off+j, k=b_k_off+0
        C: i in [0,1], k=0 -> global i=c_i_off+i, k=c_k_off+0
        """
        # C_i = A_i0 * B_0 + A_i1 * B_1
        # Terms:
        # 1. A00 * B0 -> C0
        # 2. A01 * B1 -> C0
        # 3. A10 * B0 -> C1
        # 4. A11 * B1 -> C1
        
        patterns = [
            # Term 1
            ([1, 0, 0, 0], [1, 0], [1, 0]),
            # Term 2
            ([0, 1, 0, 0], [0, 1], [1, 0]),
            # Term 3
            ([0, 0, 1, 0], [1, 0], [0, 1]),
            # Term 4
            ([0, 0, 0, 1], [0, 1], [0, 1]),
        ]

        for u_pat, v_pat, w_pat in patterns:
            # Fill U
            for idx, val in enumerate(u_pat):
                if val != 0:
                    li = idx // 2
                    lj = idx % 2
                    gi = a_i_off + li
                    gj = a_j_off + lj
                    U[gi * m + gj, col_off] = val
            
            # Fill V
            for idx, val in enumerate(v_pat):
                if val != 0:
                    lj = idx
                    gk = 0
                    gj = b_j_off + lj
                    gk_global = b_k_off + gk
                    V[gj * p + gk_global, col_off] = val
            
            # Fill W
            for idx, val in enumerate(w_pat):
                if val != 0:
                    li = idx
                    gk = 0
                    gi = c_i_off + li
                    gk_global = c_k_off + gk
                    W[gk_global * n + gi, col_off] = val
            
            col_off += 1
        
        return col_off

    col = 0
    
    # Block 1: A1 (cols 0,1) * B1 (rows 0,1)
    # B1 cols 0,1 -> Strassen
    col = add_2x2x2_block(0, 0, 0, 0, 0, 0, col)
    # B1 cols 2,3 -> Strassen
    col = add_2x2x2_block(0, 0, 0, 2, 0, 2, col)
    # B1 col 4 -> Schoolbook
    col = add_2x2x1_block(0, 0, 0, 4, 0, 4, col)

    # Block 2: A2 (cols 2,3) * B2 (rows 2,3)
    # B2 cols 0,1 -> Strassen
    col = add_2x2x2_block(0, 2, 2, 0, 0, 0, col)
    # B2 cols 2,3 -> Strassen
    col = add_2x2x2_block(0, 2, 2, 2, 0, 2, col)
    # B2 col 4 -> Schoolbook
    col = add_2x2x1_block(0, 2, 2, 4, 0, 4, col)

    if col != total_rank:
        raise RuntimeError(f"Rank mismatch: expected {total_rank}, got {col}")

    # 3. Verification using vectorized operations
    def verify_decomposition(U_mat, V_mat, W_mat):
        # Build the full tensor contraction to check correctness
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We need T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        # And T[a,b,c] = 0 for all other (a,b,c) where the indices don't match
        
        # Efficient check: for each r, compute outer product U[:,r] * V[:,r] * W[:,r]
        # and accumulate
        # But that's O(rank * nm * mp * pn) = 36 * 8 * 20 * 10 = 57600, which is fine
        
        # Check diagonal terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(U_mat[a], V_mat[b] * W_mat[c])
                    if abs(s - 1.0) > 1e-9:
                        return False
        
        # Check off-diagonal: for each a,b with matching j, check all c
        for a in range(nm):
            i_a = a // m
            j_a = a % m
            for b in range(mp):
                j_b = b // p
                k_b = b % p
                if j_a != j_b:
                    continue
                for c in range(pn):
                    k_c = c // n
                    i_c = c % n
                    if k_b == k_c and i_a == i_c:
                        continue  # diagonal, already checked
                    s = np.dot(U_mat[a], V_mat[b] * W_mat[c])
                    if abs(s) > 1e-9:
                        return False
        return True

    if time.time() < deadline:
        if verify_decomposition(U, V, W):
            # Convert to list of lists with integer/half-integer values
            U_list = [[int(round(U[i, r])) for r in range(total_rank)] for i in range(nm)]
            V_list = [[int(round(V[i, r])) for r in range(total_rank)] for i in range(mp)]
            W_list = [[int(round(W[i, r])) for r in range(total_rank)] for i in range(pn)]
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook if verification fails or time runs out
    return best_result

# EVOLVE-BLOCK-END
