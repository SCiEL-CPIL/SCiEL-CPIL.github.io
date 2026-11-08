"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time

def _make_2x3x9_rank_50():
    # Returns U (6x50), V (27x50), W (18x50) as lists of lists.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=2, m=3, p=9.
    n, m, p = 2, 3, 9
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    c = 0

    def add_term(U_entries, V_entries, W_entries, c_idx):
        for (u_idx, u_val) in U_entries:
            U[u_idx][c_idx] += u_val
        for (v_idx, v_val) in V_entries:
            V[v_idx][c_idx] += v_val
        for (w_idx, w_val) in W_entries:
            W[w_idx][c_idx] += w_val

    # --- Part A: 2x2x9 (rank 32) ---
    # The tensor decomposes into independent parts:
    # 1. A_top (2x2, cols j=0,1) * B_top (2x9, rows j=0,1) -> contributes to C
    # 2. A_bot (2x1, col j=2) * B_bot (1x9, row j=2) -> contributes to C
    # These are independent because the sets of A and B indices are disjoint.
    
    # Part A: 2x2x9
    # We can split the 9 columns of B_top into 4 pairs and 1 single.
    # For each pair (k, k+1), we use Strassen (rank 7).
    # For the single column (k=8), we use schoolbook (rank 4).
    # Total rank for 2x2x9 = 4 * 7 + 4 = 32.
    
    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1
        # Strassen for (2x2) * (2x2)
        # A = [[a00, a01], [a10, a11]]  (indices i*m+j)
        # B = [[b00, b01], [b10, b11]]  (indices j*p+k)
        # C = [[c00, c01], [c10, c11]]  (indices k*n+i)
        
        # M1 = (a00 + a11) * (b00 + b11)
        add_term(
            [(0*m+0, 1), (1*m+1, 1)],
            [(0*p+k0, 1), (1*p+k1, 1)],
            [(k0*n+0, 1), (k0*n+1, 1), (k1*n+0, 1), (k1*n+1, 1)],
            c
        )
        c += 1
        
        # M2 = (a10 + a11) * b00
        add_term(
            [(1*m+0, 1), (1*m+1, 1)],
            [(0*p+k0, 1)],
            [(k0*n+1, 1), (k1*n+1, -1)],
            c
        )
        c += 1
        
        # M3 = a00 * (b11 - b01)
        add_term(
            [(0*m+0, 1)],
            [(1*p+k1, 1), (0*p+k1, -1)],
            [(k1*n+0, 1), (k1*n+1, 1)],
            c
        )
        c += 1
        
        # M4 = a11 * (b00 - b10)
        add_term(
            [(1*m+1, 1)],
            [(0*p+k0, 1), (1*p+k0, -1)],
            [(k0*n+0, -1), (k0*n+1, 1)],
            c
        )
        c += 1
        
        # M5 = (a00 + a01) * b11
        add_term(
            [(0*m+0, 1), (0*m+1, 1)],
            [(1*p+k1, 1)],
            [(k0*n+0, 1), (k1*n+0, 1)],
            c
        )
        c += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        add_term(
            [(1*m+0, 1), (0*m+0, -1)],
            [(0*p+k0, 1), (0*p+k1, 1)],
            [(k1*n+1, 1)],
            c
        )
        c += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        add_term(
            [(0*m+1, 1), (1*m+1, -1)],
            [(1*p+k0, 1), (1*p+k1, 1)],
            [(k0*n+0, -1)],
            c
        )
        c += 1

    # Part A remainder: k=8 (2x2x1, rank 4)
    k8 = 8
    # C_{0,8} = a00*b08 + a01*b18
    # C_{1,8} = a10*b08 + a11*b18
    add_term([(0*m+0, 1)], [(0*p+k8, 1)], [(k8*n+0, 1)], c)
    c += 1
    add_term([(0*m+1, 1)], [(1*p+k8, 1)], [(k8*n+0, 1)], c)
    c += 1
    add_term([(1*m+0, 1)], [(0*p+k8, 1)], [(k8*n+1, 1)], c)
    c += 1
    add_term([(1*m+1, 1)], [(1*p+k8, 1)], [(k8*n+1, 1)], c)
    c += 1

    # --- Part B: 2x1x9 (rank 18) ---
    # A_bot (2x1) * B_bot (1x9)
    # C_{i,k} += a_{i2} * b_{2k}
    for i in range(2):
        for k in range(9):
            add_term(
                [(i*m+2, 1)],
                [(2*p+k, 1)],
                [(k*n+i, 1)],
                c
            )
            c += 1

    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    try:
        U, V, W = _make_2x3x9_rank_50()
        if time.time() > deadline:
            raise TimeoutError()
        return {"U": U, "V": V, "W": W}
    except Exception:
        # Fallback to schoolbook if something goes wrong
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
