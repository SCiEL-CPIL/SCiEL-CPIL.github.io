"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time

def _make_2x3x9_rank_51():
    # Returns U (6x51), V (27x51), W (18x51) as lists of lists.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=2, m=3, p=9.
    n, m, p = 2, 3, 9
    rank = 51
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
    # Terms for k in {0..7} (4 blocks of 2):
    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # M1 = (a00+a11) * (b00+b11)
        # C00 += M1, C11 += M1
        # U: a00, a11
        # V: b00, b11
        # W: C00 (k0*n+0), C11 (k1*n+1)
        add_term(
            [(0*m+0, 1), (1*m+1, 1)], 
            [(0*p+k0, 1), (1*p+k1, 1)], 
            [(k0*n+0, 1), (k1*n+1, 1)], 
            c
        )
        c += 1
        
        # M2 = (a10+a11) * b00
        # C10 += M2, C11 -= M2
        add_term(
            [(1*m+0, 1), (1*m+1, 1)], 
            [(0*p+k0, 1)], 
            [(k0*n+1, 1), (k1*n+1, -1)], 
            c
        )
        c += 1
        
        # M3 = a00 * (b11-b01)
        # C01 += M3, C11 += M3
        add_term(
            [(0*m+0, 1)], 
            [(1*p+k1, 1), (0*p+k1, -1)], 
            [(k1*n+0, 1), (k1*n+1, 1)], 
            c
        )
        c += 1
        
        # M4 = a11 * (b00-b10)
        # C00 -= M4, C10 += M4
        add_term(
            [(1*m+1, 1)], 
            [(0*p+k0, 1), (1*p+k0, -1)], 
            [(k0*n+0, -1), (k0*n+1, 1)], 
            c
        )
        c += 1
        
        # M5 = (a00+a01) * b11
        # C00 += M5, C01 += M5
        add_term(
            [(0*m+0, 1), (0*m+1, 1)], 
            [(1*p+k1, 1)], 
            [(k0*n+0, 1), (k1*n+0, 1)], 
            c
        )
        c += 1
        
        # M6 = (a10-a00) * (b00+b01)
        # C11 += M6
        add_term(
            [(1*m+0, 1), (0*m+0, -1)], 
            [(0*p+k0, 1), (0*p+k1, 1)], 
            [(k1*n+1, 1)], 
            c
        )
        c += 1
        
        # M7 = (a01-a11) * (b10+b11)
        # C00 -= M7
        add_term(
            [(0*m+1, 1), (1*m+1, -1)], 
            [(1*p+k0, 1), (1*p+k1, 1)], 
            [(k0*n+0, -1)], 
            c
        )
        c += 1

    # Terms for k=8 (2x2x1, rank 4)
    k8 = 8
    # C08 = a00*b08
    add_term([(0*m+0, 1)], [(0*p+k8, 1)], [(k8*n+0, 1)], c)
    c += 1
    # C08 += a01*b18
    add_term([(0*m+1, 1)], [(1*p+k8, 1)], [(k8*n+0, 1)], c)
    c += 1
    # C18 = a10*b08
    add_term([(1*m+0, 1)], [(0*p+k8, 1)], [(k8*n+1, 1)], c)
    c += 1
    # C18 += a11*b18
    add_term([(1*m+1, 1)], [(1*p+k8, 1)], [(k8*n+1, 1)], c)
    c += 1

    # --- Part B: 2x1x9 (rank 18) ---
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

def _validate_decomposition(U, V, W, n, m, p):
    """
    Verifies that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    Returns True if valid, False otherwise.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try the rank 51 decomposition
    try:
        U, V, W = _make_2x3x9_rank_51()
        # Validate before returning to ensure we don't score 0
        if _validate_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # Fallback to schoolbook if something goes wrong or validation fails
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
