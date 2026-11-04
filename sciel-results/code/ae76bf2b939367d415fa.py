"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START


def _build_trivial(n, m, p):
    """Return the rank-nmp trivial decomposition as (U, V, W) lists."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r in range(rank):
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        U[idx_u][r] = 1
        V[idx_v][r] = 1
        W[idx_w][r] = 1
    return U, V, W, rank


def _validate(n, m, p, U, V, W):
    """Verify that sum_r u_{ij}^{(r)} v_{jk}^{(r)} = w_{ki}^{(r)} for all i,j,k."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = U_arr[i * m + j] @ V_arr[j * p + k]
                rhs = W_arr[k * n + i]
                if not np.allclose(lhs, rhs, atol=1e-9):
                    return False
    return True


def _build_strassen(n, m, p):
    """
    Build a rank-55 decomposition for 2x3 * 3x10 using Strassen on 2x2x2 blocks.
    
    Split:
      A (2x3) = [A_left (2x2) | A_right (2x1)]
      B (3x10) = [B_top (2x10); B_bottom (1x10)]
    
    C = A_left @ B_top + A_right @ B_bottom
    
    For A_left @ B_top (2x2 x 2x10): split B_top into 5 blocks of 2x2 columns.
      Each 2x2 x 2x2 block uses Strassen (7 rank-1 terms).
      Total: 5 * 7 = 35 terms.
    
    For A_right @ B_bottom (2x1 x 1x10): schoolbook, 2*1*10 = 20 terms.
    
    Total rank: 55.
    """
    assert n == 2 and m == 3 and p == 10
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    # Indexing:
    # U row: i*m + j, i in [0,1], j in [0,1,2]
    # V row: j*p + k, j in [0,1,2], k in [0..9]
    # W row: k*n + i, k in [0..9], i in [0,1]
    
    r = 0  # current rank index
    
    # --- Part 1: A_left (2x2) @ B_top (2x10) via Strassen on 5 2x2 blocks ---
    # A_left entries: A[0,0], A[0,1], A[1,0], A[1,1]
    #   U indices: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4
    # B_top entries: B[0,k], B[1,k] for k=0..9
    #   V indices: 0*10+k = k, 1*10+k = 10+k
    
    # Strassen's 7 products for 2x2 x 2x2:
    # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5],
    #      [m2+m4,        m1-m2+m3+m6]]
    
    # For block b (b=0..4), columns 2b and 2b+1 of B_top:
    # A_left = [[A00, A01], [A10, A11]]
    # B_block = [[B0_{2b}, B0_{2b+1}], [B1_{2b}, B1_{2b+1}]]
    
    # Strassen coefficients for U (A_left part):
    # m1: A00 + A11  -> U[0] += 1, U[4] += 1
    # m2: A10 + A11  -> U[3] += 1, U[4] += 1
    # m3: A00        -> U[0] += 1
    # m4: A11        -> U[4] += 1
    # m5: A00 + A01  -> U[0] += 1, U[1] += 1
    # m6: A10 - A00  -> U[3] += 1, U[0] += -1
    # m7: A01 - A11  -> U[1] += 1, U[4] += -1
    
    # Strassen coefficients for V (B_block part):
    # m1: B0_{2b} + B1_{2b+1}  -> V[2b] += 1, V[10+2b+1] += 1
    # m2: B0_{2b}              -> V[2b] += 1
    # m3: B0_{2b+1} - B1_{2b+1} -> V[2b+1] += 1, V[10+2b+1] += -1
    # m4: B1_{2b} - B0_{2b}    -> V[10+2b] += 1, V[2b] += -1
    # m5: B1_{2b+1}            -> V[10+2b+1] += 1
    # m6: B0_{2b} + B0_{2b+1}  -> V[2b] += 1, V[2b+1] += 1
    # m7: B1_{2b} + B1_{2b+1}  -> V[10+2b] += 1, V[10+2b+1] += 1
    
    # W contributions:
    # C[0, 2b]   = m1 + m4 - m5 + m7
    # C[0, 2b+1] = m3 + m5
    # C[1, 2b]   = m2 + m4
    # C[1, 2b+1] = m1 - m2 + m3 + m6
    
    # W row index for C[i, k] is k*2 + i
    
    strassen_u = {
        0: [(0, 1), (4, 1)],       # m1
        1: [(3, 1), (4, 1)],       # m2
        2: [(0, 1)],               # m3
        3: [(4, 1)],               # m4
        4: [(0, 1), (1, 1)],       # m5
        5: [(3, 1), (0, -1)],      # m6
        6: [(1, 1), (4, -1)],      # m7
    }
    
    strassen_v = {
        0: [(0, 1), (11, 1)],      # m1: B0_{2b} + B1_{2b+1}
        1: [(0, 1)],               # m2: B0_{2b}
        2: [(1, 1), (11, -1)],     # m3: B0_{2b+1} - B1_{2b+1}
        3: [(10, 1), (0, -1)],     # m4: B1_{2b} - B0_{2b}
        4: [(11, 1)],              # m5: B1_{2b+1}
        5: [(0, 1), (1, 1)],       # m6: B0_{2b} + B0_{2b+1}
        6: [(10, 1), (11, 1)],     # m7: B1_{2b} + B1_{2b+1}
    }
    
    # W contributions: (w_row, coeff) for each m_s
    # C[0, 2b]   = m1 + m4 - m5 + m7  -> W row (2b)*2 + 0 = 4b
    # C[0, 2b+1] = m3 + m5            -> W row (2b+1)*2 + 0 = 4b+2
    # C[1, 2b]   = m2 + m4            -> W row (2b)*2 + 1 = 4b+1
    # C[1, 2b+1] = m1 - m2 + m3 + m6  -> W row (2b+1)*2 + 1 = 4b+3
    
    strassen_w = {
        0: [(0, 1), (3, 1)],       # m1 -> C[0,2b] and C[1,2b+1]
        1: [(1, 1), (3, -1)],      # m2 -> C[1,2b] and C[1,2b+1]
        2: [(2, 1), (3, 1)],       # m3 -> C[0,2b+1] and C[1,2b+1]
        3: [(0, 1), (1, 1)],       # m4 -> C[0,2b] and C[1,2b]
        4: [(0, -1), (2, 1)],      # m5 -> C[0,2b] and C[0,2b+1]
        5: [(3, 1)],               # m6 -> C[1,2b+1]
        6: [(0, 1)],               # m7 -> C[0,2b]
    }
    
    for b in range(5):
        col0 = 2 * b
        col1 = 2 * b + 1
        # V indices for this block:
        # B0_{col0} -> V row col0
        # B0_{col1} -> V row col1
        # B1_{col0} -> V row 10 + col0
        # B1_{col1} -> V row 10 + col1
        
        for s in range(7):
            # U contributions
            for (u_row, coeff) in strassen_u[s]:
                U[u_row][r] += coeff
            
            # V contributions (adjust indices for block b)
            for (v_row_offset, coeff) in strassen_v[s]:
                # v_row_offset is in [0,1,10,11] representing B0_{col0}, B0_{col1}, B1_{col0}, B1_{col1}
                if v_row_offset == 0:
                    v_row = col0
                elif v_row_offset == 1:
                    v_row = col1
                elif v_row_offset == 10:
                    v_row = 10 + col0
                elif v_row_offset == 11:
                    v_row = 10 + col1
                else:
                    raise ValueError(f"Unexpected v_row_offset {v_row_offset}")
                V[v_row][r] += coeff
            
            # W contributions (adjust row indices for block b)
            for (w_row_offset, coeff) in strassen_w[s]:
                # w_row_offset in [0,1,2,3] representing C[0,col0], C[1,col0], C[0,col1], C[1,col1]
                if w_row_offset == 0:
                    w_row = col0 * 2 + 0
                elif w_row_offset == 1:
                    w_row = col0 * 2 + 1
                elif w_row_offset == 2:
                    w_row = col1 * 2 + 0
                elif w_row_offset == 3:
                    w_row = col1 * 2 + 1
                else:
                    raise ValueError(f"Unexpected w_row_offset {w_row_offset}")
                W[w_row][r] += coeff
            
            r += 1
    
    # --- Part 2: A_right (2x1) @ B_bottom (1x10) schoolbook ---
    # A_right: A[0,2], A[1,2] -> U rows 0*3+2=2, 1*3+2=5
    # B_bottom: B[2,k] for k=0..9 -> V rows 2*10+k = 20+k
    # C[i,k] for i in {0,1}, k in 0..9 -> W rows k*2+i
    
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2][r] = 1
            V[2 * 10 + k][r] = 1
            W[k * 2 + i][r] = 1
            r += 1
    
    assert r == rank, f"Expected rank {rank}, got {r}"
    
    return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin under 120s cap

    n, m, p = DIMENSIONS

    # Use Strassen-based decomposition for 2x3*3x10: rank 55
    U, V, W, rank = _build_strassen(n, m, p)

    # Validate
    if not _validate(n, m, p, U, V, W):
        # Fall back to trivial if validation fails
        U, V, W, rank = _build_trivial(n, m, p)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
