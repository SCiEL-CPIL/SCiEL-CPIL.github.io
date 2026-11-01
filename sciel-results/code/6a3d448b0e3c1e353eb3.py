import time

_DEADLINE = 140  # seconds, safety margin before 150s kill

def _deadline_exceeded(t0):
    return (time.time() - t0) > _DEADLINE

def _build_rank21():
    """Construct the rank-21 Strassen-based decomposition."""
    n, m, p = 2, 2, 6
    rank = 21

    # Strassen 7 factors for 2x2 multiplication.
    # U_coeffs[k]: dict mapping A-index (i*m+j) to coefficient
    # V_coeffs[k]: list of (j_local, k_local, coeff) for B block
    # W_map[k]: list of (i, k_local, coeff) for C block
    
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U_coeffs = [
        {0: 1, 3: 1},       # M1 = a00 + a11
        {2: 1, 3: 1},       # M2 = a10 + a11
        {0: 1},             # M3 = a00
        {3: 1},             # M4 = a11
        {0: 1, 1: 1},       # M5 = a00 + a01
        {2: 1, 0: -1},      # M6 = a10 - a00
        {1: 1, 3: -1},      # M7 = a01 - a11
    ]
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],   # b00 + b11
        [(0, 0, 1)],               # b00
        [(0, 1, 1), (1, 1, -1)],  # b01 - b11
        [(1, 0, 1), (0, 0, -1)],  # b10 - b00
        [(1, 1, 1)],               # b11
        [(0, 0, 1), (0, 1, 1)],   # b00 + b01
        [(1, 0, 1), (1, 1, 1)],   # b10 + b11
    ]
    W_map = [
        [(0, 0, 1), (1, 1, 1)],   # c00 += M1, c11 += M1
        [(1, 0, 1), (1, 1, -1)],  # c10 += M2, c11 -= M2
        [(0, 1, 1), (1, 1, 1)],   # c01 += M3, c11 += M3
        [(0, 0, 1), (1, 0, 1)],   # c00 += M4, c10 += M4
        [(0, 0, -1), (0, 1, 1)],  # c00 -= M5, c01 += M5
        [(1, 1, 1)],              # c11 += M6
        [(0, 0, 1)],              # c00 += M7
    ]

    U = [[0] * rank for _ in range(n * m)]   # (4, 21)
    V = [[0] * rank for _ in range(m * p)]   # (12, 21)
    W = [[0] * rank for _ in range(p * n)]   # (12, 21)

    for r in range(3):       # block index (which 2 columns of B)
        for k in range(7):   # Strassen factor index
            col = r * 7 + k

            # Fill U: A coefficients (same for all blocks)
            for idx, val in U_coeffs[k].items():
                U[idx][col] = val

            # Fill V: B coefficients for block r (columns 2r, 2r+1)
            # V shape is (m*p, rank) = (12, 21)
            # Index in V is j * p + k_global
            for j_local, k_local, val in V_coeffs[k]:
                global_k = 2 * r + k_local
                V[j_local * p + global_k][col] = val

            # Fill W: C coefficients for block r (columns 2r, 2r+1)
            # W shape is (p*n, rank) = (12, 21)
            # Index in W is k_global * n + i
            for i, k_local, val in W_map[k]:
                global_k = 2 * r + k_local
                W[global_k * n + i][col] = val

    return {"U": U, "V": V, "W": W}

def construct():
    """Return {'U','V','W'} factor matrices exactly decomposing the 2,2,6 MM tensor."""
    t0 = time.time()
    result = _build_rank21()
    return result
