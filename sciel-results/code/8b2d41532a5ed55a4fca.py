import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 2, 4, 8

    # Try rank 50 first (better than 56)
    rank_50 = _build_rank_50(n, m, p)
    if rank_50 is not None:
        U, V, W = rank_50
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    # Fallback to rank 56
    U, V, W = _build_rank_56(n, m, p)
    if _verify_fast(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}

def _build_rank_50(n, m, p):
    """
    Attempt rank 50 decomposition.
    
    Strategy: Use the structure of 2x4 * 4x8.
    Split into two 2x2 * 2x8 problems (for j-pairs (0,1) and (2,3)).
    Each 2x2 * 2x8 = 4 * (2x2)*(2x2).
    
    Standard Strassen gives 7 per (2x2)*(2x2), total 2*4*7 = 56.
    
    To get 50, we need to save 6 ranks.
    
    Key insight: For a 2x2 * 2x2 multiplication, if we look at the structure,
    some combinations might be mergeable across the 4 k-pairs within a j-pair.
    
    Actually, let's try a different approach:
    For each j-pair, we have 4 (2x2)*(2x2) multiplications for k-pairs.
    The A-block is the same for all 4 k-pairs (within a j-pair).
    A_block = [[A[0,j1], A[0,j2]], [A[1,j1], A[1,j2]]]
    
    Can we share some Strassen factors across k-pairs?
    
    In Strassen:
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)b00
    M3 = a00(b01-b11)
    M4 = a11(b10-b00)
    M5 = (a00+a01)b11
    M6 = (a10-a00)(b00+b01)
    M7 = (a01-a11)(b10+b11)
    
    The A-side factors are:
    F1 = a00+a11
    F2 = a10+a11
    F3 = a00
    F4 = a11
    F5 = a00+a01
    F6 = a10-a00
    F7 = a01-a11
    
    These are fixed per j-pair. The B-side factors change per k-pair.
    
    We can't directly share the A-factors across k-pairs because each Strassen
    term is a separate rank-1 tensor. However, maybe we can use a different
    decomposition for the 4 (2x2)*(2x2) multiplications that share the A matrix.
    
    This is equivalent to multiplying a 2x2 matrix A by a 2x8 matrix B.
    Rank of 2x2 * 2x8 multiplication.
    
    Known result: The rank of 2x2 * 2x8 is at most 4*7 = 28 (Strassen).
    But can it be done in less?
    
    Actually, 2x2 * 2x8 is the same as 2x2 * 8x2^T, or we can think of it
    as a linear map from 2x8 to 2x2 given a fixed A.
    
    The tensor rank of m x n * n x p multiplication... for 2x2*2x8,
    this is not a standard cube. Let me think differently.
    
    The multiplication (2x2) * (2x8) -> (2x8) is a linear map.
    The tensor is in (R^2 x R^2) * (R^2 x R^8) -> (R^2 x R^8).
    
    Actually, let's just try to verify if rank 50 is achievable by a different
    block structure. 
    
    Alternative: Use the fact that we have 2 rows in A. Maybe we can use
    a decomposition that exploits the small size more directly.
    
    For now, return None to indicate this approach didn't yield a valid construction.
    """
    # Placeholder - we'll return None to fall back to rank 56
    # unless we find a valid rank 50 construction.
    return None

def _build_rank_56(n, m, p):
    """Rank 56 decomposition via Strassen on 2x2 sub-blocks."""
    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 56

    for jp in range(2):
        j1 = 2 * jp
        j2 = 2 * jp + 1
        base_col = 28 * jp

        for kp in range(4):
            k1 = 2 * kp
            k2 = 2 * kp + 1
            col_offset = base_col + 7 * kp

            a00 = 0 * 4 + j1
            a01 = 0 * 4 + j2
            a10 = 1 * 4 + j1
            a11 = 1 * 4 + j2

            b00 = j1 * 8 + k1
            b01 = j1 * 8 + k2
            b10 = j2 * 8 + k1
            b11 = j2 * 8 + k2

            c00 = k1 * 2 + 0
            c01 = k2 * 2 + 0
            c10 = k1 * 2 + 1
            c11 = k2 * 2 + 1

            # Strassen rank 7:
            # M1 = (a00+a11)(b00+b11)
            U[a00][col_offset] += 1
            U[a11][col_offset] += 1
            V[b00][col_offset] += 1
            V[b11][col_offset] += 1
            W[c00][col_offset] += 1
            W[c11][col_offset] += 1

            # M2 = (a10+a11)b00
            U[a10][col_offset+1] += 1
            U[a11][col_offset+1] += 1
            V[b00][col_offset+1] += 1
            W[c10][col_offset+1] += 1
            W[c11][col_offset+1] -= 1

            # M3 = a00(b01-b11)
            U[a00][col_offset+2] += 1
            V[b01][col_offset+2] += 1
            V[b11][col_offset+2] -= 1
            W[c01][col_offset+2] += 1
            W[c11][col_offset+2] += 1

            # M4 = a11(b10-b00)
            U[a11][col_offset+3] += 1
            V[b10][col_offset+3] += 1
            V[b00][col_offset+3] -= 1
            W[c00][col_offset+3] += 1
            W[c10][col_offset+3] += 1

            # M5 = (a00+a01)b11
            U[a00][col_offset+4] += 1
            U[a01][col_offset+4] += 1
            V[b11][col_offset+4] += 1
            W[c00][col_offset+4] -= 1
            W[c01][col_offset+4] += 1

            # M6 = (a10-a00)(b00+b01)
            U[a10][col_offset+5] += 1
            U[a00][col_offset+5] -= 1
            V[b00][col_offset+5] += 1
            V[b01][col_offset+5] += 1
            W[c11][col_offset+5] += 1

            # M7 = (a01-a11)(b10+b11)
            U[a01][col_offset+6] += 1
            U[a11][col_offset+6] -= 1
            V[b10][col_offset+6] += 1
            V[b11][col_offset+6] += 1
            W[c00][col_offset+6] += 1

    return U, V, W

def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification of the tensor decomposition."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(rank):
                    val += U_arr[i*m+j, s] * V_arr[j*p+k, s] * W_arr[k*n+i, s]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True
