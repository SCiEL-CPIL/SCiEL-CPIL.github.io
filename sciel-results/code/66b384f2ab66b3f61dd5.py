"""Winograd/Kronecker structural decomposition for 3x5x6 matmul tensor."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np


def _verify(U, V, W, n, m, p):
    """Verify exact decomposition of the matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True


def _strassen_2x2():
    """Return 7 Strassen rank-1 factors for 2x2x2.
    Each factor: (U_dict, V_dict, W_dict) with keys (i,j), (j,k), (k,i) -> coeff.
    Convention: U[i,j], V[j,k], W[k,i] so that C[i,k] gets contribution.
    """
    # M1 = (a11+a22)(b11+b22) -> C11+=1, C22+=1
    # M2 = (a21+a22)b11 -> C21+=1, C22+=-1
    # M3 = a11(b12-b22) -> C12+=1, C22+=1
    # M4 = a22(b21-b11) -> C11+=1, C21+=1
    # M5 = (a11+a12)b22 -> C11+=-1, C12+=1
    # M6 = (a21-a11)(b11+b12) -> C22+=1
    # M7 = (a12-a22)(b21+b22) -> C11+=1
    return [
        # M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
        # M3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        # M5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
        # M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]


def _construct():
    n, m, p = 3, 5, 6
    total_rank = 15  # 3 (i-block) + 7 (Strassen 2x2) + 4 (schoolbook 5x2) + 1 (tail)

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col = 0

    # --- Part 1: 3x1x1 for i=0,1,2 with j=0, k=0 ---
    # T[i,0,0] = 1 for i in {0,1,2}
    for i in range(3):
        U[i * m + 0, col] = 1  # A[i,0]
        V[0 * p + 0, col] = 1  # B[0,0]
        W[0 * n + i, col] = 1  # C[i,0]
        col += 1
    # col now 3

    # --- Part 2: Strassen 2x2x2 for i in {1,2}, j in {1,2}, k in {1,2} ---
    strassen = _strassen_2x2()
    i0, j0, k0 = 1, 1, 1
    for r in range(7):
        u_list, v_list, w_list = strassen[r]
        for (di, dj, val) in u_list:
            i = i0 + di
            j = j0 + dj
            U[i * m + j, col] += val
        for (dj, dk, val) in v_list:
            j = j0 + dj
            k = k0 + dk
            V[j * p + k, col] += val
        for (dk, di, val) in w_list:
            k = k0 + dk
            i = i0 + di
            W[k * n + i, col] += val
        col += 1
    # col now 10

    # --- Part 3: Schoolbook for i in {1,2}, j in {1,2}, k in {3,4} ---
    # T[i,j,k] = 1 for i in {1,2}, j in {1,2}, k in {3,4}
    for i in [1, 2]:
        for j in [1, 2]:
            for k in [3, 4]:
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    # col now 14

    # --- Part 4: Tail for i in {1,2}, j in {1,2}, k=5 ---
    # T[i,j,5] = 1 for i in {1,2}, j in {1,2}
    # This is a 2x2x1 tensor. Rank of 2x2x1 is 4 (since it's a 2x2 matrix of 1s, rank 1 as a matrix but as a 3-tensor with fixed k, it's just a 2x2 matrix, which has matrix rank 1, but tensor rank for T[i,j,k] with k fixed is the matrix rank of the 2x2 slice = 1).
    # Wait: T[i,j,5] = 1 for i in {1,2}, j in {1,2}. This is a 2x2x1 tensor.
    # A 2x2x1 tensor with all 1s has tensor rank 1? No.
    # T[i,j,k] = 1 if i in {1,2}, j in {1,2}, k=5.
    # This is equivalent to a 2x2 matrix of 1s for the (i,j) plane at k=5.
    # The tensor rank of a 2x2 matrix of 1s (viewed as a 2x2x1 tensor) is 1? 
    # No, tensor rank of a matrix A is its matrix rank. A 2x2 matrix of 1s has matrix rank 1.
    # So T[i,j,5] = 1 for i,j in {1,2} can be written as 1 rank-1 term:
    # U[i,1,0] = 1, V[1,5,0] = 1, W[5,i,0] = 1? No, that would give T[i,j,k] = U[i,j,c]*V[j,k,c]*W[k,i,c].
    # If c=0: U[i,1,0]=1 for i in {1,2}, V[1,5,0]=1, W[5,i,0]=1 for i in {1,2}.
    # Then T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    # For i in {1,2}, j=1, k=5: U[i,1,0]*V[1,5,0]*W[5,i,0] = 1*1*1 = 1. Correct.
    # For other (i,j,k): either j!=1 or k!=5, so V[j,k,0]=0 or U[i,j,0]=0. Correct.
    # So rank 1 suffices for the tail!
    
    # Wait, but the directive said rank 4 schoolbook remainder for 5x2. Let me re-examine.
    # The 5-column block is k in {1,2,3,4,5}. I split into k in {1,2} (Strassen) and k in {3,4,5} (remainder).
    # The remainder is i in {1,2}, j in {1,2}, k in {3,4,5}. That's 2x2x3 = 12 entries.
    # I used 4 schoolbook for k in {3,4} and 1 for k=5. Total 5 for the remainder.
    # But 2x2x3 tensor rank: the slice for each k is a 2x2 matrix of 1s, which has rank 1.
    # So we can do it in 3 rank-1 terms (one per k value), not 5.
    # Let me fix: use 3 terms for k=3,4,5 instead of 4+1=5.
    
    # Actually, let me recount. The current col is 14, and I need 1 more for the tail.
    # But I said total_rank = 15. Let me check: 3 + 7 + 4 + 1 = 15. 
    # But if I use 3 for the remainder (k=3,4,5), then total is 3 + 7 + 3 = 13.
    # Let me adjust total_rank to 13 and use 3 terms for k=3,4,5.
    
    # For k=3: U[i,1,0]=1 for i in {1,2}, V[1,3,0]=1, W[3,i,0]=1 for i in {1,2}
    # For k=4: U[i,1,1]=1 for i in {1,2}, V[1,4,1]=1, W[4,i,1]=1 for i in {1,2}
    # For k=5: U[i,1,2]=1 for i in {1,2}, V[1,5,2]=1, W[5,i,2]=1 for i in {1,2}
    
    # But wait, U[i,1,c] is already used in the Strassen part for c in 3..9.
    # U[i * m + j, col] where j=1. In Strassen, j is in {1,2} (j0=1, dj in {0,1}).
    # So U[i*m+1, col] for col in 3..9 has non-zero entries from Strassen.
    # If I add to U[i*m+1, col] for col in 10..12, that's fine since those columns are 0 so far.
    
    # Let me redo with total_rank = 13.
    pass


def _construct_corrected():
    n, m, p = 3, 5, 6
    total_rank = 13  # 3 (i-block) + 7 (Strassen 2x2) + 3 (schoolbook for k=3,4,5)

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col = 0

    # --- Part 1: 3x1x1 for i=0,1,2 with j=0, k=0 ---
    for i in range(3):
        U[i * m + 0, col] = 1
        V[0 * p + 0, col] = 1
        W[0 * n + i, col] = 1
        col += 1
    # col = 3

    # --- Part 2: Strassen 2x2x2 for i in {1,2}, j in {1,2}, k in {1,2} ---
    strassen = _strassen_2x2()
    i0, j0, k0 = 1, 1, 1
    for r in range(7):
        u_list, v_list, w_list = strassen[r]
        for (di, dj, val) in u_list:
            i = i0 + di
            j = j0 + dj
            U[i * m + j, col] += val
        for (dj, dk, val) in v_list:
            j = j0 + dj
            k = k0 + dk
            V[j * p + k, col] += val
        for (dk, di, val) in w_list:
            k = k0 + dk
            i = i0 + di
            W[k * n + i, col] += val
        col += 1
    # col = 10

    # --- Part 3: Rank-1 terms for i in {1,2}, j=1, k in {3,4,5} ---
    for k in [3, 4, 5]:
        for i in [1, 2]:
            U[i * m + 1, col] = 1
        V[1 * p + k, col] = 1
        for i in [1, 2]:
            W[k * n + i, col] = 1
        col += 1
    # col = 13

    return U, V, W


def construct():
    start_time = time.time()
    
    U, V, W = _construct_corrected()
    
    # Verify
    if _verify(U, V, W, 3, 5, 6):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    n, m, p = 3, 5, 6
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
