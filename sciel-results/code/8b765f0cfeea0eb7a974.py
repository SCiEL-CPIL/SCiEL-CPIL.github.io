"""Matrix multiplication 2x3x8: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8

    # Build explicit rank-11 decomposition for 2x3x2
    # A is 2x3, B is 3x2, C is 2x2
    # U index: i*3+j (i in {0,1}, j in {0,1,2}) -> 6 entries
    # V index: j*2+k (j in {0,1,2}, k in {0,1}) -> 6 entries
    # W index: k*2+i (k in {0,1}, i in {0,1}) -> 4 entries

    U_sub = np.zeros((6, 11))
    V_sub = np.zeros((6, 11))
    W_sub = np.zeros((4, 11))

    # Strassen term 1: M1 = (A00+A11)(B00+B11)
    U_sub[0, 0] = 1; U_sub[3, 0] = 1
    V_sub[0, 0] = 1; V_sub[3, 0] = 1
    W_sub[0, 0] = 1; W_sub[3, 0] = 1

    # Strassen term 2: M2 = (A10+A11)B00
    U_sub[3, 1] = 1; U_sub[4, 1] = 1
    V_sub[0, 1] = 1
    W_sub[2, 1] = 1

    # Strassen term 3: M3 = A00(B10-B11)
    U_sub[0, 2] = 1
    V_sub[2, 2] = 1; V_sub[3, 2] = -1
    W_sub[1, 2] = 1

    # Strassen term 4: M4 = A11(B01-B10)
    U_sub[3, 3] = 1
    V_sub[1, 3] = 1; V_sub[2, 3] = -1
    W_sub[0, 3] = 1; W_sub[2, 3] = 1

    # Strassen term 5: M5 = (A00+A01)B11
    U_sub[0, 4] = 1; U_sub[1, 4] = 1
    V_sub[3, 4] = 1
    W_sub[0, 4] = -1; W_sub[1, 4] = 1

    # Strassen term 6: M6 = (A10-A00)(B00+B10)
    U_sub[3, 5] = 1; U_sub[0, 5] = -1
    V_sub[0, 5] = 1; V_sub[2, 5] = 1
    W_sub[3, 5] = 1

    # Strassen term 7: M7 = (A01-A11)(B01+B11)
    U_sub[1, 6] = 1; U_sub[4, 6] = -1
    V_sub[1, 6] = 1; V_sub[3, 6] = 1
    W_sub[0, 6] = 1

    # Rank-1 term 8: A02 * B20 -> C00
    U_sub[2, 7] = 1
    V_sub[4, 7] = 1
    W_sub[0, 7] = 1

    # Rank-1 term 9: A02 * B21 -> C01
    U_sub[2, 8] = 1
    V_sub[5, 8] = 1
    W_sub[1, 8] = 1

    # Rank-1 term 10: A12 * B20 -> C10
    U_sub[5, 9] = 1
    V_sub[4, 9] = 1
    W_sub[2, 9] = 1

    # Rank-1 term 11: A12 * B21 -> C11
    U_sub[5, 10] = 1
    V_sub[5, 10] = 1
    W_sub[3, 10] = 1

    # Verify the 2x3x2 decomposition
    T_sub = np.zeros((6, 6, 4))
    for i in range(2):
        for j in range(3):
            for k in range(2):
                T_sub[i*3+j, j*2+k, k*2+i] = 1.0

    recon_sub = np.zeros((6, 6, 4))
    for c in range(11):
        recon_sub += np.einsum('i,j,k->ijk', U_sub[:,c], V_sub[:,c], W_sub[:,c])

    if np.linalg.norm(T_sub - recon_sub) > 1e-10:
        # Fallback: schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build 2x3x4 decomposition by tiling 2 copies of 2x3x2
    rank_234 = 2 * 11  # = 22
    U_234 = np.zeros((6, rank_234))
    V_234 = np.zeros((12, rank_234))  # 3*4 = 12
    W_234 = np.zeros((8, rank_234))   # 4*2 = 8

    for g in range(2):
        col_offset = g * 11
        for c in range(11):
            U_234[:, col_offset + c] = U_sub[:, c]
            for j in range(3):
                for k_sub in range(2):
                    V_234[j*4 + g*2 + k_sub, col_offset + c] = V_sub[j*2 + k_sub, c]
            for k_sub in range(2):
                for i in range(2):
                    W_234[(g*2 + k_sub)*2 + i, col_offset + c] = W_sub[k_sub*2 + i, c]

    # Verify 2x3x4 decomposition
    T_234 = np.zeros((6, 12, 8))
    for i in range(2):
        for j in range(3):
            for k in range(4):
                T_234[i*3+j, j*4+k, k*2+i] = 1.0

    recon_234 = np.zeros((6, 12, 8))
    for c in range(rank_234):
        recon_234 += np.einsum('i,j,k->ijk', U_234[:,c], V_234[:,c], W_234[:,c])

    if np.linalg.norm(T_234 - recon_234) > 1e-10:
        # Fallback: schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build full 2x3x8 decomposition by tiling 2 copies of 2x3x4
    rank_total = 2 * rank_234  # = 44... still 44!
    
    # Actually, we need to think differently. Let me try a direct approach.
    # The issue is that tiling doesn't reduce rank. We need a genuinely different decomposition.
    
    # Let me try: use the 2x3x2 rank-11 for the first 2 columns of B,
    # and find a way to handle the remaining 6 columns more efficiently.
    
    # Actually, the fundamental issue is that each (i,j,k) triple needs its own
    # rank-1 term unless we can combine them. The Strassen trick only works
    # for the 2x2x2 core.
    
    # Let me try a different approach: partition B into 4 blocks of 3x2,
    # but share the U factors more cleverly.
    
    # For now, return the rank-44 solution (same as before)
    rank_total = 4 * 11
    U_full = np.zeros((6, rank_total))
    V_full = np.zeros((24, rank_total))
    W_full = np.zeros((16, rank_total))

    for g in range(4):
        col_offset = g * 11
        for c in range(11):
            U_full[:, col_offset + c] = U_sub[:, c]
            for j in range(3):
                for k_sub in range(2):
                    V_full[j*8 + g*2 + k_sub, col_offset + c] = V_sub[j*2 + k_sub, c]
            for k_sub in range(2):
                for i in range(2):
                    W_full[(g*2 + k_sub)*2 + i, col_offset + c] = W_sub[k_sub*2 + i, c]

    # Verify full decomposition
    T_full = np.zeros((6, 24, 16))
    for i in range(2):
        for j in range(3):
            for k in range(8):
                T_full[i*3+j, j*8+k, k*2+i] = 1.0

    recon_full = np.zeros((6, 24, 16))
    for c in range(rank_total):
        recon_full += np.einsum('i,j,k->ijk', U_full[:,c], V_full[:,c], W_full[:,c])

    if np.linalg.norm(T_full - recon_full) > 1e-10:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
