import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: split p=7 into {0,1,2} (size 3) and {3,4,5,6} (size 4)
    # For each j-block (j=0,1 and j=2,3):
    #   - 2x2x3 subproblem: rank 8 (schoolbook)
    #   - 2x2x4 subproblem: two 2x2x2 Strassen blocks, rank 14
    # Total per j-block: 22, total rank: 44
    
    rank = 44
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    def add_strassen_block(j0, k0):
        """Add a 2x2x2 Strassen block for A rows 0-1, cols j0..j0+1, B rows j0..j0+1, cols k0..k0+1."""
        nonlocal col
        a_idx = 0 * m + j0
        b_idx = 0 * m + (j0 + 1)
        c_idx = 1 * m + j0
        d_idx = 1 * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        c00_idx = k0 * n + 0
        c01_idx = (k0 + 1) * n + 0
        c10_idx = k0 * n + 1
        c11_idx = (k0 + 1) * n + 1
        
        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_223_block(j0, k0):
        """Add a 2x2x3 schoolbook block for A rows 0-1, cols j0..j0+1, B rows j0..j0+1, cols k0..k0+2."""
        nonlocal col
        a_idx = 0 * m + j0
        b_idx = 0 * m + (j0 + 1)
        c_idx = 1 * m + j0
        d_idx = 1 * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = j0 * p + (k0 + 2)
        h_idx = (j0 + 1) * p + k0
        i_idx = (j0 + 1) * p + (k0 + 1)
        j_idx = (j0 + 1) * p + (k0 + 2)
        
        c0_idx = k0 * n + 0
        c1_idx = (k0 + 1) * n + 0
        c2_idx = (k0 + 2) * n + 0
        c3_idx = k0 * n + 1
        c4_idx = (k0 + 1) * n + 1
        c5_idx = (k0 + 2) * n + 1
        
        # Row 0 of C:
        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c0_idx, col] += 1; col += 1
        U[a_idx, col] += 1; V[f_idx, col] += 1; W[c1_idx, col] += 1; col += 1
        U[a_idx, col] += 1; V[g_idx, col] += 1; W[c2_idx, col] += 1; col += 1
        U[b_idx, col] += 1; V[h_idx, col] += 1; W[c0_idx, col] += 1; col += 1
        U[b_idx, col] += 1; V[i_idx, col] += 1; W[c1_idx, col] += 1; col += 1
        U[b_idx, col] += 1; V[j_idx, col] += 1; W[c2_idx, col] += 1; col += 1
        
        # Row 1 of C:
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c3_idx, col] += 1; col += 1
        U[c_idx, col] += 1; V[f_idx, col] += 1; W[c4_idx, col] += 1; col += 1
        U[c_idx, col] += 1; V[g_idx, col] += 1; W[c5_idx, col] += 1; col += 1
        U[d_idx, col] += 1; V[h_idx, col] += 1; W[c3_idx, col] += 1; col += 1
        U[d_idx, col] += 1; V[i_idx, col] += 1; W[c4_idx, col] += 1; col += 1
        U[d_idx, col] += 1; V[j_idx, col] += 1; W[c5_idx, col] += 1; col += 1

    for j_start in [0, 2]:
        # 2x2x3 block: k = 0,1,2
        add_schoolbook_223_block(j_start, 0)
        # 2x2x4 block: k = 3,4,5,6 -> two Strassen blocks
        add_strassen_block(j_start, 3)
        add_strassen_block(j_start, 5)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify correctness
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    expected = np.ones((n, m, p))
    if not np.allclose(T, expected, atol=1e-9):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=int)
        V_fb = np.zeros((m * p, rank_fb), dtype=int)
        W_fb = np.zeros((p * n, rank_fb), dtype=int)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m + j, idx] = 1
                    V_fb[j*p + k, idx] = 1
                    W_fb[k*n + i, idx] = 1
                    idx += 1
        return {
            "U": U_fb.tolist(),
            "V": V_fb.tolist(),
            "W": W_fb.tolist()
        }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
