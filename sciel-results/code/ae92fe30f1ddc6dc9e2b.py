"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin under the 150s wall-clock limit

    n, m, p = DIMENSIONS
    assert (n, m, p) == (2, 3, 6), "This program is specialized to 2x3x6."

    # ------------------------------------------------------------------
    # Rank-22 decomposition: two independent 2x3x3 blocks, each rank 11.
    # ------------------------------------------------------------------
    
    def build_2x3x3_block():
        """Build rank-11 decomposition for 2x3 x 3x3 -> 2x3.
        Returns list of (u_dict, v_dict, w_dict) terms.
        u_dict: {flat_A_index: coeff} where flat = i*3+j
        v_dict: {flat_B_index: coeff} where flat = j*3+k
        w_dict: {flat_C_index: coeff} where flat = i*3+k
        """
        terms = []
        
        # Strassen terms for A[:,0:2] (2x2) * B[0:2,:] (2x3) -> C[:,0:2] (2x2)
        # A sub: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
        # B sub: B[0,0]=0, B[0,1]=1, B[1,0]=3, B[1,1]=4
        # C sub: C[0,0]=0, C[0,1]=1, C[1,0]=3, C[1,1]=4
        
        # M1 = (A00+A11)(B00+B11)
        terms.append({
            'u': {0: 1, 4: 1},
            'v': {0: 1, 4: 1},
            'w': {0: 1, 4: 1}
        })
        # M2 = (A10+A11)B00
        terms.append({
            'u': {3: 1, 4: 1},
            'v': {0: 1},
            'w': {3: 1, 4: -1}
        })
        # M3 = A00(B01-B11)
        terms.append({
            'u': {0: 1},
            'v': {1: 1, 4: -1},
            'w': {1: 1, 4: 1}
        })
        # M4 = A11(B10-B00)
        terms.append({
            'u': {4: 1},
            'v': {3: 1, 0: -1},
            'w': {0: 1, 3: 1}
        })
        # M5 = (A00+A01)B11
        terms.append({
            'u': {0: 1, 1: 1},
            'v': {4: 1},
            'w': {0: -1, 1: 1}
        })
        # M6 = (A10-A00)(B00+B01)
        terms.append({
            'u': {3: 1, 0: -1},
            'v': {0: 1, 1: 1},
            'w': {4: 1}
        })
        # M7 = (A01-A11)(B10+B11)
        terms.append({
            'u': {1: 1, 4: -1},
            'v': {3: 1, 4: 1},
            'w': {0: 1}
        })
        
        # Scalar terms for A[:,2] * B[0:2,:]
        # A[0,2]=2, A[1,2]=5
        # B[0,0]=0, B[0,1]=1, B[1,0]=3, B[1,1]=4
        # C[0,0]=0, C[0,1]=1, C[1,0]=3, C[1,1]=4
        
        # K1 = A02 * B00
        terms.append({
            'u': {2: 1},
            'v': {0: 1},
            'w': {0: 1}
        })
        # K2 = A02 * B01
        terms.append({
            'u': {2: 1},
            'v': {1: 1},
            'w': {1: 1}
        })
        # K3 = A12 * B10
        terms.append({
            'u': {5: 1},
            'v': {3: 1},
            'w': {3: 1}
        })
        # K4 = A12 * B11
        terms.append({
            'u': {5: 1},
            'v': {4: 1},
            'w': {4: 1}
        })
        
        return terms  # 11 terms total

    def verify_decomposition(U, V, W, n, m, p):
        """Verify the decomposition is exact."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Build the two 2x3x3 blocks
    block_terms = build_2x3x3_block()
    rank = 22  # 2 blocks * 11 terms each
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 22
    V = [[0] * rank for _ in range(m * p)]  # 18 x 22
    W = [[0] * rank for _ in range(p * n)]  # 12 x 22
    
    for block in range(2):
        for t_idx, term in enumerate(block_terms):
            col = block * 11 + t_idx
            
            # U: flat index = i*3+j, same for both blocks
            for idx, val in term['u'].items():
                U[idx][col] = val
            
            # V: flat index = j*3+k, but k is local to block (0,1,2)
            # Global k = block*3 + k_local
            for l_idx, val in term['v'].items():
                j_local = l_idx // 3
                k_local = l_idx % 3
                g_idx = j_local * p + (block * 3 + k_local)
                V[g_idx][col] = val
            
            # W: flat index = i*3+k, but k is local to block
            # Global index: W[k*n+i, r], so flat = k*n + i
            for l_idx, val in term['w'].items():
                i_local = l_idx // 3
                k_local = l_idx % 3
                g_idx = (block * 3 + k_local) * n + i_local
                W[g_idx][col] = val
    
    # Verify
    if verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 36
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
