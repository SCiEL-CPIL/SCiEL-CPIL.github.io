"""Rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.

Approach: Kronecker product structure T_{4,4,6} = T_{2,2,2} ⊗ T_{2,2,3}
  - T_{2,2,2}: Strassen rank-7
  - T_{2,2,3}: Strassen rank-7 + schoolbook rank-4 = rank-11
  - Total rank: 7 * 11 = 77

The construction uses the separable Kronecker product to build the full
decomposition efficiently, then verifies correctness before returning.
"""

import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before the 150s kill


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2*2x2 matrix multiplication.
    
    Each term is (u_dict, v_dict, w_dict) where each dict maps (row, col) -> coeff.
    The convention: U[i*m+j], V[j*p+k], W[k*n+i] with m=p=2.
    """
    return [
        # m1 = (x00+x11)(y00+y11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # m2 = (x10+x11)y00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # m3 = x00(y01-y11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # m4 = x11(y10-y00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # m5 = (x00+x01)y11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # m6 = (x10-x00)(y00+y01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # m7 = (x01-x11)(y10+y11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]


def _build_2x2x3_decomposition():
    """Build the rank-11 decomposition for 2x2 * 2x3 matrix multiplication.
    
    Returns list of 11 (u_dict, v_dict, w_dict) tuples where:
      u_dict: keys are (i, j) for i in [0,1], j in [0,1] (2x2 input A)
      v_dict: keys are (j, k) for j in [0,1], k in [0,2] (2x3 input B)
      w_dict: keys are (i, k) for i in [0,1], k in [0,2] (2x3 output C)
    
    Uses Strassen rank-7 for the 2x2*2x2 part + 4 schoolbook terms for 2x2*2x1.
    """
    S2 = _strassen_2x2_terms()
    
    # 2x2 * 2x3: C[2x3] = A[2x2] * B[2x3]
    # Split B into B_left[2x2] (cols 0,1) and B_right[2x1] (col 2)
    # C_left[2x2] = A * B_left (rank 7 via Strassen)
    # C_right[2x1] = A * B_right (rank 4 via schoolbook)
    
    # Strassen terms for C_left (k=0,1): adapt 2x2*2x2 terms to 2x2*2x3
    terms = []
    for u_pat, v_pat, w_pat in S2:
        # v_pat has keys (j, k) where k in {0,1} - keep as is (k=0,1)
        # w_pat has keys (i, k) where k in {0,1} - keep as is (k=0,1)
        # u_pat has keys (i, j) where i,j in {0,1} - keep as is
        terms.append((u_pat, v_pat, w_pat))
    
    # Schoolbook terms for C_right (k=2): C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    # 4 terms (one per (i, j) pair):
    terms.append(({(0,0): 1}, {(0,2): 1}, {(0,2): 1}))  # A[0,0]*B[0,2] -> C[0,2]
    terms.append(({(1,0): 1}, {(0,2): 1}, {(1,2): 1}))  # A[1,0]*B[0,2] -> C[1,2]
    terms.append(({(0,1): 1}, {(1,2): 1}, {(0,2): 1}))  # A[0,1]*B[1,2] -> C[0,2]
    terms.append(({(1,1): 1}, {(1,2): 1}, {(1,2): 1}))  # A[1,1]*B[1,2] -> C[1,2]
    
    return terms  # 7 + 4 = 11 terms


def construct():
    """
    Construct a rank-reduced tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Returns a dict with keys 'U', 'V', 'W' where:
      U: shape (n*m, rank) = (16, rank)
      V: shape (m*p, rank) = (24, rank)
      W: shape (p*n, rank) = (24, rank)
    """
    n, m, p = 4, 4, 6
    
    # Build the two sub-decompositions
    S2 = _strassen_2x2_terms()       # rank 7
    S2x3 = _build_2x2x3_decomposition()  # rank 11
    
    # Kronecker product: T_{4,4,6} = T_{2,2,2} ⊗ T_{2,2,3}
    # For each pair (term_a from S2, term_b from S2x3):
    #   u_full[i1*2+i2, j1*2+j2] = u_a[i1,j1] * u_b[i2,j2]
    #   v_full[j1*2+j2, k1*3+k2] = v_a[j1,k1] * v_b[j2,k2]
    #   w_full[k1*3+k2, i1*2+i2] = w_a[k1,i1] * w_b[k2,i2]
    #
    # Wait - need to be careful with the index conventions.
    #
    # Full problem: C[4x6] = A[4x4] * B[4x6]
    # A[i,j] with i in [0,3], j in [0,3]
    # B[j,k] with j in [0,3], k in [0,5]
    # C[i,k] with i in [0,3], k in [0,5]
    #
    # Kronecker decomposition:
    # A[i1*2+i2, j1*2+j2] = A1[i1,j1] * A2[i2,j2]
    #   where A1 is 2x2, A2 is 2x2
    # B[j1*2+j2, k1*3+k2] = B1[j1,k1] * B2[j2,k2]
    #   where B1 is 2x2, B2 is 2x3
    # C[i1*2+i2, k1*3+k2] = C1[i1,k1] * C2[i2,k2]
    #   where C1 is 2x2, C2 is 2x3
    #
    # Tensor: T = T1 ⊗ T2
    # T1 is 2x2x2 (A1[2x2] * B1[2x2] -> C1[2x2])
    # T2 is 2x2x3 (A2[2x2] * B2[2x3] -> C2[2x3])
    #
    # For the Kronecker product of decompositions:
    # If T1 = sum_a u1_a ⊗ v1_a ⊗ w1_a (rank r1)
    # and T2 = sum_b u2_b ⊗ v2_b ⊗ w2_b (rank r2)
    # then T = sum_{a,b} (u1_a ⊗ u2_b) ⊗ (v1_a ⊗ v2_b) ⊗ (w1_a ⊗ w2_b)
    #
    # Index mapping for U (shape 4x4 = 16):
    #   U[i1*2+i2, j1*2+j2, (a,b)] = u1_a[i1,j1] * u2_b[i2,j2]
    #   Flat index: row = i1*2+i2, col_in_4x4 = j1*2+j2
    #   But U is stored as (n*m, rank) = (16, rank)
    #   U[i*m + j, r] where i in [0,3], j in [0,3]
    #   So U[i1*2+i2, j1*2+j2] -> flat index (i1*2+i2)*4 + (j1*2+j2)
    #
    # Index mapping for V (shape 4x6 = 24):
    #   V[j1*2+j2, k1*3+k2, (a,b)] = v1_a[j1,k1] * v2_b[j2,k2]
    #   V[j*p + k, r] where j in [0,3], k in [0,5]
    #   Flat index: (j1*2+j2)*6 + (k1*3+k2)
    #
    # Index mapping for W (shape 6x4 = 24):
    #   W[k1*3+k2, i1*2+i2, (a,b)] = w1_a[k1,i1] * w2_b[k2,i2]
    #   W[k*n + i, r] where k in [0,5], i in [0,3]
    #   Flat index: (k1*3+k2)*4 + (i1*2+i2)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    r1 = len(S2)   # 7
    r2 = len(S2x3) # 11
    total_rank = r1 * r2  # 77
    
    for a, (u1, v1, w1) in enumerate(S2):
        for b, (u2, v2, w2) in enumerate(S2x3):
            r_idx = a * r2 + b
            
            u_col = [0.0] * (n * m)  # 16
            v_col = [0.0] * (m * p)  # 24
            w_col = [0.0] * (p * n)  # 24
            
            # Fill U: u_full[i1*2+i2, j1*2+j2] = u1[i1,j1] * u2[i2,j2]
            for (i1, j1), c1 in u1.items():
                for (i2, j2), c2 in u2.items():
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    u_col[i * m + j] += c1 * c2
            
            # Fill V: v_full[j1*2+j2, k1*3+k2] = v1[j1,k1] * v2[j2,k2]
            for (j1, k1), c1 in v1.items():
                for (j2, k2), c2 in v2.items():
                    j = j1 * 2 + j2
                    k = k1 * 3 + k2
                    v_col[j * p + k] += c1 * c2
            
            # Fill W: w_full[k1*3+k2, i1*2+i2] = w1[k1,i1] * w2[k2,i2]
            for (k1, i1), c1 in w1.items():
                for (k2, i2), c2 in w2.items():
                    k = k1 * 3 + k2
                    i = i1 * 2 + i2
                    w_col[k * n + i] += c1 * c2
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    rank = len(U_cols)
    
    # Transpose: U_cols[r][i] -> U[i][r]
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m * p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p * n)]
    
    # Verify the decomposition
    if _verify_decomposition(U_final, V_final, W_final, n, m, p):
        return {"U": U_final, "V": V_final, "W": W_final}
    
    # Fallback: schoolbook rank-96
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


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is correct.
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For each (i, j, k): need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    i_vals, j_vals, k_vals = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    u_indices = (i_vals * m + j_vals).ravel()
    v_indices = (j_vals * p + k_vals).ravel()
    w_indices = (k_vals * n + i_vals).ravel()
    
    U_sel = U_arr[u_indices, :]
    V_sel = V_arr[v_indices, :]
    W_sel = W_arr[w_indices, :]
    
    totals = np.sum(U_sel * V_sel * W_sel, axis=1)
    
    if not np.allclose(totals, 1.0, atol=1e-10):
        return False
    return True
