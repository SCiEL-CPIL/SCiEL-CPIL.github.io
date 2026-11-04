"""Matrix multiplication 4x4x9 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    U_rows = n * m  # 16
    V_rows = m * p  # 36
    W_rows = p * n  # 36
    
    total_rank = 114
    U = np.zeros((U_rows, total_rank))
    V = np.zeros((V_rows, total_rank))
    W = np.zeros((W_rows, total_rank))
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Rank-7 Strassen for 2x2*2x2
    # Indices: A: (i,j) i,j in {0,1}, B: (j,k) j,k in {0,1}, C: (k,i) k,i in {0,1}
    # U index for 2x2: i*2+j, V index: j*2+k, W index: k*2+i
    s2 = [
        # (U_coeffs {idx:val}, V_coeffs {idx:val}, W_coeffs {idx:val})
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1},       {2:1}),
        ({0:1},       {1:1, 3:-1},{1:1, 3:1}),
        ({3:1},       {2:1, 0:-1},{0:1, 2:1}),
        ({0:1, 1:1},  {3:1},      {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1, 3:-1}),
    ]
    
    # For 4x4*4x4, the indices are:
    # A: (i,j) i,j in {0,1,2,3} -> U index: i*4+j
    # B: (j,k) j,k in {0,1,2,3} -> V index: j*4+k (but k offset for B1/B2)
    # C: (k,i) k,i in {0,1,2,3} -> W index: k*4+i (but k offset for C1/C2)
    
    # The 4x4*4x4 tensor is the Kronecker product of 2x2*2x2 with itself.
    # If we write i = i1*2 + i2, j = j1*2 + j2, k = k1*2 + k2 where i1,i2,j1,j2,k1,k2 in {0,1}
    # Then the 4x4*4x4 tensor = sum over all (i1,i2,j1,j2,k1,k2) with the product structure.
    # The rank-49 decomposition: for each pair (r,s) of rank-1 terms from 2x2*2x2,
    # we get a rank-1 term for 4x4*4x4.
    
    # For 2x2*2x2 rank-1 term r:
    # U_r: coefficients for (i1,j1) pairs -> maps to 2x2 U indices
    # V_r: coefficients for (j1,k1) pairs -> maps to 2x2 V indices  
    # W_r: coefficients for (k1,i1) pairs -> maps to 2x2 W indices
    
    # For the Kronecker product term (r,s):
    # U_{r,s}: for each (i1,j1) in U_r with coeff cu, and (i2,j2) in U_s with coeff cu2,
    #          U index for 4x4: (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2
    #          coeff = cu * cu2
    # Similarly for V and W.
    
    def add_4x4_strassen_kron(offset, bco, wco):
        """Add rank-49 Kronecker product decomposition for 4x4*4x4."""
        for r in range(7):
            Ur, Vr, Wr = s2[r]
            for s in range(7):
                Us, Vs, Ws = s2[s]
                # Build U coefficients for 4x4
                u_coeffs = {}
                for (i1, j1), cu in Ur.items():
                    for (i2, j2), cu2 in Us.items():
                        i = i1 * 2 + i2
                        j = j1 * 2 + j2
                        idx = u_idx(i, j)
                        u_coeffs[idx] = u_coeffs.get(idx, 0) + cu * cu2
                v_coeffs = {}
                for (j1, k1), cv in Vr.items():
                    for (j2, k2), cv2 in Vs.items():
                        j = j1 * 2 + j2
                        k = k1 * 2 + k2 + bco
                        idx = v_idx(j, k)
                        v_coeffs[idx] = v_coeffs.get(idx, 0) + cv * cv2
                w_coeffs = {}
                for (k1, i1), cw in Wr.items():
                    for (k2, i2), cw2 in Ws.items():
                        k = k1 * 2 + k2 + wco
                        i = i1 * 2 + i2
                        idx = w_idx(k, i)
                        w_coeffs[idx] = w_coeffs.get(idx, 0) + cw * cw2
                
                for idx, val in u_coeffs.items():
                    U[idx, offset] += val
                for idx, val in v_coeffs.items():
                    V[idx, offset] += val
                for idx, val in w_coeffs.items():
                    W[idx, offset] += val
                offset += 1
        return offset
    
    # Part 1: A * B1 (B cols 0-3, C cols 0-3)
    offset = add_4x4_strassen_kron(offset, 0, 0)
    # Part 2: A * B2 (B cols 4-7, C cols 4-7)
    offset = add_4x4_strassen_kron(offset, 4, 4)
    # Part 3: A * B3 (B col 8, C col 8) - schoolbook rank 16
    for i in range(4):
        for j in range(4):
            U[u_idx(i, j), offset] = 1
            V[v_idx(j, 8), offset] = 1
            W[w_idx(8, i), offset] = 1
            offset += 1
    
    assert offset == 114, f"Expected 114, got {offset}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
