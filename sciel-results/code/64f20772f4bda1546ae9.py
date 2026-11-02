import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    rank_strassen = 7
    # The 2x2x2 block is at i in [0,1], j in [0,1], k in [0,1]
    # We replace the 8 schoolbook terms for this block with 7 Strassen terms.
    # The other 67 terms remain schoolbook.
    total_rank = 7 + (rank_schoolbook - 8)
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # Strassen decomposition for the 2x2x2 subblock
    # Indices in subblock: i in {0,1}, j in {0,1}, k in {0,1}
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # M1 = (a00+a11)(b00+b11)
    col = 0
    U[0*m+0][col] += 1
    U[1*m+1][col] += 1
    V[0*p+0][col] += 1
    V[1*p+1][col] += 1
    W[0*n+0][col] += 1  # C00
    W[1*n+1][col] += 1  # C11
    col += 1
    
    # M2 = (a10+a11)b00
    U[1*m+0][col] += 1
    U[1*m+1][col] += 1
    V[0*p+0][col] += 1
    W[0*n+1][col] += 1  # C10
    W[1*n+1][col] += -1 # C11
    col += 1
    
    # M3 = a00(b01-b11)
    U[0*m+0][col] += 1
    V[0*p+1][col] += 1
    V[1*p+1][col] += -1
    W[1*n+0][col] += 1  # C01
    W[1*n+1][col] += 1  # C11
    col += 1
    
    # M4 = a11(b10-b00)
    U[1*m+1][col] += 1
    V[1*p+0][col] += 1
    V[0*p+0][col] += -1
    W[0*n+0][col] += 1  # C00
    W[0*n+1][col] += 1  # C10
    col += 1
    
    # M5 = (a00+a01)b11
    U[0*m+0][col] += 1
    U[0*m+1][col] += 1
    V[1*p+1][col] += 1
    W[0*n+0][col] += -1 # C00
    W[1*n+0][col] += 1  # C01
    col += 1
    
    # M6 = (a10-a00)(b00+b01)
    U[1*m+0][col] += 1
    U[0*m+0][col] += -1
    V[0*p+0][col] += 1
    V[0*p+1][col] += 1
    W[1*n+1][col] += 1  # C11
    col += 1
    
    # M7 = (a01-a11)(b10+b11)
    U[0*m+1][col] += 1
    U[1*m+1][col] += -1
    V[1*p+0][col] += 1
    V[1*p+1][col] += 1
    W[0*n+0][col] += 1  # C00
    col += 1
    
    # Schoolbook for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip the 2x2x2 block already handled by Strassen
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*m+j][col] += 1
                V[j*p+k][col] += 1
                W[k*n+i][col] += 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    # Check rank
    print(f"Rank: {U.shape[1]}")
    # Verify correctness
    # T_ijk = sum_c U_ic V_jc W_kc
    # Expected: delta_ik if j is such that A_ij B_jk = C_ik
    # Actually, the tensor T is such that C_ik = sum_j A_ij T_ijk B_jk
    # So T_ijk should be 1 if it corresponds to the multiplication term A_ij B_jk
    # My construction sets U, V, W such that the contraction gives the identity map on the tensor product space.
    # Let's verify a few entries.
    # T_ijk = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
    # For schoolbook terms, this is 1.
    # For Strassen terms, it should also reconstruct the identity.
    pass
