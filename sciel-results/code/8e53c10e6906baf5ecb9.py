import numpy as np

def construct():
    n, m, p = 2, 2, 2
    # Target tensor T[i*m+j, j*p+k] -> W[k*n+i]
    # T is 0 except for 8 entries
    # We want sum_r U[:,r] V[:,r] W[:,r] = T
    
    U = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 1, 0, 0, 1, -1, 0],
        [0, 1, 0, 1, 0, 0, -1]
    ])
    V = np.array([
        [1, 1, 0, -1, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0]
    ])
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Verify
    # Tensor product sum
    # T[im+j, jp+k] contributes to W[kn+i]
    # Actually, the decomposition is:
    # C = A B
    # C_{ki} = sum_{j,l} A_{jl} B_{lk} T_{j,l,k,i}? No.
    # Standard: C_{ik} = sum_j A_{ij} B_{jk}.
    # Prompt W index: k*n + i. So W index corresponds to (k, i).
    # U index: i*m + j. (i, j).
    # V index: j*p + k. (j, k).
    
    # The term in the tensor is A_{ij} B_{jk} contributing to C_{ik}.
    # So T_{i,j,k} = 1.
    # The decomposition is sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = T_{i,j,k}.
    
    T_calc = np.zeros((4, 4, 4))
    for r in range(7):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    T_calc[i, j, k] += U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
                    
    T_target = np.zeros((2, 2, 2))
    # A_{00}B_{00} -> C_{00} (i=0, j=0, k=0)
    T_target[0, 0, 0] = 1
    # A_{01}B_{10} -> C_{00} (i=0, j=1, k=0)
    T_target[0, 1, 0] = 1
    # A_{00}B_{01} -> C_{01} (i=0, j=0, k=1)
    T_target[0, 0, 1] = 1
    # A_{01}B_{11} -> C_{01} (i=0, j=1, k=1)
    T_target[0, 1, 1] = 1
    # A_{10}B_{00} -> C_{10} (i=1, j=0, k=0)
    T_target[1, 0, 0] = 1
    # A_{11}B_{10} -> C_{10} (i=1, j=1, k=0)
    T_target[1, 1, 0] = 1
    # A_{10}B_{01} -> C_{11} (i=1, j=0, k=1)
    T_target[1, 0, 1] = 1
    # A_{11}B_{11} -> C_{11} (i=1, j=1, k=1)
    T_target[1, 1, 1] = 1
    
    # Check
    if np.allclose(T_calc, T_target):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to rank 8
        rank = 8
        U_fb = [[0]*rank for _ in range(4)]
        V_fb = [[0]*rank for _ in range(4)]
        W_fb = [[0]*rank for _ in range(4)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U_fb[i*2+j][col] = 1
                    V_fb[j*2+k][col] = 1
                    W_fb[k*2+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
