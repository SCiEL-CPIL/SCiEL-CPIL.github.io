"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Decompose M_{2,3,9} into M_{2,3,3} (otimes) M_{1,1,3}
    # Rank(M_{2,3,3}) = 18
    # Rank(M_{1,1,3}) = 1
    
    n1, m1, p1 = 2, 3, 3
    n2, m2, p2 = 1, 1, 3
    
    cols_U = []
    cols_V = []
    cols_W = []
    
    # Precompute M_{1,1,3} vectors
    u2 = [1]
    v2 = [1, 1, 1]
    w2 = [1, 1, 1]
    
    V2_mat = np.array(v2).reshape(m2, p2)
    W2_trans = np.array(w2).reshape(p2, n2)
    W2_mat = W2_trans.T
    
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                # M_{2,3,3} vectors
                u1 = [0] * (n1 * m1)
                u1[i * m1 + j] = 1
                
                v1 = [0] * (m1 * p1)
                v1[j * p1 + k] = 1
                
                w1 = [0] * (p1 * n1)
                w1[k * n1 + i] = 1
                
                # Combine
                # U
                u_global = u1 # Since u2 is [1]
                
                # V
                V1_mat = np.array(v1).reshape(m1, p1)
                V_global_mat = np.kron(V1_mat, V2_mat)
                v_global = [int(x) for x in V_global_mat.flatten().tolist()]
                
                # W
                W1_trans = np.array(w1).reshape(p1, n1)
                W1_mat = W1_trans.T
                W_global_mat = np.kron(W1_mat, W2_mat)
                W_stored_mat = W_global_mat.T
                w_global = [int(x) for x in W_stored_mat.flatten().tolist()]
                
                cols_U.append(u_global)
                cols_V.append(v_global)
                cols_W.append(w_global)
                
    rank = len(cols_U)
    
    U = [[cols_U[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[cols_V[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[cols_W[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
