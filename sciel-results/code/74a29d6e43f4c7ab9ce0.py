"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 52 decomposition
    # J_{2,5,6} = J_{2,4,4} (rank 28) + J_{2,1,4} (rank 4) + J_{2,5,2} (rank 20)
    
    rank = 52
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Block 1: J_{2,4,4} (Rank 28)
    # Uses A[0:2, 0:4], B[0:4, 0:4], C[0:4, 0:2]
    # Constructed from 4 copies of J_{2,2,2} (Rank 7)
    # J_{2,2,2} factors (Strassen)
    # U_222: (4, 7), V_222: (4, 7), W_222: (4, 7)
    # Indices: i in {0,1}, j in {0,1} for U. j in {0,1}, k in {0,1} for V. k in {0,1}, i in {0,1} for W.
    
    # Strassen decomposition for 2x2x2
    # U rows: (i,j)
    # V rows: (j,k)
    # W rows: (k,i)
    
    # Term 1: (0,0) * (0,0) -> (0,0)
    # Term 2: (0,0) * (1,1) -> (1,0)
    # Term 3: (1,0) * (0,0) -> (0,1)
    # Term 4: (1,1) * (1,1) -> (1,1)
    # Term 5: (0,0) + (1,1) * (0,1) + (1,0) -> (0,1) + (1,0)
    # Term 6: (0,1) + (1,0) * (1,1) + (0,0) -> (1,1) + (0,0)
    # Term 7: (0,0) + (0,1) * (1,0) + (1,1) -> (1,0) + (0,1)
    
    # We need the explicit factors for J_{2,2,2}.
    # U:
    # c=0: U[0,0]=1
    # c=1: U[0,0]=1
    # c=2: U[1,0]=1
    # c=3: U[1,1]=1
    # c=4: U[0,0]=1, U[1,1]=1
    # c=5: U[0,1]=1, U[1,0]=1
    # c=6: U[0,0]=1, U[0,1]=1
    
    # V:
    # c=0: V[0,0]=1
    # c=1: V[1,1]=1
    # c=2: V[0,0]=1
    # c=3: V[1,1]=1
    # c=4: V[0,1]=1, V[1,0]=1
    # c=5: V[1,1]=1, V[0,0]=1
    # c=6: V[1,0]=1, V[1,1]=1
    
    # W:
    # c=0: W[0,0]=1
    # c=1: W[1,0]=1
    # c=2: W[0,1]=1
    # c=3: W[1,1]=1
    # c=4: W[0,1]=1, W[1,0]=1
    # c=5: W[1,1]=1, W[0,0]=1
    # c=6: W[1,0]=1, W[0,1]=1
    
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # c=0
    U_222[0, 0] = 1
    V_222[0, 0] = 1
    W_222[0, 0] = 1
    # c=1
    U_222[0, 1] = 1
    V_222[3, 1] = 1
    W_222[2, 1] = 1
    # c=2
    U_222[2, 2] = 1
    V_222[0, 2] = 1
    W_222[1, 2] = 1
    # c=3
    U_222[3, 3] = 1
    V_222[3, 3] = 1
    W_222[3, 3] = 1
    # c=4
    U_222[0, 4] = 1
    U_222[3, 4] = 1
    V_222[1, 4] = 1
    V_222[2, 4] = 1
    W_222[1, 4] = 1
    W_222[2, 4] = 1
    # c=5
    U_222[1, 5] = 1
    U_222[2, 5] = 1
    V_222[3, 5] = 1
    V_222[0, 5] = 1
    W_222[3, 5] = 1
    W_222[0, 5] = 1
    # c=6
    U_222[0, 6] = 1
    U_222[1, 6] = 1
    V_222[2, 6] = 1
    V_222[3, 6] = 1
    W_222[2, 6] = 1
    W_222[1, 6] = 1
    
    # Block 1: 4 copies of J_{2,2,2}
    # (i,j) in 2x4, (j,k) in 4x4, (k,i) in 4x2
    # Copy 1: i=0,1; j=0,1; k=0,1
    # Copy 2: i=0,1; j=0,1; k=2,3
    # Copy 3: i=0,1; j=2,3; k=0,1
    # Copy 4: i=0,1; j=2,3; k=2,3
    
    for c_idx in range(4):
        for c_local in range(7):
            c = col + c_local
            # i offset 0
            # j offset: c_idx // 2 * 2
            # k offset: c_idx % 2 * 2
            
            j_off = (c_idx // 2) * 2
            k_off = (c_idx % 2) * 2
            
            # U: i in {0,1}, j in {0,1} -> i*m + (j + j_off)
            for i in range(2):
                for j in range(2):
                    if U_222[i*2+j, c_local] != 0:
                        U[i*m + (j + j_off), c] = U_222[i*2+j, c_local]
            
            # V: j in {0,1}, k in {0,1} -> (j + j_off)*p + (k + k_off)
            for j in range(2):
                for k in range(2):
                    if V_222[j*2+k, c_local] != 0:
                        V[(j + j_off)*p + (k + k_off), c] = V_222[j*2+k, c_local]
            
            # W: k in {0,1}, i in {0,1} -> (k + k_off)*n + i
            for k in range(2):
                for i in range(2):
                    if W_222[k*2+i, c_local] != 0:
                        W[(k + k_off)*n + i, c] = W_222[k*2+i, c_local]
            
    col += 28
    
    # Block 2: J_{2,1,4} (Rank 4)
    # Uses A[0:2, 4], B[4, 0:4], C[0:4, 0:2]
    # U: i in {0,1}, j=4
    # V: j=4, k in {0..3}
    # W: k in {0..3}, i in {0,1}
    
    for k in range(4):
        c = col + k
        for i in range(2):
            U[i*m + 4, c] = 1
            V[4*p + k, c] = 1
            W[k*n + i, c] = 1
            
    col += 4
    
    # Block 3: J_{2,5,2} (Rank 20)
    # Uses A[0:2, 0:5], B[0:5, 4:6], C[4:6, 0:2]
    # U: i in {0,1}, j in {0..4}
    # V: j in {0..4}, k in {4,5}
    # W: k in {4,5}, i in {0,1}
    
    for j in range(5):
        for k in range(2):
            c = col + j*2 + k
            for i in range(2):
                U[i*m + j, c] = 1
                V[j*p + (4 + k), c] = 1
                W[(4 + k)*n + i, c] = 1
                
    col += 20
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
