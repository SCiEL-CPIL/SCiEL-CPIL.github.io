import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _construct_strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indices:
    U: i*2 + j
    V: j*2 + k
    W: k*2 + i
    """
    # Strassen's 7 products
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # We need to map these to the tensor form:
    # C_ik = sum_r U_i,r M_r V_k,r ? No.
    # The tensor form is sum_r U_ij,r V_jk,r W_ki,r = 1.
    
    # Let's define the 7 rank-1 tensors.
    # Each tensor is defined by U, V, W vectors.
    
    # M1:
    # A part: A00 + A11 -> U indices (0,0) and (1,1)
    # B part: B00 + B11 -> V indices (0,0) and (1,1)
    # C part: C00 += M1, C01 += M1, C10 += M1, C11 += M1
    # W indices: (0,0), (1,0), (0,1), (1,1) -> (k,i)
    
    # Let's just hardcode the 7 sets of (U, V, W) indices and values.
    # U index: i*2+j
    # V index: j*2+k
    # W index: k*2+i
    
    # 1. (A00+A11)(B00+B11)
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1  (k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1)
    
    # 2. (A10+A11)B00
    # U: (1,0)=1, (1,1)=1
    # V: (0,0)=1
    # W: (0,1)=1 (k=0, i=1)
    
    # 3. A00(B01-B11)
    # U: (0,0)=1
    # V: (0,1)=1, (1,1)=-1
    # W: (1,0)=1 (k=1, i=0)
    
    # 4. A11(B10-B00)
    # U: (1,1)=1
    # V: (1,0)=1, (0,0)=-1
    # W: (1,1)=1 (k=1, i=1)
    
    # 5. (A00+A01)B11
    # U: (0,0)=1, (0,1)=1
    # V: (1,1)=1
    # W: (1,0)=1, (1,1)=1 (k=1, i=0; k=1, i=1)
    
    # 6. (A10-A00)(B00+B01)
    # U: (1,0)=1, (0,0)=-1
    # V: (0,0)=1, (0,1)=1
    # W: (0,1)=1, (1,1)=1 (k=0, i=1; k=1, i=1)
    
    # 7. (A01-A11)(B10+B11)
    # U: (0,1)=1, (1,1)=-1
    # V: (1,0)=1, (1,1)=1
    # W: (0,0)=1, (0,1)=1 (k=0, i=0; k=0, i=1)

    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # 1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # 2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # 3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # 4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[3, 3] = 1
    
    # 5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[2, 4] = 1; W[3, 4] = 1
    
    # 6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[1, 5] = 1; W[3, 5] = 1
    
    # 7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = 44
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Part 1: i in {0,1}, j in {0,1}, k in {0..5}
    # This is 2x2x6. Split k into 3 blocks of 2.
    for k_start in [0, 2, 4]:
        U_s, V_s, W_s = _construct_strassen_222()
        # Map indices
        # U index: i*2+j. Original i,j.
        # V index: j*2+k. Original j, k_local.
        # W index: k*2+i. Original k_local, i.
        
        for r in range(7):
            for u_idx, val in np.ndenumerate(U_s[:, r]):
                if val != 0:
                    i, j = u_idx
                    U[i*m + j, col] += val
            for v_idx, val in np.ndenumerate(V_s[:, r]):
                if val != 0:
                    j, k_loc = v_idx
                    V[j*p + (k_start + k_loc), col] += val
            for w_idx, val in np.ndenumerate(W_s[:, r]):
                if val != 0:
                    k_loc, i = w_idx
                    W[(k_start + k_loc)*n + i, col] += val
            col += 1
            
    # Part 2: i in {0,1}, j in {2,3}, k in {0..5}
    # Similar, but j is offset by 2.
    for k_start in [0, 2, 4]:
        U_s, V_s, W_s = _construct_strassen_222()
        for r in range(7):
            for u_idx, val in np.ndenumerate(U_s[:, r]):
                if val != 0:
                    i, j_loc = u_idx
                    U[i*m + (2 + j_loc), col] += val
            for v_idx, val in np.ndenumerate(V_s[:, r]):
                if val != 0:
                    j_loc, k_loc = v_idx
                    V[(2 + j_loc)*p + (k_start + k_loc), col] += val
            for w_idx, val in np.ndenumerate(W_s[:, r]):
                if val != 0:
                    k_loc, i = w_idx
                    W[(k_start + k_loc)*n + i, col] += val
            col += 1
            
    # Part 3: i in {0,1}, j in {4}, k in {0..5}
    # Rank 1.
    # U[i, 4] = 1
    # V[4, k] = 1
    # W[k, i] = 1
    for i in range(2):
        U[i*m + 4, col] = 1
    for k in range(6):
        V[4*p + k, col] = 1
        W[k*n + i, col] = 1 # Wait, W index is k*n + i.
    # Correction: W[k*n + i] for i in 0,1.
    for i in range(2):
        for k in range(6):
            W[k*n + i, col] = 1
    col += 1
    
    # Part 4: i in {2}, j in {0..4}, k in {0..5}
    # Rank 1.
    # U[2, j] = 1
    # V[j, k] = 1
    # W[k, 2] = 1
    for j in range(5):
        U[2*m + j, col] = 1
    for j in range(5):
        for k in range(6):
            V[j*p + k, col] = 1
    for k in range(6):
        W[k*n + 2, col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
