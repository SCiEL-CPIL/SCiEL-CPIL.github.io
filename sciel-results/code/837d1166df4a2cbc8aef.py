import time
import numpy as np

# Dimensions: n=2 rows of A, m=5 cols of A (rows of B), p=5 cols of B
N, M, P = 2, 5, 5

# Index mappings
def A_idx(i, j): return i * M + j
def B_idx(j, k): return j * P + k
def C_idx(k, i): return k * N + i

def build_tensor():
    T = np.zeros((N*M, M*P, P*N), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[A_idx(i,j), B_idx(j,k), C_idx(k,i)] = 1.0
    return T

def strassen_222():
    """
    Return U, V, W for 2x2x2 matrix multiplication.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: M1 = (A11 - A22)(B21 + B22)
    U[0,0] = 1.0
    U[3,0] = -1.0
    V[2,0] = 1.0
    V[3,0] = 1.0
    W[0,0] = 1.0
    
    # r=1: M2 = (A11 + A22)B22
    U[0,1] = 1.0
    U[3,1] = 1.0
    V[3,1] = 1.0
    W[2,1] = 1.0
    W[3,1] = 1.0
    
    # r=2: M3 = A11(B11 - B21)
    U[0,2] = 1.0
    V[0,2] = 1.0
    V[2,2] = -1.0
    W[1,2] = 1.0
    W[3,2] = -1.0
    
    # r=3: M4 = A22(B12 - B11)
    U[3,3] = 1.0
    V[1,3] = 1.0
    V[0,3] = -1.0
    W[0,3] = 1.0
    W[3,3] = 1.0
    
    # r=4: M5 = (A11 + A12)B11
    U[0,4] = 1.0
    U[1,4] = 1.0
    V[0,4] = 1.0
    W[0,4] = -1.0
    W[1,4] = 1.0
    
    # r=5: M6 = (A21 - A11)(B11 + B12)
    U[2,5] = 1.0
    U[0,5] = -1.0
    V[0,5] = 1.0
    V[1,5] = 1.0
    W[3,5] = 1.0
    
    # r=6: M7 = (A12 - A22)(B21 + B22)
    U[1,6] = 1.0
    U[3,6] = -1.0
    V[2,6] = 1.0
    V[3,6] = 1.0
    W[0,6] = 1.0
    
    return U, V, W

def construct():
    start_time = time.time()
    
    # Total rank estimate: 28 (core) + 1 (edge1) + 1 (edge2) = 30
    max_rank = 30
    U = np.zeros((N*M, max_rank))
    V = np.zeros((M*P, max_rank))
    W = np.zeros((P*N, max_rank))
    
    current_rank = 0
    
    # Part 1: 2x4x4 core
    U_s, V_s, W_s = strassen_222()
    
    for j_off in [0, 2]:
        for k_off in [0, 2]:
            for r in range(7):
                for i in range(2):
                    for j_local in range(2):
                        j_global = j_off + j_local
                        U[A_idx(i, j_global), current_rank] += U_s[i*2 + j_local, r]
                for j_local in range(2):
                    for k_local in range(2):
                        j_global = j_off + j_local
                        k_global = k_off + k_local
                        V[B_idx(j_global, k_global), current_rank] += V_s[j_local*2 + k_local, r]
                for i in range(2):
                    for k_local in range(2):
                        k_global = k_off + k_local
                        W[C_idx(k_global, i), current_rank] += W_s[k_local*2 + i, r]
                current_rank += 1
    
    # Part 2: 2x4x1 edge (j in 0..3, k=4)
    # Rank 1
    for i in range(2):
        for j in range(4):
            U[A_idx(i, j), current_rank] = 1.0
    for j in range(4):
        V[B_idx(j, 4), current_rank] = 1.0
    for i in range(2):
        W[C_idx(4, i), current_rank] = 1.0
    current_rank += 1
    
    # Part 3: 2x1x5 edge (j=4, k in 0..4)
    # Rank 1
    for i in range(2):
        U[A_idx(i, 4), current_rank] = 1.0
    for k in range(5):
        V[B_idx(4, k), current_rank] = 1.0
    for k in range(5):
        for i in range(2):
            W[C_idx(k, i), current_rank] = 1.0
    current_rank += 1
    
    # Trim matrices
    U = U[:, :current_rank]
    V = V[:, :current_rank]
    W = W[:, :current_rank]
    
    # Verify
    T = build_tensor()
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.sum((recon - T)**2)
    
    if err > 1e-9:
        # Fallback to schoolbook
        U = np.zeros((N*M, N*M*P))
        V = np.zeros((M*P, N*M*P))
        W = np.zeros((P*N, N*M*P))
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[A_idx(i,j), col] = 1.0
                    V[B_idx(j,k), col] = 1.0
                    W[C_idx(k,i), col] = 1.0
                    col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
