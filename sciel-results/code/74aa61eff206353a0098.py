import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = 120.0

    n, m, p = 4, 4, 4

    # Strassen rank-7 factors for 2x2 matrix multiplication
    # Convention: for 2x2, U has rows indexed by (i,j) -> i*2+j
    # V has rows indexed by (j,k) -> j*2+k
    # W has rows indexed by (k,i) -> k*2+i
    U2 = np.array([
        [1,  0,  1,  0,  1, -1,  0],
        [0,  0,  0,  0,  1,  0,  1],
        [0,  1,  0,  0,  0,  1,  0],
        [1,  1,  0,  1,  0,  0, -1]
    ], dtype=float)

    V2 = np.array([
        [1,  1,  0, -1,  0,  1,  0],
        [0,  0,  1,  0,  0,  1,  0],
        [0,  0,  0,  1,  0,  0,  1],
        [1,  0, -1,  0,  1,  0,  1]
    ], dtype=float)

    W2 = np.array([
        [1,  0,  0,  1, -1,  0,  1],
        [0,  1,  0,  1,  0,  0,  0],
        [0,  0,  1,  0,  1,  0,  0],
        [1, -1, 1,  0,  0,  1,  0]
    ], dtype=float)

    # Kronecker product gives 4x4 from 2x2: rank 49
    U_kron = np.kron(U2, U2)  # shape (16, 49)
    V_kron = np.kron(V2, V2)  # shape (16, 49)
    W_kron = np.kron(W2, W2)  # shape (16, 49)

    # Permute rows to standard row-major ordering
    # For U: row index S corresponds to (i,j) where i=S//4, j=S%4
    #   i = i_out*2 + i_in, j = j_out*2 + j_in
    #   Kron row = (i_out*2+j_out)*4 + (i_in*2+j_in)
    # For V: row index S corresponds to (j,k) where j=S//4, k=S%4
    #   j = j_out*2 + j_in, k = k_out*2 + k_in
    #   Kron row = (j_out*2+k_out)*4 + (j_in*2+k_in)
    # For W: row index S corresponds to (k,i) where k=S//4, i=S%4
    #   k = k_out*2 + k_in, i = i_out*2 + i_in
    #   Kron row = (k_out*2+i_out)*4 + (k_in*2+i_in)

    perm_U = []
    perm_V = []
    perm_W = []
    
    for S in range(16):
        # U: (i, j)
        i = S // 4
        j = S % 4
        i_out, i_in = i // 2, i % 2
        j_out, j_in = j // 2, j % 2
        perm_U.append((i_out * 2 + j_out) * 4 + (i_in * 2 + j_in))
        
        # V: (j, k)
        j = S // 4
        k = S % 4
        j_out, j_in = j // 2, j % 2
        k_out, k_in = k // 2, k % 2
        perm_V.append((j_out * 2 + k_out) * 4 + (j_in * 2 + k_in))
        
        # W: (k, i)
        k = S // 4
        i = S % 4
        k_out, k_in = k // 2, k % 2
        i_out, i_in = i // 2, i % 2
        perm_W.append((k_out * 2 + i_out) * 4 + (k_in * 2 + i_in))

    U = U_kron[perm_U, :]
    V = V_kron[perm_V, :]
    W = W_kron[perm_W, :]

    # Verification
    if time.time() - t0 > deadline - 5.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    max_err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = np.sum(U[i*4+j, :] * V[j*4+k, :] * W[k*4+i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
                    
    if max_err > 1e-9:
        # Fallback to schoolbook rank 64 if verification fails
        rank_fb = 64
        U_fb = np.zeros((16, rank_fb), dtype=float)
        V_fb = np.zeros((16, rank_fb), dtype=float)
        W_fb = np.zeros((16, rank_fb), dtype=float)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i*4+j, col] = 1.0
                    V_fb[j*4+k, col] = 1.0
                    W_fb[k*4+i, col] = 1.0
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
