import time
import numpy as np

def _construct_rank_56():
    """
    Constructs a rank-56 decomposition for 2x4 * 4x8 matrix multiplication
    using the Kronecker product of Strassen (2x2x2) and Schoolbook (1x2x4).
    """
    n, m, p = 2, 4, 8
    
    # Strassen factors for 2x2x2
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # Schoolbook factors for 1x2x4
    U_b = np.zeros((2, 8))
    V_b = np.zeros((8, 8))
    W_b = np.zeros((4, 8))
    
    col = 0
    for j in range(2):
        for k in range(4):
            U_b[j, col] = 1
            V_b[j * 4 + k, col] = 1
            W_b[k, col] = 1
            col += 1
    
    rank = 7 * 8
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for s in range(7):
        for b in range(8):
            col = s * 8 + b
            
            for i in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        val_s = U_s[i * 2 + j1, s]
                        val_b = U_b[j2, b]
                        if val_s != 0 and val_b != 0:
                            global_idx = i * 4 + (2 * j1 + j2)
                            U[global_idx, col] += val_s * val_b
            
            for j1 in range(2):
                for j2 in range(2):
                    for c1 in range(2):
                        for c2 in range(4):
                            j = 2 * j1 + j2
                            k = 4 * c1 + c2
                            val_s = V_s[j1 * 2 + c1, s]
                            val_b = V_b[j2 * 4 + c2, b]
                            if val_s != 0 and val_b != 0:
                                global_idx = j * 8 + k
                                V[global_idx, col] += val_s * val_b
            
            for i in range(2):
                for c1 in range(2):
                    for c2 in range(4):
                        k = 4 * c1 + c2
                        val_s = W_s[c1 * 2 + i, s]
                        val_b = W_b[c2, b]
                        if val_s != 0 and val_b != 0:
                            global_idx = k * 2 + i
                            W[global_idx, col] += val_s * val_b
    
    return U, V, W

def construct():
    deadline = time.time() + 110
    U, V, W = _construct_rank_56()
    rank = U.shape[1]
    
    rng = np.random.default_rng(42)
    A_test = rng.standard_normal((2, 4))
    B_test = rng.standard_normal((4, 8))
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((2, 8))
    for r in range(rank):
        u_vec = U[:, r].reshape(2, 4)
        v_vec = V[:, r].reshape(4, 8)
        w_vec = W[:, r].reshape(8, 2)
        C_computed += (u_vec @ v_vec) * w_vec.T
    
    if not np.allclose(C_computed, C_expected, atol=1e-10, rtol=1e-10):
        rank_fb = 64
        U_fb = np.zeros((8, rank_fb))
        V_fb = np.zeros((32, rank_fb))
        W_fb = np.zeros((16, rank_fb))
        col = 0
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    U_fb[i * 4 + j, col] = 1.0
                    V_fb[j * 8 + k, col] = 1.0
                    W_fb[k * 2 + i, col] = 1.0
                    col += 1
        return {
            "U": U_fb.tolist(),
            "V": V_fb.tolist(),
            "W": W_fb.tolist()
        }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
