import time
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = 140  # Safety margin

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify if the decomposition is exact."""
    rank = U.shape[1]
    U_ = U.reshape(n, m, rank)
    V_ = V.reshape(m, p, rank)
    W_ = W.reshape(p, n, rank)
    
    # The tensor T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # We want T_{i,j,k} = 1 for all i,j,k
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        result += np.einsum('ij,jk,ki->ijk', U_[:, :, r], V_[:, :, r], W_[:, :, r])
    
    if np.max(np.abs(result - 1.0)) > tol:
        return False
    return True

def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen's 7 products for 2x2:
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = -1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[2, 2] = 1
    
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_2x2x1():
    """Return a rank-4 decomposition for 2x2x1 matrix-vector multiplication."""
    rank = 4
    U = np.zeros((4, rank))
    V = np.zeros((2, rank))
    W = np.zeros((2, rank))
    
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
    U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
    
    return U, V, W

def _construct_schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _construct_optimal_block(n, m, p):
    """Construct the decomposition using the optimal block splits found by search."""
    # Based on search, optimal is m=2+2, p=3+4 for 2x2x7
    # And 2x2x3 is 1+2, 2x2x4 is 2+2
    
    rank = 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_block(u_off, v_off, w_off, n_b, m_b, p_b, base_U, base_V, base_W, base_rank):
        nonlocal col
        for r in range(base_rank):
            for i in range(n_b):
                for j in range(m_b):
                    if base_U[i * m_b + j, r] != 0:
                        U[(u_off[0] + i) * m + (u_off[1] + j), col + r] = base_U[i * m_b + j, r]
            for j in range(m_b):
                for k in range(p_b):
                    if base_V[j * p_b + k, r] != 0:
                        V[(v_off[0] + j) * p + (v_off[1] + k), col + r] = base_V[j * p_b + k, r]
            for k in range(p_b):
                for i in range(n_b):
                    if base_W[k * n_b + i, r] != 0:
                        W[(w_off[0] + k) * n + (w_off[1] + i), col + r] = base_W[k * n_b + i, r]
        col += base_rank

    # Block 1: 2x2x1 (A_left, B_top_left_1)
    add_block((0, 0), (0, 0), (0, 0), 2, 2, 1, *_strassen_2x2x1(), 4)
    
    # Block 2: 2x2x2 (A_left, B_top_left_2)
    add_block((0, 0), (0, 1), (1, 0), 2, 2, 2, *_strassen_2x2x2(), 7)
    
    # Block 3: 2x2x2 (A_left, B_top_right_1)
    add_block((0, 0), (0, 3), (3, 0), 2, 2, 2, *_strassen_2x2x2(), 7)
    
    # Block 4: 2x2x2 (A_left, B_top_right_2)
    add_block((0, 0), (0, 5), (5, 0), 2, 2, 2, *_strassen_2x2x2(), 7)
    
    # Block 5: 2x2x1 (A_right, B_bot_left_1)
    add_block((0, 2), (2, 0), (0, 0), 2, 2, 1, *_strassen_2x2x1(), 4)
    
    # Block 6: 2x2x2 (A_right, B_bot_left_2)
    add_block((0, 2), (2, 1), (1, 0), 2, 2, 2, *_strassen_2x2x2(), 7)
    
    # Block 7: 2x2x2 (A_right, B_bot_right_1)
    add_block((0, 2), (2, 3), (3, 0), 2, 2, 2, *_strassen_2x2x2(), 7)
    
    # Block 8: 2x2x2 (A_right, B_bot_right_2)
    add_block((0, 2), (2, 5), (5, 0), 2, 2, 2, *_strassen_2x2x2(), 7)
    
    if _verify(U, V, W, n, m, p):
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    result = _construct_optimal_block(n, m, p)
    if result is not None:
        U, V, W = result
    else:
        U, V, W = _construct_schoolbook(n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
