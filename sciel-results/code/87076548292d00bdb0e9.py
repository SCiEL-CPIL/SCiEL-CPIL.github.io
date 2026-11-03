import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2 * 2x2 = 2x2 using Strassen (rank 7).
    Indices:
    U: i*2+j (0..3)
    V: j*2+k (0..3)
    W: k*2+i (0..3)
    """
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 1, 0, 0, 0]
    ], dtype=float)
    V = np.array([
        [1, 1, -1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, -1, 1, 0, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 1, 0, -1, -1, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [1, -1, 0, 1, 0, 0, 1]
    ], dtype=float)
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, num_trials=5):
    """
    Verify if U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    """
    R = U.shape[1]
    for _ in range(num_trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p), dtype=float)
        
        A_flat = A.flatten() # i*m+j
        B_flat = B.flatten() # j*p+k
        
        for r in range(R):
            u_val = np.dot(U[:, r], A_flat)
            v_val = np.dot(V[:, r], B_flat)
            if u_val == 0 or v_val == 0:
                continue
            
            w_vec = W[:, r]
            # w_vec index k*n+i
            # C[i, k] += w_vec[k*n+i] * u_val * v_val
            w_mat = w_vec.reshape(p, n)
            C_computed += u_val * v_val * w_mat.T
        
        if not np.allclose(C_computed, C_expected, atol=1e-8):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 10 # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Construct Rank 42 decomposition using 6 Strassen blocks
    # A (2x4) = [A0, A1] (2x2 each)
    # B (4x6) = [[B00, B01, B02], [B10, B11, B12]] (2x2 blocks)
    # C (2x6) = [C0, C1, C2] (2x2 blocks)
    # Cc = A0 * B0c + A1 * B1c
    
    U22, V22, W22 = _strassen_2x2x2()
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col_idx = 0
    # 6 Strassen multiplications
    for source in range(2): # 0: A0, 1: A1
        for c in range(3):  # 0, 1, 2: column blocks
        
            for t in range(7): # 7 Strassen terms
                
                # U factors from A_source
                # A_source is 2x2.
                # A0: A[i, j] for j in 0,1 -> U[i*4 + j]
                # A1: A[i, j+2] for j in 0,1 -> U[i*4 + (j+2)]
                # U22 indices: i*2+j
                for i in range(2):
                    for j in range(2):
                        u_row = i * 4 + (j + 2 * source)
                        U[u_row, col_idx] += U22[i * 2 + j, t]
                
                # V factors from B_part
                # B_part is B[2*source : 2*source+2, 2*c : 2*c+2]
                # B indices: (2*source + qi, 2*c + qj)
                # V index: (2*source + qi) * 6 + (2*c + qj)
                # V22 indices: qi*2 + qj
                for qi in range(2):
                    for qj in range(2):
                        v_row = (2 * source + qi) * 6 + (2 * c + qj)
                        V[v_row, col_idx] += V22[qi * 2 + qj, t]
                
                # W factors to C_part
                # C_part is C[:, 2*c : 2*c+2]
                # C indices: (i, 2*c + rj)
                # W index: (2*c + rj) * 2 + i
                # W22 indices: rj*2 + i
                for rj in range(2):
                    for i in range(2):
                        w_row = (2 * c + rj) * 2 + i
                        W[w_row, col_idx] += W22[rj * 2 + i, t]
                
                col_idx += 1
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p, num_trials=3):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Schoolbook Rank 48
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
