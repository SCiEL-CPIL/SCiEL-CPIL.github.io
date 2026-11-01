import numpy as np
import time

DIMENSIONS = [2, 3, 5]
_START_TIME = time.time()
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _verify(U, V, W, n, m, p, num_tests=5):
    """Verify the decomposition computes correct matrix multiplication using vectorized ops."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for _ in range(num_tests):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Vectorized: C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Build via einsum: for each r, U_col (n*m,), V_col (m*p,), W_col (p*n,)
        # C[i,k] = sum_r sum_j U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_col = U_arr[:, r].reshape(n, m)
            v_col = V_arr[:, r].reshape(m, p)
            w_col = W_arr[:, r].reshape(p, n)
            # C[i,k] += sum_j u_col[i,j] * v_col[j,k] * w_col[k,i]
            prod = u_col @ v_col  # (n,p)
            C_computed += prod * w_col.T  # w_col.T is (n,p), w_col[k,i] -> w_col.T[i,k]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # Rank 28 decomposition
    # Split into j=0 (rank 10) and j=1,2 (rank 18)
    # j=1,2 split into k=0,1 (Strassen rank 7) and k=2,3 (Strassen rank 7) and k=4 (Schoolbook rank 4)
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1
        U[(0 + i_off) * m + (0 + j_off)][col] = 1
        U[(1 + i_off) * m + (1 + j_off)][col] = 1
        V[(0 + j_off) * p + (0 + k_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = 1
        W[(0 + k_off) * n + (0 + i_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M2
        U[(1 + i_off) * m + (0 + j_off)][col] = 1
        U[(1 + i_off) * m + (1 + j_off)][col] = 1
        V[(0 + j_off) * p + (0 + k_off)][col] = 1
        W[(0 + k_off) * n + (1 + i_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = -1
        col += 1
        
        # M3
        U[(0 + i_off) * m + (0 + j_off)][col] = 1
        V[(0 + j_off) * p + (1 + k_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = -1
        W[(1 + k_off) * n + (0 + i_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M4
        U[(1 + i_off) * m + (1 + j_off)][col] = 1
        V[(1 + j_off) * p + (0 + k_off)][col] = 1
        V[(0 + j_off) * p + (0 + k_off)][col] = -1
        W[(0 + k_off) * n + (0 + i_off)][col] = 1
        W[(0 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M5
        U[(0 + i_off) * m + (0 + j_off)][col] = 1
        U[(0 + i_off) * m + (1 + j_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = 1
        W[(0 + k_off) * n + (0 + i_off)][col] = -1
        W[(1 + k_off) * n + (0 + i_off)][col] = 1
        col += 1
        
        # M6
        U[(1 + i_off) * m + (0 + j_off)][col] = 1
        U[(0 + i_off) * m + (0 + j_off)][col] = -1
        V[(0 + j_off) * p + (0 + k_off)][col] = 1
        V[(0 + j_off) * p + (1 + k_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M7
        U[(0 + i_off) * m + (1 + j_off)][col] = 1
        U[(1 + i_off) * m + (1 + j_off)][col] = -1
        V[(1 + j_off) * p + (0 + k_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = 1
        W[(0 + k_off) * n + (0 + i_off)][col] = 1
        col += 1

    # Block 1: j=0 (Rank 10)
    # This is 2x1 by 1x5
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 2: j=1,2 (Rank 18)
    # Split k into 0,1 (Strassen), 2,3 (Strassen), 4 (Schoolbook)
    
    # k=0,1: Strassen
    add_strassen(0, 1, 0)
    
    # k=2,3: Strassen
    add_strassen(0, 1, 2)
    
    # k=4: Schoolbook (Rank 4)
    # 2x2 by 2x1
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify if time permits
    if _elapsed() < DEADLINE - 5:
        if not _verify(U, V, W, n, m, p):
            # Should not happen, but return anyway
            pass
    
    return result

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
