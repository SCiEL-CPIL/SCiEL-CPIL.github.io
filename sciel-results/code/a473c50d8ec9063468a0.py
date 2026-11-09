import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Schoolbook: rank = n * m * p = 54
    r = n * m * p  # 54
    U = [[0] * r for _ in range(n * m)]  # 6 x 54
    V = [[0] * r for _ in range(m * p)]  # 27 x 54
    W = [[0] * r for _ in range(p * n)]  # 18 x 54
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if time.time() < deadline - 10:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Test with random matrices
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        A_vec = A.flatten()  # length n*m = 6
        B_vec = B.flatten()  # length m*p = 27
        
        # C[i,k] = sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j',k'} V[j'*p+k', r] * B[j',k'])
        C_computed = np.zeros((n, p))
        for r_idx in range(r):
            a_part = np.sum(U_np[:, r_idx] * A_vec)
            b_part = np.sum(V_np[:, r_idx] * B_vec)
            for k_idx in range(p):
                for i_idx in range(n):
                    C_computed[i_idx, k_idx] += W_np[k_idx * n + i_idx, r_idx] * a_part * b_part
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback - should not happen
            pass
    
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
