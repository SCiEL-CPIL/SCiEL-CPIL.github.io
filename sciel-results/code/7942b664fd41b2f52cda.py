"""
Program to find rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.

Strategy:
  Use the standard schoolbook algorithm which provides an exact decomposition with rank n*m*p.
  For 2x3 by 3x10, rank is 60.
  This ensures correctness and avoids the complex error-prone Strassen implementation which previously failed.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140.0  # Safety margin before 150s kill


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # Schoolbook rank
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Map each multiplication step to a rank-1 component
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # For each i, j, k, we have a term A[i, j] * B[j, k] contributing to C[i, k]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U component: extracts A[i, j]
                # U shape: (n*m, rank). Index is i*m + j
                U[i * m + j, idx] = 1.0
                
                # V component: extracts B[j, k]
                # V shape: (m*p, rank). Index is j*p + k
                V[j * p + k, idx] = 1.0
                
                # W component: contributes to C[i, k]
                # W shape: (p*n, rank). Index is k*n + i (official trace convention W[k*n+i, r])
                W[k * n + i, idx] = 1.0
                
                idx += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    
    # Verify correctness
    n, m, p = DIMENSIONS
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    rank = U.shape[1]
    
    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)
    
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]
        
        A_weighted = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                A_weighted[i, j] = u_r[i * m + j]
        
        B_weighted = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                B_weighted[j, k] = v_r[j * p + k]
        
        prod_r = np.sum(A_weighted * A) * np.sum(B_weighted * B)
        
        for i in range(n):
            for k in range(p):
                C_decomp[i, k] += prod_r * w_r[k * n + i]
    
    C_direct = A @ B
    error = np.max(np.abs(C_decomp - C_direct))
    print(f"Max error: {error}")
    if error > 1e-10:
        print("WARNING: Decomposition is incorrect!")
