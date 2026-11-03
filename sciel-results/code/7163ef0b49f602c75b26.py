import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def _verify_exact(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Returns True if exact, False otherwise."""
    rank = len(U[0])
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Convert to numpy for faster computation
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Check all entries
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            # Expected: 1 if i==ip and j==jp and k==kp, else 0
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            
                            # Computed: sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r]
                            computed = 0.0
                            for r in range(rank):
                                computed += U_np[i*m+j, r] * V_np[jp*p+k, r] * W_np[kp*n+ip, r]
                            
                            if abs(computed - expected) > 1e-9:
                                return False
    return True

def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m  # 6
    dim_V = m * p  # 12
    dim_W = p * n  # 8
    rank_schoolbook = n * m * p  # 24
    
    # Direct construction of the schoolbook decomposition
    # This is always valid and has rank n*m*p
    U = [[0] * rank_schoolbook for _ in range(dim_U)]
    V = [[0] * rank_schoolbook for _ in range(dim_V)]
    W = [[0] * rank_schoolbook for _ in range(dim_W)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
