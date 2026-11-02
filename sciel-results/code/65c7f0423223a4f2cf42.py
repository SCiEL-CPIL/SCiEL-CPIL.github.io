"""Matrix multiplication 4x4x8 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, -1, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ])
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1],
    ])
    W2 = np.array([
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0],
    ])
    
    # Schoolbook 1x1x2 decomposition
    # U (1x1=1 row), V (1x2=2 rows), W (2x1=2 rows)
    U1 = np.array([[1, 1]])
    V1 = np.array([[1, 0], [0, 1]])
    W1 = np.array([[1, 0], [0, 1]])
    
    # Kronecker product helper
    def kron_decomp(U1, V1, W1, U2, V2, W2):
        # U has shape (n1*m1, r1) and (n2*m2, r2) -> (n1*m1*n2*m2, r1*r2)
        # Index mapping: (i1, j1, i2, j2) -> i1*m1_total + j1? 
        # Standard Kronecker: A (n1 x m1) and B (n2 x m2) -> n1*n2 x m1*m2
        # Here we have tensors.
        # U_total[i1*m1_total + j1, i2*m2_total + j2, c1*r2 + c2] = U1[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
        # But we need to flatten the indices correctly for the final U, V, W.
        
        n1, m1, p1 = 2, 2, 2
        n2, m2, p2 = 1, 1, 2
        
        # Final dimensions
        n = n1 * n2
        m = m1 * m2
        p = p1 * p2
        
        r = U1.shape[1] * U2.shape[1]
        
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        for i1 in range(n1):
            for j1 in range(m1):
                for i2 in range(n2):
                    for j2 in range(m2):
                        # Index in U: (i1, j1, i2, j2)
                        # Flattened index for U: (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
                        idx_u = (i1 * n2 + i2) * (m1 * m2) + (j1 * m2 + j2)
                        
                        for c1 in range(U1.shape[1]):
                            for c2 in range(U2.shape[1]):
                                idx_c = c1 * U2.shape[1] + c2
                                U[idx_u, idx_c] = U1[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
        
        for j1 in range(m1):
            for k1 in range(p1):
                for j2 in range(m2):
                    for k2 in range(p2):
                        # Index in V: (j1, k1, j2, k2)
                        # Flattened index for V: (j1*m2 + j2) * (p1*p2) + (k1*p2 + k2)
                        idx_v = (j1 * m2 + j2) * (p1 * p2) + (k1 * p2 + k2)
                        
                        for c1 in range(V1.shape[1]):
                            for c2 in range(V2.shape[1]):
                                idx_c = c1 * V2.shape[1] + c2
                                V[idx_v, idx_c] = V1[j1*p1+k1, c1] * V2[j2*p2+k2, c2]
        
        for k1 in range(p1):
            for i1 in range(n1):
                for k2 in range(p2):
                    for i2 in range(n2):
                        # Index in W: (k1, i1, k2, i2)
                        # Flattened index for W: (k1*p2 + k2) * (n1*n2) + (i1*n2 + i2)
                        idx_w = (k1 * p2 + k2) * (n1 * n2) + (i1 * n2 + i2)
                        
                        for c1 in range(W1.shape[1]):
                            for c2 in range(W2.shape[1]):
                                idx_c = c1 * W2.shape[1] + c2
                                W[idx_w, idx_c] = W1[k1*n1+i1, c1] * W2[k2*n2+i2, c2]
        
        return U, V, W
    
    # Construct 4x4x4 using 2x2x2 and 2x2x2 (Strassen x Strassen)
    # Wait, 4x4x4 = (2x2) x (2x2) x (2x2).
    # We have Strassen for 2x2x2.
    # So 4x4x4 rank = 7 * 7 = 49.
    U444, V444, W444 = kron_decomp(U2, V2, W2, U2, V2, W2)
    
    # Construct 4x4x8 using 4x4x4 and 1x1x2 (Schoolbook)
    # 4x4x8 = (4x4) x (4x1) x (4x2)? No.
    # Dimensions: n=4, m=4, p=8.
    # We can view as (2x2) x (2x2) x (2x2x2)? No.
    # We can view as (4x4x4) x (1x1x2).
    # n=4*1=4, m=4*1=4, p=4*2=8.
    # So we combine 4x4x4 (Strassen^2) and 1x1x2 (Schoolbook).
    U448, V448, W448 = kron_decomp(U444, V444, W444, U1, V1, W1)
    
    # Verify
    def verify(U, V, W, n, m, p):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if verify(U448, V448, W448, n, m, p):
        return {"U": U448.tolist(), "V": V448.tolist(), "W": W448.tolist()}
    else:
        # Fallback to Schoolbook
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    r = len(result["U"][0])
    print(f"Rank: {r}")
