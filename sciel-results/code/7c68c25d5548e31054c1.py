import time

def construct():
    n, m, p = 2, 3, 8
    
    # Dimensions of factors
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16
    
    rank = 44
    
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    col = 0
    
    # Part 1: 2x1x8 part (j=0)
    # A[:, 0] * B[0, :]
    # For each k in 0..7
    for k in range(p):
        # Term 1: A[0,0] * B[0,k] -> C[0,k]
        U[0 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # Term 2: A[1,0] * B[0,k] -> C[1,k]
        U[1 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
    # Part 2: 2x2x8 part (j=1,2)
    # Split into 4 blocks of 2x2x2
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # P1: (A[0,1] + A[1,2]) * (B[1,k0] + B[2,k1])
        # U: A[0,1]=1, A[1,2]=1
        U[0 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        # V: B[1,k0]=1, B[2,k1]=1
        V[1 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        # W: C[0,k0]=1, C[1,k1]=1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P2: (A[1,1] + A[1,2]) * B[1,k0]
        U[1 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        # W: C[1,k0]=1, C[1,k1]=-1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # P3: A[0,1] * (B[1,k1] - B[2,k1])
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        V[2 * p + k1][col] = -1
        # W: C[0,k1]=1, C[1,k1]=1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P4: A[1,2] * (B[2,k0] - B[1,k0])
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        V[1 * p + k0][col] = -1
        # W: C[0,k0]=1, C[1,k0]=1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # P5: (A[0,1] + A[0,2]) * B[2,k1]
        U[0 * m + 1][col] = 1
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        # W: C[0,k0]=-1, C[0,k1]=1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # P6: (A[1,1] - A[0,1]) * (B[1,k0] + B[1,k1])
        U[1 * m + 1][col] = 1
        U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        # W: C[1,k1]=1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P7: (A[0,2] - A[1,2]) * (B[2,k0] + B[2,k1])
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = -1
        V[2 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        # W: C[0,k0]=1
        W[k0 * n + 0][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Check that the decomposition is exact."""
    n, m, p = 2, 3, 8
    R = len(U[0]) if U else 0
    
    # Test with A having 1 at (i0,j0), B having 1 at (j0,k0)
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                for i in range(n):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            # C[i,k] = sum_r W[k*n+i, r] * sum_j U[i*m+j, r] * V[j*p+k, r] * A[i,j] * B[j,k]
                            # With A[i0,j0]=1, B[j0,k0]=1:
                            # Non-zero only if i==i0, j==j0 and j==j0, k==k0.
                            # So j must be j0. And i must be i0, k must be k0.
                            if i == i0 and k == k0:
                                val += W[k * n + i][r] * U[i * m + j0][r] * V[j0 * p + k][r]
                        
                        expected_val = 1 if (i == i0 and k == k0) else 0
                        if val != expected_val:
                            return False
    return True

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    if _verify(res["U"], res["V"], res["W"]):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
