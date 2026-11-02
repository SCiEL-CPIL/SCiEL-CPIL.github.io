"""Matrix multiplication 2x2x15: rank-minimal exact tensor decomposition."""
import time
import numpy as np
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Rank 53 decomposition: 7 Strassen blocks (14 cols) + 4 schoolbook (1 col)
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53
    
    # A indices in U: 0:A00, 1:A01, 2:A10, 3:A11
    # B indices in V: j*p + k
    # C indices in W: k*n + i
    
    def add_strassen_block(base_r, k0, k1):
        r1, r2, r3, r4, r5, r6, r7 = base_r, base_r+1, base_r+2, base_r+3, base_r+4, base_r+5, base_r+6
        
        # P1: (A00+A11)(B0k0+B1k1)
        U[0][r1] = 1; U[3][r1] = 1
        V[0*p+k0][r1] = 1; V[1*p+k1][r1] = 1
        # P2: (A10+A11)B0k0
        U[2][r2] = 1; U[3][r2] = 1
        V[0*p+k0][r2] = 1
        # P3: A00(B0k1-B1k1)
        U[0][r3] = 1
        V[0*p+k1][r3] = 1; V[1*p+k1][r3] = -1
        # P4: A11(B1k0-B0k0)
        U[3][r4] = 1
        V[1*p+k0][r4] = 1; V[0*p+k0][r4] = -1
        # P5: (A00+A01)B1k1
        U[0][r5] = 1; U[1][r5] = 1
        V[1*p+k1][r5] = 1
        # P6: (A10-A00)(B0k0+B0k1)
        U[2][r6] = 1; U[0][r6] = -1
        V[0*p+k0][r6] = 1; V[0*p+k1][r6] = 1
        # P7: (A01-A11)(B1k0+B1k1)
        U[1][r7] = 1; U[3][r7] = -1
        V[1*p+k0][r7] = 1; V[1*p+k1][r7] = 1
        
        # Reconstruction:
        # C0k0 = P1 + P4 - P5 + P7
        W[2*k0][r1] = 1; W[2*k0][r4] = 1; W[2*k0][r5] = -1; W[2*k0][r7] = 1
        # C1k0 = P2 + P4
        W[2*k0+1][r2] = 1; W[2*k0+1][r4] = 1
        # C0k1 = P3 + P5
        W[2*k1][r3] = 1; W[2*k1][r5] = 1
        # C1k1 = P1 - P2 + P3 + P6
        W[2*k1+1][r1] = 1; W[2*k1+1][r2] = -1; W[2*k1+1][r3] = 1; W[2*k1+1][r6] = 1
    
    # 7 Strassen blocks for column pairs
    for i in range(7):
        add_strassen_block(7*i, 2*i, 2*i+1)
    
    # Trivial block for column 14 (rank 4)
    base_r = 49
    U[0][base_r] = 1; V[0*p+14][base_r] = 1; W[28][base_r] = 1
    U[1][base_r+1] = 1; V[1*p+14][base_r+1] = 1; W[28][base_r+1] = 1
    U[2][base_r+2] = 1; V[0*p+14][base_r+2] = 1; W[29][base_r+2] = 1
    U[3][base_r+3] = 1; V[1*p+14][base_r+3] = 1; W[29][base_r+3] = 1
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: sum_r U[i,r] * V[j,r] * W[k,r] should equal 1 if (i,j,k) is in the tensor
    # Tensor T[i,j,k] = 1 if C[i,k] = A[i,:] @ B[:,k], i.e., sum_l A[i,l]*B[l,k]
    # So T[i,j,k] where i=(i1,i2), j=(l,k_col), k=(k_row,i2)... 
    # Actually: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # U index: i*n + m? No. Let me re-read.
    # U: shape (n*m, rank) -> index is (i, m_row) where i in [0,n), m_row in [0,m)
    #   Actually for A (n x m), U[i*m + j, r] corresponds to A[i,j]
    # V: shape (m*p, rank) -> index is (m_row, k) where m_row in [0,m), k in [0,p)
    #   V[j*p + k, r] corresponds to B[j,k]
    # W: shape (p*n, rank) -> index is (k, i) where k in [0,p), i in [0,n)
    #   W[k*n + i, r] corresponds to C[i,k]
    
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # Sample check with random values
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_test = A_test @ B_test
    
    # Compute via decomposition
    # P = U @ diag(...) @ V^T... actually:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # Wait, that's not right. The tensor decomposition means:
    # T_{a,b,c} = sum_r U[a,r] * V[b,r] * W[c,r]
    # where a indexes A entries, b indexes B entries, c indexes C entries
    # So: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k])
    # Hmm, that's not standard. Let me think again.
    # Standard: T_{(i,j),(j',k),(k',i')} = delta_{j,j'} * delta_{k,k'}
    # U indexed by (i,j), V by (j',k), W by (k',i')
    # T = sum_r U[i,j,r] * V[j',k,r] * W[k',i',r]
    # We need this to equal delta_{j,j'} * delta_{k,k'}
    
    # Let's just verify numerically
    # Build the full tensor check
    max_err = 0
    for i in range(n):
        for k in range(p):
            # C[i,k] should be sum_r W[k*n+i, r] * prod_a U[a,r]*A[a//m, a%m] * prod_b V[b,r]*B[b//p, b%p]
            # Actually simpler: for each r, compute the contribution
            # contrib_r = W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
            # Wait no. U[i*m+j, r] is the coefficient for A[i,j] in factor r.
            # V[j*p+k, r] is the coefficient for B[j,k] in factor r.
            # W[k*n+i, r] is the coefficient for C[i,k] in factor r.
            # The product for factor r is: W[k*n+i,r] * (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k])
            # Sum over r gives C[i,k]
            
            val = 0
            for r in range(rank):
                u_prod = 0
                for j in range(m):
                    u_prod += U[i*m+j][r] * A_test[i, j]
                v_prod = 0
                for j in range(m):
                    v_prod += V[j*p+k][r] * B_test[j, k]
                val += W[k*n+i][r] * u_prod * v_prod
            max_err = max(max_err, abs(val - C_test[i, k]))
    
    if max_err > 1e-10:
        # Decomposition is incorrect - but we know it should be correct
        # Fall back to trivial rank 60
        rank_trivial = n * m * p  # 60
        U_t = [[0] * rank_trivial for _ in range(n * m)]
        V_t = [[0] * rank_trivial for _ in range(m * p)]
        W_t = [[0] * rank_trivial for _ in range(p * n)]
        r_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_t[i*m+j][r_idx] = 1
                    V_t[j*p+k][r_idx] = 1
                    W_t[k*n+i][r_idx] = 1
                    r_idx += 1
        return {"U": U_t, "V": V_t, "W": W_t}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
