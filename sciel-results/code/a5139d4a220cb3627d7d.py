"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 5
    deadline = time.time() + 140.0
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower-rank decomposition
    # For T[i,j,k] = 1, try rank R and see if we can solve
    # Using the structure: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try various ranks from low to high
    for R in range(15, 61):
        if time.time() > deadline - 10:
            break
        
        # Try to find a rank-R decomposition
        # Use a structured approach: assume W has a simple form
        # Try: W[k*n+i,c] = 1 for all c (constant)
        # Then we need: sum_c U[i*m+j,c] * V[j*p+k,c] = 1 for all (i,j,k)
        # This is: for each j, the matrix U_j (n x R) * V_j^T (R x p) = ones(n,p)
        # So U_j * V_j^T = J (all-ones matrix)
        # This requires rank(U_j) * rank(V_j) >= rank(J) = 1, and specifically
        # U_j * V_j^T = J means each row of U_j times each column of V_j^T = 1
        
        # Simple solution: U_j = ones(n, 1) * a, V_j = ones(p, 1) * b with a*b=1
        # But we need this for all j, and U, V are shared across j in the c dimension
        
        # Alternative: use the fact that J = 1_n * 1_p^T, so U_j = 1_n, V_j^T = 1_p^T works
        # with U[i*m+j, c] = 1 for c=0, V[j*p+k, 0] = 1 for all j,k
        # But then sum_c U[i*m+j,c]*V[j*p+k,c] = U[i*m+j,0]*V[j*p+k,0] = 1*1 = 1 only if R=1
        # and U[i*m+j,0]=1, V[j*p+k,0]=1 for all i,j,k
        # But W[k*n+i,0] must also be 1, so W[k*n+i,0]=1
        # Check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1*1*1 = 1 ✓
        
        # So rank 1 works?? Let me verify:
        # U[i*m+j, 0] = 1 for all i,j
        # V[j*p+k, 0] = 1 for all j,k  
        # W[k*n+i, 0] = 1 for all k,i
        # Then for any (i,j,k): U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1 ✓
        
        # This gives rank 1! But wait, is this correct for matrix multiplication?
        # The tensor for matrix multiplication is T[i,j,k] = 1, yes.
        # So rank 1 should work!
        
        # Let me double-check the convention:
        # U has shape (n*m, rank) = (12, 1)
        # V has shape (m*p, rank) = (20, 1)
        # W has shape (p*n, rank) = (15, 1)
        # U[i*m+j, 0] = 1
        # V[j*p+k, 0] = 1
        # W[k*n+i, 0] = 1
        # Product: sum_{c=0}^{0} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1*1*1 = 1 ✓
        
        U = [[1] for _ in range(n * m)]
        V = [[1] for _ in range(m * p)]
        W = [[1] for _ in range(p * n)]
        
        # Verify
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(1):
                        T[i, j, k] += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
        
        if np.allclose(T, np.ones((n, m, p))):
            return {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
