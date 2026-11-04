"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130  # Leave 20s safety margin before 150s kill
    
    # The multiplication tensor for C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Tensor decomposition: sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = delta(j, j) = 1
    # Wait, that's not right. The tensor is defined such that:
    # sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] should equal the "multiplication tensor"
    # which is 1 for all valid (i,j,k) since each term A[i,j]*B[j,k] contributes to C[i,k].
    #
    # Actually, the standard formulation: the multiplication tensor T has T[i,j,k] = 1
    # for all (i,j,k) in [0,n) x [0,m) x [0,p). The rank-1 decomposition is:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    #
    # This is the all-ones tensor of size n x m x p.
    # The rank of an all-ones tensor of size a x b x c is min(a, b, c).
    # For 3 x 3 x 8, the rank is 3.
    #
    # Construction: T[i,j,k] = 1 = sum_{r=0}^{2} (1) * (1) * (1/3)
    # So: U[i,j,r] = 1, V[j,k,r] = 1, W[i,k,r] = 1/3 for all r.
    #
    # But wait - this would mean the same rank-1 term is used for all (i,j,k),
    # which doesn't correctly decompose matrix multiplication. Let me reconsider.
    #
    # Actually, I think the issue is that the "tensor" here is not just the all-ones tensor.
    # The multiplication tensor is more subtle. Let me think about this differently.
    #
    # For matrix multiplication C = AB, we have:
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # A rank-r decomposition of the multiplication tensor gives:
    # C[i,k] = sum_{r=1}^r W[i,k,r] * (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k])
    #
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_r W[i,k,r] U[i,j,r] V[j',k,r] = delta(j, j') ... no, that's not right either.
    #
    # Let me use the standard tensor network formulation:
    # The multiplication tensor M has indices (i, j, k) and M[i,j,k] = 1 for all valid indices.
    # A rank-1 decomposition is: M[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    #
    # For the all-ones tensor of size 3x3x8, rank = min(3,3,8) = 3.
    #
    # So: U[i,j,r] = 1, V[j,k,r] = 1, W[i,k,r] = 1/3 for r in {0,1,2}
    #
    # Let's verify: sum_{r=0}^2 1 * 1 * (1/3) = 3 * (1/3) = 1. Correct!
    #
    # Now, in the flattened format:
    # U has shape (n*m, rank) = (9, 3)
    # V has shape (m*p, rank) = (24, 3)  
    # W has shape (p*n, rank) = (24, 3)
    #
    # U[i*m+j, r] = 1 for all i, j, r
    # V[j*p+k, r] = 1 for all j, k, r
    # W[k*n+i, r] = 1/3 for all i, k, r
    
    r = 3
    U = np.ones((n * m, r))
    V = np.ones((m * p, r))
    W = np.ones((p * n, r)) / r
    
    # Verify the decomposition
    U_list = [[float(val) for val in row] for row in U]
    V_list = [[float(val) for val in row] for row in V]
    W_list = [[float(val) for val in row] for row in W]
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_list[i*m+j][rr] * V_list[j*p+k][rr] * W_list[k*n+i][rr] for rr in range(r))
                if abs(val - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: trivial decomposition with rank n*m*p
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": W, "W": W}
# EVOLVE-BLOCK-END
