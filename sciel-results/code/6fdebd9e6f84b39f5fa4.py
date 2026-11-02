"""Program for matrix multiplication 3x5x5 tensor decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication.
    
    The matrix multiplication tensor T has entries T[i,j,k] where:
    - i is row index of A (0..n-1, n=3)
    - j is column index of A / row index of B (0..m-1, m=5)
    - k is column index of B (0..p-1, p=5)
    
    We seek factor matrices U (n*m x rank), V (m*p x rank), W (p*n x rank)
    such that for each output position (i, k):
    sum_r U[i*m + something] * V[...] * W[k*n + i, r] = T[i,j,k]
    
    Using the standard tensor decomposition formulation:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    But the interface specifies shapes (n*m, rank), (m*p, rank), (p*n, rank),
    with W following trace convention W[k*n+i, r].
    
    So the decomposition is:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    
    # Deadline guard: 150s total, use 120s for safety
    deadline = time.time() + 120
    
    # Start with schoolbook (rank = n*m*p = 75)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    
    # Verify schoolbook
    def verify(U_list, V_list, W_list, n=n, m=m, p=p):
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        W_arr = np.array(W_list, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    computed = np.dot(U_arr[i*m+j], np.multiply(V_arr[j*p+k], W_arr[k*n+i]))
                    if abs(computed - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find lower rank decompositions
    # The tensor T[i,j,k] = 1 for all (i,j,k) in the 3x5x5 grid
    # This is a rank-1 tensor! T = ones(3,5,5)
    # So the tensor rank is 1!
    
    # Wait - let me re-read the problem. The matrix multiplication tensor
    # for multiplying A (n x m) by B (m x p) gives C (n x p) where
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The bilinear algorithm uses:
    # C[i,k] = sum_r sum_j U[i,j,r] * A[i,j] * V[j,k,r] * B[j,k]
    # 
    # Actually the standard formulation is:
    # C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k])
    # 
    # No wait. The standard matrix multiplication tensor decomposition:
    # T_{ijk} = sum_r U_{ij,r} * V_{jk,r} * W_{ki,r}
    # where T_{ijk} is the coefficient of A_{ij} * B_{jk} in C_{ki}
    # 
    # For matrix multiplication, T_{ijk} = 1 if the indices are valid, 0 otherwise.
    # Specifically, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # So T is the all-ones tensor of shape (n, m, p) = (3, 5, 5).
    # The tensor rank of the all-ones tensor is 1!
    
    # Rank-1 decomposition:
    # T[i,j,k] = 1 = 1 * 1 * 1
    # So U[i,j,0] = 1, V[j,k,0] = 1, W[k,i,0] = 1
    
    # But wait - we need to be more careful. The tensor entries are:
    # T[i,j,k] where i indexes rows of A (0..n-1), j indexes cols of A / rows of B (0..m-1),
    # k indexes cols of B (0..p-1)
    # 
    # The product is C[i,k] = sum_j A[i,j] B[j,k]
    # So the coefficient of A[i,j] B[j,k] in C[i,k] is 1.
    # 
    # In the decomposition: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = T[i,j,k] = 1
    # 
    # With rank 1: U[i,j,0] V[j,k,0] W[k,i,0] = 1 for all i,j,k
    # Setting U[i,j,0]=1, V[j,k,0]=1, W[k,i,0]=1 works!
    
    # So rank 1 should work!
    rank_1 = 1
    U1 = np.ones((n * m, rank_1))
    V1 = np.ones((m * p, rank_1))
    W1 = np.ones((p * n, rank_1))
    
    if verify(U1.tolist(), V1.tolist(), W1.tolist()):
        return {"U": U1.tolist(), "V": V1.tolist(), "W": W1.tolist()}
    
    # If rank 1 doesn't verify (shouldn't happen), fall back to schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
