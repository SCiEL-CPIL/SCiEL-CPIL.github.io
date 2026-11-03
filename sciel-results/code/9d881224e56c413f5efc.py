"""Matrix multiplication 3,3,6 - low rank tensor decomposition"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the tensor."""
    rank = len(U[0])
    # Tensor T[i,j,k] = sum over c of U[i,j,c] * V[j,k,c] ... 
    # Actually for matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: T[i,j,k] where A[i,j] and B[j,k]
    # Decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c
    # where U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * ... 
    # Wait, let me re-read the convention.
    # W[k*n+i, r] - so W has p*n rows indexed by (k,i)
    # U[i*m+j, c] - U has n*m rows indexed by (i,j)  
    # V[j*p+k, c] - V has m*p rows indexed by (j,k)
    # The tensor equation should be: sum_c U[i*m+j,c] * V[j*p+k,c] = delta-related
    # Actually for matrix multiplication tensor:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The multiplication tensor has entries T[i,j,k] = 1 if... 
    # Standard: the tensor for matmul is sum_{i,j,k} x_{i,j} y_{j,k} z_{k,i}
    # So T[i,j,k] = 1 (all entries are 1 in the sense that each (i,j,k) term appears)
    # Decomposition: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k?
    # No! That's not right either.
    
    # Let me think again. The matrix multiplication bilinear map:
    # (A, B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
    # As a tensor in U^* ⊗ V^* ⊗ W^* (or similar), the multiplication tensor is:
    # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}  ... no
    # T = sum_{i,j,k} e_i ⊗ e_{j,k} ⊗ e_{j,i} ... 
    # 
    # Standard form: T = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_{i,j}^U ⊗ e_{j,k}^V ⊗ e_{k,i}^W
    # where e_{i,j}^U is the (i,j)-th basis vector of the n*m-dimensional space for A,
    # e_{j,k}^V is the (j,k)-th basis vector of the m*p-dimensional space for B,
    # e_{k,i}^W is the (k,i)-th basis vector of the p*n-dimensional space for C^T or similar.
    #
    # The decomposition T = sum_{c=1}^rank U_c ⊗ V_c ⊗ W_c means:
    # For all (i,j,k): T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # And T[i*m+j, j*p+k, k*n+i] = 1 if this is a valid term, 0 otherwise.
    #
    # Wait, that's not right. The tensor T has trilinear form:
    # T(a,b,c) = sum_{i,j,k} a_{i,j} * b_{j,k} * c_{k,i}
    # The tensor entries are T_{i*m+j, j*p+k, k*n+i} = 1 for all valid (i,j,k), 0 otherwise.
    #
    # A rank-1 decomposition is u ⊗ v ⊗ w where:
    # sum_{i,j,k} u_{i*m+j} v_{j*p+k} w_{k*n+i} = T(a,b,c) when a=u, b=v, c=w
    # 
    # So for a decomposition T = sum_c U_c ⊗ V_c ⊗ W_c:
    # T_{i*m+j, j*p+k, k*n+i} = sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
    # 
    # The left side is 1 if (i,j,k) is a valid index triple, 0 otherwise.
    # But actually every (i,j,k) with i in [0,n), j in [0,m), k in [0,p) is valid.
    # So T_{i*m+j, j*p+k, k*n+i} = 1 for all i,j,k.
    #
    # Hmm, but that would mean we need sum_c U_c[i*m+j] V_c[j*p+k] W_c[k*n+i] = 1 for ALL i,j,k.
    # That's 54 equations for the 3x3x6 case.
    
    # Let me verify with the schoolbook: U[i*m+j, i*m*p + j*p + k] = 1, etc.
    # Then sum_c ... = 1 when c corresponds to (i,j,k), giving 1. Correct.
    
    n, m, p = n, m, p
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1.0) > 1e-10:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    U_school, V_school, W_school, rank_school = schoolbook()
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    
    # Try to find a lower rank decomposition using Strassen-like methods
    # Approach: Use the 2x2 Strassen block decomposition
    
    # For 3x3x3, we can partition into 2x2 blocks + 1x1 terms
    # Let's try an explicit construction
    
    # Key idea: Use 7 Strassen multiplications for 2x2 blocks, handle boundaries
    # A 3x3 matrix A partitioned as:
    # A = [[a11, a12],      B 3x6 partitioned as:
    #      [a21, a22]]      [[b11, b12, b13, b14, b15, b16],
    # where a11 is 2x2, a12 is 2x1, a21 is 1x2, a22 is 1x1
    #      [b21, b22, b23, b24, b25, b26],
    #      [b31, b32, b33, b34, b35, b36]]
    # b11, b12, etc. are 2x6 and 1x6 blocks
    
    # This is getting complex. Let me try a different approach:
    # Directly search for a low-rank decomposition using random initialization and verification
    
    # Actually, let me try to construct a rank-45 decomposition using known results
    # M_{3,3,3} has rank 23. For M_{3,3,6}, we can potentially do better than 54.
    
    # Try: use 23-term scheme for the 3x3 part, adapt for 6 columns
    # If B is 3x6 = 3x(3+3), we can compute C1 = A*B1 (3x3) and C2 = A*B2 (3x3)
    # Each uses 23 multiplications, total 46.
    
    # But can we share some multiplications? Possibly.
    
    # Let me try to find the rank-23 decomposition for 3x3 matmul.
    # This is non-trivial. Let me use a heuristic approach instead.
    
    # Heuristic: random search for low-rank decompositions
    import random
    random.seed(42)
    
    def try_rank(r, num_trials=100):
        """Try to find a rank-r decomposition."""
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            # Initialize with random small values
            U = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(n*m)]
            V = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(m*p)]
            W = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(p*n)]
            
            if _verify(U, V, W, n, m, p):
                return U, V, W, r
        return None
    
    # Try ranks from low to high
    for target_rank in range(30, best_rank + 1, 5):
        if time.time() > deadline - 10:
            break
        result = try_rank(target_rank, num_trials=20)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
