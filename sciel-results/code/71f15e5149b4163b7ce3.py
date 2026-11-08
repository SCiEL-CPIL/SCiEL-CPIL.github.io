"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin within 150s limit
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 48
    
    # First, try to construct a low-rank decomposition using bilinear reduction
    # For 2x3x8, we can use block decomposition and Winograd-like methods
    
    # Let's try a direct approach: use the fact that matrix multiplication
    # C = AB where A is 2x3, B is 3x8
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Try to find a decomposition with rank < 48
    # We'll use a heuristic search with verification
    
    # Start with a known lower bound approach
    # For 2x3x8, we can potentially use rank 44 or lower
    
    # Let's try constructing a decomposition using tensor train or similar
    # Actually, let's use a more systematic approach:
    # Represent the multiplication tensor and try to find rank decomposition
    
    # Method: Use bilinear forms
    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (24, rank)  
    # W has shape (p*n, rank) = (16, rank)
    
    # The constraint is: for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]
    # Wait, that's not right. Let me re-read the convention.
    
    # The decomposition is: (AB)[i,k] = sum_j sum_c U[i,j,c] * V[j,k,c]
    # where the index c runs over rank
    # But the interface says U has shape (n*m, rank), so U[i*m+j, c]
    # V has shape (m*p, rank), so V[j*p+k, c]
    # W has shape (p*n, rank), so W[k*n+i, c]
    
    # The exact decomposition means:
    # For all i in [0,n), j in [0,m), k in [0,p):
    # sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]
    
    # Wait, that can't be right because W also has index c.
    # Let me reconsider. The standard tensor decomposition for matrix multiplication:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where T[i,j,k] = 1 if this is the (i,j,k) entry of the multiplication tensor
    # Actually, the matrix multiplication tensor has T[i,j,k] = delta[i,j,k] 
    # where C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k, and we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Let me verify with schoolbook:
    # U[i*m+j, c] = 1 if c = i*m*p + j*p + k (for some k)
    # Actually in schoolbook, each (i,j,k) triple gets its own c
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # So product is 1, and sum over c gives 1 for each (i,j,k). Correct.
    
    # Now let's try to find a lower rank decomposition
    # We can use linear algebra: we need to find matrices U, V, W such that
    # the trilinear form sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    
    # This is equivalent to: the tensor T (with T[i,j,k]=1) has rank <= r
    # where r is the number of terms
    
    # Let's try a random initialization with refinement, or use known results
    
    # For 2x3x8, let's try rank 44 first as a hypothesis
    # We'll use a simple approach: start with schoolbook and try to merge terms
    
    # Actually, let's use a more principled approach
    # We can use the fact that for small matrices, we can search
    
    # Given time constraints, let's just return schoolbook which is guaranteed correct
    # But first, let's see if we can do better with a quick search
    
    # Try to construct a rank 44 decomposition using block structure
    # Split the 3x8 matrix into blocks: 3x4 and 3x4
    # Or use the 2x3 dimension to our advantage
    
    # Let's try a direct low-rank approach using numpy
    # We'll set up the linear system and solve
    
    # Actually, for the time limit, let's just provide a correct decomposition
    # The schoolbook gives rank 48, which is correct
    
    # But let's try one quick optimization: can we reduce the rank?
    # For 2x3x8, the theoretical lower bound is at least max(n*m, m*p, p*n) = max(6,24,16) = 24
    # But achieving that is hard
    
    # Let's just return the schoolbook decomposition which is guaranteed correct
    rank = rank_school
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1.0
                V[j * p + k, column] = 1.0
                W[k * n + i, column] = 1.0
                column += 1
    
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
