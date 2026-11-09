<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by checking for linear dependencies
    # The decomposition is valid if:
    # For all i in [0,n), k in [0,p):
    #   sum_j A_{i,j} * B_{j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    # Wait, that's not right. Let me reconsider the convention.
    
    # The convention is: W[k*n + i, r]
    # So the decomposition is:
    # C_{i,k} = sum_r (sum_j U_{i*m + j, r} * A_{i,j}) * (sum_l V_{l*p + k, r} * B_{l,k})
    # But that's not standard. Let me re-read the task.
    
    # "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k*n + i, r) where k is column of B / row of output, i is row of A / row of output.
    
    # The standard tensor decomposition for matrix multiplication is:
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # = sum_r U_{i,j,r} * V_{j,k,r} * W_{i,k,r}  (in some convention)
    
    # But the interface says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # So U is indexed by (i*m + j, r), V by (j*p + k, r), W by (k*n + i, r).
    
    # The decomposition should satisfy:
    # sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r} = A_{i,j} * B_{j,k} for all i,j,k
    
    # This is the standard form. So we need:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if we're decomposing the tensor T_{i,j,k}=1
    
    # Actually, the tensor for matrix multiplication has T_{i,j,k} = 1 for all valid i,j,k, and the multiplication is:
    # C_{i,k} = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A_{i,j} * B_{j,k}
    # Wait no. Let me think again.
    
    # The bilinear form is: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # A rank-r decomposition is: C_{i,k} = sum_r (sum_j u_{i,j,r} A_{i,j}) (sum_l v_{l,k,r} B_{l,k})
    # where u and v are the factor matrices.
    
    # But the interface uses U, V, W with specific indexing. Let me assume:
    # U[i*m + j, r] = u_{i,j,r}
    # V[j*p + k, r] = v_{j,k,r}  
    # W[k*n + i, r] = w_{i,k,r}
    
    # And the decomposition is: sum_r u_{i,j,r} * v_{j,k,r} * w_{i,k,r} = 1 for all i,j,k
    # (This is the tensor decomposition where each term is a product of three factors)
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p equations in the unknowns U, V, W.
    
    # For the schoolbook, U[i*m+j, i*m*p + j*p + k] = 1, etc., which satisfies this.
    
    # To reduce rank, we need to find U, V, W with fewer columns that still satisfy these equations.
    
    # This is a hard problem. Let me try a simple heuristic: start with schoolbook and try to merge columns.
    
    # Given time constraints, let me just return schoolbook for now, but try a quick search.
    
    deadline = time.time() + 100  # Leave 20s margin
    
    # Try to find a lower rank by random projection
    # This is heuristic and may not work well, but let's try
    
    import random
    random.seed(42)
    
    # Try rank from 100 down to 144, see if we can find a valid decomposition
    # This is a very simplified search
    
    for rank in range(140, 143):  # Try a few ranks below 144
        if time.time() > deadline:
            break
        # Try to construct a decomposition of this rank
        # This is very heuristic
        pass
    
    # For now, return schoolbook
    # Convert to lists as expected
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check if the decomposition is valid."""
    rank = U.shape[1]
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    # Check: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # The tensor for matrix multiplication M_{n,m,p} has entries T[i,j,k] = 1 for all i,j,k
    # A rank-r decomposition requires:
    #   sum_{r=1}^{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k
    
    # This is the "1-tensor" of shape (n, m, p)
    # The tensor rank of the all-ones tensor of shape (n,m,p) is known to be:
    #   For n=m=p: rank = n^2 (Strassen-like results exist but for 2x2 it's 7, not 4)
    #   Actually, the all-ones tensor [1]^{n x m x p} has tensor rank = max(n, m, p) 
    #   Wait, that's not right either.
    
    # Actually, the all-ones tensor of shape (n, m, p) can be decomposed as:
    #   T[i,j,k] = 1 = sum_{r=1}^{R} u_{i,r} * v_{j,r} * w_{k,r}
    # This is possible with R = max(n, m, p) using:
    #   u_{i,r} = 1 if i <= r else 0 (or similar)
    #   Actually, we can use: u_{i,r} = 1 for all, v_{j,r} = 1 for all, w_{k,r} = 1/R? No, that gives 1/R * R = 1 only if we sum R terms.
    #   Simpler: use R terms where each term is 1 * 1 * (1/R)? No.
    
    # Let me think more carefully. We want:
    #   1 = sum_{r=1}^{R} u_{i,r} v_{j,r} w_{k,r}
    # for all i, j, k.
    
    # One simple construction: Let R = max(n, m, p). 
    # Define u_{i,r} = 1 if i < r else 0 (for i in 0..n-1, r in 1..R)
    # Define v_{j,r} = 1 if j < r else 0
    # Define w_{k,r} = 1 if k < r else 0
    # Then sum_{r=1}^{R} u_{i,r} v_{j,r} w_{k,r} = number of r in {1..R} such that r > i, r > j, r > k
    # = number of r in {max(i,j,k)+1, ..., R} = R - max(i,j,k)
    # This is not 1.
    
    # Alternative: Use the fact that the all-ones matrix of size (nm) x (mp) can be factored...
    # Actually, let's think of it differently. The tensor T[i,j,k] = 1 is the outer product of 
    # three all-ones vectors: T = ones(n) ⊗ ones(m) ⊗ ones(p)
    # The tensor rank of an outer product of vectors is 1!
    # So: T[i,j,k] = 1 * 1 * 1 = 1
    # This means rank 1 suffices: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    
    # Let me verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 * 1 * 1 = 1. Yes!
    
    # So the rank is 1!
    
    U1 = np.ones((n * m, 1), dtype=float)
    V1 = np.ones((m * p, 1), dtype=float)
    W1 = np.ones((p * n, 1), dtype=float)
    
    if _validate(U1, V1, W1, n, m, p):
        best_rank = 1
        best_U, best_V, best_W = U1.copy(), V1.copy(), W1.copy()
    else:
        # Fallback to schoolbook
        pass
    
    # Convert to lists as expected
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
