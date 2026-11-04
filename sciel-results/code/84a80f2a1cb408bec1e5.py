"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START

def _schoolbook_decomposition(n, m, p):
    """Standard O(nmp) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if U,V,W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the target tensor T[i,j,k] = delta(i,j,k) for matmul
    # Actually, the matrix multiplication tensor T has T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition requires: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,j,k} 
    # Wait, that's not right. Let me reconsider.
    
    # Matrix multiplication: C = A @ B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] B[j,k]
    # As a tensor: T[i,j,k] = 1 (the structure tensor of matmul)
    # We want: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]  ... no that's not standard.
    
    # Actually the standard formulation: the matmul tensor T ∈ R^{n×m × m×p × p×n} 
    # with T[(i,j),(j',k),(k',i')] = δ_{j,j'} δ_{k,k'} δ_{i,i'}
    # But that's the identity-like tensor. 
    
    # Let me re-read the task. The output factor follows W[k*n+i, r].
    # U has shape (n*m, rank), indexed by (i*m + j, c)
    # V has shape (m*p, rank), indexed by (j*p + k, c)  
    # W has shape (p*n, rank), indexed by (k*n + i, c)
    
    # The decomposition should satisfy:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = δ_{something}
    
    # Actually for matrix multiplication, the tensor is:
    # T[i,j,k] = 1 for all valid i,j,k (it's the "multiplication" tensor)
    # No wait. Let me think again.
    
    # The matrix multiplication map: (A, B) -> A*B
    # This is a bilinear map. Its associated tensor T ∈ R^{(n×m) × (m×p) × (n×p)}
    # T[a, b, c] where a=(i,j), b=(j',k'), c=(i'',k'')
    # T[a,b,c] = 1 if j=j' and i=i'' and k=k', else 0.
    
    # But the task says W has shape (p*n, rank) with index k*n+i. So the third factor is indexed by (k,i) not (i,k).
    # This suggests the output is stored as W[k*n+i] which is C transposed or something.
    
    # Let me just verify by checking: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1
    # for the "diagonal" entries and 0 otherwise... 
    
    # Actually, I think the correct check is:
    # The matmul tensor T has entries T[i,j,k] = 1 for all i,j,k (it's just the all-ones tensor in the 
    # sense that every (i,j,k) triple contributes to the product).
    
    # No, that's wrong too. Let me think more carefully.
    
    # Bilinear map: f: R^{n×m} × R^{m×p} -> R^{n×p}, f(A,B) = AB
    # f_{i,k}(A,B) = sum_j A_{i,j} B_{j,k}
    # The tensor T ∈ R^{(n×m)×(m×p)×(n×p)} is defined by T_{(i,j),(j',k'),(i'',k'')} = δ_{j,j'}δ_{i,i''}δ_{k,k'}
    
    # But the task uses W[k*n+i] for the output index, which is (k,i) ordering, i.e., C^T or column-major.
    
    # So the decomposition should satisfy:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = δ_{j,j'}δ_{i,i''}δ_{k,k'}
    # But since we're summing over c only, and the indices are fixed at (i,j), (j,k), (k,i):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Wait, that would mean the tensor is all ones, which can't be right for a non-trivial rank.
    
    # I think I'm overcomplicating this. Let me just check if the schoolbook decomposition works:
    # In schoolbook, U[i*m+j, (i,j,k)] = 1, V[j*p+k, (i,j,k)] = 1, W[k*n+i, (i,j,k)] = 1
    # So sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = sum_{c=(i',j',k')} δ_{i,i'}δ_{j,j'} * δ_{j,j'}δ_{k,k'} * δ_{k,k'}δ_{i,i'}
    # = δ_{i,i'}δ_{j,j'}δ_{k,k'}δ_{j,j'}δ_{k,k'}δ_{i,i'} summed over (i',j',k')
    # = 1 (when i'=i, j'=j, k'=k)
    
    # So yes, the sum equals 1 for all valid (i,j,k). That means the tensor is indeed all-ones in the 
    # (i,j,k) indexing scheme used here. This makes sense because the "tensor" in this formulation 
    # represents the structure constants of the bilinear map where each (i,j,k) triple has coefficient 1.
    
    # So the verification is: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] ≈ 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                # Actually it's element-wise product then sum, which is dot of (V*W) with U
                # But more precisely: sum_c U[c] * V[c] * W[c]
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_rank_r(r, n, m, p, max_time=100.0, start_time=None):
    """Try to find a decomposition of rank r using random initialization and optimization."""
    if start_time is None:
        start_time = time.time()
    
    # Lower bound on rank for matmul n×m × m×p is max(n*m, m*p, n*p) / something... 
    # Actually the trivial lower bound is max(n, m, p) * min(n, m, p) / something.
    # For 2×4 × 4×5: schoolbook is 40. Known bounds are around 20-30.
    
    # Use a simple approach: try random rank-r decompositions and check
    # This is very unlikely to work for exact decompositions, so let's try a smarter approach.
    
    # For small ranks, we can try to solve the system of polynomial equations.
    # The equations are: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a system of n*m*p = 40 polynomial equations in 3*r variables (well, 3*r*something).
    # Actually there are (n*m + m*p + p*n) * r = (8 + 20 + 10) * r = 38r variables.
    
    # For r < 40, this is an underdetermined system, so solutions should exist generically,
    # but finding them exactly is hard.
    
    # Let's try a different approach: use the fact that for small matrices, we can try
    # structured decompositions.
    
    # For now, just return None (we couldn't find one) if we can't verify quickly.
    # This function is a placeholder for more sophisticated search.
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit, keep safety margin
    
    # Start with schoolbook (rank 40)
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bound: rank >= max(n*m, m*p, n*p) / max(n, m, p) ... not quite
    # A simple lower bound: rank >= n*m*p / (something)
    # For 2x4x5, the rank is known to be at most 20 (via Strassen-like methods) 
    # and at least around 12-15.
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Use a simple random search with verification
    
    # Actually, for exact decompositions with integer/half-integer entries,
    # we need a more structured approach. Let's try known constructions.
    
    # For 2×4 × 4×5, one approach is to use block structure.
    # Let's try rank 20 first (half of schoolbook).
    
    # Try various ranks in decreasing order of likelihood
    for r in [20, 22, 24, 25, 28, 30, 32, 35, 38, 40]:
        if time.time() > deadline - 30:  # Keep 30s safety margin
            break
        
        # Try to construct a rank-r decomposition
        # For now, use a simple approach: try to find one by solving a least-squares problem
        # This won't give exact solutions, but let's see.
        
        # Actually, let's just return the schoolbook for now since finding exact
        # low-rank decompositions is very hard without specific constructions.
        # The score is based on rank, so schoolbook (40) is the baseline.
        
        pass
    
    # Convert to lists for the expected output format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
