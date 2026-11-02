"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank as fallback
    rank_sb = n * m * p
    
    # Try to find a lower rank decomposition
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is equivalent to finding rank-r decomposition of the all-ones tensor
    # For all-ones tensor, rank = 1 suffices in terms of the tensor equation
    # BUT we need the decomposition to correctly compute matrix multiplication
    
    # Let me verify: if U[i*m+j,r]=1, V[j*p+k,r]=1, W[k*n+i,r]=1 for all,
    # then for a given (i,k): C[i,k] = sum_r sum_j U[i*m+j,r]*A[i,j] * V[j*p+k,r]*B[j,k] * W[k*n+i,r]
    # Wait, that's not how it works. The decomposition is:
    # C[i,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_k' V[j*p+k',r]*B[j,k']) * W[k*n+i,r]
    
    # Actually, I think the correct interpretation is:
    # The r-th term computes: (sum_j U[i*m+j,r]*A[i,j]) * (sum_k' V[j*p+k',r]*B[j,k']) * W[k*n+i,r]
    # and these are summed over r to give C[i,k]
    
    # But that doesn't make sense dimensionally. Let me reconsider.
    
    # The standard tensor decomposition for matrix multiplication:
    # C[i,k] = sum_r alpha_r * (sum_j u_r[i,j]*A[i,j]) * (sum_k' v_r[j,k']*B[j,k'])
    # where alpha_r is absorbed into the factors.
    
    # In the U,V,W formulation:
    # C[i,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_k' V[j*p+k',r]*B[j,k']) * W[k*n+i,r]
    
    # No wait, that's not right either. Let me think again.
    
    # The tensor T_M[i,j,k] has the property that:
    # C[i,k] = sum_j T_M[i,j,k] * A[i,j] * B[j,k]
    
    # The rank-r decomposition means:
    # T_M[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # So: C[i,k] = sum_j (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]) * A[i,j]*B[j,k]
    
    # For this to equal sum_j A[i,j]*B[j,k], we need:
    # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    
    # So yes, we need the all-ones tensor to have rank r.
    
    # The all-ones tensor of size n*m x m*p x p*n has rank 1!
    # U[i*m+j,0]=1, V[j*p+k,0]=1, W[k*n+i,0]=1 gives sum = 1 for all i,j,k.
    
    # But wait, that would mean rank 1 suffices, which is clearly wrong for matrix multiplication.
    
    # I think the issue is that the tensor T_M is NOT the all-ones tensor in the way I'm thinking.
    
    # Let me reconsider the tensor structure. The matrix multiplication tensor is:
    # T_M = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the indices are (i from A rows, j from A cols/B rows, k from B cols)
    # And T_M[i,j,k] = 1 for all valid i,j,k.
    
    # The rank of this tensor is what we want to minimize.
    
    # For the all-ones tensor of size a x b x c, the rank is min(a,b,c) if we're talking about 
    # the standard rank (minimum number of rank-1 tensors to sum to it).
    
    # Actually no. The all-ones tensor T[i,j,k]=1 can be written as:
    # T = sum_{r=1}^{min(a,b,c)} (1_{i>=r}) (1_{j>=r}) (1_{k>=r})? No.
    
    # T[i,j,k] = 1 = 1*1*1, so it's already rank 1!
    
    # I'm clearly misunderstanding something fundamental. Let me look at this differently.
    
    # Perhaps the constraint is not just T[i,j,k]=1, but that the decomposition must use
    # the specific indexing where U is indexed by (i,j) pairs, V by (j,k) pairs, etc.
    
    # In that case, the "tensor" is not the all-ones tensor in a simple sense, but rather
    # a tensor with a specific structure imposed by the matrix multiplication.
    
    # Let me just try to find a low-rank decomposition numerically.
    
    deadline = time.time() + 120
    
    # Try ranks from 1 up to schoolbook
    for rank in range(1, rank_sb + 1):
        if time.time() > deadline - 10:
            break
        if _try_rank(n, m, p, rank):
            U, V, W = _solve_decomposition(n, m, p, rank)
            if U is not None:
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(n, m, p, rank):
    """Try to determine if a rank-r decomposition exists."""
    # For now, just return False to skip to schoolbook
    # This is a placeholder for actual search
    return False

def _solve_decomposition(n, m, p, rank):
    """Solve for the decomposition matrices."""
    return None, None, None
# EVOLVE-BLOCK-END
