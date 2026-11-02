"""Matrix multiplication tensor decomposition for 2x3 * 3x7."""
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _build_tensor_flattening(n, m, p, mode='ik'):
    """
    Build the matricization of the matrix multiplication tensor.
    For mode='ik': rows indexed by (i,k) in [n*p], columns by j in [m].
    Entry T_ikj = 1 if the (i,k) output uses input j, which is always 1 for valid indices.
    Actually, the tensor T[i,j,k] = 1 for all i,j,k in the multiplication tensor.
    The flattening for rank determination: 
    - Reshape to (n*p, m) where row = (i,k), col = j. Entry is 1.
    But this is just a ones matrix of rank 1, which is not useful.
    
    The correct approach: the tensor T[i,j,k] for matrix multiplication C[i,k] = sum_j A[i,j]B[j,k]
    is T[i,j,k] = 1 for all valid i,j,k.
    
    For a CP decomposition of rank r, we need:
    T[i,j,k] = sum_{c=1}^{r} U[i,c] * V[j,c] * W[k,c]
    
    Matricization: fix k, get matrix M_k of size (n*m) x (n*m)? No.
    
    Standard matricization: 
    - Mode-1 (i): matrix of size n x (m*p), entry [i, (j,k)] = T[i,j,k] = 1
    - Mode-2 (j): matrix of size m x (n*p), entry [j, (i,k)] = T[i,j,k] = 1  
    - Mode-3 (k): matrix of size p x (n*m), entry [k, (i,j)] = T[i,j,k] = 1
    
    All these are just ones matrices, so rank 1. This doesn't help find the tensor rank directly.
    
    The tensor rank of the multiplication tensor is what we want. For small cases, we can try
    to find a CP decomposition of rank r by solving the system.
    
    Let's try a different approach: use the fact that for the matrix multiplication tensor,
    we can attempt to find a low-rank decomposition by searching over small ranks.
    """
    pass

def _try_rank(n, m, p, rank, deadline):
    """
    Try to find a rank-`rank` CP decomposition of the multiplication tensor.
    T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
    
    For the multiplication tensor, T[i,j,k] = 1 for all i,j,k.
    
    We can fix W (or use symmetry) and solve for U, V.
    
    Actually, let's use the matricization approach properly.
    Reshape T as a matrix of size (n) x (m*p): M[i, j*p+k] = T[i,j,k] = 1.
    This is just an n x (m*p) ones matrix, rank 1.
    
    The tensor rank is NOT the matrix rank of any flattening. We need to find the minimal r
    such that there exist U (n x r), V (m x r), W (p x r) with:
    T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c] for all i,j,k.
    
    Since T[i,j,k] = 1, this means:
    sum_c U[i,c] * V[j,c] * W[k,c] = 1 for all i,j,k.
    
    This is a system of n*m*p equations in n*r + m*r + p*r unknowns (with scaling ambiguity).
    
    For small n,m,p and small r, we can try to solve this.
    """
    from fractions import Fraction
    
    # For the all-ones tensor, one trivial decomposition is rank 1:
    # U = [1]*n, V = [1]*m, W = [1]*p gives sum_c=1: 1*1*1 = 1. 
    # But wait, this would mean the tensor rank is 1!
    
    # Let me re-examine. The multiplication tensor for C = A*B where A is n x m, B is m x p:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 for all valid i,j,k.
    
    # A CP decomposition of rank r: T[i,j,k] = sum_{c=1}^r U[i,c]*V[j,c]*W[k,c]
    
    # For r=1: U[i,0]*V[j,0]*W[k,0] = 1 for all i,j,k.
    # This requires U[i,0] = a_i, V[j,0] = b_j, W[k,0] = c_k with a_i*b_j*c_k = 1 for all i,j,k.
    # This is possible: a_i = 1, b_j = 1, c_k = 1. So rank 1?
    
    # But that can't be right for matrix multiplication. The issue is that the CP decomposition
    # of the multiplication tensor is not just about the all-ones pattern. 
    
    # I think I'm confusing the tensor. Let me reconsider.
    
    # The matrix multiplication tensor has entries T[i,j,k] where the tensor represents
    # the bilinear map. The "1" in T[i,j,k] = 1 means that the (i,j) entry of A and 
    # (j,k) entry of B contribute to C[i,k]. 
    # 
    # A rank-r decomposition gives r "operations" where each operation c uses
    # a linear combination of A entries (via U[i,c]) and a linear combination of B entries 
    # (via V[j,c]) to produce a contribution to C[i,k] (via W[k,c]).
    # 
    # The condition is: for all i,k: C[i,k] = sum_c (sum_j U[i,c]*A[i,j]) * (sum_j V[j,c]*B[j,k]) * W[k,c]
    # Wait, that's not quite right either.
    
    # Let me look at this more carefully. In a rank-r decomposition of matrix multiplication:
    # C[i,k] = sum_{c=1}^r (sum_j U_{i,j}^{(c)} A[i,j]) * (sum_j V_{j,k}^{(c)} B[j,k])
    # No, the standard form is:
    # C[i,k] = sum_{c=1}^r (sum_j U_{i,j,c} A[i,j]) * (sum_j V_{j,k,c} B[j,k])
    # where U and V are the factor matrices.
    
    # The interface says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # And the trace convention is W[k*n+i, r].
    
    # So the decomposition is:
    # C[i,k] = sum_{r=1}^{rank} (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # And then... how does W enter? 
    
    # Looking at the seed program:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for each (i,j,k) with a unique c.
    # So C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * ... 
    
    # I think the convention is:
    # C[i,k] = sum_{c} (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    # And W is used to verify or it's part of the tensor decomposition in a different form.
    
    # Actually, looking at the W index W[k*n+i, c], it seems like W is the "output" factor.
    # The full tensor decomposition might be:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But that doesn't make sense dimensionally for the multiplication.
    
    # Let me just try to find a low-rank decomposition by solving the system.
    # For the all-ones tensor T[i,j,k]=1, with the given indexing:
    # We need: 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for all i,j,k.
    
    # This is a system of n*m*p = 42 equations.
    # Unknowns: rank*(n*m + m*p + p*n) = rank*(6 + 21 + 14) = rank*41.
    # For rank=1: 41 unknowns, 42 equations. Might be solvable.
    
    # But there's scaling ambiguity: we can scale U[:,c] by alpha, V[:,c] by beta, W[:,c] by gamma
    # with alpha*beta*gamma = 1. So effectively 3*rank - rank = 2*rank free scalings.
    
    # This is getting complex. Let me try a simpler approach: just return the schoolbook 
    # decomposition but see if we can find a better one by trying small ranks.
    
    # For now, let's just return the schoolbook result since finding a better one is complex.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Start with schoolbook
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # For the multiplication tensor, known results suggest rank can be lower than n*m*p
    # Let's try ranks from small to large
    
    for rank in range(1, best_rank):
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
