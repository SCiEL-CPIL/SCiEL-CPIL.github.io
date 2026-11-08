"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return (U, V, W) for the trivial schoolbook decomposition, rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col_idx] = 1.0
                V[j * p + k, col_idx] = 1.0
                W[k * n + i, col_idx] = 1.0
                col_idx += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-8):
    """
    Verify that the tensor decomposition is exact.
    The multiplication tensor T[i,j,k] = 1 if and only if the rank-1 terms
    reconstruct it. We check: sum_r U[i,j,r]*V[j,k,r] == W[k,i,r] contracted.
    
    Actually, the tensor decomposition means:
    T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    should equal 1 for all (i,j,k) in the multiplication tensor.
    
    Wait, that's not right. The matrix multiplication tensor is:
    T[i, j, k] = 1 where i indexes row of A, j indexes col of A / row of B, k indexes col of B.
    The product C[i,k] = sum_j A[i,j] * B[j,k].
    
    The tensor decomposition says:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    For matrix multiplication, T[i,j,k] is not a fixed tensor - it's the bilinear map.
    The correct check is: for the tensor M[i,j,k] = e_i^T A e_j * e_k^T B ... 
    
    Let me think again. The matrix multiplication tensor is defined as:
    mu = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    
    where e_{i,j} is the (n*m)-dim vector with 1 at position i*m+j,
    e_{j,k} is the (m*p)-dim vector with 1 at position j*p+k,
    e_{i,k} is the (p*n)-dim vector with 1 at position k*n+i.
    
    So the decomposition requires:
    sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = mu
    
    Which means for each (i,j,k):
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    And for all other index combinations not in the support, it should be 0.
    But actually, the tensor mu only has non-zero entries at (i,j,k) for valid indices,
    and those are all 1. The "other" entries don't exist in the tensor - the tensor
    is only defined on the grid [n*m] x [m*p] x [p*n], and mu[a,b,c] = 1 iff
    a = i*m+j, b = j*p+k, c = k*n+i for some valid i,j,k.
    
    So we need: for all i in [0,n), j in [0,m), k in [0,p):
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_optimization(n, m, p, rank, deadline, n_restarts=5, max_iter=500):
    """
    Try to find a rank-`rank` decomposition using numerical optimization.
    Uses the fact that we need sum_r U[:,r]*V[:,r]*W[:,r] = mu (the mult tensor).
    
    We parameterize and use gradient-based or random restart approaches.
    """
    target = n * m * p
    
    # The tensor mu has exactly target non-zero entries (all 1s)
    # We need: for all (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # Strategy: use the structure. For small ranks, try to find solutions.
    # One approach: fix U and V, solve for W linearly.
    # Given U and V, for each (i,k), we need:
    #   sum_r W[k*n+i, r] * (sum_j U[i*m+j,r]*V[j*p+k,r] ... )
    # Wait, that's not separable that way.
    
    # Better: The condition is that the tensor T defined by
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # equals mu.
    
    # For fixed U and V, the condition on W is:
    # For each (i,k): sum_r W[k*n+i, r] * S[i,k,r] = 1
    # where S[i,k,r] = sum_j U[i*m+j, r] * V[j*p+k, r]
    
    # So we have a linear system for W (or rather, for the rows of W indexed by (k,i)).
    
    # Let's build this system.
    # Number of equations: n * p (one for each (i,k) pair)
    # Unknowns: p * n * rank (entries of W)
    # But W is structured: W has shape (p*n, rank), so p*n*rank unknowns.
    # Equations: for each (i,k), sum_r W[k*n+i, r] * S[i,k,r] = 1
    # That's n*p equations with p*n*rank unknowns.
    # If rank >= 1, we have more unknowns than equations, so solutions exist if consistent.
    
    # Actually, we can just set W[k*n+i, r] to satisfy the equations.
    # For each (i,k), we need a vector w_{ik} in R^rank such that w_{ik} . S_{ik} = 1.
    # If S_{ik} is not the zero vector, we can pick w_{ik} = S_{ik} / ||S_{ik}||^2.
    # If S_{ik} is zero, we have a problem.
    
    # So the strategy is:
    # 1. Randomly initialize U and V
    # 2. For each (i,k), compute S_{ik,r} = sum_j U[i*m+j,r]*V[j*p+k,r]
    # 3. If all S_{ik} are non-zero, set W[k*n+i,r] = S_{ik,r} / sum(S_{ik,r}^2)
    # 4. Verify the full decomposition
    
    # Let's try this approach with random restarts.
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        U = rng.standard_normal((n * m, rank))
        V = rng.standard_normal((m * p, rank))
        
        # Compute S[i,k,r] = sum_j U[i*m+j,r] * V[j*p+k,r]
        S = np.zeros((n, p, rank))
        for i in range(n):
            for k in range(p):
                for r in range(rank):
                    s_val = 0.0
                    for j in range(m):
                        s_val += U[i*m+j, r] * V[j*p+k, r]
                    S[i, k, r] = s_val
        
        # Check if any S[i,k] is close to zero
        S_norms = np.sqrt(np.sum(S**2, axis=2))  # shape (n, p)
        min_norm = np.min(S_norms)
        
        if min_norm < 1e-6:
            continue
        
        # Set W
        W = np.zeros((p * n, rank))
        for i in range(n):
            for k in range(p):
                s_vec = S[i, k, :]
                norm_sq = np.dot(s_vec, s_vec)
                W[k*n+i, :] = s_vec / norm_sq
        
        # Verify
        if _verify(U, V, W, n, m, p, tol=1e-8):
            return U, V, W
    
    return None

def _construct_via_block_structure(n, m, p, deadline):
    """
    Try to exploit block structure. For 2x3 * 3x5:
    
    A = [[a11, a12, a13],
         [a21, a22, a23]]
    B = [[b11, b12, b13, b14, b15],
         [b21, b22, b23, b24, b25],
         [b31, b32, b33, b34, b35]]
    
    C[0,0] = a11*b11 + a12*b21 + a13*b31
    C[0,1] = a11*b12 + a12*b22 + a13*b32
    ...
    
    Total: 2*3*5 = 30 scalar multiplications in schoolbook.
    
    Known lower bounds: For m x n * n x p, the rank is at least max(m*n, n*p, m*p) 
    ... actually that's not a standard bound. The Strassen-style lower bound for 2x2x2 is 7.
    
    For 2x3 * 3x5, let's think about what's achievable.
    
    One approach: treat it as a sum of smaller matrix multiplications.
    2x3 * 3x5 can be seen as:
    [A1 A2] * [B1; B2] where A1, A2 are 2x2 and 2x1, B1, B2 are 2x5 and 1x5.
    C = A1*B1 + A2*B2
    
    A1*B1: 2x2 * 2x5, rank at most 2*2*5=20, but maybe less.
    A2*B2: 2x1 * 1x5, rank = 2*5 = 10 (outer product, rank 1 per column... actually it's just 2*5=10 scalar mults, but as tensor rank it's 1? No.)
    
    Actually A2*B2 where A2 is 2x1 and B2 is 1x5:
    C[i,k] = A2[i,0] * B2[0,k]
    This is an outer product: C = A2 * B2 (2x1 times 1x5 = 2x5).
    The tensor for this has rank 1! Because it's a single rank-1 term:
    U[i,0] = A2[i,0], V[0,k] = B2[0,k], W[k,i] = 1.
    Wait, let me check: sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r]
    For the 2x1 * 1x5 part, the "matrix multiplication" is:
    C[i,k] = sum_j A[i,j]*B[j,k] where j only ranges over 1 value (j=0).
    So C[i,k] = A[i,0]*B[0,k].
    The tensor is: sum_{i,k} e_{i,0} ⊗ e_{0,k} ⊗ e_{i,k}
    This has rank 1: U[i,0,0]=A[i,0], V[0,k,0]=B[0,k], W[i,k,0]=1.
    Check: U[i*m+0, 0]*V[0*p+k, 0]*W[k*n+i, 0] = A[i,0]*B[0,k]*1 = C[i,k]. Yes!
    
    So the 2x1 * 1x5 part has tensor rank 1.
    
    Now A1*B1: 2x2 * 2x5.
    C[i,k] = A1[i,0]*B1[0,k] + A1[i,1]*B1[1,k]
    This is a 2x2 * 2x5 multiplication.
    
    For 2x2 * 2x2, Strassen gives rank 7.
    For 2x2 * 2x5, we can split B1 into [B1a, B1b] where B1a is 2x2 and B1b is 2x3.
    C = A1*B1a + A1*B1b
    A1*B1a: 2x2 * 2x2, rank 7 (Strassen)
    A1*B1b: 2x2 * 2x3
    
    For 2x2 * 2x3:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for k=0,1,2
    This is like two 2x2 * 2x1 multiplications plus one 2x2 * 2x1.
    Each 2x2 * 2x1: C[i] = A[i,0]*B[0] + A[i,1]*B[1], which is 2x2 * 2x1.
    Tensor rank of 2x2 * 2x1: 
    C[i] = A[i,0]*B[0] + A[i,1]*B[1]
    Terms: (i=0,j=0,k=0), (i=0,j=1,k=0), (i=1,j=0,k=0), (i=1,j=1,k=0)
    That's 4 terms, rank 4 (each is a separate rank-1 term since k is always 0).
    Actually, can we do better? 
    U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    Here n=2, m=2, p=1.
    U is 4x1, V is 2x1, W is 2x1 (p*n = 1*2 = 2).
    Wait, p=1, n=2, so W has shape (1*2, rank) = (2, rank).
    
    We need: for i in {0,1}, j in {0,1}, k in {0}:
    sum_r U[i*2+j, r] * V[j*1+0, r] * W[0*2+i, r] = 1
    
    That's 4 equations. With rank 4, we can trivially solve. Can we do rank < 4?
    The 4 equations are:
    U[0,r]*V[0,r]*W[0,r] summed over r = 1  (i=0,j=0)
    U[1,r]*V[1,r]*W[0,r] summed over r = 1  (i=0,j=1)
    U[2,r]*V[0,r]*W[1,r] summed over r = 1  (i=1,j=0)
    U[3,r]*V[1,r]*W[1,r] summed over r = 1  (i=1,j=1)
    
    With rank 1: U[0]*V[0]*W[0]=1, U[1]*V[1]*W[0]=1, U[2]*V[0]*W[1]=1, U[3]*V[1]*W[1]=1
    From first two: U[0]*V[0] = U[1]*V[1] (both = 1/W[0])
    From last two: U[2]*V[0] = U[3]*V[1] (both = 1/W[1])
    This is consistent! Set W[0]=W[1]=1, U[0]=U[1]=a, V[0]=V[1]=1/a, U[2]=U[3]=b, V[0]=V[1]=1/b.
    Wait, V[0] appears in both first and third. V[0] = 1/a and V[0] = 1/b, so a=b.
    Then U[0]=U[1]=U[2]=U[3]=a, V[0]=V[1]=1/a, W[0]=W[1]=1.
    Check: U[0]*V[0]*W[0] = a*(1/a)*1 = 1. ✓
    U[1]*V[1]*W[0] = a*(1/a)*1 = 1. ✓
    U[2]*V[0]*W[1] = a*(1/a)*1 = 1. ✓
    U[3]*V[1]*W[1] = a*(1/a)*1 = 1. ✓
    So 2x2 * 2x1 has rank 1! That's surprising but correct because p=1.
    
    Wait, that can't be right in general. Let me re-examine.
    For n=2, m=2, p=1:
    The multiplication is: C[i] = sum_j A[i,j] * B[j] where B is a 2x1 matrix (column vector).
    So C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    
    The tensor mu has entries:
    mu[i*m+j, j*p+k, k*n+i] = mu[i*2+j, j, i] for i,j in {0,1}, k=0.
    So: mu[0,0,0]=1, mu[1,1,0]=1, mu[2,0,1]=1, mu[3,1,1]=1.
    
    With rank 1: U[0,0]*V[0,0]*W[0,0]=1, U[1,0]*V[1,0]*W[0,0]=1, U[2,0]*V[0,0]*W[1,0]=1, U[3,0]*V[1,0]*W[1,0]=1.
    
    Let U[0,0]=U[1,0]=U[2,0]=U[3,0]=1, V[0,0]=V[1,0]=1, W[0,0]=W[1,0]=1.
    Then all four products are 1. Yes! Rank 1 works.
    
    This makes sense because when p=1, the "matrix" B is just a vector, and the multiplication is just a matrix-vector product, which has a very simple tensor structure.
    
    OK so let me reconsider. For 2x2 * 2x3:
    Split into 2x2 * 2x1 (rank 1) + 2x2 * 2x1 (rank 1) + 2x2 * 2x1 (rank 1) = rank 3?
    But wait, these three sub-problems share the same A matrix. The tensor decomposition needs to be for the full problem, not a sum of independent decompositions.
    
    Actually, for the full 2x2 * 2x3 problem:
    n=2, m=2, p=3.
    U is 4x3, V is 6x3, W is 6x3.
    
    We need: for i in {0,1}, j in {0,1}, k in {0,1,2}:
    sum_r U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r] = 1
    
    That's 12 equations. Schoolbook rank is 12. Can we do better?
    
    Let me try rank 6 or so.
    
    Actually, let me just try the numerical approach with the linear solve for W.
    """
    pass

def construct():
    """
    Constructs the best known tensor decomposition for matrix multiplication.
    Dimensions: (2, 3, 5)
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Leave 10s safety margin
    
    # Start with schoolbook (rank 30)
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find lower rank decompositions
    # The theoretical minimum rank for 2x3 * 3x5 is unknown but >= some bound.
    # Let's try ranks from 15 up to 29.
    
    # First, try the linear-solve approach for various ranks
    for rank in range(15, best_rank):
        if time.time() > deadline - 5.0:
            break
        
        result = _try_rank_optimization(n, m, p, rank, deadline, n_restarts=20, max_iter=100)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break  # Found a lower rank, stop
    
    return best_result
# EVOLVE-BLOCK-END
