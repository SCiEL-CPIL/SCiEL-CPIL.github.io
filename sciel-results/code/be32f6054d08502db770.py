"""Matrix multiplication 2x3x10 - rank-minimal tensor decomposition search."""
import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly computes matrix multiplication.
    U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    Convention: W[k*n+i, r]
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 iff i,j,k form valid product
    Actually the tensor is: T[i,j,k] = 1 (it's the matrix multiplication tensor)
    We need: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    Wait - the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k in the standard MM tensor.
    Actually no. The MM tensor for (n x m) * (m x p) -> (n x p) has:
    T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    And the product C[i,k] = sum_j A[i,j] * B[j,k]
    The tensor decomposition: C[i,k] = sum_r (sum_j A[i,j]*U[i*m+j,r]) * (sum_k B[j,k]*V[j*p+k,r])... 
    
    Let me re-read the convention. The output factor W[k*n+i, r] means W is indexed by (k, i) -> (k*n+i).
    
    The tensor for MM_{n,m,p} is: T[i, j, k] = 1 for all valid i,j,k (it's the structure tensor)
    A rank-r decomposition means: T[i,j,k] = sum_{r=1}^{rank} U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    Wait, that's not right either. Let me think more carefully.
    
    Matrix multiplication: C = A * B where A is n x m, B is m x p, C is n x p.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T is in (R^n x R^m)^* ⊗ (R^m x R^p)^* ⊗ (R^n x R^p)
    Actually, the structure tensor has entries T[i, j, k] where:
    - i indexes rows of A (and C): 0..n-1
    - j indexes columns of A / rows of B: 0..m-1  
    - k indexes columns of B (and C): 0..p-1
    
    And T[i,j,k] = 1 (the MM tensor is all ones in the standard basis).
    
    A rank-r decomposition: T[i,j,k] = sum_{r=1}^{rank} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    So we need: for all i in [0,n), j in [0,m), k in [0,p):
    sum_{r=1}^{rank} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if total != 1:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, rank):
    """Fast verification using numpy for large ranks."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Build the tensor check
    # T[i,j,k] should equal sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # This is a bit tricky. Let's do it element-wise.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(U3[i,j,:] * V3[j,k,:] * W3[k,i,:])
                if abs(total - 1.0) > 1e-10:
                    return False
    return True

def _try_random_decomposition(n, m, p, rank, num_trials=100, seed=42):
    """Try to find a rank-r decomposition using random initialization."""
    rng = np.random.RandomState(seed)
    
    for trial in range(num_trials):
        # Initialize with random values
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
        
        # Scale to get closer
        # The product U*V*W should sum to 1 for each (i,j,k)
        # Average magnitude per term is ~1, so with rank terms, sum ~ rank * (1/sqrt(rank))^3 = 1/sqrt(rank)
        # We want sum = 1, so scale each by rank^(1/3)
        scale = rank ** (1.0/3.0)
        U *= scale
        V *= scale
        W *= scale
        
        # Simple iterative refinement: adjust one factor at a time
        # This is a very basic approach - just check if it's close
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        U3 = U_arr.reshape(n, m, rank)
        V3 = V_arr.reshape(m, p, rank)
        W3 = W_arr.reshape(p, n, rank)
        
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U3[i,j,:] * V3[j,k,:] * W3[k,i,:])
                    err = abs(total - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err < 1e-8:
            # Found a solution! Convert to fractions
            U_frac = [[Fraction(U[i][r]).limit_denominator(10**6) for r in range(rank)] for i in range(n*m)]
            V_frac = [[Fraction(V[i][r]).limit_denominator(10**6) for r in range(rank)] for i in range(m*p)]
            W_frac = [[Fraction(W[i][r]).limit_denominator(10**6) for r in range(rank)] for i in range(p*n)]
            
            # Verify exactly
            if _verify(U_frac, V_frac, W_frac, n, m, p, rank):
                return U_frac, V_frac, W_frac
    
    return None

def _strassen_based_decomposition(n, m, p):
    """
    Try to construct a decomposition using Strassen-like ideas.
    
    For 2x2x2, Strassen gives rank 7 (vs 8 schoolbook).
    For 2x3x10, we can try to exploit structure.
    
    One approach: decompose the m=3 dimension.
    Think of the 3x10 matrix as three 1x10 rows.
    The 2x3 matrix has 2 rows, each of length 3.
    
    Actually, let me think of this differently.
    
    C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    For each i (0,1) and k (0..9):
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    Schoolbook: 2*3*10 = 60 terms.
    
    Can we do better? 
    
    One idea: group the B matrix. B is 3x10. We can write:
    B[0,k], B[1,k], B[2,k] for each k.
    
    For a fixed k, C[:,k] = A * B[:,k] where B[:,k] is a 3-vector.
    This is a 2x3 times 3x1 multiplication, which needs at most 2*3=6 scalar mults per k.
    Total: 6*10 = 60. Same as schoolbook.
    
    But wait, we can share across k values if we structure it right.
    
    Alternative: Think of it as (2x3) * (3x10).
    Let A' = A (2x3), B' = B (3x10).
    
    We can write B' = [B0, B1, B2] where Bj is 3x10... no that's not helpful.
    
    Let me try a different grouping. Write the 3x10 matrix as a sum of rank-1 matrices:
    B = sum_{t=1}^{3} u_t * v_t^T where u_t in R^3, v_t in R^10.
    Then C = A * B = sum_t (A * u_t) * v_t^T
    Each (A*u_t) is a 2-vector, and v_t^T is 1x10.
    So C[i,k] = sum_t (A*u_t)[i] * v_t[k]
    
    This requires 3 * 2 * 10 = 60 scalar multiplications if done naively,
    but the "multiplications" are (A*u_t)[i] * v_t[k].
    
    Hmm, that's still 60.
    
    Let me think about the tensor rank lower bound.
    The slice rank of MM_{n,m,p} is at least max(n*m, m*p, n*p) = max(6, 30, 20) = 30.
    The rank is at least the slice rank, so rank >= 30.
    
    Actually, I recall that for MM_{2,3,10}, there might be known results.
    Let me try a more systematic search.
    
    For now, let me try ranks from 30 upward and see if we can find a valid decomposition.
    """
    pass

def _construct_decomposition(n, m, p, rank):
    """
    Try to construct a rank-r decomposition for MM_{n,m,p}.
    
    Strategy: Use a structured approach.
    
    For MM_{2,3,10}:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    Idea: Use the fact that we can combine terms.
    
    Let's try rank = 30 first (the slice rank lower bound).
    
    Actually, let me try a constructive approach.
    
    For the 2x3 part: A is 2x3. We can write A as a sum of 2 rank-1 matrices:
    A = a1 * x1^T + a2 * x2^T where a1,a2 in R^3, x1,x2 in R^2.
    Wait, that's not quite right for arbitrary A.
    
    Actually, any 2x3 matrix A can be written as:
    A = [A[0,0] A[0,1] A[0,2]]
        [A[1,0] A[1,1] A[1,2]]
    
    Let me try a different approach. Consider the tensor as:
    T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,...,9}
    
    We want T[i,j,k] = sum_{r=1}^{rank} U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    Note the indices: U is indexed by (i,j), V by (j,k), W by (k,i).
    This forms a cycle: i->j->k->i.
    
    For a cyclic tensor like this, we can use the "cyclic" decomposition trick.
    
    Let me try: for each r, assign U[i,j,r], V[j,k,r], W[k,i,r] such that
    the product U[i,j,r]*V[j,k,r]*W[k,i,r] is "localized" to specific (i,j,k) triples.
    
    This is essentially what the schoolbook does (rank 60, one term per (i,j,k)).
    
    To reduce rank, we need some terms to cover multiple (i,j,k) triples simultaneously.
    
    Key insight: If for some r, U[i,j,r] = 0 for all but a few j, and V[j,k,r] = 0 for all but a few k, and W[k,i,r] = 0 for all but a few i, then the term r only contributes to a small subset of (i,j,k).
    
    For the product to equal 1 for all (i,j,k), we need the sum over r to give 1.
    
    Let me try a structured approach:
    
    Group the k dimension. For each i (0,1), we have:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For fixed i, this is a 3x10 matrix-vector product (treating B^T as 10x3 and A[i,:] as 1x3).
    The 3x10 multiplication can be done in... well, it's just 30 scalar mults naively.
    
    But we can use the structure that both i=0 and i=1 share the same B.
    
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    For each k, we're computing two dot products of length 3.
    Total: 10 * 2 * 3 = 60 scalar mults.
    
    Can we do better by sharing? 
    Notice that B[0,k], B[1,k], B[2,k] appear in both C[0,k] and C[1,k].
    
    Define for each k:
    s0 = B[0,k], s1 = B[1,k], s2 = B[2,k]
    C[0,k] = A[0,0]*s0 + A[0,1]*s1 + A[0,2]*s2
    C[1,k] = A[1,0]*s0 + A[1,1]*s1 + A[1,2]*s2
    
    This is: [C[0,k], C[1,k]]^T = [[A[0,0],A[0,1],A[0,2]],[A[1,0],A[1,1],A[1,2]]] * [s0,s1,s2]^T = A * [s0,s1,s2]^T
    
    So for each k, we're doing a 2x3 matrix times 3-vector product.
    That's 6 scalar mults per k, total 60.
    
    The question is whether we can reduce the 2x3 * 3 -> 2 multiplication.
    For a general 2x3 times 3-vector, the minimum number of scalar multiplications is... 
    It's at least 2*3 = 6 by slice rank (max(2*3, 3*1, 2*1) = 6).
    And 6 is achievable (schoolbook). So no improvement there.
    
    Therefore, the total minimum is at least 10 * 6 = 60? No, that's not right because we're sharing the B values.
    
    Actually, the slice rank of the whole tensor MM_{2,3,10} is max(2*3, 3*10, 10*2) = max(6,30,20) = 30.
    So rank >= 30.
    
    Let me try to find a rank-30 decomposition or see if it's possible.
    
    Actually, I think the key is that the tensor has a special structure. Let me try a computational search.
    """
    
    # Try to find a valid decomposition at the given rank
    # Use a simple but effective search strategy
    
    deadline = time.time() + 100  # Leave 20s margin
    
    # Try structured decompositions first
    # Then random search
    
    # For now, return a valid (but possibly not minimal) decomposition
    # We'll search for better ones
    
    best_result = None
    
    # Try ranks from lower bound upward
    lower_bound = max(n*m, m*p, p*n)  # slice rank lower bound = 30
    
    for r in range(lower_bound, min(n*m*p, lower_bound + 35)):  # Try up to rank 60
        if time.time() > deadline:
            break
            
        # Try random search at this rank
        result = _try_random_decomposition(n, m, p, r, num_trials=50, seed=r*1000+42)
        if result is not None:
            U_frac, V_frac, W_frac = result
            best_result = (r, U_frac, V_frac, W_frac)
            break  # Found the lowest rank, stop
    
    if best_result is not None:
        r, U_frac, V_frac, W_frac = best_result
        return U_frac, V_frac, W_frac
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3x10 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 100  # 100s budget, leave 20s margin
    
    # Try to find a low-rank decomposition
    # Start from the slice rank lower bound and go up
    
    lower_bound = max(n*m, m*p, p*n)  # = max(6, 30, 20) = 30
    
    best_rank = n * m * p  # schoolbook rank
    best_U = None
    best_V = None
    best_W = None
    
    # Try structured/random searches at various ranks
    for r in range(lower_bound, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try random decomposition at rank r
        result = _try_random_decomposition(n, m, p, r, num_trials=30, seed=r*7919+42)
        if result is not None:
            U_frac, V_frac, W_frac = result
            # Double-check
            if _verify(U_frac, V_frac, W_frac, n, m, p, r):
                best_rank = r
                best_U = U_frac
                best_V = V_frac
                best_W = W_frac
                break
    
    # If no better found, use schoolbook
    if best_U is None:
        rank = n * m * p
        best_U = [[0] * rank for _ in range(n * m)]
        best_V = [[0] * rank for _ in range(m * p)]
        best_W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j][column] = 1
                    best_V[j * p + k][column] = 1
                    best_W[k * n + i][column] = 1
                    column += 1
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
