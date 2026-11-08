"""Search for a rank-minimal exact tensor decomposition of 2x3 * 3x4 matmul.
Output factor follows the official trace convention W[k*n+i, r].
"""
import time
import itertools
import random

DIMENSIONS = [2, 3, 4]
DEADLINE = 150.0  # evaluator kills at 150s; keep margin
START = time.time()
TIME_LIMIT = 135.0  # safety margin

# EVOLVE-BLOCK-START
def _time_left():
    return TIME_LIMIT - (time.time() - START)


def _verify(U, V, W, n, m, p):
    """Check exactness: for all (i,j,k), sum_r U[i*m+j, r]*V[j*p+k, r] == W[k*n+i, r] (i.e., ==1)
    and for all other (i,j,k) combos the sum is 0.
    Actually the tensor T[i,j,k] = 1 for the matmul. The decomposition says:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Wait - that's not right. Let me re-read the contract.
    """
    pass


def _verify_correct(U, V, W, n, m, p):
    """The matrix multiplication tensor T has T[i,j,k] = 1 (the product (AB)[i,k] = sum_j A[i,j]*B[j,k]).
    The rank-r decomposition: T[i,j,k] = sum_{r} U[i,j,r] * V[j,k,r] * W[i,k,r]
    where U has shape (n*m, r) indexed by (i*m+j), V has shape (m*p, r) indexed by (j*p+k),
    W has shape (p*n, r) indexed by (k*n+i).
    So we need: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True


def _verify_all_zero_except(U, V, W, n, m, p):
    """Actually the tensor IS the matmul tensor which is 1 everywhere in (i,j,k) space.
    Wait no. The matmul tensor T[i,j,k] = delta (it's 1 for all i,j,k because (AB)[i,k] = sum_j A[i,j]B[j,k]
    means the coefficient of A[i,j]*B[j,k] in (AB)[i,k] is 1).
    So T[i,j,k] = 1 for ALL i in [0,n), j in [0,m), k in [0,p).
    So we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True


def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def _try_reduce(U, V, W, n, m, p, max_rank):
    """Try to find a lower-rank decomposition by structured search.
    
    Strategy: The matmul tensor for n=2,m=3,p=4 has rank between 12 and 24.
    We'll try random structured decompositions with rank < 24.
    
    Approach: Use the fact that T[i,j,k]=1 for all (i,j,k). We need:
    sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    This is equivalent to: the 3-tensor of all-ones (size 2x3x4) has a rank-r decomposition.
    Actually wait - the all-ones tensor of size n x m x p has rank... 
    The all-ones tensor J_{n,m,p} has rank min(n,m,p) * something? No.
    Actually J_{n,m,p}[i,j,k] = 1 for all i,j,k. This can be written as:
    J[i,j,k] = sum_r u[i,r]*v[j,r]*w[k,r]
    The rank of the all-ones tensor is 1! Because J[i,j,k] = 1*1*1.
    
    But that's NOT the matmul tensor. The matmul tensor is different.
    Let me reconsider.
    
    The matrix multiplication tensor for (n x m) * (m x p) -> (n x p):
    T[i, j, k] = 1 if we're computing (AB)[i,k] = sum_j A[i,j] * B[j,k]
    The coefficient of A[i,j]*B[j,k] in (AB)[i,k] is exactly 1.
    So T[i,j,k] = 1 for all valid i,j,k.
    
    Hmm, so it IS the all-ones tensor? That seems too simple. Let me re-read.
    
    Actually I think I'm confusing myself. The matrix multiplication tensor is a 3-tensor where:
    - First mode: (i, j) pairs, size n*m
    - Second mode: (j, k) pairs, size m*p  
    - Third mode: (i, k) pairs, size n*p
    
    And T[(i,j), (j',k), (i',k')] = 1 if i=i' and j=j' and k=k', else 0.
    
    So T[(i,j),(j',k),(i',k')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    This is NOT the all-ones tensor. It's a structured tensor where entries are 1 only when the indices are consistent.
    
    So the condition is:
    sum_r U[i*m+j, r] * V[j'*p+k, r] * W[i'*n+k', r] = 1 if i=i', j=j', k=k' else 0
    
    Let me re-verify with the schoolbook: U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1 for col corresponding to (i,j,k).
    For a given (i,j),(j',k),(i',k'):
    sum over all (a,b,c) of U[i*m+j, (a,b,c)] * V[j'*p+k, (a,b,c)] * W[i'*n+k', (a,b,c)]
    = sum over (a,b,c) of [i==a and j==b] * [j'==b and k==c] * [i'==a and k'==c]
    = 1 if i==j'... wait no.
    
    U[i*m+j, (a,b,c)] = 1 iff i==a and j==b
    V[j'*p+k, (a,b,c)] = 1 iff j'==b and k==c
    W[i'*n+k', (a,b,c)] = 1 iff i'==a and k'==c
    
    So the product is 1 iff: i==a, j==b, j'==b, k==c, i'==a, k'==c
    = iff i==i', j==j', k==k'
    
    Great, so the schoolbook works.
    
    Now for a lower-rank decomposition, we need to find U, V, W with rank < 24 such that:
    sum_r U[i*m+j, r] * V[j'*p+k, r] * W[i'*n+k', r] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    """
    pass


def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 24)
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p  # 24
    
    # The matmul tensor T[(i,j),(j',k),(i',k')] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # Lower bound: max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # Actually the true rank of matmul for 2x3*3x4 is unknown but >= 12 and <= 24.
    
    # Strategy: Try to find decompositions with rank < 24 using structured search.
    # Key insight: We can try "block" decompositions or use known results.
    # For small matrices, we can do an exhaustive/semi-exhaustive search over integer/half-integer entries.
    
    # Approach 1: Try to merge schoolbook terms.
    # The schoolbook has 24 terms. Can we combine some?
    # T[i,j,k] contribution: each (i,j,k) triple gives one term.
    # We need sum_r U[i,j,r]*V[j',k,r]*W[i',k',r] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
    
    # Let's try a parameterized search with small rank.
    # For rank r, we have r columns in U (6 x r), V (12 x r), W (8 x r).
    # Total unknowns: 6r + 12r + 8r = 26r
    # Constraints: for each (i,j,j',k,i',k'), the sum must equal delta_{i,i'}delta_{j,j'}delta_{k,k'}
    # Number of constraints: n^2 * m^2 * p^2 = 4 * 9 * 16 = 576
    # But many are zero constraints.
    
    # Actually, let's think about this differently.
    # The tensor is: T[i,j,j',k,i',k'] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # This factors as: T = (sum_i e_i ⊗ e_i) ⊗ (sum_j e_j ⊗ e_j) ⊗ (sum_k e_k ⊗ e_k)
    # = I_n ⊗ I_m ⊗ I_p (in some sense)
    
    # The rank of this tensor: since it's a Kronecker product of identity-like tensors,
    # rank(T) = rank(I_n tensor) * rank(I_m tensor) * rank(I_p tensor)?
    # No, that's not how tensor rank works for Kronecker products in general.
    
    # Let me just do a direct search.
    
    # For a given rank r, the conditions are:
    # For all (i,j,j',k,i',k'): sum_{r=0}^{R-1} U[i*m+j, r] * V[j'*p+k, r] * W[i'*n+k', r] = delta_{i,i'}*delta_{j,j'}*delta_{k,k'}
    
    # Non-zero constraints: i=i', j=j', k=k' -> 2*3*4 = 24 constraints equal 1
    # Zero constraints: all others -> 576 - 24 = 552 constraints equal 0
    
    # Let's try structured decompositions.
    # One approach: use the fact that the tensor separates in a certain way.
    
    # Actually, for this specific case, let me try a "grouped" approach.
    # Group the j index. For each j, we have a 2x4 sub-problem (i x k).
    # T[i,j,j',k,i',k'] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    
    # If we fix j=j', then for that j, we need:
    # sum_r U[i,j,r] * V[j,k,r] * W[i',k',r] = delta_{i,i'} * delta_{k,k'} for all i,k,i',k'
    # This is like a 2x4 identity tensor for each j.
    
    # The rank of a 2x4 "identity" tensor (delta_{i,i'}delta_{k,k'}) is... 
    # It's the tensor that maps (i,k) to (i',k') as identity. 
    # This has rank min(2*4, ...) hmm.
    
    # Actually, let me just try random/structured searches with time limit.
    
    def verify(U, V, W, rank):
        """Verify the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for kp in range(p):
                                expected = 1 if (i == ip and j == jp and k == kp) else 0
                                s = 0
                                for r in range(rank):
                                    s += U[i*m+j][r] * V[jp*p+k][r] * W[kp*n+ip][r]
                                if s != expected:
                                    return False
        return True
    
    # Try to find lower rank decompositions
    # Start from rank 24 and try to reduce.
    
    # Method: Try random structured decompositions.
    # For each candidate rank, try random integer/half-integer matrices and check.
    
    # Actually, a better approach: use the structure.
    # The tensor T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    # (where e_{i,j} is the basis vector for position (i,j) in the first mode, etc.)
    
    # This is exactly the "matrix multiplication tensor" and its rank is what we're looking for.
    
    # Known: rank of matmul for (n,m,p) is at least max(n*m, m*p, n*p) and at most n*m*p.
    # For (2,3,4): at least 12, at most 24.
    
    # Let me try a systematic search over small-rank decompositions.
    # Given the complexity, I'll try a few structured approaches.
    
    # Approach: Try to use "separable" structures.
    # If we can write T as sum of rank-1 tensors, each rank-1 tensor is u ⊗ v ⊗ w.
    
    # Let's try: for each rank r from 12 to 23, try to find a decomposition.
    # This is a hard problem in general, but for small sizes we can try.
    
    # Given time constraints, let me try a heuristic approach:
    # 1. Start with schoolbook
    # 2. Try to merge pairs of terms (linear combinations)
    
    # Actually, let me try a direct algebraic approach for small ranks.
    # For rank r, we need 26r variables and 576 equations (but only 24 non-trivial + 552 zero).
    
    # Let me try a different tactic: exploit the Kronecker structure.
    # T[i,j,j',k,i',k'] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # = (delta_{i,i'})(delta_{j,j'})(delta_{k,k'})
    
    # This is a product of three "identity" tensors.
    # The rank of a product of tensors is bounded by the product of ranks,
    # but can be less.
    
    # For the 2x2 identity tensor: rank = 2 (it's just I_2)
    # For the 3x3 identity tensor: rank = 3
    # For the 4x4 identity tensor: rank = 4
    # Product rank <= 2*3*4 = 24 (which is what we have)
    
    # Can we do better? The tensor rank of a Kronecker product can be strictly less than the product of ranks.
    
    # Let me just try a computational search with the time limit.
    
    # For efficiency, let me try to find a decomposition with rank < 24.
    # I'll use a randomized approach with structured initialization.
    
    best_U, best_V, best_W = U, V, W
    
    # Try structured search: use the fact that we can combine terms.
    # For the schoolbook, term (i,j,k) has U[i,j]=1, V[j,k]=1, W[i,k]=1.
    # Can we replace two terms with one?
    # Term (i,j,k) + term (i,j,k') for same i,j but different k:
    # U[i,j] would need to be 1+1=2? No, that doesn't work simply.
    
    # Let me try a more systematic approach: assume a specific structure.
    # For example, assume U depends only on (i,j), V only on (j,k), W only on (i,k).
    # Then we need: sum_r U[i,j,r]*V[j,k,r]*W[i,k,r] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
    # Wait, that's not right either because the left side doesn't have i',j',k'.
    
    # I think I need to be more careful. Let me re-examine.
    
    # The decomposition is:
    # T[(i,j),(j',k),(i',k')] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[i'*n+k', r]
    
    # And T[(i,j),(j',k),(i',k')] = 1 if i=i' and j=j' and k=k', else 0.
    
    # So: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[i'*n+k', r] = delta_{i,i'}*delta_{j,j'}*delta_{k,k'}
    
    # For the "diagonal" case i=i', j=j', k=k':
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # For off-diagonal, it must be 0.
    
    # This is a system of polynomial equations. Let me try to solve it for small ranks.
    
    # Given the time limit, let me try a few specific rank values with random restarts.
    
    def try_rank(r, n_attempts=100):
        """Try to find a rank-r decomposition via random search."""
        if r < 12:  # below lower bound
            return None
        for attempt in range(n_attempts):
            if _time_left() < 5:
                return None
            # Random initialization with small integers/half-integers
            values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
            U_rand = [[random.choice(values) for _ in range(r)] for _ in range(n*m)]
            V_rand = [[random.choice(values) for _ in range(r)] for _ in range(m*p)]
            W_rand = [[random.choice(values) for _ in range(r)] for _ in range(p*n)]
            
            if verify(U_rand, V_rand, W_rand, r):
                return (U_rand, V_rand, W_rand)
        return None
    
    # Try ranks from 12 upward
    for r in range(12, best_rank):
        if _time_left() < 10:
            break
        result = try_rank(r, n_attempts=50)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = r
            break  # Found the minimum, stop
    
    # If we didn't find anything better, return schoolbook
    if best_rank == n * m * p:
        best_U, best_V, best_W = _schoolbook(n, m, p)
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
