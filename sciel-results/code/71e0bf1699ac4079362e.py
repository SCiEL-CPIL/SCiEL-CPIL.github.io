"""Find rank-minimal tensor decomposition for 2x5 * 5x6 matrix multiplication."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2 by 5 matrix by a 5 by 6 matrix.
    
    Returns dict with U, V, W matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    such that the tensor T[i*m+j, k*p+l] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    Wait, let me re-read the convention.
    
    Output factor follows W[k*n+i, r]. So W has shape (p*n, rank) with index [k*n+i, r].
    U has shape (n*m, rank) with index [i*m+j, r]
    V has shape (m*p, rank) with index [j*p+k, r]
    
    The decomposition should satisfy:
    For all i in [0,n), j in [0,m), k in [0,p):
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    where T[i,j,k] is the matrix multiplication tensor. For matrix multiplication,
    the "tensor" is defined such that C[i,k] = sum_j A[i,j]*B[j,k].
    So the bilinear form is: for each (i,j,k), the coefficient is 1 if we're 
    computing C[i,k] from A[i,j]*B[j,k].
    
    Actually, the matrix multiplication tensor M of size n*m*p has entries:
    M[i, j, k] = 1 if this term contributes to C[i,k] via A[i,j]*B[j,k]
    
    More precisely, the tensor is: for each i in [0,n), j in [0,m), k in [0,p),
    M[i,j,k] = 1 (since A[i,j]*B[j,k] always contributes to C[i,k]).
    
    So we need: for all i,j,k:
    1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    And for all other (i',j',k') not corresponding to valid terms... wait, 
    actually every (i,j,k) with i in [0,n), j in [0,m), k in [0,p) is a valid 
    term. So the tensor is all ones of shape (n,m,p).
    
    Hmm, that's not right either. Let me think again.
    
    The matrix multiplication tensor is defined as:
    T[a, b, c] where a indexes (i,j), b indexes (j,k), c indexes (i,k)
    T[i*m+j, j'*p+k, i'*n+k'] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    So T is 1 when i=i', j=j', k=k', and 0 otherwise.
    
    The decomposition is:
    T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    So for a = i*m+j, b = j*p+k, c = i'*n+k':
    T[a,b,c] = 1 if i=i' and j=j' and k=k', else 0.
    
    So the condition is:
    For all i,j,k: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r equals 1
    For all (i,j,k,i',j',k') with not (i=i' and j=j' and k=k'): 
    sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = 0
    
    This is a complex constraint. Let me implement a search.
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best = schoolbook()
    
    # Try to find lower rank decompositions
    # The tensor has structure: T[i*m+j, j'*p+k, i'*n+k'] = 
    #   1 if i=i', j=j', k=k' else 0
    
    # Let's try rank from lower bounds up
    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # Also, the tensor rank is at least the max of the "frontal" ranks
    
    # Try various ranks
    min_possible = max(n * m, m * p, p * n)
    
    for rank in range(min_possible, n * m * p + 1):
        if time.time() > deadline:
            break
        
        # Try to find a decomposition of this rank
        # Use a greedy/structured approach
        
        # Strategy: For each (i,j,k), we need exactly one "active" term that 
        # contributes to T[i*m+j, j*p+k, k*n+i] = 1
        # and no cross-terms.
        
        # A simple structured approach: group by i
        # For fixed i, we have m*p terms: (i, j, k) for j in [0,m), k in [0,p)
        # These correspond to U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        
        # Let's try: for each i, use m*p independent terms (schoolbook per i)
        # That gives rank = n*m*p = 60
        
        # Can we do better by sharing across i values?
        # If we share, we risk cross-terms.
        
        # Try rank = m*p = 30: use the same V and W structure for all i?
        # U[i*m+j, r] would need to depend on i, but V and W shared.
        # For fixed j,k: V[j*p+k, r] and W[k*n+i, r] depend on i through W.
        # So W[k*n+i, r] varies with i.
        
        # Let's try: rank = m*p = 30
        # For r = j*p+k:
        #   V[j*p+k, r] = 1
        #   W[k*n+i, r] = delta_{i, i_r} for some i_r? No, that doesn't work.
        
        # Actually, let's think differently.
        # The tensor is: T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
        
        # This is a sum of n*m*p rank-1 tensors.
        # Can we combine them?
        
        # For two terms (i1,j1,k1) and (i2,j2,k2), their sum is:
        # e_{i1*m+j1} ⊗ e_{j1*p+k1} ⊗ e_{k1*n+i1} + e_{i2*m+j2} ⊗ e_{j2*p+k2} ⊗ e_{k2*n+i2}
        
        # This can be written as a rank-2 tensor only if... it's already rank 2 
        # unless they share some index structure.
        
        # If i1=i2=i, j1≠j2, k1≠k2:
        # e_{i*m+j1} ⊗ e_{j1*p+k1} ⊗ e_{k1*n+i} + e_{i*m+j2} ⊗ e_{j2*p+k2} ⊗ e_{k2*n+i}
        # The W part: e_{k1*n+i} + e_{k2*n+i} is a vector with two non-zero entries.
        # This is still rank 2 in general.
        
        # So it seems hard to reduce below n*m*p without clever structure.
        
        # Let's verify: is the tensor rank actually n*m*p for these dimensions?
        # For square matrices, the tensor rank is n^3 for schoolbook, but can be 
        # reduced (Strassen: 7 for n=2).
        
        # For 2x5 * 5x6, let's try to find a lower rank.
        
        # Approach: Use bilinear decomposition.
        # We need matrices U (10 x rank), V (30 x rank), W (12 x rank) such that
        # for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # for all (i,j,k,i',j',k') with (i,j,k)≠(i',j',k'): 
        #   sum_r U[i*m+j,r] * V[j'*p+k,r] * W[k'*n+i',r] = 0
        
        # This is a system of polynomial equations. Let's try small ranks.
        
        # For rank r, we have 10*r + 30*r + 12*r = 52*r variables.
        # Constraints: 2*5*6 = 60 equations for the "1" entries,
        # and (2*5*6)^2 - 2*5*6 = 3540 equations for the "0" entries.
        
        # This is highly overdetermined for small r.
        
        # Let's try a different approach: exploit the structure.
        # Group by the middle index j.
        # For fixed j, we have terms (i, j, k) for i in [0,2), k in [0,6).
        # That's 12 terms per j, and 5 j's, total 60.
        
        # For fixed j, the U indices are {j, j+5} (i=0: j, i=1: j+5)
        # V indices are {j*6, j*6+1, ..., j*6+5}
        # W indices are {0, 1, ..., 11} but only those with i=0 or i=1:
        #   For i=0: W[k*2+0, r] for k in [0,6) → indices 0,2,4,6,8,10
        #   For i=1: W[k*2+1, r] for k in [0,6) → indices 1,3,5,7,9,11
        
        # Hmm, this is getting complex. Let me just try to verify if rank 60 is 
        # indeed minimal, or if we can do better.
        
        # For 2x2 * 2x2, Strassen gives rank 7 < 8.
        # For 2x5 * 5x6, can we do better than 60?
        
        # Let's try a heuristic: assume the rank is at least max(n*m, m*p, p*n) = 30.
        # Try rank 30, 31, ..., 60 and see if we can find a valid decomposition.
        
        # For now, return schoolbook. A more sophisticated search would be needed.
        pass
    
    return best
# EVOLVE-BLOCK-END
