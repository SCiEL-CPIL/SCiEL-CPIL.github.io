"""Matrix multiplication 3,5,5 - rank-minimal tensor decomposition search."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 75
    
    # Build the target tensor: T[i,j,k] = 1 if this is a valid (i,j,k) contribution
    # For matrix multiplication C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid i,j,k (it's the identity tensor for this multiplication)
    # Actually, the multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] gets contribution A[i,j]*B[j,k] for each j
    
    # The decomposition requires: for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # (since the tensor entry is 1 for all valid indices)
    
    # Schoolbook: rank = n*m*p, with U[i*m+j, i*m+j] = 1, V[j*p+k, i*m+j] = 1, W[k*n+i, i*m+j] = 1
    # Wait, that's not right. Let me re-read the convention.
    
    # W[k*n+i, r] - so W is indexed by (k, i) where k is column of B / row of C, i is row of A / column of C
    # U[i*m+j, r] - U indexed by (i, j) where i is row of A, j is column of A / row of B
    # V[j*p+k, r] - V indexed by (j, k) where j is row of B, k is column of B
    
    # The product: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] should equal delta-like structure
    
    # Actually for matrix multiplication tensor, the entry T[i,j,k] = 1 for all i,j,k
    # because every (i,j,k) triple contributes to the multiplication.
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Schoolbook: one term per (i,j,k) triple, with U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # where c indexes the (i,j,k) triple.
    
    # Let me verify: for a given (i,j,k), only the term where c corresponds to (i,j,k) has all three factors = 1
    # So the sum is 1. Correct.
    
    # Now, to reduce rank, we need to find linear combinations.
    # Key insight: if we have two terms that differ only in one index, we can sometimes merge them.
    
    # For example, terms for (i, j, k1) and (i, j, k2) have:
    # U: U[i*m+j, c1] = 1, U[i*m+j, c2] = 1 (same U row)
    # V: V[j*p+k1, c1] = 1, V[j*p+k2, c2] = 1 (different V rows)
    # W: W[k1*n+i, c1] = 1, W[k2*n+i, c2] = 1 (different W rows)
    
    # These can't be directly merged because V and W rows differ.
    
    # However, we can use the fact that the constraint is only that the SUM equals 1 for each (i,j,k).
    # So we can have overlapping contributions as long as they sum correctly.
    
    # Strategy: Use random linear combinations of the 75 basis terms.
    # Let the basis terms be T_0, T_1, ..., T_74.
    # A rank-r decomposition is: sum_{c=0}^{r-1} U_c ⊗ V_c ⊗ W_c
    # where each (U_c, V_c, W_c) is a linear combination of the basis terms.
    
    # Actually, more precisely: we want to find r vectors (u_c, v_c, w_c) such that
    # sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1 for all i,j,k
    
    # The schoolbook gives 75 such vectors. We want to find fewer.
    
    # Approach: Start with 75 terms, and try to "compress" by finding linear dependencies.
    
    # Let me try a different, more direct approach:
    # Use the fact that for fixed i, the sub-tensor over (j,k) is a rank-m tensor (identity-like).
    # Specifically, for fixed i, we need: sum_c sum_j' U[i*m+j', c] * V[j'*p+k, c] * W[k*n+i, c] = delta_{j,k} ... no wait.
    
    # Let me re-derive. C[i,k] = sum_j A[i,j] * B[j,k].
    # The bilinear form is: sum_{j} A[i,j] * B[j,k].
    # In tensor decomposition: sum_c (sum_j A[i,j] * U[i*m+j, c]) * (sum_k' ... ) 
    # No, that's not right either.
    
    # The correct tensor decomposition for matrix multiplication:
    # We want: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    # This means the (i,j,k) entry of the sum of outer products equals 1.
    
    # OK so my original formulation is correct. Let me just do a search.
    
    # Given the complexity, let me try known results or a smart construction.
    
    # For 3×5×5, one approach is to use the 3×3×3 result and extend.
    # But I don't know the exact low-rank for 3×3×3 off the top of my head.
    
    # Let me try a practical search: attempt to find a decomposition of rank r by
    # solving a system of equations.
    
    # For a given rank r, we have r*15 + r*25 + r*15 = 60r unknowns.
    # We have 3*5*5 = 75 equations (one for each (i,j,k)).
    # The equations are cubic (trilinear), so this is hard.
    
    # Alternative: Use the fact that we can fix some variables.
    # For example, fix U to be the schoolbook U (15×75 with identity-like structure),
    # then solve for V and W.
    
    # Actually, let me try the simplest possible improvement: check if rank 74 works.
    # We can try to merge two schoolbook terms into one by taking a linear combination.
    
    # Let me implement a verification function and a search.
    
    def verify(U, V, W):
        """Verify that the decomposition is exact: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k"""
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if total != 1:
                        return False
        return True
    
    # Schoolbook solution
    def schoolbook():
        n, m, p = DIMENSIONS
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
    
    U0, V0, W0 = schoolbook()
    
    # Try to reduce rank by merging terms.
    # Idea: take two schoolbook terms (c1, c2) and replace them with one term
    # that is a linear combination. But a single rank-1 term can't generally
    # replace two rank-1 terms unless they share structure.
    
    # Better idea: Use a "block" approach. For each pair of (i,j) indices,
    # we have p terms (one for each k). Can we represent all p terms with fewer?
    
    # For fixed (i,j), the terms are:
    # Term k: U[i*m+j, c_k] = 1, V[j*p+k, c_k] = 1, W[k*n+i, c_k] = 1
    # We need: for this (i,j), sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for each k.
    
    # If we use a single term for all k in this (i,j) block, we'd need:
    # U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
    # This requires V[j*p+k,c] * W[k*n+i,c] = 1/U[i*m+j,c] for all k.
    # So V[j*p+k,c] and W[k*n+i,c] must be constant across k (up to reciprocal).
    # But they're indexed by different k, so this only works if we set them to constants.
    # E.g., U[i*m+j,c] = 1, V[j*p+k,c] = 1, W[k*n+i,c] = 1 for all k.
    # But then this term also contributes to other (i',j') blocks!
    
    # So a single term contributes to multiple (i,j,k) triples. This is the key to reduction.
    
    # Let me try a random search: generate random rank-r decompositions and check.
    # But the space is huge.
    
    # Alternative: Use integer linear algebra. The equations are:
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is a system of 75 trilinear equations in 60r unknowns.
    
    # For small r, we can try to solve this using a greedy approach or random initialization.
    
    # Given time constraints, let me try a few specific constructions:
    
    # Construction 1: Use the fact that n=3 is small. For each row i of A,
    # the contribution to C[i,k] is sum_j A[i,j]*B[j,k].
    # We can think of this as: for each i, we need a bilinear form in (j,k).
    
    # There are 3 such bilinear forms (one per i). Each is a 5×5 matrix multiplication "slice".
    
    # The tensor is the sum over i of (e_i ⊗ M_i ⊗ e_i^T) where M_i is the identity-like structure.
    # Actually, for fixed i, the (j,k) slice is the identity matrix? No, it's all ones? 
    # Wait, T[i,j,k] = 1 for all i,j,k. So each (i,j,k) slice... 
    
    # I think I'm overcomplicating this. Let me just try to find a lower rank by search.
    
    # Given the 120s limit, let me try a few promising ranks and use random search.
    
    import numpy as np
    
    best_rank = schoolbook_rank
    best_solution = (U0, V0, W0)
    
    # Try ranks from 74 down, with a time limit per rank
    for target_rank in range(schoolbook_rank - 1, 40, -1):
        if time.time() > deadline:
            break
        
        # For this rank, try random initializations
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with small integers
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p*n)]
            
            if verify(U, V, W):
                best_rank = target_rank
                best_solution = (U, V, W)
                break
        
        if best_rank < schoolbook_rank:
            break
    
    U, V, W = best_solution
    return {"U": U, "V": V, "W": W}
