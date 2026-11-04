"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]
MAX_STEPS = 8

def _schoolbook(n, m, p):
    """
    Standard schoolbook matrix multiplication tensor decomposition.
    Rank = n * m * p.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    Indexing:
    U[i*m+j, c]
    V[j*p+k, c]
    W[k*n+i, c]
    """
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
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill switch
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # 1. Establish the safe baseline (Schoolbook)
    best = _schoolbook(n, m, p)
    best_rank = base_rank
    
    # 2. Attempt to find rank-59 decomposition
    # Strategy: Use the fact that for 3x4x5, we can potentially eliminate one term
    # by finding a non-trivial linear dependency in the tensor space.
    
    def _verify_decomposition(U, V, W, n, m, p, rank):
        """Verify that the decomposition exactly computes matrix multiplication."""
        # For each (i, j, k), check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        # And for all other combinations, the sum should be 0.
        # Actually, we need to verify the full tensor identity.
        # The decomposition gives: C[i,k] = sum_j A[i,j]*B[j,k]
        # In tensor form: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We need T[i,j,k] = 1 if this is a valid multiplication term, else 0.
        # Wait, that's not right. The tensor for matrix mult is:
        # T[i,j,k] contributes A[i,j]*B[j,k] to C[i,k]
        # So the decomposition should satisfy:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta for the right structure
        
        # Let me think more carefully. The standard decomposition has:
        # U[i*m+j, c] = 1 if c corresponds to (i,j,k)
        # V[j*p+k, c] = 1 if c corresponds to (i,j,k)  
        # W[k*n+i, c] = 1 if c corresponds to (i,j,k)
        # Then sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 if there exists c where all three are 1
        # For schoolbook, this happens exactly when c=(i,j,k), so the sum is 1 for all (i,j,k)
        # But that's not right either - the tensor should be 1 for each (i,j,k) independently
        
        # Actually, the correct check is:
        # The product tensor T has T[i,j,k] = 1 for all i,j,k (it's the identity in the sense
        # that each term A[i,j]*B[j,k] contributes to C[i,k])
        # No wait, T[i,j,k] should be such that C[i,k] = sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]*A[i,j]*B[j,k]
        # = sum_j A[i,j]*B[j,k] * (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])
        # For this to equal sum_j A[i,j]*B[j,k], we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
        
        # So the verification is: for all i,j,k, the sum equals 1.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def _try_find_rank59(n, m, p, deadline):
        """
        Try to find a rank-59 decomposition by starting from schoolbook (rank 60)
        and trying to eliminate one term.
        
        Key idea: If we can find coefficients such that one schoolbook term's contribution
        can be replicated by adjusting other terms, we reduce rank by 1.
        
        For term (i_e, j_e, k_e) to be eliminated, we need:
        For all (i,j,k): 
          sum_{c != e} U'[i*m+j,c] * V'[j*p+k,c] * W'[k*n+i,c] 
          + U'[i*m+j,e] * V'[j*p+k,e] * W'[k*n+i,e] = 1
        
        If we set U'[i*m+j,e] = 0 for all (i,j) except (i_e,j_e), etc., this gets complex.
        
        Alternative approach: Try small perturbations. Start with schoolbook, set one term's
        coefficients to 0, and see if we can solve for adjustments in other terms.
        """
        # Build the schoolbook terms
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append((i, j, k))
        
        # Try each possible term to eliminate
        for elim_idx in range(base_rank):
            if time.time() > deadline - 20.0:
                return None
            
            i_e, j_e, k_e = terms[elim_idx]
            
            # Create a modified decomposition where term elim_idx is "removed"
            # and we try to find compensating adjustments in other terms.
            
            # Initialize with schoolbook
            U = [[0] * (base_rank - 1) for _ in range(n * m)]
            V = [[0] * (base_rank - 1) for _ in range(m * p)]
            W = [[0] * (base_rank - 1) for _ in range(p * n)]
            
            # Copy all terms except elim_idx
            col = 0
            for idx in range(base_rank):
                if idx == elim_idx:
                    continue
                i_t, j_t, k_t = terms[idx]
                U[i_t * m + j_t][col] = 1
                V[j_t * p + k_t][col] = 1
                W[k_t * n + i_t][col] = 1
                col += 1
            
            # Now check if this already works (it shouldn't, since we removed a term)
            # For (i_e, j_e, k_e), the sum will be 0 instead of 1.
            # We need to add corrections to other terms to compensate.
            
            # The compensation must come from terms that share at least one index with (i_e, j_e, k_e)
            # Terms that can contribute to fixing (i_e, j_e, k_e):
            # - Terms with same i_e, j_e but different k: (i_e, j_e, k') for k' != k_e
            # - Terms with same i_e, k_e but different j: (i_e, j', k_e) for j' != j_e  
            # - Terms with same j_e, k_e but different i: (i', j_e, k_e) for i' != i_e
            
            # Let's try to find a linear combination of these that gives 1 at (i_e, j_e, k_e)
            # and 0 at all other (i,j,k) positions.
            
            # This is a system of equations. Let's set up the variables.
            # Let x_{i,j,k} be the coefficient adjustment for term (i,j,k) (where (i,j,k) != (i_e,j_e,k_e))
            
            # The constraint is: for all (i,j,k),
            #   sum_{(i',j',k') in support} x_{i',j',k'} * U[i*m+j,i',j',k'] * V[j*p+k,i',j',k'] * W[k*n+i,i',j',k'] = 1 - base_sum
            # where base_sum is the sum from the remaining schoolbook terms.
            
            # This is getting complex. Let me try a simpler heuristic:
            # Try to find if there's a "cycle" of terms that can compensate.
            
            # For now, let's try a random search with verification
            import random
            rng = random.Random(42 + elim_idx)
            
            max_attempts = 500
            for attempt in range(max_attempts):
                if time.time() > deadline - 10.0:
                    return None
                
                # Try random small integer coefficients for a subset of terms
                # Pick up to 5 terms to adjust
                candidates = [t for t in range(base_rank) if t != elim_idx]
                num_adjust = min(5, len(candidates))
                adj_indices = rng.sample(candidates, num_adjust)
                
                # Try random coefficients from {-1, 0, 1, 2, -2}
                coeffs = [rng.choice([-2, -1, 0, 1, 2]) for _ in range(num_adjust)]
                
                # Build the modified decomposition
                U_mod = [row[:] for row in U]  # Deep copy
                V_mod = [row[:] for row in V]
                W_mod = [row[:] for row in W]
                
                # Apply adjustments
                for adj_idx, coeff in zip(adj_indices, coeffs):
                    if coeff == 0:
                        continue
                    i_t, j_t, k_t = terms[adj_idx]
                    # Find the column index for this term in the reduced decomposition
                    col_idx = None
                    for c in range(base_rank - 1):
                        if U[i_t * m + j_t][c] == 1 and V[j_t * p + k_t][c] == 1 and W[k_t * n + i_t][c] == 1:
                            col_idx = c
                            break
                    if col_idx is not None:
                        U_mod[i_t * m + j_t][col_idx] = 1 + coeff
                        V_mod[j_t * p + k_t][col_idx] = 1 + coeff
                        W_mod[k_t * n + i_t][col_idx] = 1 + coeff
                
                # Verify
                if _verify_decomposition(U_mod, V_mod, W_mod, n, m, p, base_rank - 1):
                    return {"U": U_mod, "V": V_mod, "W": W_mod}
            
            # If random search failed, try a more systematic approach
            # Focus on terms that share indices with the eliminated term
            relevant_terms = []
            for idx in range(base_rank):
                if idx == elim_idx:
                    continue
                i_t, j_t, k_t = terms[idx]
                # Include if shares at least one index
                if i_t == i_e or j_t == j_e or k_t == k_e:
                    relevant_terms.append(idx)
            
            if len(relevant_terms) <= 6:
                # Try all combinations of coefficients from {-1, 0, 1}
                from itertools import product as prod
                coeff_range = [-1, 0, 1]
                for coeff_combo in prod(coeff_range, repeat=len(relevant_terms)):
                    if time.time() > deadline - 5.0:
                        return None
                    
                    U_mod = [row[:] for row in U]
                    V_mod = [row[:] for row in V]
                    W_mod = [row[:] for row in W]
                    
                    for idx, coeff in zip(relevant_terms, coeff_combo):
                        if coeff == 0:
                            continue
                        i_t, j_t, k_t = terms[idx]
                        col_idx = None
                        for c in range(base_rank - 1):
                            if U[i_t * m + j_t][c] == 1 and V[j_t * p + k_t][c] == 1 and W[k_t * n + i_t][c] == 1:
                                col_idx = c
                                break
                        if col_idx is not None:
                            U_mod[i_t * m + j_t][col_idx] = 1 + coeff
                            V_mod[j_t * p + k_t][col_idx] = 1 + coeff
                            W_mod[k_t * n + i_t][col_idx] = 1 + coeff
                    
                    if _verify_decomposition(U_mod, V_mod, W_mod, n, m, p, base_rank - 1):
                        return {"U": U_mod, "V": V_mod, "W": W_mod}
        
        return None
    
    # Try the rank-59 search
    result_59 = _try_find_rank59(n, m, p, deadline)
    if result_59 is not None:
        best = result_59
        best_rank = 59
    
    # Check time to ensure we don't accidentally timeout
    if time.time() > deadline - 5.0:
        return best

    return best
# EVOLVE-BLOCK-END
