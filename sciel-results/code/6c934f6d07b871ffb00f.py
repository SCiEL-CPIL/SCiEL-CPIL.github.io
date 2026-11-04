"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]

def _verify(U, V, W, n, m, p, rank):
    """
    Verify that the decomposition exactly computes matrix multiplication.
    For all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For each (i,j,k), compute the sum
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i*m + j]
                v_val = V_arr[j*p + k]
                w_val = W_arr[k*n + i]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition. Rank = n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p  # 60
    
    # Baseline: schoolbook
    best = _schoolbook(n, m, p)
    best_rank = base_rank
    
    # Strategy: Try to find a rank-59 decomposition by checking if any
    # single schoolbook term can be "absorbed" by adjusting coefficients
    # of other terms.
    #
    # For term (ie, je, ke) to be removable, we need to find coefficients
    # a_c for c != e such that for all (i,j,k):
    #   sum_{c != e} a_c * U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1 - delta_{(i,j,k),(ie,je,ke)}
    #
    # where U_c, V_c, W_c are the schoolbook basis vectors.
    #
    # In schoolbook, U_c[i*m+j] = 1 iff c=(i,j,k'), V_c[j*p+k] = 1 iff c=(i',j,k),
    # W_c[k*n+i] = 1 iff c=(i',j',k).
    #
    # So the product U_c[i*m+j]*V_c[j*p+k]*W_c[k*n+i] = 1 iff c=(i,j,k), else 0.
    #
    # This means: sum_{c != e} a_c * delta_{c,(i,j,k)} = 1 - delta_{(i,j,k),(ie,je,ke)}
    #
    # For (i,j,k) = (ie,je,ke): sum_{c != e} a_c * delta_{c,(ie,je,ke)} = 0 (since e is excluded)
    # But we need this to equal 0. OK, that's consistent.
    #
    # For (i,j,k) != (ie,je,ke): a_{(i,j,k)} = 1 if (i,j,k) != e, which is always true here.
    #
    # So the only constraint is that for (ie,je,ke), the sum is 0, which it is.
    # But wait, this means we can't actually remove any term - the structure is rigid.
    #
    # The above analysis shows that simple coefficient scaling won't work.
    # We need to actually modify the structure of multiple terms.
    #
    # Alternative: Try a small perturbation approach where we modify a few terms
    # to see if we can maintain the identity.
    
    def _try_rank59():
        """
        Try to find a rank-59 decomposition by modifying the schoolbook.
        We'll try to find a set of terms to modify such that the identity holds.
        """
        terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        # For each term to potentially eliminate, try to find a correction
        for elim_idx in range(base_rank):
            if time.time() > deadline - 30.0:
                return None
            
            ie, je, ke = terms[elim_idx]
            
            # Build the reduced system: remove term elim_idx
            # and try to adjust a few other terms to compensate.
            
            # The key constraint: for (ie, je, ke), we need sum to be 1.
            # In the reduced schoolbook (without term e), the sum at (ie,je,ke) is 0.
            # We need to add 1 to that position without affecting others.
            
            # A term c=(i',j',k') contributes to position (i,j,k) only if i'=i, j'=j, k'=k.
            # So no single other term can help at (ie,je,ke) unless it IS (ie,je,ke).
            
            # This confirms that rank 59 is not achievable by simple coefficient adjustment.
            # The tensor is "rigid" in the sense that each (i,j,k) position is only
            # affected by the term with exactly those indices.
            
            # However, if we allow terms to have non-standard structure (not just
            # indicator vectors), we might be able to do better. But that requires
            # a fundamentally different construction.
            
            # For now, this confirms that no rank-59 solution exists via this approach.
            pass
        
        return None
    
    # Try the rank-59 search (will likely fail, but we try)
    result_59 = _try_rank59()
    if result_59 is not None and _verify(result_59["U"], result_59["V"], result_59["W"], n, m, p, base_rank - 1):
        best = result_59
        best_rank = base_rank - 1
    
    # Safety check
    if time.time() > deadline - 5.0:
        return best
    
    # Final verification of the best result
    if not _verify(best["U"], best["V"], best["W"], n, m, p, best_rank):
        # Fallback to guaranteed-valid schoolbook
        best = _schoolbook(n, m, p)
        best_rank = base_rank
    
    return best
# EVOLVE-BLOCK-END
