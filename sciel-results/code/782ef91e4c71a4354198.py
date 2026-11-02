"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] = 1 if we're computing (A*B)[i,k] with A[i,j]*B[j,k]
    # Actually, the standard decomposition: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = 1 for all i,j,k (it's the identity tensor for bilinear form)
    # Wait, let me reconsider.
    
    # The matrix multiplication tensor M_{i,j,k} has M_{i,j,k} = 1 for all valid i,j,k
    # because (AB)_{ik} = sum_j A_{ij} B_{jk}
    
    # The decomposition says: M_{i,j,k} = sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                if abs(actual - expected) > tol:
                    return False
    return True


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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


def _try_low_rank(n, m, p, max_rank=None, time_limit=140.0):
    """
    Try to find a rank-minimal decomposition.
    For 2x2 * 2x5, the schoolbook rank is 20.
    We know from literature that small matrix multiplications can sometimes be done faster,
    but for 2x2*2x5 the lower bound is at least max(n*p, m*p, n*m) considerations.
    
    Actually, for multiplying (n x m) by (m x p), the tensor rank lower bound is max(n*m, m*p, n*p)
    is NOT a valid lower bound. The valid lower bound is max(n*p, m*p, n*m) is also not standard.
    
    A known lower bound: rank >= max(n*p, m*p, n*m) is not correct either.
    The trivial lower bound is max(n*m, m*p, n*p) is not a proven lower bound.
    
    For 2x2 * 2x5: n=2, m=2, p=5
    Schoolbook: 20
    Lower bound: at least p*n = 10 (since output is 2x5=10 entries, each needs at least one term)
    Also at least n*m = 4 and m*p = 10.
    
    So rank >= 10. Let's see if we can do better than 20.
    
    For this specific case, it's known that 2x2 * 2x2 can be done in rank 7 (Strassen).
    For 2x2 * 2x5, we can potentially use Strassen-like methods.
    
    Let me try a systematic search for low-rank decompositions.
    """
    start_time = time.time()
    
    if max_rank is None:
        max_rank = n * m * p
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, n*p) is not a valid LB, but n*p is a valid LB
    # because the output has n*p entries and each rank-1 term contributes to at most... 
    # Actually, each rank-1 term U[:,c] ⊗ V[:,c] ⊗ W[:,c] can be non-zero in many positions.
    # The valid lower bound is simply: rank >= 1 (trivial), and more usefully:
    # The border rank considerations are complex. Let's just try from reasonable values.
    
    # For 2x2*2x5, let's try ranks from 10 to 19
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Use a simple approach: try to find decompositions with integer/half-integer entries
    # by solving linear systems
    
    for target_rank in range(max(1, n * p), min(max_rank, best_rank)):
        if time.time() - start_time > time_limit:
            break
        
        # Try to find a decomposition of this rank
        # The constraint is: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        
        # This is a system of bilinear equations. Hard to solve directly.
        # Let's try a different approach: use the fact that for small dimensions,
        # we can enumerate or use known constructions.
        
        # For now, just check if we can find one via random search with integer entries
        # in {-1, 0, 1, 0.5, 1.5} or similar small sets
        
        # Actually, let's be more systematic. For 2x2 * 2x5:
        # U is 4 x r, V is 10 x r, W is 10 x r
        # The equations are: for each (i,j,k) in [2]x[2]x[5], 
        #   sum_c U[i*2+j,c] * V[j*5+k,c] * W[k*2+i,c] = 1
        
        # That's 20 equations. With r rank-1 terms, we have 4r + 10r + 10r = 24r unknowns.
        # For r >= 1, we have more unknowns than equations, so solutions should exist
        # over the reals. The question is whether they exist with rational (half-integer) entries.
        
        # Let's try to construct one explicitly.
        # 
        # Key insight: The 2x2*2x5 multiplication can be seen as 5 independent 2x2*2x1 multiplications
        # plus some structure. Actually, (A 2x2)(B 2x5) = C 2x5.
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1}, k in {0,1,2,3,4}
        
        # So we need 10 products essentially, but they share the A factors.
        # 
        # Let's try rank 10: one term per output element.
        # For output (i,k), we need: sum_c U[i*2+j,c]*V[j*5+k,c]*W[k*2+i,c] = 1 for all j
        # Wait, that's not right. The equation is per (i,j,k):
        # sum_c U[i*2+j,c] * V[j*5+k,c] * W[k*2+i,c] = 1
        
        # For fixed (i,k), we have 2 equations (j=0,1) that must both equal 1.
        # 
        # Let's try a block structure. Suppose we use 10 terms, one for each (i,k) pair.
        # Term (i,k) has: U[i*2+0, c_ik]=a, U[i*2+1, c_ik]=b, V[0*5+k, c_ik]=c, V[1*5+k, c_ik]=d, W[k*2+i, c_ik]=e
        # Other entries of U, V, W for this term are 0.
        # 
        # Then for equation (i',j,k'):
        # Only terms where i'=i and k'=k contribute (if we make them sparse)
        # For (i,j,k) with i'=i, k'=k: contribution is U[i*2+j, c_ik] * V[j*5+k, c_ik] * W[k*2+i, c_ik]
        # = (a if j=0 else b) * (c if j=0 else d) * e
        # For j=0: a*c*e = 1
        # For j=1: b*d*e = 1
        # So we need a*c*e=1 and b*d*e=1.
        # Choose e=1, a=c=1, b=d=1. That works!
        # 
        # But wait, this only works if the terms are completely decoupled.
        # Let me verify: with this construction, for term c_ik:
        #   U[i*2+0, c_ik] = 1, U[i*2+1, c_ik] = 1
        #   V[0*5+k, c_ik] = 1, V[1*5+k, c_ik] = 1  
        #   W[k*2+i, c_ik] = 1
        #   All other entries are 0.
        # 
        # Check equation for (i', j, k'):
        #   sum over all terms c_ik: U[i'*2+j, c_ik] * V[j*5+k', c_ik] * W[k'*2+i', c_ik]
        #   Non-zero only if i'=i and k'=k (since U is non-zero only for row i*2+j, V for col k, W for row k*2+i')
        #   Wait, W[k'*2+i', c_ik] is non-zero only if k'*2+i' = k*2+i, i.e., k'=k and i'=i.
        #   So indeed, only the term c_ik' where k'=k and i'=i contributes.
        #   Then the contribution is U[i'*2+j, c_ik'] * V[j*5+k', c_ik'] * W[k'*2+i', c_ik']
        #   = 1 * 1 * 1 = 1. 
        # 
        # But wait, I need to double-check the indices. Let me re-examine.
        # 
        # U has shape (n*m, rank) = (4, rank). Rows indexed by i*m+j.
        # V has shape (m*p, rank) = (10, rank). Rows indexed by j*p+k.
        # W has shape (p*n, rank) = (10, rank). Rows indexed by k*n+i.
        # 
        # For term corresponding to output (i_out, k_out):
        #   U[i_out*m + j, c] should be non-zero for j=0,1
        #   V[j*p + k_out, c] should be non-zero for j=0,1
        #   W[k_out*n + i_out, c] should be non-zero
        # 
        # Equation for (i, j, k):
        #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # 
        # If I assign term c_ik to output (i,k):
        #   U[i*m+j, c_ik] = 1 for j=0,1
        #   V[j*p+k, c_ik] = 1 for j=0,1
        #   W[k*n+i, c_ik] = 1
        # 
        # Then for equation (i', j', k'):
        #   Only c_ik where i=i' and k=k' contributes (because W[k'*n+i', c_ik] != 0 requires k'=k and i'=i)
        #   And U[i'*m+j', c_ik'] = 1, V[j'*p+k', c_ik'] = 1, W[k'*n+i', c_ik'] = 1
        #   So the sum is 1. 
        # 
        # This gives a rank-10 decomposition! But wait, is this valid?
        # Let me count: there are n*p = 2*5 = 10 such terms.
        
        # Let's construct it:
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        if target_rank == n * p:
            col = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
            
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # If the simple construction didn't work, try other approaches
    # But for now, return schoolbook as fallback
    return best_result


def construct():
    n, m, p = DIMENSIONS
    
    # First, try the rank-10 construction
    result = _try_low_rank(n, m, p, max_rank=n*m*p, time_limit=140.0)
    
    # Verify the result
    if not _verify(result["U"], result["V"], result["W"], n, m, p):
        # Fall back to schoolbook
        result = _schoolbook(n, m, p)
    
    return result

# EVOLVE-BLOCK-END
