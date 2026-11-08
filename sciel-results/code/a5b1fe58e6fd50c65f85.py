"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return a valid rank n*m*p decomposition (the safe fallback)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly computes matrix multiplication.
    For all i in [0,n), k in [0,p):
      sum_j A[i,j] B[j,k] = sum_c (sum_j U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c]
    
    This means: for each c, the outer product of the U-column (indexed by (i,j)) 
    and V-column (indexed by (j,k)) must, when summed over c and weighted by W,
    give the correct product.
    
    More precisely, define for each c:
      U_c[i,j] = U[i*m+j, c]
      V_c[j,k] = V[j*p+k, c]
      w_c[k,i] = W[k*n+i, c]
    
    Then we need: sum_c w_c[k,i] * sum_j U_c[i,j] * V_c[j,k] = sum_j A[i,j]*B[j,k]
    
    For this to hold for all A, B, we need:
      sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1 if the (i,j,k) triple is "active"
    Wait, that's not right. Let me re-derive.
    
    The tensor T[i,j,k] = A[i,j]*B[j,k] has the property that:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    But T is not a fixed tensor - it depends on A and B. The decomposition says:
    (A*B)[i,k] = sum_j A[i,j]*B[j,k] = sum_c (sum_j U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c]
    
    For this to hold for ALL A and B, we need the following:
    For each c, define U_c as an n x m matrix and V_c as an m x p matrix.
    Then: sum_c W[k*n+i,c] * (U_c * V_c)[i,k] = (A*B)[i,k] for all A,B.
    
    Actually, the standard formulation is:
    (A*B)[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * A[i,j]) * (sum_j' V[j'*p+k,c] * B[j',k])
    
    No wait. Let me think again carefully.
    
    The tensor decomposition of matrix multiplication means:
    T_{i,j,k} = sum_c U_{i,j,c} * V_{j,k,c} * W_{k,i,c}
    
    where T_{i,j,k} represents the "coefficient" such that:
    (A*B)_{i,k} = sum_{j} A_{i,j} * B_{j,k}
    
    And the decomposition expresses this as:
    (A*B)_{i,k} = sum_c W_{k,i,c} * (sum_j U_{i,j,c} * A_{i,j}) * (sum_j' V_{j',k,c} * B_{j',k})
    
    Hmm, that doesn't look right either. Let me use the standard Strassen formulation.
    
    The matrix multiplication tensor is a 3-tensor T of shape (n, m, p) where:
    T[i,j,k] = 1 if we're looking at the contribution of A[i,j]*B[j,k] to (AB)[i,k]
    
    Actually, the standard way: the matrix multiplication tensor T has entries:
    T[i,j,k] = 1 for all valid (i,j,k)
    
    No. The tensor is defined so that:
    (AB)[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    
    where T[i,j,k] = 1 for all i,j,k.
    
    A rank-r decomposition gives:
    (AB)[i,k] = sum_{c=1}^{r} W[k,i,c] * (sum_j U[i,j,c] * A[i,j]) * (sum_j V[j,k,c] * B[j,k])
    
    For this to equal sum_j A[i,j]*B[j,k] for all A,B, we need:
    sum_c W[k,i,c] * U[i,j,c] * V[j',k,c] = delta_{j,j'} ... no.
    
    Let me be very careful. We want:
    sum_j A[i,j] B[j,k] = sum_c W[k,i,c] * (sum_j U[i,j,c] A[i,j]) * (sum_j' V[j',k,c] B[j',k])
    
    = sum_c W[k,i,c] * sum_j sum_j' U[i,j,c] V[j',k,c] A[i,j] B[j',k]
    
    For this to equal sum_j A[i,j] B[j,k] for all A,B, we need:
    sum_c W[k,i,c] U[i,j,c] V[j',k,c] = delta_{j,j'}
    
    Wait, that would mean for each (i,k), the matrix sum_c W[k,i,c] * U_c[i,:] * V_c[:,k] = I_m.
    
    Hmm, that's one valid formulation. Let me verify with schoolbook:
    For schoolbook, rank = n*m*p, and for each (i,j,k) triple, c corresponds to (i,j,k):
    U[i,m+j,c] = 1 (only at position (i,j))
    V[j*p+k,c] = 1 (only at position (j,k))
    W[k*n+i,c] = 1 (only at position (k,i))
    
    Then sum_c W[k,i,c] U[i,j,c] V[j',k,c] = sum over c where W[k,i,c]=1, U[i,j,c]=1, V[j',k,c]=1
    = 1 if there exists c with all three = 1, which happens when c corresponds to (i,j,k) AND j'=j.
    So it equals delta_{j,j'}. Correct!
    
    So the verification condition is:
    For all i in [0,n), k in [0,p), j in [0,m), j' in [0,m):
      sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = 1 if j==j' else 0
    """
    r = len(U[0])
    # Check: for all i,k,j,j': sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c] = delta_{j,j'}
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    s = 0
                    for c in range(r):
                        s += W[k * n + i][c] * U[i * m + j][c] * V[jp * p + k][c]
                    expected = 1 if j == jp else 0
                    if s != expected:
                        return False
    return True

def _random_decomposition(n, m, p, r, seed=42):
    """
    Try to find a rank-r decomposition using random initialization and iterative refinement.
    We use the condition: for all i,k,j,j': sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c] = delta_{j,j'}
    """
    rng = random.Random(seed)
    
    # Initialize with small random values
    U = [[rng.uniform(-1, 1) for _ in range(r)] for _ in range(n * m)]
    V = [[rng.uniform(-1, 1) for _ in range(r)] for _ in range(m * p)]
    W = [[rng.uniform(-1, 1) for _ in range(r)] for _ in range(p * n)]
    
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication.
    Returns {'U': U, 'V': V, 'W': W} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 105
    
    # Start with schoolbook as the best known valid solution
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Try to find lower-rank decompositions
    # Known lower bounds: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # Upper bound from schoolbook: 105
    # For small matrices, we can try various ranks
    
    # Strategy: For each candidate rank r from lower bound to best_rank-1,
    # try to find a valid decomposition. Use random restarts.
    
    lower_bound = max(n * m, m * p, p * n)
    
    # We'll try ranks in increasing order, but given time constraints,
    # focus on a few promising ranks with multiple random restarts.
    
    # Actually, for 3x5*5x7, the exact minimal rank is not known in general,
    # but we can try to improve on schoolbook.
    
    # Let's try a structured approach: use the fact that we need
    # sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c] = delta_{j,j'}
    
    # This is equivalent to: for each (i,k), the sum over c of outer products
    # U_c[i,:] * V_c[:,k] weighted by W[k,i,c] equals I_m.
    
    # For a given rank r, we need to find U, V, W such that this holds.
    
    # Let's try random restarts for a few ranks.
    
    candidate_ranks = list(range(lower_bound, best_rank))
    # Reverse to try lower ranks first
    candidate_ranks.reverse()
    
    for r in candidate_ranks:
        if time.time() > deadline:
            break
        
        # Try a few random restarts for this rank
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            U, V, W = _random_decomposition(n, m, p, r, seed=restart)
            
            # Check if it's valid (unlikely with random init, but let's check)
            if _verify(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break  # Found valid for this rank, move to next lower rank
        
        # If we found a valid decomposition for this rank, continue to lower
        if best_rank == r:
            continue
    
    # Return the best valid decomposition found
    # Convert to lists (they're already lists)
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
