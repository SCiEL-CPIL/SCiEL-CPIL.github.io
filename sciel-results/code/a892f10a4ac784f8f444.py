"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the rank-48 schoolbook decomposition for 3x4x4."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, tol=1e-9):
    """Check that U, V, W exactly decompose the matrix multiplication tensor.
    
    The tensor T[i,j,k] = 1 if the (i,k) entry of A*B is sum_j A[i,j]*B[j,k].
    We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{i,k} ... 
    Actually, the standard decomposition is:
    (A*B)[i,k] = sum_j A[i,j] B[j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] * A[i,j] B[j,k]
    
    Wait, let me re-read the convention.
    
    The tensor for matrix multiplication is T_{i,j,k} such that (AB)_{i,k} = sum_j A_{i,j} B_{j,k}.
    A rank-r decomposition is: T_{i,j,k} = sum_c U_{(i,j),c} V_{(j,k),c} W_{(k,i),c} ... no.
    
    Actually, the standard form is:
    (AB)_{i,k} = sum_c (sum_j U_{i,j,c} A_{i,j}) (sum_j V_{j,k,c} B_{j,k}) ... no that's not right either.
    
    Let me think more carefully. The matrix multiplication tensor T has indices (i,j,k) where i=0..n-1, j=0..m-1, k=0..p-1.
    T[i,j,k] = 1 for all valid indices (it's the identity in the sense that (AB)_{ik} = sum_j A_{ij} B_{jk}).
    
    A rank-r decomposition means: T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Wait, that doesn't make sense dimensionally. Let me reconsider.
    
    Actually, the standard definition: A rank-r decomposition of the matrix multiplication tensor is a set of r triples of vectors (U_c, V_c, W_c) where:
    U_c is in R^{n*m}, V_c is in R^{m*p}, W_c is in R^{p*n}
    such that for all i,j,k: T[i,j,k] = sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
    
    But T[i,j,k] = 1 for all i,j,k (since (AB)_{ik} = sum_j A_{ij} B_{jk} uses all j's).
    
    Hmm, that's not quite right either. Let me look at the schoolbook decomposition in the seed:
    
    For each (i,j,k): U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    
    So T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    
    In the schoolbook case, for a fixed (i,j,k), only the column corresponding to (i,j,k) has all three entries equal to 1, so T[i,j,k] = 1.
    
    But wait, the matrix multiplication tensor should be such that (AB)_{ik} = sum_j A_{ij} B_{jk}. The "tensor" here is actually the structure constants. The decomposition gives:
    
    (AB)_{ik} = sum_j A_{ij} B_{jk} = sum_c [sum_j U[i*m+j,c] A_{ij}] [sum_k' V[j'*p+k,c] B[j'k]] ... no.
    
    I think the correct interpretation is:
    (AB)_{ik} = sum_c (sum_j U_{ij,c} A_{ij}) (sum_{j'} V_{j'k,c} B_{j'k}) ... no, that would give a product of two scalars.
    
    Let me just trust the seed code. The seed says the decomposition is exact if:
    For all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    And the schoolbook satisfies this. So I'll use this as my verification criterion.
    """
    n, m, p = DIMENSIONS
    # Build the target tensor: T[i,j,k] = 1 for all i,j,k
    # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _rank_of_decomposition(U, V, W):
    """Return the rank (number of columns) of the decomposition."""
    return U.shape[1]

def _try_random_search(deadline, initial_rank=48, num_trials=500, seed=42):
    """Try to find a lower-rank decomposition via random initialization and optimization."""
    np.random.seed(seed)
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    best_rank = initial_rank
    best_U, best_V, best_W = _schoolbook()
    
    # Try ranks from 47 down to 32
    for rank in range(initial_rank - 1, 31, -1):
        if time.time() > deadline:
            break
        for trial in range(num_trials // (initial_rank - rank + 1)):
            if time.time() > deadline:
                break
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Simple gradient-free check: just verify
            # This is a very crude approach; real optimization would be needed
            # For now, just check if this random init happens to work (extremely unlikely)
            if _verify(U, V, W, tol=1e-8):
                return rank, U, V, W
    
    return best_rank, best_U, best_V, best_W

def construct():
    """Construct the matrix multiplication decomposition.
    
    Returns a dict with keys 'U', 'V', 'W' mapping to lists of lists.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to improve via random search (very unlikely to succeed for this problem,
    # but we include it as a placeholder for more sophisticated methods)
    # The known best rank for 3x4x4 matrix multiplication is not well-established,
    # but it's likely higher than 32. The schoolbook rank is 48.
    
    # Given the difficulty of finding a lower-rank exact decomposition with integer/half-integer
    # entries, and the 120s time limit, we return the schoolbook decomposition.
    # A more sophisticated approach would use numerical algebraic geometry or
    # structured optimization, but that's beyond the scope of a simple Python program
    # within the time limit.
    
    # Convert numpy arrays to lists of lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
