import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

def _verify(U, V, W):
    """Verify that the decomposition exactly computes matrix multiplication."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Check dimensions
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # and check it equals the standard matrix multiplication tensor
    # where T[i,j,k] = 1 if we're computing (AB)[i,k] = sum_j A[i,j]B[j,k]
    # Actually the tensor for matrix multiplication is:
    # T[i,j,k] = 1 (it's the structure tensor)
    # The decomposition says: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r[i,j] = U[i*m+j, r], v_r[j,k] = V[j*p+k, r], w_r[k,i] = W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # The structure tensor entry for (i,j,k) should be 1
                # because (AB)[i,k] = sum_j A[i,j] B[j,k], and the tensor 
                # T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in (AB)[i,k]
                # which is 1 for all valid i,j,k
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook_decomposition():
    """Return the schoolbook rank-72 decomposition."""
    n, m, p = DIMENSIONS
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

def _try_strassen_2x2_block():
    """
    Try to use Strassen's algorithm on a 2x2 block structure.
    Partition the 3x3 * 3x8 multiplication using block structure.
    
    A is 3x3, B is 3x8, C = AB is 3x8.
    
    We can try to split B into B1 (3x4) and B2 (3x4), then C = [A*B1, A*B2].
    Each A*B is 3x3 * 3x4.
    
    For 3x3 * 3x4: split into 2x2, 2x1, 1x2, 1x1 blocks of A and B.
    This gets complex. Let's try a different approach.
    
    Alternative: Use the known rank-21 decomposition for 3x3x3 if we can find it.
    Since I don't have it memorized, let me try a random search for a low-rank
    decomposition using the structure of the tensor.
    """
    pass

def _random_search(max_rank, n_iters=1000, deadline=None):
    """
    Try to find a low-rank decomposition via random initialization + optimization.
    This is unlikely to work well for exact decompositions, but let's try.
    """
    n, m, p = DIMENSIONS
    rank = max_rank
    best_result = None
    best_err = float('inf')
    
    for it in range(n_iters):
        if deadline and time.time() > deadline:
            break
        
        # Random initialization with small values
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Compute the error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += (val - 1.0)**2
        
        if err < best_err:
            best_err = err
            if err < 1e-9:
                # Found an exact decomposition!
                U_list = [[int(round(U[i, r])) if abs(U[i, r] - round(U[i, r])) < 1e-6 
                           else round(U[i, r] * 2) / 2 
                           for r in range(rank)] for i in range(n*m)]
                V_list = [[int(round(V[i, r])) if abs(V[i, r] - round(V[i, r])) < 1e-6 
                           else round(V[i, r] * 2) / 2 
                           for r in range(rank)] for i in range(m*p)]
                W_list = [[int(round(W[i, r])) if abs(W[i, r] - round(W[i, r])) < 1e-6 
                           else round(W[i, r] * 2) / 2 
                           for r in range(rank)] for i in range(p*n)]
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                return best_result
    
    return best_result

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank exact decomposition for 3x3 * 3x8 matrix multiplication.
    Starts with schoolbook rank-72, then tries to improve.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s margin before 150s kill
    
    # Start with schoolbook
    best = _schoolbook_decomposition()
    best_rank = n * m * p  # 72
    
    # Try to find a better decomposition
    # Known bounds: M(3,3,8) <= M(3,3,3) + M(3,3,5) or other splits
    # M(3,3,3) <= 21 (best known), M(3,3,5) <= ?
    # M(3,3,8) <= 21 + 21 + M(3,3,2) = 21 + 21 + 8 = 50? 
    # But we need to verify the split works.
    
    # Let's try splitting p=8 into 3+3+2:
    # C = A * B where B = [B1 | B2 | B3] with B1: 3x3, B2: 3x3, B3: 3x2
    # C = [A*B1 | A*B2 | A*B3]
    # Each A*Bi is 3x3 * 3xpi
    # For pi=3: use rank-21 decomposition (if we have it)
    # For pi=2: use schoolbook rank-6
    
    # Since I don't have the rank-21 tensors, let me try a different approach.
    # Use the fact that M(3,3,8) can be bounded by various methods.
    
    # For now, let's try random search for rank < 72
    for target_rank in [60, 55, 50, 45]:
        if time.time() > deadline - 30:
            break
        result = _random_search(target_rank, n_iters=50, deadline=deadline - 10)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"]):
                best = result
                best_rank = target_rank
                break
    
    return best
# EVOLVE-BLOCK-END
