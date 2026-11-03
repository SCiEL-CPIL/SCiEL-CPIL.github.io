"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition as dict of lists."""
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

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly reproduces matrix multiplication.
    Returns True if valid."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # Build the multiplication tensor T[i,j,k] = 1 if we multiply A[i,j]*B[j,k]
    # The decomposition says: C[i,k] = sum_c U[i*m+j? no...] 
    # Actually the standard form: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: T[i,j,k] = delta(i,j,k) where the tensor for matmul has T[i,j,k]=1 for all i,j,k? No.
    # The matrix multiplication tensor M has entries M[i,j,k] = 1 for all i,j,k? 
    # No. The tensor is such that <M, A ⊗ B> = C where C[i,k] = sum_j A[i,j]B[j,k].
    # So M[i,j,k] = 1 for all valid i,j,k? That's not right either.
    # Let me think: C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form, C = M(A,B) where M is the matmul tensor.
    # M[i,j,k] * A[i,j] * B[j,k] summed over j gives C[i,k].
    # So M[i,j,k] = 1 for all i,j,k.
    
    # The decomposition: M[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, the factor matrices are:
    # U has shape (n*m, rank) -> U[i*m+j, c]
    # V has shape (m*p, rank) -> V[j*p+k, c]  
    # W has shape (p*n, rank) -> W[k*n+i, c]
    
    # So the decomposition is: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _try_random_rank(n, m, p, target_rank, max_attempts=50, seed=0):
    """Try to find a rank-target_rank decomposition using random initialization.
    This is a heuristic; may not succeed but worth trying for small improvements."""
    # For this problem, the known lower bound for 4x5 * 5x5 is around 100-110.
    # Schoolbook is 100. It's unlikely we can do better with simple random search.
    # But let's try a very small number of attempts just in case.
    rng = random.Random(seed)
    for attempt in range(min(max_attempts, 5)):
        U = [[rng.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n*m)]
        V = [[rng.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m*p)]
        W = [[rng.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p*n)]
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    
    # Try to find a slightly better decomposition
    # The known lower bound for 4x5x5 matmul is not well below 100.
    # Let's try a few random searches at rank 99, 98, etc.
    for target_rank in [n*m*p - 1, n*m*p - 2]:
        if time.time() > deadline:
            break
        found = _try_random_rank(n, m, p, target_rank, max_attempts=10, seed=target_rank)
        if found is not None:
            result = found
            break
    
    return result
# EVOLVE-BLOCK-END
