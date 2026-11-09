"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition of matrix multiplication."""
    rank = len(U[0])
    # The tensor T[i,j,k] = 1 for all i,j,k (matrix mult tensor)
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _try_rank(rank, n, m, p, deadline):
    """Try to find a decomposition of given rank."""
    # If rank is the trivial rank, we know the solution
    if rank == n * m * p:
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
                    
        return U.tolist(), V.tolist(), W.tolist()

    # For lower ranks, do a very quick random search
    # This is unlikely to succeed for exact integer solutions in <1s, 
    # but prevents hanging and allows the loop to proceed to 45.
    if time.time() > deadline:
        return None
        
    # Simple heuristic: try a few random integer matrices
    for _ in range(20): 
        if time.time() > deadline:
            return None
        U = np.random.randint(-2, 3, size=(n * m, rank))
        V = np.random.randint(-2, 3, size=(m * p, rank))
        W = np.random.randint(-2, 3, size=(p * n, rank))
        
        # Fast check using numpy einsum
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        
        try:
            # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            prod = np.einsum('ijr,jkr,kin->ijk', U_t, V_t, W_t)
            if np.all(prod == 1):
                return U.tolist(), V.tolist(), W.tolist()
        except Exception:
            pass
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with trivial rank 45
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # But actual minimum is likely higher
    
    # Try ranks from 15 upward
    for rank in range(15, best_rank + 1):
        if time.time() > deadline:
            break
            
        result = _try_rank(rank, n, m, p, deadline)
        if result is not None:
            best_rank = rank
            best_U, best_V, best_W = result
            if rank == 15:  # Hit lower bound
                break
    
    # If no improvement found, use trivial
    if best_U is None:
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
                    
        best_U = U.tolist()
        best_V = V.tolist()
        best_W = W.tolist()
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
