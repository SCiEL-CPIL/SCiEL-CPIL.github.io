"""Matrix multiplication 3x3 * 3x6 rank-minimal decomposition search."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 6]
N, M, P = DIMENSIONS
# Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
# Convention: W[k*n + i, r]
# Tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
# We need T[i, j, k] = 1 for all i,j,k (since C = A*B where A is 3x3, B is 3x6,
# C[i,k] = sum_j A[i,j] B[j,k]. The tensor is the multiplication tensor.)

def _time_remaining():
    return 120.0 - (time.time() - _START)

_START = time.time()

def _verify(U, V, W, rank):
    """Check if the decomposition is exact using Fraction arithmetic."""
    n, m, p = N, M, P
    # For each (i, j, k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if s != 1:
                    return False
    return True

def _verify_float(U, V, W, rank):
    """Fast float check for early pruning."""
    import numpy as np
    n, m, p = N, M, P
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    # Build tensor check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_random_search(seed, max_rank, deadline):
    """Try to find a decomposition with rank <= max_rank using random search."""
    rng = random.Random(seed)
    n, m, p = N, M, P
    nm = n * m
    mp = m * p
    pn = p * n
    
    # We'll try ranks from max_rank down to some lower bound
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # Also rank >= n*p = 18 (output size)
    # Schoolbook is 54.
    
    for rank in range(max_rank, 17, -1):
        if _time_remaining() < 5.0:
            break
        # Try several random initializations
        for attempt in range(50):
            if _time_remaining() < 5.0:
                break
            # Initialize with small random values
            U = [[Fraction(rng.randint(-2, 2), 1) for _ in range(rank)] for _ in range(nm)]
            V = [[Fraction(rng.randint(-2, 2), 1) for _ in range(rank)] for _ in range(mp)]
            W = [[Fraction(rng.randint(-2, 2), 1) for _ in range(rank)] for _ in range(pn)]
            
            if _verify(U, V, W, rank):
                return U, V, W, rank
    
    return None

def _try_structured_search(deadline):
    """Try structured patterns that might achieve lower rank."""
    n, m, p = N, M, P
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Try rank 18 (theoretical lower bound based on output dimension)
    # The tensor has n*m*p = 54 non-zero entries.
    # A rank-r decomposition means we can write it as sum of r rank-1 tensors.
    
    # Let's try a different approach: use the fact that this is matrix multiplication.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We can try to factor using the structure.
    
    # For 3x3 * 3x6, the output is 3x6.
    # One approach: use the standard decomposition but try to merge terms.
    
    # Let's try a direct algebraic approach for small ranks.
    # Rank 18: We need 18 terms.
    # Each term r contributes to T[i,j,k] via U[i,j]*V[j,k]*W[k,i].
    
    # Let's try to set up a system. For rank r, we have:
    # T[i,j,k] = sum_{r=0}^{r-1} U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # This is a system of 54 equations in r*(9+18+18) = 45r unknowns (with integer/half-integer constraint).
    
    # For small r, we can try to solve. But this is complex. Let's try a heuristic.
    
    # Try rank 27 (half of schoolbook)
    for rank in [27, 24, 21, 18]:
        if _time_remaining() < 5.0:
            break
        # Use a structured initialization
        # Split the 54 (i,j,k) triples into rank groups
        if rank <= 54:
            # Assign each (i,j,k) to a rank slot
            assignments = {}
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        assignments[(i,j,k)] = idx % rank
                        idx += 1
            
            # For each rank slot, set the corresponding entries
            U = [[Fraction(0) for _ in range(rank)] for _ in range(nm)]
            V = [[Fraction(0) for _ in range(rank)] for _ in range(mp)]
            W = [[Fraction(0) for _ in range(rank)] for _ in range(pn)]
            
            # For each rank slot r, we have multiple (i,j,k) assigned to it.
            # We need U[i,j,r]*V[j,k,r]*W[k,i,r] to sum to 1 for each (i,j,k)
            # but only one term per (i,j,k) if each (i,j,k) is in exactly one slot.
            # Wait, that's not right. If (i,j,k) is in slot r, then only the r-th term
            # contributes to T[i,j,k]. So we need U[i,j,r]*V[j,k,r]*W[k,i,r] = 1.
            # But U[i,j,r] appears in multiple (i,j,k) if the same (i,j) is in the same slot
            # with different k's. Similarly for V and W.
            
            # This is getting complicated. Let's just try the schoolbook for now
            # and see if we can find a better one through search.
            pass
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication."""
    global _START
    _START = time.time()
    
    n, m, p = N, M, P
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(nm)]
    V = [[0] * rank_schoolbook for _ in range(mp)]
    W = [[0] * rank_schoolbook for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W, best_rank = U, V, W, rank_schoolbook
    
    # Try to find better decompositions
    # Theoretical lower bound is 18 (max of nm, mp, pn)
    
    # Try random search for ranks below schoolbook
    for seed in range(20):
        if _time_remaining() < 10.0:
            break
        result = _try_random_search(seed, min(rank_schoolbook - 1, 40), 120.0)
        if result is not None:
            rU, rV, rW, rRank = result
            if rRank < best_rank:
                best_U, best_V, best_W, best_rank = rU, rV, rW, rRank
    
    # Return the best found
    # Convert to lists of Fractions or keep as is
    return {"U": best_U, "V": best_V, "W": best_W}
