"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE_SECONDS = 140  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor T[i,j,k] = 1 if (i,j,k) is a valid triple
    # For each column c: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal
    # 1 if this (i,j,k) corresponds to a multiplication, 0 otherwise.
    # Actually, the tensor for matrix multiplication M_{n,m,p} has:
    # T[i, j, k] = 1 for all valid (i,j,k) in [0,n) x [0,m) x [0,p)
    # and the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Convert to numpy for speed
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum over all columns
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                prod = u_val * v_val * w_val
                s = np.sum(prod)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _search_random_rank(n, m, p, target_rank, max_iters, deadline):
    """Try to find a decomposition of rank=target_rank with random search."""
    np.random.seed(42)
    random.seed(42)
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            return None
        
        # Generate random integer/half-integer entries in {-1, 0, 1} or {-2, -1, 0, 1, 2}
        # Try small values first
        max_val = 2
        U = np.random.randint(-max_val, max_val + 1, size=(n * m, target_rank))
        V = np.random.randint(-max_val, max_val + 1, size=(m * p, target_rank))
        W = np.random.randint(-max_val, max_val + 1, size=(p * n, target_rank))
        
        # Verify
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_structured_search(n, m, p, target_rank, deadline):
    """Try structured patterns for lower rank decompositions."""
    # For small matrices, try specific known patterns
    # The rank of M_{3,4,5} is known to be at most 3*4*5=60 (schoolbook)
    # But can it be lower? For 2x2x2, Strassen gives rank 7 < 8.
    # For 3x4x5, the minimal rank is not well-known but likely close to 60.
    
    # Try a few structured approaches
    # Approach: use the fact that some entries can be shared
    
    # For now, this is a placeholder for more sophisticated search
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    deadline = time.time() + DEADLINE_SECONDS
    
    # Start with schoolbook
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Search from lower ranks upward
    for target_rank in range(1, schoolbook_rank):
        if time.time() > deadline:
            break
        
        # Limit iterations per rank to stay within deadline
        remaining_time = deadline - time.time()
        if remaining_time < 5:  # Need at least 5 seconds left
            break
        
        max_iters = min(1000, int(remaining_time * 100))  # ~100 attempts per second
        if max_iters < 10:
            break
        
        result = _search_random_rank(n, m, p, target_rank, max_iters, deadline)
        if result is not None:
            best = result
            best_rank = target_rank
            # Found a lower rank, continue searching for even lower
            continue
    
    return best
# EVOLVE-BLOCK-END
