"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from fractions import Fraction

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using numpy tensor contraction."""
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Reshape
    U_reshaped = U_np.reshape(n, m, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    W_reshaped = W_np.reshape(p, n, rank)
    
    # Compute R[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Use einsum for efficient contraction
    # First: A[j,k,c] = V[j,k,c] * W[k,i,c] doesn't work directly
    # Let's do it step by step
    # B[j,k,i] = sum_c V[j,k,c] * W[k,i,c]
    B = np.einsum('jkc,kic->jki', V_reshaped, W_reshaped)
    # R[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # = sum_c U[i,j,c] * B[j,k,i]
    R = np.einsum('ijc,jki->ijk', U_reshaped, B)
    
    # Check if R[i,j,k] ≈ 1 for all i,j,k
    return np.all(np.abs(R - 1.0) < tol)

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using fractions."""
    rank = len(U[0])
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                if total != 1:
                    return False
    return True

def _random_decomposition(n, m, p, rank, max_val=3):
    """Generate a random decomposition with small integer entries."""
    U = [[random.randint(-max_val, max_val) for _ in range(rank)] for _ in range(n*m)]
    V = [[random.randint(-max_val, max_val) for _ in range(rank)] for _ in range(m*p)]
    W = [[random.randint(-max_val, max_val) for _ in range(rank)] for _ in range(p*n)]
    return {"U": U, "V": V, "W": W}

def _structured_search(n, m, p, deadline, target_rank):
    """
    Search for a rank-target_rank decomposition using structured patterns.
    We exploit the symmetry of 3x3x8 matmul.
    """
    best_result = None
    
    # Try various structured patterns
    # Pattern 1: Block diagonal with 3x3 blocks
    # Pattern 2: Random with constraints
    
    # For 3x3x8, the schoolbook rank is 72.
    # Known lower bounds suggest rank >= 70 for 3x3x8? Actually not sure.
    # Let's try to find rank 70, 69, 68, ...
    
    for r in range(target_rank, 64, -1):
        if time.time() > deadline - 5:
            break
        
        # Try random search for this rank
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            decomp = _random_decomposition(n, m, p, r, max_val=2)
            
            # Fast verification first
            if _verify_fast(decomp["U"], decomp["V"], decomp["W"], n, m, p, tol=1e-6):
                # Exact verification
                if _verify_exact(decomp["U"], decomp["V"], decomp["W"], n, m, p):
                    best_result = decomp
                    return best_result, r
        
        if best_result:
            return best_result, r
    
    return None, None

def _iterative_deepening_search(n, m, p, deadline):
    """
    Search for minimal rank decomposition using iterative deepening.
    """
    # Start from a reasonable upper bound
    # Schoolbook is 72, try going down
    max_rank_to_try = 71
    min_rank_to_try = 60
    
    best_decomp = None
    best_rank = None
    
    # Try ranks from high to low
    for r in range(max_rank_to_try, min_rank_to_try - 1, -1):
        if time.time() > deadline - 10:
            break
        
        # Quick random search
        attempts = 0
        max_attempts = 20
        
        while attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            decomp = _random_decomposition(n, m, p, r, max_val=2)
            
            if _verify_fast(decomp["U"], decomp["V"], decomp["W"], n, m, p, tol=1e-6):
                if _verify_exact(decomp["U"], decomp["V"], decomp["W"], n, m, p):
                    best_decomp = decomp
                    best_rank = r
                    break
        
        if best_decomp:
            break
    
    return best_decomp, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Fallback: schoolbook solution (always valid)
    fallback = _schoolbook()
    
    # Try to find a better (lower rank) solution
    # First, try structured search with a tight deadline
    if time.time() < deadline - 20:
        decomp, rank = _iterative_deepening_search(n, m, p, deadline)
        if decomp and rank < n * m * p:
            # Found a better solution
            return decomp
    
    # Return the schoolbook solution as fallback
    return fallback
# EVOLVE-BLOCK-END
