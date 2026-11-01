"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    # Start with schoolbook rank
    best_rank = n * m * p  # 42
    best_decomp = _schoolbook_decomp(n, m, p, best_rank)
    
    # The tensor T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # We need sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1 for all i,j,k
    
    # Strategy: try structured decompositions that exploit the fact that
    # matrix multiplication has algebraic structure (not just arbitrary tensors)
    
    # Try ranks from best_rank-1 down, with time-limited search
    for target_rank in range(best_rank - 1, max(20, best_rank - 20), -1):
        if time.time() > deadline - 5:
            break
        
        # Multiple attempts with different strategies
        for attempt in range(3):
            if time.time() > deadline - 3:
                break
            
            decomp = _try_rank_search(n, m, p, target_rank, attempt)
            if decomp is not None:
                best_rank = target_rank
                best_decomp = decomp
                break
    
    # Final polish: try to verify and clean up
    if best_decomp is not None:
        best_decomp = _clean_decomp(best_decomp, n, m, p)
    
    return best_decomp

def _schoolbook_decomp(n, m, p, rank):
    """Standard schoolbook decomposition with rank n*m*p."""
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

def _verify_decomp_fast(U, V, W, n, m, p):
    """Fast verification using numpy matrix operations."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For each r, the contribution is outer product structure
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(val - 1.0) > 1e-8:
                    return False
    return True

def _verify_decomp_exact(U, V, W, n, m, p):
    """Exact verification using fractions."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0, 1)
                for r in range(rank):
                    u_val = Fraction(str(U[i*m+j][r]))
                    v_val = Fraction(str(V[j*p+k][r]))
                    w_val = Fraction(str(W[k*n+i][r]))
                    val += u_val * v_val * w_val
                if val != 1:
                    return False
    return True

def _try_rank_search(n, m, p, target_rank, seed):
    """Try to find a rank-target_rank decomposition using structured search."""
    random.seed(seed * 1000 + target_rank * 7)
    
    # Initialize with a structured pattern that respects the algebraic structure
    U = [[0] * target_rank for _ in range(n*m)]
    V = [[0] * target_rank for _ in range(m*p)]
    W = [[0] * target_rank for _ in range(p*n)]
    
    # Use a hybrid initialization: some direct terms + some structured terms
    direct_terms = min(target_rank // 2, n * m * p)
    structured_terms = target_rank - direct_terms
    
    # Direct terms: standard basis
    for r in range(direct_terms):
        i = r % n
        j = (r // n) % m
        k = (r // (n * m)) % p
        U[i * m + j][r] = 1
        V[j * p + k][r] = 1
        W[k * n + i][r] = 1
    
    # Structured terms: try to create cancellations that reduce effective rank
    for r in range(direct_terms, target_rank):
        # Random but structured initialization
        i1 = random.randint(0, n-1)
        j1 = random.randint(0, m-1)
        k1 = random.randint(0, p-1)
        i2 = random.randint(0, n-1)
        j2 = random.randint(0, m-1)
        k2 = random.randint(0, p-1)
        
        # Create a term that might help with cancellations
        sign1 = random.choice([1, -1])
        sign2 = random.choice([1, -1])
        
        U[i1 * m + j1][r] = sign1
        V[j1 * p + k1][r] = sign1
        W[k1 * n + i1][r] = 1
        
        U[i2 * m + j2][r] = sign2
        V[j2 * p + k2][r] = sign2
        W[k2 * n + i2][r] = 1
    
    # Local search to find a valid decomposition
    for iteration in range(200):
        if _verify_decomp_fast(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
        # Modify a random entry with small changes
        matrix_choice = random.randint(0, 2)
        col = random.randint(0, target_rank-1)
        if matrix_choice == 0:
            idx = random.randint(0, n*m-1)
            U[idx][col] = random.randint(-2, 2)
        elif matrix_choice == 1:
            idx = random.randint(0, m*p-1)
            V[idx][col] = random.randint(-2, 2)
        else:
            idx = random.randint(0, p*n-1)
            W[idx][col] = random.randint(-2, 2)
    
    return None

def _clean_decomp(decomp, n, m, p):
    """Clean up the decomposition to ensure exact integer/half-integer values."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    
    # Round to nearest half-integer if close
    for i in range(len(U)):
        for j in range(len(U[0])):
            U[i][j] = round(U[i][j] * 2) / 2
            V[i][j] = round(V[i][j] * 2) / 2
            W[i][j] = round(W[i][j] * 2) / 2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
