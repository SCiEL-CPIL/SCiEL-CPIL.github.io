"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_fallback = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fallback = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fallback = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][col] = 1
                V_fallback[j * p + k][col] = 1
                W_fallback[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_fallback, "V": V_fallback, "W": W_fallback}
    best_rank = rank_schoolbook
    
    # Try to find a lower-rank decomposition using structured search
    # For 2x2x14, we can try to bundle the k-dimension using linear combinations
    
    # Strategy: For each (i,j) pair in U, we need to represent the tensor.
    # The tensor T[i,j,k] = delta (for matrix multiplication)
    # Actually, for matrix multiplication (A 2x2) * (B 2x14) = (C 2x14):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 if this term contributes, which is always for valid indices.
    
    # Let's try a different approach: use the fact that we can have fractional coefficients
    # and try to find a decomposition with rank < 56
    
    # Try rank from 28 up to 55
    for target_rank in range(28, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try to find a decomposition of this rank
        # Use a simple heuristic: try to pair up k-values
        
        # For 2x2x14, one approach is to use the fact that:
        # We can write the multiplication as a sum of rank-1 tensors
        # Each rank-1 tensor is U[i,j] * V[j,k] * W[k,i]
        
        # Let's try a structured approach:
        # Group the 14 k-values into pairs and use Strassen-like ideas
        
        if target_rank >= 28 and target_rank < 56:
            # Try to find a valid decomposition
            success = _try_find_decomposition(n, m, p, target_rank, deadline)
            if success:
                best_rank = target_rank
                # Extract the result
                U, V, W = _get_current_decomposition()
                best_result = {"U": U, "V": V, "W": W}
                if target_rank <= 30:  # Early exit for very good results
                    break
    
    return best_result

def _try_find_decomposition(n, m, p, target_rank, deadline):
    """Try to find a rank-`target_rank` decomposition for n x m x p matrix multiplication."""
    # This is a placeholder for actual search logic
    # For now, return False to indicate not found
    # In a real implementation, this would use linear algebra or search
    
    # Quick check: is target_rank even possible?
    # Lower bound: max(n*m, m*p, p*n) / something... 
    # Actually, rank >= max(n*m, m*p, p*n) is not true
    # Rank >= n*p is a lower bound for matrix multiplication
    # For 2x2x14, lower bound is at least 2*14 = 28
    
    if target_rank < n * p:
        return False
    
    # Try a simple construction for specific ranks
    # For rank 28-55, we need to find if a decomposition exists
    
    # One known approach: use the fact that 2x2 multiplication has rank 7 (Strassen)
    # But here we have 2x2 * 2x14, which is different
    
    # Let's try to use a block approach:
    # Split the 14 columns into 7 pairs
    # For each pair of k-values, use a 2x2x2 multiplication (rank 7)
    # Total rank: 7 * 7 = 49
    
    if target_rank == 49:
        return _construct_49_rank(n, m, p)
    
    if target_rank == 42:
        return _construct_42_rank(n, m, p)
    
    # Try other ranks with a simple search
    return _search_for_rank(n, m, p, target_rank, deadline)

def _construct_49_rank(n, m, p):
    """Construct a rank-49 decomposition using 7 pairs of k-values with Strassen."""
    # Split 14 k-values into 7 pairs: (0,1), (2,3), ..., (12,13)
    # For each pair, use a 2x2x2 Strassen decomposition (rank 7)
    
    U = [[0] * 49 for _ in range(n * m)]  # 4x49
    V = [[0] * 49 for _ in range(m * p)]  # 28x49
    W = [[0] * 49 for _ in range(p * n)]  # 28x49
    
    # Strassen for 2x2x2:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # For our case:
    # A is 2x2 (indices i,j)
    # B is 2x2 for each pair (indices j,k_pair)
    # C is 2x2 for each pair (indices i,k_pair)
    
    for pair_idx in range(7):
        k1 = 2 * pair_idx
        k2 = 2 * pair_idx + 1
        
        base_col = pair_idx * 7
        
        # M1: (a11+a22)(b11+b22)
        # U contribution: a11 and a22
        U[0][base_col] = 1  # i=0, j=0
        U[3][base_col] = 1  # i=1, j=1
        # V contribution: b11 and b22
        V[0 * p + k1][base_col] = 1  # j=0, k=k1
        V[0 * p + k2][base_col] = 1  # j=0, k=k2
        V[1 * p + k1][base_col] = 1  # j=1, k=k1
        V[1 * p + k2][base_col] = 1  # j=1, k=k2
        # W contribution: c11 and c22
        W[k1 * n + 0][base_col] = 1  # k=k1, i=0
        W[k1 * n + 1][base_col] = 1  # k=k1, i=1
        W[k2 * n + 0][base_col] = 1  # k=k2, i=0
        W[k2 * n + 1][base_col] = 1  # k=k2, i=1
        
        # M2: (a21+a22)b11
        U[2][base_col + 1] = 1  # i=1, j=0
        U[3][base_col + 1] = 1  # i=1, j=1
        V[0 * p + k1][base_col + 1] = 1  # j=0, k=k1
        W[k1 * n + 1][base_col + 1] = 1  # k=k1, i=1
        
        # M3: a11(b12-b22)
        U[0][base_col + 2] = 1  # i=0, j=0
        V[0 * p + k2][base_col + 2] = 1  # j=0, k=k2
        V[1 * p + k2][base_col + 2] = -1  # j=1, k=k2
        W[k2 * n + 0][base_col + 2] = 1  # k=k2, i=0
        
        # M4: a22(b21-b11)
        U[3][base_col + 3] = 1  # i=1, j=1
        V[1 * p + k1][base_col + 3] = 1  # j=1, k=k1
        V[0 * p + k1][base_col + 3] = -1  # j=0, k=k1
        W[k1 * n + 1][base_col + 3] = 1  # k=k1, i=1
        
        # M5: (a11+a12)b22
        U[0][base_col + 4] = 1  # i=0, j=0
        U[1][base_col + 4] = 1  # i=0, j=1
        V[1 * p + k2][base_col + 4] = 1  # j=1, k=k2
        W[k2 * n + 0][base_col + 4] = 1  # k=k2, i=0
        
        # M6: (a21-a11)(b11+b12)
        U[2][base_col + 5] = 1  # i=1, j=0
        U[0][base_col + 5] = -1  # i=0, j=0
        V[0 * p + k1][base_col + 5] = 1  # j=0, k=k1
        V[0 * p + k2][base_col + 5] = 1  # j=0, k=k2
        W[k1 * n + 1][base_col + 5] = 1  # k=k1, i=1
        W[k2 * n + 1][base_col + 5] = 1  # k=k2, i=1
        
        # M7: (a12-a22)(b21+b22)
        U[1][base_col + 6] = 1  # i=0, j=1
        U[3][base_col + 6] = -1  # i=1, j=1
        V[1 * p + k1][base_col + 6] = 1  # j=1, k=k1
        V[1 * p + k2][base_col + 6] = 1  # j=1, k=k2
        W[k1 * n + 0][base_col + 6] = 1  # k=k1, i=0
        W[k2 * n + 0][base_col + 6] = 1  # k=k2, i=0
    
    # Verify and store
    return _verify_and_store(U, V, W, n, m, p)

def _construct_42_rank(n, m, p):
    """Try to construct a rank-42 decomposition."""
    # This is more complex; return False for now
    return False

def _search_for_rank(n, m, p, target_rank, deadline):
    """Search for a decomposition of the given rank."""
    # Placeholder for actual search
    return False

def _verify_and_store(U, V, W, n, m, p):
    """Verify the decomposition and store it if valid."""
    # Check if the decomposition is valid
    # For each (i,j,k), sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(len(U[0])):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    
    # Store the valid decomposition
    global _current_U, _current_V, _current_W
    _current_U = U
    _current_V = V
    _current_W = W
    return True

def _get_current_decomposition():
    """Get the currently stored decomposition."""
    return _current_U, _current_V, _current_W

# Initialize global variables
_current_U = None
_current_V = None
_current_W = None
# EVOLVE-BLOCK-END
