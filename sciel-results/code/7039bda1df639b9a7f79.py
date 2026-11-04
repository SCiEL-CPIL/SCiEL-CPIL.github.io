"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import sys
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    
    def make_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank_school
    
    # Verify a candidate decomposition
    def verify(U, V, W, rank):
        """Check if U, V, W correctly decompose the multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check all entries: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True
    
    # Try to find lower rank decompositions
    best_U, best_V, best_W, best_rank = make_schoolbook()
    
    # Structured approach: exploit the small dimension n=2
    # The tensor has structure: T[i,j,k] = 1 for all valid (i,j,k)
    # We can try to find decompositions where U, V, W have special structure
    
    # Strategy 1: Try rank = n*m*p - d for small d, using structured perturbations
    # Strategy 2: Use the fact that for n=2, we can potentially merge some terms
    
    # Key insight: For matrix multiplication A(2x5) * B(5x6) = C(2x6),
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    
    # The tensor T has entries T[i,j,k] = 1 for i in {0,1}, j in {0..4}, k in {0..5}
    # Total 60 entries, each must equal sum of rank-1 terms
    
    # Let's try to find rank < 60 by exploiting linear dependencies
    
    # Approach: Try ranks from a lower bound up to schoolbook
    # Lower bound for matrix multiplication rank: at least max(n*m, m*p, p*n) is not quite right
    # Actually, a better lower bound: rank >= max(n, m, p) * something
    # For now, try from 40 to 60 (conservative)
    
    np.random.seed(42)
    
    # Try structured decompositions first
    # For n=2, we can try to use the structure that U has only 2 rows per "block"
    
    # Try rank values from 40 to 60
    for target_rank in range(40, rank_school + 1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations with structure
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize with structured random values
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Apply scaling
            U /= np.sqrt(target_rank)
            V /= np.sqrt(target_rank)
            W /= np.sqrt(target_rank)
            
            # Verify
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify(U_list, V_list, W_list, target_rank):
                best_U, best_V, best_W, best_rank = U_list, V_list, W_list, target_rank
                break
        
        if best_rank < target_rank:
            break
    
    # If we found something better, return it; otherwise return schoolbook
    if best_rank < rank_school:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        U, V, W, _ = make_schoolbook()
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
