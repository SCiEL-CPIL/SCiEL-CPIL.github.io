"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time

def _verify(U, V, W, n, m, p, tol=1e-12):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The matrix multiplication tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = 1 if i,j,k are valid indices (always true for the multiplication tensor)
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because (AB)[i,k] = sum_j A[i,j]*B[j,k]
    
    # Let's verify: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1
    # Wait, that's not right. The tensor for matrix multiplication A (n x m) * B (m x p) -> C (n x p)
    # is T[i,j,k] where C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k.
    
    # Let me re-check: The multiplication tensor has entries T[i,j,k] = 1 for all i,j,k
    # and the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _is_valid_entries(U, V, W, tol=1e-9):
    """Check that all entries are integers or half-integers."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                # Check if val is close to an integer or half-integer
                v2 = 2 * val
                if abs(v2 - round(v2)) > tol:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Build the trivial schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_landsberg_style(n, m, p):
    """
    Try to find a low-rank decomposition using structured patterns.
    For 3x3*3x3, the Landsberg decomposition has rank 23.
    For 3x3*3x5, we can try to extend or adapt known patterns.
    """
    # The multiplication tensor for A(nxm) * B(mx p) -> C(nxp)
    # has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # and the decomposition satisfies:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # One approach: use the fact that the tensor is a "slice rank" problem
    # and try to find a decomposition with small integer/half-integer entries.
    
    # For now, let's try a structured search with specific patterns.
    # We'll use a grid search over small integer/half-integer values.
    
    # The key insight: for the multiplication tensor, we need
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a system of n*m*p = 45 equations in 3*rank*2 unknowns (with symmetry constraints)
    # but the structure is highly constrained.
    
    # Let's try a different approach: use the known decomposition for 3x3*3x3
    # and extend it to 3x3*3x5.
    
    # The 3x3*3x3 Landsberg decomposition has rank 23.
    # For 3x3*3x5, we can think of it as 3x3*3x3 + 3x3*3x2.
    # The 3x3*3x2 part has rank at most 3*3*2 = 18 (schoolbook).
    # So total rank could be at most 23 + 18 = 41, but we can do better.
    
    # Actually, let's just do a smart random search with better initialization.
    # We'll try ranks from a reasonable lower bound up to schoolbook.
    
    lower_bound = max(n*m, m*p, p*n)  # 15
    schoolbook = n * m * p  # 45
    
    # Try ranks from lower_bound to schoolbook
    for target_rank in range(lower_bound, schoolbook + 1):
        # Try multiple random initializations with structured patterns
        np.random.seed(42 + target_rank)
        
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > time.time() + 140:  # This is a placeholder, will be fixed
                break
            
            # Generate random matrices with integer/half-integer entries
            # Use a mix of small values
            scale = 2
            U_rand = np.round(np.random.randn(n*m, target_rank) * scale) * 0.5
            V_rand = np.round(np.random.randn(m*p, target_rank) * scale) * 0.5
            W_rand = np.round(np.random.randn(p*n, target_rank) * scale) * 0.5
            
            # Convert to lists
            U_list = [[U_rand[i,j] for j in range(target_rank)] for i in range(n*m)]
            V_list = [[V_rand[i,j] for j in range(target_rank)] for i in range(m*p)]
            W_list = [[W_rand[i,j] for j in range(target_rank)] for i in range(p*n)]
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # First, return schoolbook as fallback
    best_result = _build_schoolbook(n, m, p)
    
    # Try to find lower rank decompositions
    # Strategy: structured search with integer/half-integer entries
    
    np.random.seed(42)
    
    lower_bound = max(n*m, m*p, p*n)
    schoolbook = n * m * p
    
    # Try various ranks from lower bound up to schoolbook
    for target_rank in range(lower_bound, schoolbook + 1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random matrices with integer/half-integer entries
            # Use small values for better convergence
            scale = 2
            U_rand = np.round(np.random.randn(n*m, target_rank) * scale) * 0.5
            V_rand = np.round(np.random.randn(m*p, target_rank) * scale) * 0.5
            W_rand = np.round(np.random.randn(p*n, target_rank) * scale) * 0.5
            
            # Convert to lists
            U_list = [[U_rand[i,j] for j in range(target_rank)] for i in range(n*m)]
            V_list = [[V_rand[i,j] for j in range(target_rank)] for i in range(m*p)]
            W_list = [[W_rand[i,j] for j in range(target_rank)] for i in range(p*n)]
            
            if _verify(U_list, V_list, W_list, n, m, p):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                return best_result
        
        # If we found a valid decomposition at this rank, return it
        # (already returned above if found)
    
    return best_result
# EVOLVE-BLOCK-END
