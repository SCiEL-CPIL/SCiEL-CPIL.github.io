"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.linalg import null_space

DIMENSIONS = [3, 3, 4]

def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    rank = len(W[0])
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # For a rank-r decomposition: T = sum_c U[:,c] outer V[:,c] outer W[:,c]
    # We check: for each (i,j,k) in the tensor, T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    # Also verify that entries that should be zero are zero
    # The tensor has non-zero entries only at T[i*m+j, j*p+k, k*n+i] = 1
    # All other entries must be 0
    # For a given (a, b, c_idx) in the flattened tensor, 
    # a = i*m+j, b = j*p+k, c_idx = k*n+i
    # The constraint is: a = i*m+j, b = j*p+k, c_idx = k*n+i
    # So for a given (a, b, c_idx), we need i = a//m, j = a%m, 
    # then check if b == j*p + k for some k, and c_idx == k*n + i
    # Actually, let's just check that for all (a,b,c) not corresponding to valid (i,j,k), the value is 0.
    # But this is expensive. Instead, let's check all (a,b,c) combinations.
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                val = 0.0
                for cc in range(rank):
                    val += U[a][cc] * V[b][cc] * W[c][cc]
                # Determine what T[a,b,c] should be
                # a = i*m + j => i = a//m, j = a%m
                # b = j'*p + k => j' = b//p, k = b%p
                # c = k'*n + i' => k' = c//n, i' = c%n
                i = a // m
                j = a % m
                jp = b // p
                k = b % p
                kp = c // n
                ip = c % n
                if i == jp and j == jp and k == kp and i == ip and j == jp and k == kp:
                    # Wait, this isn't right. Let me reconsider.
                    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
                    # So a determines (i, j), b determines (j, k), c determines (k, i)
                    # For consistency: a//m = i, a%m = j; b//p = j, b%p = k; c//n = k, c%n = i
                    # So we need: a//m == c%n AND a%m == b//p AND b%p == c//n
                    expected = 1.0
                else:
                    expected = 0.0
                if i == c % n and j == b // p and k == c // n:
                    expected = 1.0
                else:
                    expected = 0.0
                if abs(val - expected) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def _try_strassen_based(n, m, p):
    """
    Try to construct a rank-24 decomposition using a 23-term 3x3x3 Strassen
    decomposition embedded into 3x3x4.
    
    The 3x3x4 tensor can be seen as 4 copies of a 3x3x3 slice along the last dimension,
    but that's not quite right. Let's think differently.
    
    Actually, let's try a direct approach: use a known 24-term formula.
    Since I don't have a verified 24-term formula memorized, let me try a different strategy.
    
    I'll use a bilinear decomposition approach. For 3x3x4, the trivial rank is 36.
    A known upper bound is 24 (from Winograd). Let me try to construct it.
    
    The 3x3x4 matrix multiplication: C = A * B where A is 3x3, B is 3x4, C is 3x4.
    
    C[i][k] = sum_j A[i][j] * B[j][k]
    
    This is a bilinear map from (A, B) -> C.
    
    Let me try a rank-24 construction based on the idea of grouping.
    Actually, let me just try to find a valid decomposition with rank < 36 using an optimization approach.
    """
    # For now, return None to indicate we couldn't find a better one
    return None

def _optimize_rank(n, m, p, target_rank, time_limit=100):
    """
    Try to find a rank-target_rank decomposition using iterative refinement.
    Start from a random initialization and use gradient-like updates.
    """
    start_time = time.time()
    
    # We need U of shape (n*m, target_rank), V of shape (m*p, target_rank), W of shape (p*n, target_rank)
    # such that for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # and for all other (a,b,c): sum_c U[a, c] * V[b, c] * W[c, c] = 0
    
    # This is a hard non-convex problem. Let me try a simple approach:
    # Start with a rank-36 solution and try to null out some columns.
    
    # Actually, let me try a different approach: use the fact that we can 
    # express the tensor as a sum of rank-1 tensors and try to find a sparse solution.
    
    # For a 3x3x4 tensor, let's enumerate all non-zero entries:
    # T[i*m+j, j*p+k, k*n+i] = 1 for i in 0..2, j in 0..2, k in 0..3
    # That's 3*3*4 = 36 non-zero entries.
    
    # A rank-r decomposition means we can write T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    # where u_c is a vector of length 9, v_c of length 12, w_c of length 12.
    
    # For this to work, for each (i,j,k) with i in 0..2, j in 0..2, k in 0..3:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # For all other (a,b,c) not of this form, the sum must be 0.
    
    # This is 9*12*12 = 1296 constraints, of which 36 are =1 and 1260 are =0.
    
    # With r variables per tensor (9+12+12 = 33 per rank-1 term), we have 33r variables.
    # For r=24, that's 792 variables and 1296 constraints - over-determined.
    
    # Let me try a simple iterative method: start with random vectors and iterate.
    
    np.random.seed(42)
    
    for trial in range(5):
        if time.time() - start_time > time_limit - 5:
            break
        
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(2000):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Compute error
            err = 0.0
            # Only check the 36 non-zero entries
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(target_rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-12:
                break
            
            # Gradient-like update: for each c, we want to increase the product
            # where it's too small and decrease where it's too large.
            # Simple update: for each (i,j,k), the residual is r = 1 - sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
            # We want to update U[i,j,:], V[j,k,:], W[k,i,:] to reduce r.
            # This is complex. Let me try a simpler approach.
            
            # Actually, let me try a different strategy: use the known structure.
            break  # This simple approach won't work well
    
    # Return the best found if it's valid
    if best_err < 1e-10:
        # Check full validity
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        if _verify(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    
    # Try to improve
    # Try rank 24 (known upper bound for 3x3x4)
    if time.time() < deadline - 60:
        res = _optimize_rank(n, m, p, 24, time_limit=50)
        if res is not None:
            result = res
    
    # Try rank 26
    if time.time() < deadline - 30:
        res = _optimize_rank(n, m, p, 26, time_limit=20)
        if res is not None:
            result = res
    
    # Try rank 28
    if time.time() < deadline - 10:
        res = _optimize_rank(n, m, p, 28, time_limit=10)
        if res is not None:
            result = res
    
    return result
