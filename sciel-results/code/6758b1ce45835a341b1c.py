"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_linear(rank, n, m, p, seed=42):
    """
    Try to find a rank-r decomposition using linear algebra.
    
    Key idea: Fix two of the three matrices and solve linearly for the third.
    Use iterative refinement: fix V and W, solve for U; fix U and W, solve for V; etc.
    
    The constraint is: for all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    If we fix V and W, then for each (i,j), the constraint becomes:
    sum_k [sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]] = 1 for each k
    
    Actually, for fixed V and W, each (i,j,k) gives a linear constraint on U[i*m+j,:]:
    sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
    
    This is a linear system in the r entries of U[i*m+j,:].
    """
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.random(p * n, rank)  # Use non-negative init sometimes
    
    for iteration in range(500):
        # Solve for U: for each (i,j), solve linear system
        for i in range(n):
            for j in range(m):
                a_idx = i * m + j
                # Constraints: for each k, sum_c U[a_idx,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                A = np.zeros((p, rank))
                b = np.ones(p)
                for k in range(p):
                    A[k] = V[j * p + k] * W[k * n + i]
                # Solve A @ U[a_idx] = b
                try:
                    U[a_idx] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Solve for V: for each (j,k), solve linear system
        for j in range(m):
            for k in range(p):
                v_idx = j * p + k
                # Constraints: for each i, sum_c U[i*m+j,c] * V[v_idx,c] * W[k*n+i,c] = 1
                A = np.zeros((n, rank))
                b = np.ones(n)
                for i in range(n):
                    A[i] = U[i * m + j] * W[k * n + i]
                try:
                    V[v_idx] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Solve for W: for each (k,i), solve linear system
        for k in range(p):
            for i in range(n):
                w_idx = k * n + i
                # Constraints: for each j, sum_c U[i*m+j,c] * V[j*p+k,c] * W[w_idx,c] = 1
                A = np.zeros((m, rank))
                b = np.ones(m)
                for j in range(m):
                    A[j] = U[i * m + j] * V[j * p + k]
                try:
                    W[w_idx] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        
        # Check convergence
        if iteration % 50 == 0:
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        max_err = max(max_err, abs(s - 1.0))
            if max_err < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _try_rank_strassen_based(rank, n, m, p, seed=42):
    """
    Try to construct a decomposition based on Strassen's 2x2x2 algorithm.
    
    For 2x2x2, Strassen uses 7 multiplications:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    For our 2x2x11 case, we have 11 output columns. The idea is to use
    a shared Strassen-like structure but allow the intermediate multipliers
    to be linear combinations that work for all 11 k-values.
    
    Actually, let's think about this more carefully. We need:
    For each k in 0..10:
      C[0,0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]  (wait, no)
    
    Actually the matrix multiplication is:
    (2x2) * (2x11) = (2x11)
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    This is 2*11 = 22 products, but with Strassen-like sharing we can do better.
    """
    rng = np.random.RandomState(seed)
    
    # Use the linear approach as a starting point
    return _try_rank_linear(rank, n, m, p, seed)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Try ranks from a reasonable lower bound
    # Lower bound considerations:
    # - We have 4 U entries, 22 V entries, 22 W entries
    # - 44 constraints total
    # - Each rank term provides 1 constraint satisfaction
    # - But the structure is bilinear, so it's more complex
    # 
    # Known results: 2x2x2 has rank 7 (Strassen)
    # For 2x2x11, a trivial bound is 7*11 = 77, but we can share
    # 
    # Let's try from rank 20 up to 44 (schoolbook is 44)
    schoolbook_rank = n * m * p  # 44
    
    best_result = None
    
    for rank in range(20, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple seeds
        for seed in [42, 123, 456, 789, 1000]:
            if time.time() > deadline - 5:
                break
            
            result = _try_rank_linear(rank, n, m, p, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
        
        # If we found a valid result at this rank, return it
        if best_result is not None and rank == 20:
            break
    
    # Fallback: schoolbook decomposition
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
