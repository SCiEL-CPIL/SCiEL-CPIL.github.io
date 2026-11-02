"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from collections import defaultdict

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # T should equal 1 iff i,j,k form a valid multiplication term, else 0
    # Actually, the tensor for matrix multiplication is:
    # M_{ijk} = 1 if (i,j,k) corresponds to (row, col_of_first, col_of_second) where
    # the product contributes to output[i,k] via input1[i,j]*input2[j,k]
    # So M_{ijk} = 1 for all valid (i,j,k), and the decomposition is:
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    # Wait, that's not right. Let me reconsider.
    
    # The standard convention: we want
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor M has entries M[i,j,k] such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # No wait. Let me re-read the task.
    
    # "W[k*n+i,r]" - so W is indexed by (k, i) where k is the column of B (and row of output)
    # and i is the row of A (and row of output).
    
    # The decomposition should satisfy:
    # For the matrix multiplication tensor, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    
    # Actually, the multiplication tensor T is defined such that:
    # T[i, j, k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # And the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i,j,k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank(n, m, p, target_rank, deadline):
    """Try to find a decomposition of the given rank using random initialization + refinement."""
    n, m, p = n, m, p
    nm = n * m
    mp = m * p
    pn = p * n
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We want U (nm x r), V (mp x r), W (pn x r) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is equivalent to: the 3-tensor T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r[i*m+j] = U[i*m+j, r], etc.
    
    # Use iterative refinement: start with random or structured init, then optimize
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    np.random.seed(42)
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Initialize with small random values
        U = np.random.randn(nm, target_rank) * 0.1
        V = np.random.randn(mp, target_rank) * 0.1
        W = np.random.randn(pn, target_rank) * 0.1
        
        # Iterative refinement: alternating least squares style
        # For fixed V and W, solve for U:
        # For each (i,j), we need sum_r U[i*m+j, r] * (V[:,r] * W[:,r] mapped to (k)) = 1 for all k
        # This is: U[i*m+j, :] @ diag(V[j*p+k, r] * W[k*n+i, r] for each k) ... 
        # Actually, let's think of it as: for each (i,j), the vector over k is:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
        # Let A_{k,r} = V[j*p+k, r] * W[k*n+i, r]
        # Then we need U[i*m+j, :] @ A = 1 (vector of ones of length p)
        # Similarly for V and W.
        
        for iteration in range(50):
            if time.time() > deadline:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    A = V[j*p:(j+1)*p, :] * W[0:p*n:(n), :]  # This is wrong indexing
                    # W[k*n+i] for k in range(p)
                    W_sub = np.array([W[k*n + i, :] for k in range(p)])
                    A = V[j*p:(j+1)*p, :] * W_sub
                    # Solve U[i*m+j, :] @ A = 1
                    # A is p x r, we want a vector x of length r such that x @ A = ones(p)
                    # i.e., A.T @ x = ones(p)
                    try:
                        sol, residuals, rank_a, sv = np.linalg.lstsq(A.T, np.ones(p), rcond=None)
                        U[i*m + j, :] = sol
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # A[i, r] = U[i*m+j, r] * W[k*n+i, r] for i in range(n)
                    U_sub = np.array([U[i*m + j, :] for i in range(n)])
                    W_sub = np.array([W[k*n + i, :] for i in range(n)])
                    A = U_sub * W_sub
                    try:
                        sol, residuals, rank_a, sv = np.linalg.lstsq(A.T, np.ones(n), rcond=None)
                        V[j*p + k, :] = sol
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # A[j, r] = U[i*m+j, r] * V[j*p+k, r] for j in range(m)
                    U_sub = np.array([U[i*m + j, :] for j in range(m)])
                    V_sub = np.array([V[j*p + k, :] for j in range(m)])
                    A = U_sub * V_sub
                    try:
                        sol, residuals, rank_a, sv = np.linalg.lstsq(A.T, np.ones(m), rcond=None)
                        W[k*n + i, :] = sol
                    except:
                        pass
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        err += (s - 1.0) ** 2
            if err < best_error:
                best_error = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if best_error < 1e-12:
                break
    
    if best_error < 1e-9 and best_U is not None:
        # Try to round to integers/half-integers
        U_round = np.round(best_U * 2) / 2
        V_round = np.round(best_V * 2) / 2
        W_round = np.round(best_W * 2) / 2
        
        # Check if rounded version works
        if _verify(U_round.tolist(), V_round.tolist(), W_round.tolist(), n, m, p, tol=1e-6):
            return U_round.tolist(), V_round.tolist(), W_round.tolist()
        
        # If rounding fails, check if the unrounded version is close enough
        # and try integer rounding
        U_int = np.round(best_U)
        V_int = np.round(best_V)
        W_int = np.round(best_W)
        if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p, tol=1e-6):
            return U_int.tolist(), V_int.tolist(), W_int.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Start with schoolbook
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try ranks from low to high
    # Known lower bounds suggest rank >= 40 or so for 3x4x8
    # Try from 40 up to schoolbook rank
    for rank in range(40, min(best_rank, 60)):
        if time.time() > deadline:
            break
        
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            break  # Found a valid decomposition, return it
    
    return best_result
# EVOLVE-BLOCK-END
