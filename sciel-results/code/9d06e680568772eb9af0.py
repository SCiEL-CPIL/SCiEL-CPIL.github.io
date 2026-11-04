import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, 10s buffer

    # Schoolbook solution as fallback
    def get_schoolbook():
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

    # Known lower bounds for matrix multiplication rank
    # For 2x4 x 4x7, the rank is at least max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # But there are better bounds. Let's start from a reasonable lower bound.
    
    # The trivial upper bound is n*m*p = 56
    # A better upper bound comes from rectangular Strassen-like algorithms
    # For now, let's try to find a decomposition with rank < 56
    
    # We'll use a structured approach: try to find a decomposition
    # where the tensor can be expressed as a sum of rank-1 tensors.
    
    # The matrix multiplication tensor T_{i,j,k}^{i'} = delta_{i,i'} 
    # can be written as: sum_c U_{i,j,c} V_{j,k,c} W_{k,i',c} = delta_{i,i'}
    
    # One approach: use the fact that for small matrices, we can sometimes
    # find decompositions using bilinear or trilinear forms.
    
    # Let's try a different strategy: use integer linear algebra.
    # We want to find integer (or half-integer) matrices U, V, W such that
    # for all i, j, k, i': sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i', c] = delta[i, i']
    
    # This is a system of polynomial equations in the entries of U, V, W.
    # For small ranks, we can try to solve this using a combination of
    # random initialization and least-squares, but with exact rational arithmetic.
    
    # Let's try ranks from a lower bound up to 56.
    # Lower bound: for 2x4 x 4x7, the rank is at least 28 (from max dimension product)
    # but actually the rank must be at least n*p + m - 1 or similar bounds.
    # Let's just try from 20 to 56.
    
    best_result = None
    best_rank = n * m * p + 1  # Start above schoolbook
    
    # Try to find a decomposition with rank r
    for r in range(20, n * m * p + 1):
        if time.time() > deadline:
            break
            
        # For each rank, try multiple random restarts
        max_restarts = 30 if r < 35 else 10
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                    break
                    
            # Random initialization with small integers
            # Use values from {-1, 0, 1} for U and V
            U = np.random.randint(-1, 2, size=(n * m, r))
            V = np.random.randint(-1, 2, size=(m * p, r))
            
            # For each (k, i') pair, we need to find W[k*n+i', :] such that
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i', c] = delta[i, i']
            # for all i, j
            
            # This gives us a linear system for each (k, i')
            W = np.zeros((p * n, r))
            valid = True
            
            for k in range(p):
                for ip in range(n):
                    # Build the linear system: A * w = b
                    # where A has rows for each (i, j) pair
                    # A[i*m+j, c] = U[i*m+j, c] * V[j*p+k, c]
                    # b[i*m+j] = 1 if i == ip else 0
                    
                    A = np.zeros((n * m, r))
                    b = np.zeros(n * m)
                    
                    for i in range(n):
                        for j in range(m):
                            A[i * m + j, :] = U[i * m + j, :] * V[j * p + k, :]
                            b[i * m + j] = 1.0 if i == ip else 0.0
                    
                    # Solve A * w = b using least squares
                    # But we need w to be half-integer, so let's solve and then round
                    w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + ip, :] = w
            
            # Round W to nearest half-integer
            W_rounded = np.round(W * 2) / 2.0
            
            # Check if the rounding is close enough
            if not np.allclose(W, W_rounded, atol=1e-8):
                continue
            
            # Verify the decomposition exactly
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i', c] = delta[i, i']
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for ip in range(n):
                            sum_val = 0.0
                            for c in range(r):
                                sum_val += U[i * m + j, c] * V[j * p + k, c] * W_rounded[k * n + ip, c]
                            expected = 1.0 if i == ip else 0.0
                            if not np.isclose(sum_val, expected, atol=1e-10):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            if valid:
                # Found a valid decomposition!
                # Convert to lists, ensuring integer/half-integer values
                U_list = []
                for row in U:
                    U_list.append([int(x) if x == int(x) else float(x) for x in row])
                V_list = []
                for row in V:
                    V_list.append([int(x) if x == int(x) else float(x) for x in row])
                W_list = []
                for row in W_rounded:
                    W_list.append([int(x) if x == int(x) else float(x) for x in row])
                
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # If we found a valid decomposition with this rank, we could break
        # but let's continue to try to find a better one if time permits
        # Actually, since we're iterating from low to high, the first valid one is optimal
        # for this search strategy. But we might miss lower ranks if random init fails.
        # Let's keep searching if time allows.

    # If we didn't find anything better, return schoolbook
    return get_schoolbook()
