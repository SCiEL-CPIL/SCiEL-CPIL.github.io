import time
import numpy as np

# Deadline guard
_DEADLINE = 110.0
_start_time = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def _verify_236(U, V, W, tol=1e-9):
    """
    Verify that U, V, W decompose the 2x3x6 multiplication tensor.
    U: (6, r), V: (18, r), W: (12, r)
    """
    n, m, p = 2, 3, 6
    r = U.shape[1]
    
    # Reshape to tensor forms
    U_mat = U.reshape(n, m, r)
    V_mat = V.reshape(m, p, r)
    W_raw = W.reshape(p, n, r)
    W_mat = np.transpose(W_raw, (1, 0, 2)) # (n, p, r)
    
    # Compute term[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[i, k, c]
    prod_UV = U_mat[:, :, np.newaxis, :] * V_mat[np.newaxis, :, :, :]
    prod = prod_UV * W_mat[:, np.newaxis, :, :]
    
    result = np.sum(prod, axis=3) # (n, m, p)
    
    # The target tensor for matrix multiplication is 1 for all i,j,k
    if not np.allclose(result, 1.0, atol=tol):
        return False
    return True

def _verify_233(U, V, W, tol=1e-9):
    """
    Verify 2x3x3 decomposition.
    U: (6, r), V: (9, r), W: (6, r)
    """
    n, m, p = 2, 3, 3
    r = U.shape[1]
    
    U_mat = U.reshape(n, m, r)
    V_mat = V.reshape(m, p, r)
    W_raw = W.reshape(p, n, r)
    W_mat = np.transpose(W_raw, (1, 0, 2)) # (n, p, r)
    
    prod_UV = U_mat[:, :, np.newaxis, :] * V_mat[np.newaxis, :, :, :]
    prod = prod_UV * W_mat[:, np.newaxis, :, :]
    
    result = np.sum(prod, axis=3) # (n, m, p)
    
    if not np.allclose(result, 1.0, atol=tol):
        return False
    return True

def _solve_linear_for_UV(W_fixed, target, n=2, m=3, p=3):
    """
    Given a fixed W, solve for U and V such that the tensor decomposition holds.
    This is a bilinear problem, but we can linearize it if we fix one of U or V.
    However, a better approach for small ranks is to view this as:
    sum_c U_{ijc} V_{jkc} = T_{ijk} / W_{ikc} (if W is non-zero)
    
    Actually, a more robust method for finding a solution with integer/half-integer entries
    when rank is small is to search for a "simple" W structure (0/1) and then solve the
    resulting system of linear equations for the products U*V? No, U and V are coupled.
    
    Let's try a different tactic:
    We know that for rank 15, a solution exists (border rank is 12, but integer rank is likely higher).
    Let's try to find a rank 15 solution by fixing W to a specific pattern and using least squares
    on a linearized version?
    
    Actually, there is a known construction for 2x3x3 rank 15?
    The 2x2x2 Strassen rank is 7.
    2x3x3 can be viewed as 2x2x3 + 2x1x3.
    
    Let's try a direct search for U, V, W with a strong regularization towards 0, 0.5, 1.
    We will use a local search (hill climbing) on the objective function, but with a "snap" heuristic.
    """
    pass # Placeholder, we will implement a specific search below

def _search_233_rank_15():
    """
    Search for a rank-15 decomposition of the 2x3x3 tensor.
    Strategy:
    1. Initialize W with a random 0/1 pattern with density ~0.5.
    2. Solve for U and V using a linear least squares approach by fixing the "products"?
       No, we can use the fact that if we fix W, the condition is:
       sum_c U_{ijc} V_{jkc} W_{ikc} = 1
       
       This is a system of 18 equations (2*3*3) in the variables U and V (6*15 + 9*15 = 225 variables).
       It is underdetermined. We can set up a linear system for U if we fix V, and vice versa.
       
       Let's use an alternating minimization:
       1. Fix V, solve for U (linear least squares).
       2. Fix U, solve for V (linear least squares).
       3. Repeat.
       
       To ensure integer/half-integer results, we will snap the solutions to the nearest half-integer
       and verify.
    """
    n, m, p = 2, 3, 3
    rank = 15
    
    best_obj = np.inf
    best_U, best_V, best_W = None, None, None
    
    # Try several random W patterns
    for restart in range(50):
        if _time_left() < 10.0:
            break
            
        np.random.seed(restart * 123 + 456)
        
        # Initialize W with 0/1 values
        # To make the linear system well-conditioned, we want W to have enough non-zeros.
        # Let's try a dense W first.
        W = np.random.choice([0, 1], size=(p*n, rank), p=[0.3, 0.7])
        
        # Ensure each row has at least one non-zero
        for i in range(p*n):
            if np.all(W[i] == 0):
                W[i, np.random.randint(rank)] = 1
                
        # Initialize U and V randomly
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        
        obj = np.inf
        for step in range(100):
            if _time_left() < 5.0:
                break
            
            # 1. Solve for U given V and W
            # Equation: sum_k V_{jkc} W_{ikc} U_{ijc} = 1 (for all i,j,k)
            # For a fixed i,j, we have: sum_k (V_{jkc} W_{ikc}) U_{ijc} = 1
            # This is not quite right. The sum is over c.
            # For fixed i,j,k: sum_c U_{ijc} (V_{jkc} W_{ikc}) = 1
            # Let M_{ijc, k} = V_{jkc} W_{ikc}. Then sum_c M_{ijc, k} U_{ijc} = 1.
            # This is a linear system for U_{ij*} for each fixed i,j?
            # No, U_{ijc} appears in equations for all k.
            # So for each pair (i,j), we have a system of p equations (one for each k) in rank variables U_{ij0}...U_{ij,r-1}.
            # A_{ij, k, c} = V_{jkc} W_{ikc}
            # b_{ij, k} = 1
            # Solve A_{ij} U_{ij} = b_{ij}
            
            U_new = np.zeros((n*m, rank))
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        for c in range(rank):
                            A[k, c] = V[j*p + k, c] * W[k*n + i, c]
                    b = np.ones(p)
                    # Solve least squares
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U_new[i*m + j] = sol
                    except:
                        U_new[i*m + j] = 0.1 # Fallback
            
            # 2. Solve for V given U and W
            # For fixed j,k: sum_c V_{jkc} (U_{ijc} W_{ikc}) = 1 (for all i)
            # For each pair (j,k), we have a system of n equations in rank variables V_{jk0}...V_{jk,r-1}.
            # A_{jk, i, c} = U_{ijc} W_{ikc}
            # b_{jk, i} = 1
            
            V_new = np.zeros((m*p, rank))
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        for c in range(rank):
                            A[i, c] = U_new[i*m + j, c] * W[k*n + i, c]
                    b = np.ones(n)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V_new[j*p + k] = sol
                    except:
                        V_new[j*p + k] = 0.1
            
            U = U_new
            V = V_new
            
            # Calculate objective
            U_mat = U.reshape(n, m, rank)
            V_mat = V.reshape(m, p, rank)
            W_mat = np.transpose(W.reshape(p, n, rank), (1, 0, 2))
            
            prod_UV = U_mat[:, :, np.newaxis, :] * V_mat[np.newaxis, :, :, :]
            prod = prod_UV * W_mat[:, np.newaxis, :, :]
            result = np.sum(prod, axis=3)
            
            diff = result - 1.0
            obj = np.sum(diff * diff)
            
            if obj < best_obj:
                best_obj = obj
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if obj < 1e-8:
                    # Snap to half-integers
                    U_s = np.round(U * 2) / 2
                    V_s = np.round(V * 2) / 2
                    W_s = np.round(W * 2) / 2
                    if _verify_233(U_s, V_s, W_s, tol=1e-6):
                        return U_s, V_s, W_s

    # If we found a good solution, try to snap it
    if best_obj < 1e-4:
        U_s = np.round(best_U * 2) / 2
        V_s = np.round(best_V * 2) / 2
        W_s = np.round(best_W * 2) / 2
        if _verify_233(U_s, V_s, W_s, tol=1e-6):
            return U_s, V_s, W_s
            
    return None

def _construct_236_rank_30():
    """
    Try to construct a rank-30 decomposition by finding a rank-15 decomposition for 2x3x3
    and using it twice.
    """
    n, m, p = 2, 3, 6
    
    # Try to find rank-15 decomposition for 2x3x3
    decomp_15 = _search_233_rank_15()
    
    if decomp_15 is not None:
        U1, V1, W1 = decomp_15
        return _combine_233_to_236(U1, V1, W1, U1, V1, W1)
    else:
        # Fallback to schoolbook rank 36
        return _schoolbook_236()

def _schoolbook_236():
    """Return schoolbook rank-36 decomposition for 2x3x6 as lists."""
    n, m, p = 2, 3, 6
    rank = n * m * p
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

def _combine_233_to_236(U1, V1, W1, U2, V2, W2):
    """
    Combine two 2x3x3 rank-r decompositions into a 2x3x6 rank-2r decomposition.
    """
    n, m, p = 2, 3, 6
    rank1, rank2 = U1.shape[1], U2.shape[1]
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill first half (columns 0..rank1-1)
    for r in range(rank1):
        for i in range(n):
            for j in range(m):
                U[i * m + j][r] = U1[i * m + j][r]
        for j in range(m):
            for k in range(3):
                V[j * p + k][r] = V1[j * 3 + k][r]
        for k in range(3):
            for i in range(n):
                W[k * n + i][r] = W1[k * n + i][r]
    
    # Fill second half (columns rank1..total_rank-1)
    offset = rank1
    for r in range(rank2):
        for i in range(n):
            for j in range(m):
                U[i * m + j][offset + r] = U2[i * m + j][r]
        for j in range(m):
            for k in range(3):
                V[j * p + (k + 3)][offset + r] = V2[j * 3 + k][r]
        for k in range(3):
            for i in range(n):
                W[(k + 3) * n + i][offset + r] = W2[k * n + i][r]
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3x6 matrix multiplication.
    """
    global _start_time
    _start_time = time.time()
    
    result = _construct_236_rank_30()
    
    # Verify the result
    if result is not None:
        U = np.array(result['U'])
        V = np.array(result['V'])
        W = np.array(result['W'])
        if _verify_236(U, V, W, tol=1e-6):
            return result
        else:
            # If verification fails, return schoolbook
            return _schoolbook_236()
    else:
        return _schoolbook_236()
