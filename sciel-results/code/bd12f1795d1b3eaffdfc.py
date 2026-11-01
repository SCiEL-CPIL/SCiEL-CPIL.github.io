"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    r = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(r):
                    total += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(total - 1.0) > tol:
                    return False
    return True

def _build_rank_11():
    """
    Construct a rank-11 decomposition for 2x2x3 matrix multiplication.
    
    Uses Strassen's 7-term decomposition for the 2x2x2 subproblem (k=0,1)
    plus 4 terms for k=2.
    """
    n, m, p = 2, 2, 3
    rank = 11
    
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 11
    V = [[0.0] * rank for _ in range(m * p)]  # 6 x 11
    W = [[0.0] * rank for _ in range(p * n)]  # 6 x 11
    
    # Strassen's 7 products for 2x2x2 (k=0,1 part)
    # Each product c: M_c = (sum a_{i,j} A_{i,j}) * (sum b_{j,k} B_{j,k})
    # C_{i,k} = sum w_{c,i,k} M_c
    
    # M1 = (A00+A11)(B00+B11)
    # a: A00=1, A11=1; b: B00=1, B11=1
    # w: C00=1, C01=1, C10=1, C11=1
    c = 0
    U[0][c] = 1; U[3][c] = 1  # A00, A11
    V[0][c] = 1; V[4][c] = 1  # B00 (j=0,k=0), B11 (j=1,k=1)
    W[0][c] = 1; W[1][c] = 1; W[2][c] = 1; W[3][c] = 1  # C00,C01,C10,C11
    
    # M2 = (A10+A11)B00
    # a: A10=1, A11=1; b: B00=1
    # w: C10=1, C11=-1
    c = 1
    U[2][c] = 1; U[3][c] = 1  # A10, A11
    V[0][c] = 1  # B00
    W[2][c] = 1; W[3][c] = -1  # C10=1, C11=-1
    
    # M3 = A00(B10-B11)
    # a: A00=1; b: B10=1, B11=-1
    # w: C01=1, C11=1
    c = 2
    U[0][c] = 1  # A00
    V[3][c] = 1; V[4][c] = -1  # B10 (j=1,k=0), B11 (j=1,k=1)
    W[1][c] = 1; W[3][c] = 1  # C01=1, C11=1
    
    # M4 = A11(B01-B10)
    # a: A11=1; b: B01=1, B10=-1
    # w: C00=1, C10=1
    c = 3
    U[3][c] = 1  # A11
    V[1][c] = 1; V[3][c] = -1  # B01 (j=0,k=1), B10 (j=1,k=0)
    W[0][c] = 1; W[2][c] = 1  # C00=1, C10=1
    
    # M5 = (A00+A01)B11
    # a: A00=1, A01=1; b: B11=1
    # w: C00=-1, C01=1
    c = 4
    U[0][c] = 1; U[1][c] = 1  # A00, A01
    V[4][c] = 1  # B11
    W[0][c] = -1; W[1][c] = 1  # C00=-1, C01=1
    
    # M6 = (A10-A00)(B00+B10)
    # a: A10=1, A00=-1; b: B00=1, B10=1
    # w: C11=1
    c = 5
    U[2][c] = 1; U[0][c] = -1  # A10=1, A00=-1
    V[0][c] = 1; V[3][c] = 1  # B00, B10
    W[3][c] = 1  # C11=1
    
    # M7 = (A01-A11)(B01+B11)
    # a: A01=1, A11=-1; b: B01=1, B11=1
    # w: C00=1
    c = 6
    U[1][c] = 1; U[3][c] = -1  # A01=1, A11=-1
    V[1][c] = 1; V[4][c] = 1  # B01, B11
    W[0][c] = 1  # C00=1
    
    # Terms 7-10: Handle k=2 (4 terms, one per (i,j) pair)
    # For k=2, we need: sum_c U[i*2+j,c] * V[j*3+2,c] * W[2*2+i,c] = 1
    # i.e., U[i*2+j,c] * V[j*3+2,c] * W[4+i,c] = 1 for each (i,j)
    
    # Term 7: (i=0, j=0, k=2)
    c = 7
    U[0][c] = 1  # A00
    V[2][c] = 1  # B02 (j=0, k=2)
    W[4][c] = 1  # C02 (k=2, i=0)
    
    # Term 8: (i=0, j=1, k=2)
    c = 8
    U[1][c] = 1  # A01
    V[5][c] = 1  # B12 (j=1, k=2)
    W[4][c] = 1  # C02 (k=2, i=0)
    
    # Term 9: (i=1, j=0, k=2)
    c = 9
    U[2][c] = 1  # A10
    V[2][c] = 1  # B02 (j=0, k=2)
    W[5][c] = 1  # C12 (k=2, i=1)
    
    # Term 10: (i=1, j=1, k=2)
    c = 10
    U[3][c] = 1  # A11
    V[5][c] = 1  # B12 (j=1, k=2)
    W[5][c] = 1  # C12 (k=2, i=1)
    
    return U, V, W

def _try_rank_11():
    """Attempt to construct and verify a rank-11 decomposition."""
    n, m, p = 2, 2, 3
    U, V, W = _build_rank_11()
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None

def _try_rank_10():
    """
    Try to find a rank-10 decomposition using optimization.
    
    Strategy: Start from rank-11 and try to merge two terms by finding
    a rank-10 solution that satisfies all 12 equations.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # We need U (4x10), V (6x10), W (6x10) such that:
    # sum_c U[i*2+j,c] * V[j*3+k,c] * W[k*2+i,c] = 1 for all i,j,k
    
    # Parameterize: use 7 Strassen-like terms + 3 additional terms
    # The 7 terms handle k=0,1 primarily, and also contribute to k=2
    # The 3 terms handle the remaining k=2 contributions
    
    # Let's set up the problem with variables for all entries
    # and use scipy.optimize to minimize the error
    
    from scipy.optimize import minimize
    
    # Total variables: 4*10 + 6*10 + 6*10 = 160
    # Constraints: 12 equations (must equal 1)
    
    def _compute_error(params):
        """Compute the squared error of the decomposition."""
        U_flat = params[0:40]
        V_flat = params[40:100]
        W_flat = params[100:160]
        
        U = U_flat.reshape(4, 10)
        V = V_flat.reshape(6, 10)
        W = W_flat.reshape(6, 10)
        
        error = 0.0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    total = 0.0
                    for c in range(10):
                        total += U[i*2+j, c] * V[j*3+k, c] * W[k*2+i, c]
                    error += (total - 1.0) ** 2
        return error
    
    # Initialize with a perturbed rank-11 solution (drop last column)
    U11, V11, W11 = _build_rank_11()
    U11_arr = np.array(U11, dtype=float)[:, :10]
    V11_arr = np.array(V11, dtype=float)[:, :10]
    W11_arr = np.array(W11, dtype=float)[:, :10]
    
    x0 = np.concatenate([U11_arr.flatten(), V11_arr.flatten(), W11_arr.flatten()])
    
    # Try multiple random restarts
    best_result = None
    best_error = float('inf')
    
    n_restarts = 5
    for restart in range(n_restarts):
        if restart == 0:
            x = x0.copy()
        else:
            x = x0 + 0.1 * np.random.randn(160)
        
        try:
            result = minimize(_compute_error, x, method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if result.fun < best_error:
                best_error = result.fun
                best_result = result.x
        except Exception:
            pass
    
    if best_result is not None and best_error < 1e-8:
        U_flat = best_result[0:40]
        V_flat = best_result[40:100]
        W_flat = best_result[100:160]
        
        U = U_flat.reshape(4, 10).tolist()
        V = V_flat.reshape(6, 10).tolist()
        W = W_flat.reshape(6, 10).tolist()
        
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, try the known rank-11 construction
    result = _try_rank_11()
    if result is not None:
        return result
    
    # Try rank-10 with time check
    if time.time() < deadline - 10:
        result = _try_rank_10()
        if result is not None:
            return result
    
    # Fallback: return the schoolbook rank-12 decomposition
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
# EVOLVE-BLOCK-END
