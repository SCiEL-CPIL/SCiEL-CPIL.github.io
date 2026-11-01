"""Tensor rank minimization for 2x3 x 3x10 matrix multiplication."""
import numpy as np
import time
import itertools
from scipy.optimize import linprog
from scipy.linalg import norm

DIMENSIONS = [2, 3, 10]
DEADLINE = 115  # seconds, safety margin before 120s limit

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    schoolbook_rank = n * m * p  # 60
    
    # Known lower bound: max(n*m, m*p, n*p) = max(6, 30, 20) = 30
    # Also, rank >= max(n, m, p) * something...
    # For 2x3x10, the trivial decomposition is rank 60.
    
    # Strategy: Try to find a decomposition of rank r < 60
    # We'll use a randomized approach with linear algebra.
    
    # The matrix multiplication tensor T has components T[i,j,k] where
    # Z_{ik} = sum_j X_{ij} Y_{jk}
    # In tensor form: T_{i,j,k} is the coefficient such that
    # T = sum_r U_r \otimes V_r \otimes W_r
    # where U_r has entries U[i,j,r], V_r has entries V[j,k,r], W_r has entries W[k,i,r]
    
    # The trace convention: W[k*n+i, r]
    # So W is indexed as W[k*n + i, r]
    
    # Let's set up the problem:
    # We want to find rank r such that there exist U (n*m x r), V (m*p x r), W (p*n x r)
    # with integer or half-integer entries such that
    # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{...}
    
    # Actually, the tensor decomposition means:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # But wait, T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term
    # The matrix multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # No, that's not right.
    
    # Let me reconsider. The tensor for matrix multiplication M_{n,m,p} is defined as:
    # T[i,j,k] where i indexes rows of A and rows of C, j indexes cols of A and rows of B, k indexes cols of B and cols of C
    # T[i,j,k] = 1 (it's a 3-tensor with all entries 1 in the appropriate positions)
    
    # Actually, the standard tensor decomposition of matrix multiplication:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # The tensor T has entries T[i,j,k] = 1 for all valid (i,j,k)
    # And we want T = sum_r u_r \otimes v_r \otimes w_r where
    # u_r[i,j] = U[i,j,r], v_r[j,k] = V[j,k,r], w_r[k,i] = W[k,i,r]
    
    # So T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # With the given indexing:
    # U is (n*m) x rank, indexed as U[i*m+j, r]
    # V is (m*p) x rank, indexed as V[j*p+k, r]  
    # W is (p*n) x rank, indexed as W[k*n+i, r]
    
    # So T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # And T[i,j,k] = 1 for all i,j,k in the appropriate ranges.
    
    # Wait, that can't be right. If T[i,j,k] = 1 for all i,j,k, then we need
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # That's a very specific condition.
    
    # Actually, I think I'm confusing myself. Let me re-read the problem.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 10 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries that exactly decompose the matrix multiplication tensor"
    
    # So the tensor T for multiplying an n x m matrix by an m x p matrix to get an n x p matrix.
    # T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 (the coefficient of A_{ij} B_{jk} in C_{ik})
    
    # So yes, T[i,j,k] = 1 for all valid (i,j,k).
    
    # And we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a multiplicative decomposition.
    
    # The schoolbook solution has rank n*m*p = 60, with U[i*m+j, i*m+j] = 1, V[j*p+k, i*m+j] = 1, W[k*n+i, i*m+j] = 1, and zeros elsewhere.
    
    # Let's verify: for fixed (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # = U[i*m+j, i*m+j] * V[j*p+k, i*m+j] * W[k*n+i, i*m+j]
    # = 1 * 1 * 1 = 1. Good.
    
    # Now, can we do better than 60?
    
    # Known results: The rank of the 2x3x10 matrix multiplication tensor.
    # For small matrices, exact ranks are known.
    # 2x2x2 has rank 7 (Strassen).
    # 2x3x3, 3x3x3, etc. have known or bounded ranks.
    
    # For 2x3x10, I don't know the exact rank off the top of my head.
    # Let's try to find a decomposition with rank < 60 using a search approach.
    
    # Approach: Use random initialization and optimize
    # We want to minimize ||T - sum_r u_r \otimes v_r \otimes w_r||_F^2
    # where T[i,j,k] = 1 for all i,j,k
    
    # Let's set up the optimization.
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        return _schoolbook()
    
    # Try rank 55 first (a reasonable target)
    for target_rank in [55, 50, 45, 40, 35, 30]:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
        
        result = _try_decompose(n, m, p, target_rank, time.time())
        if result is not None:
            return result
    
    return _schoolbook()

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _try_decompose(n, m, p, target_rank, start_time):
    """Try to find a rank target_rank decomposition."""
    elapsed = time.time() - start_time
    if elapsed > 20:
        return None
    
    n_um = n * m
    n_vm = m * p
    n_wm = p * n
    
    # T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's use a randomized approach:
    # 1. Initialize U, V, W randomly with small values
    # 2. Use multiplicative updates to converge
    
    np.random.seed(42)
    
    best_error = float('inf')
    best_U = best_V = best_W = None
    
    for attempt in range(5):
        elapsed = time.time() - start_time
        if elapsed > 15:
            break
        
        # Initialize
        U = np.random.randn(n_um, target_rank) * 0.1
        V = np.random.randn(n_vm, target_rank) * 0.1
        W = np.random.randn(n_wm, target_rank) * 0.1
        
        # Use iterative optimization
        for iteration in range(200):
            elapsed = time.time() - start_time
            if elapsed > 15:
                break
            
            # Update W: For each (k,i), we want
            # sum_r V[j*p+k, r] * U[i*m+j, r] * W[k*n+i, r] = 1 for all j
            # This is: sum_r (sum_j' U[i*m+j', r] * delta_{j',j}) * V[j*p+k, r] * W[k*n+i, r]
            # Actually, let's think of it differently.
            
            # For fixed (k,i), define A_{r,j} = U[i*m+j, r] * V[j*p+k, r]
            # Then sum_r A_{r,j} * W[k*n+i, r] = 1 for all j
            # This is a linear system: W[k*n+i, :] dot A_{:,j} = 1 for all j
            
            # Actually, let me use a different approach.
            # Fix U and V, solve for W.
            # For each (k,i), we need:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
            
            # Let B_{j,r} = U[i*m+j, r] * V[j*p+k, r]
            # Then sum_r B_{j,r} * W[k*n+i, r] = 1 for all j
            # This is: B @ W[k*n+i, :] = 1 (vector of ones)
            
            # So W[k*n+i, :] = B^{-1} @ 1 (if B is invertible)
            # Or use least squares
            
            for k in range(p):
                for i in range(n):
                    B = np.zeros((m, target_rank))
                    for j in range(m):
                        for r in range(target_rank):
                            B[j, r] = U[i * m + j, r] * V[j * p + k, r]
                    
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(B, np.ones(m), rcond=None)[0]
                    except:
                        W[k * n + i, :] = np.linalg.pinv(B) @ np.ones(m)
            
            # Similarly update U and V
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, target_rank))
                    for i in range(n):
                        for r in range(target_rank):
                            B[i, r] = W[k * n + i, r] * U[i * m + j, r]
                    
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                    except:
                        V[j * p + k, :] = np.linalg.pinv(B) @ np.ones(n)
            
            for i in range(n):
                for j in range(m):
                    B = np.zeros((p, target_rank))
                    for k in range(p):
                        for r in range(target_rank):
                            B[k, r] = W[k * n + i, r] * V[j * p + k, r]
                    
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                    except:
                        U[i * m + j, :] = np.linalg.pinv(B) @ np.ones(p)
        
        # Check error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(target_rank))
                    error += (val - 1.0) ** 2
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        elapsed = time.time() - start_time
        if elapsed > 15 or error < 1e-8:
            break
    
    # Check if we can round to integers or half-integers
    if best_error < 1e-6 and best_U is not None:
        result = _try_round(best_U, best_V, best_W, n, m, p, target_rank)
        if result is not None:
            return result
    
    return None

def _try_round(U, V, W, n, m, p, target_rank):
    """Try to round the solution to integer or half-integer values."""
    n_um = n * m
    n_vm = m * p
    n_wm = p * n
    
    # Try rounding each matrix to nearest half-integer
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    # Check if the rounded solution is exact
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_rounded[i*m+j, r] * V_rounded[j*p+k, r] * W_rounded[k*n+i, r] for r in range(target_rank))
                error += (val - 1.0) ** 2
    
    if error < 1e-10:
        # Convert to list of lists with integer or half-integer entries
        def to_list(matrix):
            result = []
            for row in matrix:
                row_list = []
                for val in row:
                    # Check if it's an integer or half-integer
                    val = round(val * 2) / 2
                    if abs(val - round(val)) < 1e-10:
                        row_list.append(int(round(val)))
                    elif abs(val * 2 - round(val * 2)) < 1e-10:
                        row_list.append(val)
                    else:
                        row_list.append(val)
                result.append(row_list)
            return result
        
        return {"U": to_list(U_rounded), "V": to_list(V_rounded), "W": to_list(W_rounded)}
    
    return None
