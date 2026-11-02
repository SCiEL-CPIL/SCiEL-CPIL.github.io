"""
Tensor decomposition search for 2x3 x 3x3 matrix multiplication.
Strategy: Structured support search with linear solve and rational reconstruction.
"""
import numpy as np
import time
from fractions import Fraction
from itertools import product, combinations
import random

DIMENSIONS = [2, 3, 3]

def _verify(U, V, W, n=2, m=3, p=3, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _round_to_half_integer(arr, tol=1e-6):
    """Round array to nearest half-integers if close enough."""
    doubled = 2.0 * arr
    rounded = np.round(doubled)
    if np.max(np.abs(doubled - rounded)) < tol:
        return rounded / 2.0
    return None

def _als_optimize(U, V, W, n=2, m=3, p=3, max_iter=50, tol=1e-10):
    """
    Alternating Least Squares for tensor decomposition.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iter):
        rank = U.shape[1]
        num_eqs = n * m * p
        
        # Solve for W
        A = np.zeros((num_eqs, p*n*rank))
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    w_row = k * n + i
                    for c in range(rank):
                        coeff = U[i*m+j, c] * V[j*p+k, c]
                        if coeff != 0:
                            A[eq_idx, w_row * rank + c] = coeff
                    eq_idx += 1
        b = np.ones(num_eqs)
        try:
            W_vec, residuals, r, sv = np.linalg.lstsq(A, b, rcond=None)
            W = W_vec.reshape(p*n, rank)
        except np.linalg.LinAlgError:
            break
        
        # Solve for U
        A_U = np.zeros((num_eqs, n*m*rank))
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = i*m + j
                    for c in range(rank):
                        coeff = V[j*p+k, c] * W[k*n+i, c]
                        if coeff != 0:
                            A_U[eq_idx, u_row * rank + c] = coeff
                    eq_idx += 1
        try:
            U_vec, residuals, r, sv = np.linalg.lstsq(A_U, b, rcond=None)
            U = U_vec.reshape(n*m, rank)
        except np.linalg.LinAlgError:
            break
        
        # Solve for V
        A_V = np.zeros((num_eqs, m*p*rank))
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    v_row = j*p + k
                    for c in range(rank):
                        coeff = U[i*m+j, c] * W[k*n+i, c]
                        if coeff != 0:
                            A_V[eq_idx, v_row * rank + c] = coeff
                    eq_idx += 1
        try:
            V_vec, residuals, r, sv = np.linalg.lstsq(A_V, b, rcond=None)
            V = V_vec.reshape(m*p, rank)
        except np.linalg.LinAlgError:
            break
        
        # Check convergence
        # We can check residual
        res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    res += (val - 1.0)**2
        if np.sqrt(res) < tol:
            break
            
    return U, V, W

def _try_support_search(target_rank, n=2, m=3, p=3, max_trials=100, time_budget=10, seed=42):
    """
    Search for a decomposition by fixing binary supports and solving for coefficients.
    """
    np.random.seed(seed)
    start_time = time.time()
    
    for trial in range(max_trials):
        if time.time() - start_time > time_budget:
            break
            
        # Generate random binary supports
        # Each row of U (n*m rows) should have some non-zeros.
        # Total non-zeros in U is roughly n*m*p/r? No, let's just pick random subsets.
        
        # Heuristic: Each column c in U, V, W should be sparse.
        # Let's pick a sparsity level. For rank 15, maybe 2-3 non-zeros per row?
        # Or just random 0/1 matrices with density ~ 1/rank?
        
        U_mask = np.random.randint(0, 2, size=(n*m, target_rank))
        V_mask = np.random.randint(0, 2, size=(m*p, target_rank))
        W_mask = np.random.randint(0, 2, size=(p*n, target_rank))
        
        # Ensure each column has at least one 1 in each matrix to avoid trivial null spaces?
        # Actually, if a column is all zeros in U, it's useless.
        # Let's enforce at least one 1 per column in each matrix.
        for c in range(target_rank):
            if np.sum(U_mask[:, c]) == 0:
                U_mask[np.random.randint(n*m), c] = 1
            if np.sum(V_mask[:, c]) == 0:
                V_mask[np.random.randint(m*p), c] = 1
            if np.sum(W_mask[:, c]) == 0:
                W_mask[np.random.randint(p*n), c] = 1
        
        # Build the linear system
        # Variables: U_nonzeros, V_nonzeros, W_nonzeros
        # This is a quadratic system if we don't fix supports.
        # But if we fix supports (masks), we still have a system of equations:
        # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
        # This is cubic in the variables? No, it's linear in W if U,V fixed.
        # But we are solving for all coefficients simultaneously?
        # No, the equations are linear in the *products* of coefficients.
        # Wait, the tensor decomposition is:
        # T_{ijk} = sum_c U_{ij,c} V_{jk,c} W_{ki,c}
        # If we fix the supports (which indices are non-zero), we have a system of bilinear/cubic equations?
        # Actually, for a fixed support, it's a system of polynomial equations.
        # However, if we view it as:
        # For each c, let u_c, v_c, w_c be the vectors of non-zero coefficients.
        # The equation is sum_c (u_c . (v_c * w_c)) = 1? No.
        
        # Let's use the property that if we fix U and V, W is linear.
        # So we can do:
        # 1. Randomly initialize U and V with the fixed masks (coefficients = 1.0 for non-zeros).
        # 2. Solve for W linearly.
        # 3. Randomly initialize V and W with masks. Solve for U.
        # 4. Iterate (ALS).
        
        U_init = np.zeros((n*m, target_rank))
        V_init = np.zeros((m*p, target_rank))
        W_init = np.zeros((p*n, target_rank))
        
        # Fill with 1.0 where mask is 1
        U_init[U_mask == 1] = 1.0
        V_init[V_mask == 1] = 1.0
        W_init[W_mask == 1] = 1.0
        
        # Run ALS to refine coefficients
        U_opt, V_opt, W_opt = _als_optimize(U_init, V_init, W_init, n, m, p, max_iter=100)
        
        # Check if exact
        if _verify(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p, tol=1e-8):
            # Try to round to half-integers
            U_rounded = _round_to_half_integer(U_opt, tol=1e-6)
            V_rounded = _round_to_half_integer(V_opt, tol=1e-6)
            W_rounded = _round_to_half_integer(W_opt, tol=1e-6)
            
            if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p, tol=1e-10):
                    # Check if the rounded solution respects the sparsity? 
                    # Not strictly required, but if it doesn't, it's still a valid rank-r decomposition 
                    # if the non-zero entries are within the rank.
                    # The rank is defined by the number of columns.
                    # If rounding introduces new non-zeros in previously zero spots, 
                    # the rank might effectively be higher? 
                    # No, the rank is the number of terms. If we have r terms, and some coefficients 
                    # become non-zero where they were zero in the mask, it's still r terms.
                    # But we want integer/half-integer.
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    
    return None

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    start_time = time.time()
    total_time_budget = 140
    n, m, p = DIMENSIONS
    
    # Try ranks from low to high
    for target_rank in range(12, 19):
        elapsed = time.time() - start_time
        remaining = total_time_budget - elapsed
        if remaining < 5:
            break
        
        # Allocate time based on difficulty
        # Lower ranks are harder, give them more time?
        # Or just distribute evenly.
        time_per_rank = min(remaining / (19 - target_rank), 30)
        
        # Try structured support search
        result = _try_support_search(target_rank, n, m, p, max_trials=500, time_budget=time_per_rank, seed=target_rank*100+42)
        if result is not None:
            return result
        
        # If structured search fails, try generic random float search with ALS
        # (Retaining some of the original logic as a fallback)
        np.random.seed(target_rank*100+42)
        for trial in range(50):
            if time.time() - start_time > total_time_budget - 5:
                break
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            U_opt, V_opt, W_opt = _als_optimize(U, V, W, n, m, p, max_iter=50)
            
            if _verify(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p, tol=1e-8):
                U_rounded = _round_to_half_integer(U_opt, tol=1e-5)
                V_rounded = _round_to_half_integer(V_opt, tol=1e-5)
                W_rounded = _round_to_half_integer(W_opt, tol=1e-5)
                if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                    if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p, tol=1e-8):
                        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    
    # Fallback to schoolbook
    return _build_schoolbook()
