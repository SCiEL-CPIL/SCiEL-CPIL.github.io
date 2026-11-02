"""Matrix multiplication 2x2 by 2x5 - rank-minimal tensor decomposition."""
import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 2, 5]

def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i1 in range(n):
        for r in range(m):
            for j2 in range(p):
                i = i1 * m + r
                j = r * p + j2
                k = i1 * n + j2
                T[i, j, k] = 1.0
    return T

def _check_decomposition(U, V, W, T, tol=1e-6):
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    err = np.max(np.abs(recon - T))
    return err < tol

def _is_half_integer(mat, tol=1e-6):
    return np.allclose(2 * mat, np.round(2 * mat), atol=tol)

def _build_rank18():
    """Construct the known rank-18 decomposition."""
    n, m, p = DIMENSIONS
    rank = 18
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1; W[col1*n+1, r] = 1
        
        r = base + 1
        U[2, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1
        W[col0*n+1, r] = 1; W[col1*n+1, r] = -1
        
        r = base + 2
        U[0, r] = 1
        V[0*p+col1, r] = 1; V[1*p+col1, r] = -1
        W[col1*n+0, r] = 1; W[col1*n+1, r] = 1
        
        r = base + 3
        U[3, r] = 1
        V[1*p+col0, r] = 1; V[0*p+col0, r] = -1
        W[col0*n+0, r] = 1; W[col0*n+1, r] = 1
        
        r = base + 4
        U[0, r] = 1; U[1, r] = 1
        V[1*p+col1, r] = 1
        W[col0*n+0, r] = -1; W[col1*n+0, r] = 1
        
        r = base + 5
        U[2, r] = 1; U[0, r] = -1
        V[0*p+col0, r] = 1; V[0*p+col1, r] = 1
        W[col1*n+1, r] = 1
        
        r = base + 6
        U[1, r] = 1; U[3, r] = -1
        V[1*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    r = 14
    U[0, r] = 1; V[0*p+4, r] = 1; W[4*n+0, r] = 1
    r = 15
    U[1, r] = 1; V[1*p+4, r] = 1; W[4*n+0, r] = 1
    r = 16
    U[2, r] = 1; V[0*p+4, r] = 1; W[4*n+1, r] = 1
    r = 17
    U[3, r] = 1; V[1*p+4, r] = 1; W[4*n+1, r] = 1
    
    return U, V, W

def _try_rank17_algebraic():
    """
    Try to find rank-17 by sharing one term between Strassen blocks.
    
    Key idea: The 2x2x5 tensor can be viewed as having structure where
    we use 7 Strassen terms for cols 0-1, and for cols 2-4 we use a 
    modified 5-term block that shares one U-vector with the first block.
    
    Alternative: Use a different partition. Instead of (0,1)+(2,3)+(4),
    try (0,1,2) + (3,4) with a 9-term block for 3 cols and 8 for 2 cols,
    or find a 17-term structure directly.
    
    Let's try: fix the first 7 terms as Strassen for cols 0,1.
    Then for the remaining 10 terms, we need to handle cols 2,3,4.
    The residual after 7 Strassen terms has specific structure.
    """
    n, m, p = DIMENSIONS
    T = _build_target()
    R = 17
    
    # Fix first 7 terms as standard Strassen for cols 0,1
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Standard Strassen for 2x2 matrix multiply (cols 0,1)
    # C00 = (A00+A11)(B00+B11) - A11(B10-B00) - A00(B01-B11) + (A01-A11)(B10+B11)
    # C01 = A00(B01-B11) + (A00+A01)B11
    # C10 = A11(B10-B00) + (A10+A11)(B00-B10)
    # C11 = (A10-A00)(B00+B01) + (A00+A11)(B00+B11) - (A10+A11)(B00-B10) - A00(B01-B11)
    
    # Term 0: (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[1,1] = 1  # V[0*p+0], V[1*p+1]
    W[0,0] = 1; W[1,1] = 1  # W[0*n+0], W[1*n+1]
    
    # Term 1: (A10+A11)(B00-B10)
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1  # V[0*p+0]
    W[2,1] = 1; W[3,1] = -1  # W[1*n+0], W[1*n+1]
    
    # Term 2: A00*(B01-B11)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1  # V[0*p+1], V[1*p+1]
    W[1,2] = 1; W[3,2] = 1  # W[0*n+1], W[1*n+1]
    
    # Term 3: A11*(B10-B00)
    U[3,3] = 1
    V[5,3] = 1; V[0,3] = -1  # V[1*p+0], V[0*p+0]
    W[0,3] = 1; W[2,3] = 1  # W[0*n+0], W[1*n+0]
    
    # Term 4: (A00+A01)*B11
    U[0,4] = 1; U[1,4] = 1
    V[7,4] = 1  # V[1*p+1]
    W[0,4] = -1; W[1,4] = 1  # W[0*n+0], W[0*n+1]
    
    # Term 5: (A10-A00)*(B00+B01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1  # V[0*p+0], V[0*p+1]
    W[3,5] = 1  # W[1*n+1]
    
    # Term 6: (A01-A11)*(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[5,6] = 1; V[7,6] = 1  # V[1*p+0], V[1*p+1]
    W[0,6] = 1  # W[0*n+0]
    
    # Compute residual after 7 terms
    recon7 = np.einsum('ir,jr,kr->ijk', U[:, :7], V[:, :7], W[:, :7])
    residual = T - recon7
    
    # Now we need 10 more terms to cover the residual
    # The residual should be zero for k=0,1 (cols 0,1 done)
    # and non-zero for k=2,3,4,5,6,7,8,9 (cols 2,3,4)
    
    # Let's check what the residual looks like
    # For k in {2,3,4,5,6,7,8,9}, residual[i,j,k] should be the 
    # contribution from cols 2,3,4
    
    # Strategy: Use a 10-term decomposition for the 2x2x3 sub-problem (cols 2,3,4)
    # But we only have 10 terms, and the minimum rank for 2x2x3 is... let's compute
    
    # Actually, let's try a different approach: use 7 terms for cols 0,1 (Strassen)
    # and 10 terms for cols 2,3,4. But 10 is more than needed for 2x2x3.
    # The issue is that the U-vectors for cols 2,3,4 might need to be different
    # from those used for cols 0,1.
    
    # Let's try: use 7 Strassen for cols 0,1, and then for cols 2,3,4 use
    # a structure where some U-vectors are shared.
    
    # Alternative: Try to find a 17-term decomposition where the U-matrix
    # has rank 4 (which it must, since we have 4 input elements).
    
    # Let's set up the problem differently. We'll parameterize the last 10 terms
    # and solve for them.
    
    # For the last 10 terms (indices 7-16), let's fix V and W to have a 
    # specific sparse structure and solve for U.
    
    # Structure for cols 2,3,4:
    # We need: C[i1, j2] = sum_r U[i1*m+r, 7+r'] * V[r*p+j2, 7+r'] * W[i1*n+j2, 7+r']
    # for j2 in {2,3,4}
    
    # Let's try: for the last 10 terms, use a pattern where:
    # - Terms 7-9 handle col 2 (3 terms)
    # - Terms 10-12 handle col 3 (3 terms) 
    # - Terms 13-16 handle col 4 (4 terms)
    # But this gives 10 terms total, which is what we have.
    
    # Actually, let's try a smarter approach. The 2x2x3 matrix multiply
    # (A is 2x2, B is 2x3) has tensor rank at most 2*3 = 6 (schoolbook)
    # but could be less. Let's try rank 6 for the 2x2x3 part.
    
    # If we can do cols 2,3,4 in 6 terms, total would be 7+6 = 13.
    # But that's probably not possible. The minimum rank for 2x2x3 is likely 8 or 9.
    
    # Let's just try to solve the system numerically.
    
    # Fix V and W for the last 10 terms with a specific pattern, solve for U.
    
    # Pattern: For cols 2,3,4, use a structure similar to Strassen but adapted.
    
    # Let's try: 10 terms where V has entries only in rows 2,3,4 (col 2)
    # rows 5,6,7 (col 3) and rows 8,9 (col 4).
    
    # Actually, let's just do a full ALS optimization for the last 10 terms.
    
    U_try = U.copy()
    V_try = V.copy()
    W_try = W.copy()
    
    # Initialize last 10 terms
    np.random.seed(42)
    for r in range(7, 17):
        U_try[:, r] = np.random.uniform(-1, 1, size=n*m)
        V_try[:, r] = np.random.uniform(-1, 1, size=m*p)
        W_try[:, r] = np.random.uniform(-1, 1, size=p*n)
    
    # ALS for last 10 terms only
    for it in range(100):
        # Update U for last 10 terms
        for i in range(n*m):
            b = residual[i, :, :].flatten()
            A = np.zeros((m*p * p*n, 10))
            for r in range(10):
                outer = np.outer(V_try[:, 7+r], W_try[:, 7+r])
                A[:, r] = outer.flatten()
            try:
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U_try[i, 7:17] = x
            except:
                pass
        
        # Update V for last 10 terms
        for j in range(m*p):
            b = residual[:, j, :].transpose(1, 0).flatten()
            A = np.zeros((n*m * p*n, 10))
            for r in range(10):
                outer = np.outer(U_try[:, 7+r], W_try[:, 7+r])
                A[:, r] = outer.flatten()
            try:
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V_try[j, 7:17] = x
            except:
                pass
        
        # Update W for last 10 terms
        for k in range(p*n):
            b = residual[:, :, k].flatten()
            A = np.zeros((n*m * m*p, 10))
            for r in range(10):
                outer = np.outer(U_try[:, 7+r], V_try[:, 7+r])
                A[:, r] = outer.flatten()
            try:
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W_try[k, 7:17] = x
            except:
                pass
        
        # Check convergence
        recon_try = np.einsum('ir,jr,kr->ijk', U_try, V_try, W_try)
        err_try = np.max(np.abs(recon_try - T))
        if err_try < 1e-8:
            # Try to round to half-integers
            U_r = np.round(2 * U_try) / 2
            V_r = np.round(2 * V_try) / 2
            W_r = np.round(2 * W_try) / 2
            if _check_decomposition(U_r, V_r, W_r, T):
                return U_r, V_r, W_r
            if _is_half_integer(U_try) and _is_half_integer(V_try) and _is_half_integer(W_try):
                if _check_decomposition(U_try, V_try, W_try, T):
                    return U_try, V_try, W_try
    
    return None

def _try_rank17_full_als(max_iters=30, attempts=20):
    """Full ALS for rank 17 from random initializations."""
    n, m, p = DIMENSIONS
    T = _build_target()
    R = 17
    start_time = time.time()
    deadline = start_time + 45
    
    for attempt in range(attempts):
        if time.time() > deadline:
            break
        
        np.random.seed(attempt * 1000 + 42)
        U = np.random.randint(-2, 3, size=(n*m, R)).astype(float)
        V = np.random.randint(-2, 3, size=(m*p, R)).astype(float)
        W = np.random.randint(-2, 3, size=(p*n, R)).astype(float)
        
        for r in range(R):
            if np.all(U[:, r] == 0): U[0, r] = 1
            if np.all(V[:, r] == 0): V[0, r] = 1
            if np.all(W[:, r] == 0): W[0, r] = 1
        
        for it in range(max_iters):
            # ALS step for U
            U_new = np.zeros_like(U)
            for i in range(n*m):
                b = T[i].flatten()
                A = np.zeros((m*p * p*n, R))
                for r in range(R):
                    outer = np.outer(V[:, r], W[:, r])
                    A[:, r] = outer.flatten()
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i] = x
                except:
                    U_new[i] = U[i]
            U = U_new
            
            # ALS step for V
            V_new = np.zeros_like(V)
            for j in range(m*p):
                b = T[:, j, :].transpose(1, 0).flatten()
                A = np.zeros((n*m * p*n, R))
                for r in range(R):
                    outer = np.outer(U[:, r], W[:, r])
                    A[:, r] = outer.flatten()
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j] = x
                except:
                    V_new[j] = V[j]
            V = V_new
            
            # ALS step for W
            W_new = np.zeros_like(W)
            for k in range(p*n):
                b = T[:, :, k].flatten()
                A = np.zeros((n*m * m*p, R))
                for r in range(R):
                    outer = np.outer(U[:, r], V[:, r])
                    A[:, r] = outer.flatten()
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k] = x
                except:
                    W_new[k] = W[k]
            W = W_new
            
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.max(np.abs(recon - T))
            if err < 1e-5:
                U_rounded = np.round(2 * U) / 2
                V_rounded = np.round(2 * V) / 2
                W_rounded = np.round(2 * W) / 2
                if _check_decomposition(U_rounded, V_rounded, W_rounded, T):
                    return U_rounded, V_rounded, W_rounded
                if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                    if _check_decomposition(U, V, W, T):
                        return U, V, W
    
    return None

def _try_rank17_hybrid():
    """
    Hybrid approach: Use 7 Strassen terms for cols 0,1 and a structured 
    10-term decomposition for cols 2,3,4 with specific sparsity patterns.
    """
    n, m, p = DIMENSIONS
    T = _build_target()
    R = 17
    
    # First 7 terms: Strassen for cols 0,1
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[1,1] = 1
    W[0,0] = 1; W[1,1] = 1
    
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = -1
    
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1
    
    U[3,3] = 1
    V[5,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1
    
    U[0,4] = 1; U[1,4] = 1
    V[7,4] = 1
    W[0,4] = -1; W[1,4] = 1
    
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    U[1,6] = 1; U[3,6] = -1
    V[5,6] = 1; V[7,6] = 1
    W[0,6] = 1
    
    # For cols 2,3,4, try a 10-term structure
    # Let's use a pattern where we share some U-vectors
    
    # Try: Terms 7-16 for cols 2,3,4
    # Use a structure where V entries are sparse
    
    # Col 2: 3 terms
    # Col 3: 3 terms  
    # Col 4: 4 terms
    
    # Term 7: A00 * B02
    U[0,7] = 1; V[2,7] = 1; W[2,7] = 1  # W[0*n+2]
    
    # Term 8: A01 * B12
    U[1,8] = 1; V[7,8] = 1; W[2,8] = 1  # W[0*n+2]
    
    # Term 9: A10 * B02
    U[2,9] = 1; V[2,9] = 1; W[4,9] = 1  # W[1*n+2]
    
    # Term 10: A11 * B12
    U[3,10] = 1; V[7,10] = 1; W[4,10] = 1  # W[1*n+2]
    
    # Col 3:
    # Term 11: A00 * B03
    U[0,11] = 1; V[3,11] = 1; W[3,11] = 1  # W[0*n+3]
    
    # Term 12: A01 * B13
    U[1,12] = 1; V[8,12] = 1; W[3,12] = 1  # W[0*n+3]
    
    # Term 13: A10 * B03
    U[2,13] = 1; V[3,13] = 1; W[5,13] = 1  # W[1*n+3]
    
    # Term 14: A11 * B13
    U[3,14] = 1; V[8,14] = 1; W[5,14] = 1  # W[1*n+3]
    
    # Col 4:
    # Term 15: A00 * B04
    U[0,15] = 1; V[4,15] = 1; W[4,15] = 1  # W[0*n+4]
    
    # Term 16: A01 * B14
    U[1,16] = 1; V[9,16] = 1; W[4,16] = 1  # W[0*n+4]
    
    # But we're missing A10*B04 and A11*B14 for C14!
    # This only gives 16 terms. We need to check if this works.
    
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    err = np.max(np.abs(recon - T))
    
    if err < 1e-6:
        U_r = np.round(2 * U) / 2
        V_r = np.round(2 * V) / 2
        W_r = np.round(2 * W) / 2
        if _check_decomposition(U_r, V_r, W_r, T):
            return U_r, V_r, W_r
    
    # Try another structure for col 4 with shared terms
    # Reset col 4 terms
    U[0,15] = 0; V[4,15] = 0; W[4,15] = 0
    U[1,16] = 0; V[9,16] = 0; W[4,16] = 0
    
    # Term 15: (A00+A10) * B04
    U[0,15] = 1; U[2,15] = 1; V[4,15] = 1
    W[4,15] = 1; W[5,15] = 1  # W[0*n+4], W[1*n+4]
    
    # Term 16: (A01+A11) * B14
    U[1,16] = 1; U[3,16] = 1; V[9,16] = 1
    W[4,16] = 1; W[5,16] = 1  # W[0*n+4], W[1*n+4]
    
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    err = np.max(np.abs(recon - T))
    
    if err < 1e-6:
        U_r = np.round(2 * U) / 2
        V_r = np.round(2 * V) / 2
        W_r = np.round(2 * W) / 2
        if _check_decomposition(U_r, V_r, W_r, T):
            return U_r, V_r, W_r
    
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_target()
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Try hybrid structured approach
    if time.time() < deadline - 60:
        result = _try_rank17_hybrid()
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try algebraic approach
    if time.time() < deadline - 50:
        result = _try_rank17_algebraic()
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try full ALS
    if time.time() < deadline - 10:
        result = _try_rank17_full_als()
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to Rank 18
    U, V, W = _build_rank18()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
