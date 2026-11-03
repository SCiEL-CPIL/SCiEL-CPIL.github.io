import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 4]

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _check_exact(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Build tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Expected: T[i,j,k] = 1 if i,j,k valid (always 1 for matrix mult tensor)
    # Actually for matrix mult: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor: T[i,j,k] = 1 for all i,j,k
    # Check: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol, max_err

def _try_rank(r, deadline, rng):
    """Try to find a rank-r decomposition. Returns (U,V,W) or None."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Try multiple random initializations
    n_trials = max(3, int((deadline - time.time()) / 2.0))
    n_trials = min(n_trials, 50)
    
    for trial in range(n_trials):
        if time.time() > deadline - 5:
            break
        
        # Random initialization with small integers
        U = rng.integers(-2, 3, size=(nm, r))
        V = rng.integers(-2, 3, size=(mp, r))
        W = rng.integers(-2, 3, size=(pn, r))
        
        # Alternating minimization: fix two, solve for third
        # For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is a 3-way tensor completion problem
        
        # Use a simple iterative approach:
        # For each c, the contribution is U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        # We want sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] = T where T[i,j,k]=1
        
        # Iterative: fix U,V, solve for W (linear system)
        for it in range(20):
            if time.time() > deadline - 5:
                break
            
            # Solve for W: for each (k,i), sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*V[j*p+k,c]) = 1
            # This is: for each k,i: sum_c W[k*n+i,c] * S[k,i,c] = 1 where S[k,i,c] = sum_j U[i*m+j,c]*V[j*p+k,c]
            # But S depends on j... let me think differently.
            
            # Actually, the constraint is: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # Fix U and V. Then for each (i,k): sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*V[j*p+k,c]) = 1
            # Wait, that's not right either because the sum over c is coupled.
            
            # Let me use a different approach: treat it as a linear system in W entries.
            # Unknowns: W[k*n+i, c] for all k,i,c -> pn * r unknowns
            # Equations: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 -> n*m*p equations
            
            # Build the system: A * w = b where w is flattened W
            # Row (i,j,k): for each c: coeff of W[k*n+i, c] is U[i*m+j,c]*V[j*p+k,c]
            # So row (i,j,k) only has non-zero entries in positions corresponding to W[k*n+i, :]
            
            # This means the system is block-diagonal in (i,k)! For each (i,k), we have m equations
            # (one per j) and r unknowns (W[k*n+i, 0..r-1]).
            
            # For each (i,k): solve m x r system: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for j=0..m-1
            # If m <= r, we can potentially solve this.
            
            success = True
            new_W = np.zeros((pn, r))
            for i in range(n):
                for k in range(p):
                    # Build m x r matrix A_ik where A_ik[j,c] = U[i*m+j,c]*V[j*p+k,c]
                    A_ik = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A_ik[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    # Solve A_ik @ new_W[k*n+i, :] = ones(m)
                    # Use least squares
                    try:
                        sol, residuals, rank_ik, sv = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                        new_W[k*n+i, :] = sol
                    except:
                        success = False
                        break
                if not success:
                    break
            
            if not success:
                break
            
            W = new_W.astype(int)  # Round to integers
            
            # Check exactness
            exact, err = _check_exact(U.tolist(), V.tolist(), W.tolist())
            if exact:
                return U.tolist(), V.tolist(), W.tolist()
            
            # If not exact, try to round and re-check
            W_rounded = np.round(W).astype(int)
            exact, err = _check_exact(U.tolist(), V.tolist(), W_rounded.tolist())
            if exact:
                return U.tolist(), V.tolist(), W_rounded.tolist()
            
            # Update: try solving for U and V similarly
            # For each (j,k): solve for V[j*p+k, :]
            new_V = np.zeros((mp, r))
            for j in range(m):
                for k in range(p):
                    # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i
                    # Fix U, W. For each (j,k): sum_c V[j*p+k,c] * (sum_i U[i*m+j,c]*W[k*n+i,c]) = 1 for i=0..n-1
                    # n equations, r unknowns
                    A_jk = np.zeros((n, r))
                    for i in range(n):
                        for c in range(r):
                            A_jk[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    try:
                        sol, residuals, rank_jk, sv = np.linalg.lstsq(A_jk, np.ones(n), rcond=None)
                        new_V[j*p+k, :] = sol
                    except:
                        pass
            V = new_V
            
            # For each (i,j): solve for U[i*m+j, :]
            new_U = np.zeros((nm, r))
            for i in range(n):
                for j in range(m):
                    # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all k
                    # p equations, r unknowns
                    A_ij = np.zeros((p, r))
                    for k in range(p):
                        for c in range(r):
                            A_ij[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    try:
                        sol, residuals, rank_ij, sv = np.linalg.lstsq(A_ij, np.ones(p), rcond=None)
                        new_U[i*m+j, :] = sol
                    except:
                        pass
            U = new_U
        
        # Final check after iterations
        U_int = np.round(U).astype(int)
        V_int = np.round(V).astype(int)
        W_int = np.round(W).astype(int)
        exact, err = _check_exact(U_int.tolist(), V_int.tolist(), W_int.tolist())
        if exact:
            return U_int.tolist(), V_int.tolist(), W_int.tolist()
    
    return None

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Try ranks from low to high
    # Known bounds: for 2x4x4, the tensor rank is at most 32 (schoolbook)
    # Lower bound: at least max(n*m, m*p, p*n) = max(8, 16, 8) = 16
    # Actually the trivial lower bound is max(n*m, m*p, p*n) but there might be better
    # Let's try from 16 upward
    
    rng = np.random.default_rng(42)
    
    # First, try a few specific structured approaches
    # Approach 1: Try rank = n*m*p / 2 = 16 (if possible)
    # Approach 2: Try rank = 17, 18, ...
    
    for r in range(16, 33):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(r, deadline, rng)
        if result is not None:
            U, V, W = result
            # Verify one more time
            exact, err = _check_exact(U, V, W)
            if exact:
                return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    return _schoolbook()
