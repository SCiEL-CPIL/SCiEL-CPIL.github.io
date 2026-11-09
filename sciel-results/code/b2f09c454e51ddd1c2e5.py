import time
import numpy as np
from scipy.optimize import nnls # Might not be needed, np.linalg.lstsq is fine

DIMENSIONS = [3, 4, 8]

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
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

def _build_tensor(n, m, p):
    """Build the target tensor T as a list of (a,b,c,value) tuples for non-zero entries."""
    T = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T.append((a, b, c, 1.0))
    return T

def _als_decomposition(n, m, p, rank, max_iters=30, tol=1e-8):
    """
    Try to find a rank-r decomposition using Alternating Least Squares.
    Returns (U, V, W) if found, else None.
    """
    # Precompute index groupings for efficiency
    # For each a, list of (b, c) pairs where T[a,b,c] = 1
    a_to_bc = {}
    b_to_ac = {}
    c_to_ab = {}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if a not in a_to_bc:
                    a_to_bc[a] = []
                a_to_bc[a].append((b, c))
                if b not in b_to_ac:
                    b_to_ac[b] = []
                b_to_ac[b].append((a, c))
                if c not in c_to_ab:
                    c_to_ab[c] = []
                c_to_ab[c].append((a, b))
    
    # Initialize with structured random
    np.random.seed(42)
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    best_err = float('inf')
    
    for it in range(max_iters):
        # Solve for U
        for a in range(n * m):
            if a not in a_to_bc:
                continue
            bc_list = a_to_bc[a]
            b_sub = np.array([x[0] for x in bc_list])
            c_sub = np.array([x[1] for x in bc_list])
            V_sub = V[b_sub]
            W_sub = W[c_sub]
            M = V_sub * W_sub
            if M.size == 0:
                continue
            sol, _, _, _ = np.linalg.lstsq(M, np.ones(len(bc_list)), rcond=None)
            U[a] = sol
            
        # Solve for V
        for b in range(m * p):
            if b not in b_to_ac:
                continue
            ac_list = b_to_ac[b]
            a_sub = np.array([x[0] for x in ac_list])
            c_sub = np.array([x[1] for x in ac_list])
            U_sub = U[a_sub]
            W_sub = W[c_sub]
            M = U_sub * W_sub
            if M.size == 0:
                continue
            sol, _, _, _ = np.linalg.lstsq(M, np.ones(len(ac_list)), rcond=None)
            V[b] = sol
            
        # Solve for W
        for c in range(p * n):
            if c not in c_to_ab:
                continue
            ab_list = c_to_ab[c]
            a_sub = np.array([x[0] for x in ab_list])
            b_sub = np.array([x[1] for x in ab_list])
            U_sub = U[a_sub]
            V_sub = V[b_sub]
            M = U_sub * V_sub
            if M.size == 0:
                continue
            sol, _, _, _ = np.linalg.lstsq(M, np.ones(len(ab_list)), rcond=None)
            W[c] = sol
        
        # Check error on all tensor entries (both 1 and 0)
        error = 0.0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # Determine true value
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    true_val = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                    recon = np.dot(U[a], V[b] * W[c])
                    error += (recon - true_val) ** 2
        
        if error < best_err:
            best_err = error
        
        if error < tol:
            return U, V, W
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(U)) or not np.all(np.isfinite(V)) or not np.all(np.isfinite(W)):
            break
            
    return None

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            j2 = b // p
            k = b % p
            for c in range(p * n):
                k2 = c // n
                i2 = c % n
                true_val = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                recon = np.dot(U[a], V[b] * W[c])
                if abs(recon - true_val) > tol:
                    return False
    return True

def _round_to_half_integers(U, V, W, n, m, p):
    """
    Try to round U, V, W to half-integers and check if it still works.
    """
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
        return U_rounded, V_rounded, W_rounded
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s kill
    
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # Known results suggest rank is likely between 40-60 for 3x4x8
    # Try ranks from a reasonable range
    best_result = None
    best_rank = n * m * p  # schoolbook rank as fallback
    
    # Try to find decompositions at various ranks
    # Start with smaller ranks and work up
    for rank in range(40, min(96, n*m*p + 1)):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # ALS
            result = _als_decomposition(n, m, p, rank, max_iters=15, tol=1e-8)
            if result is not None:
                U, V, W = result
                rounded = _round_to_half_integers(U, V, W, n, m, p)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    # Final verification
                    if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-6):
                        U_list = U_r.tolist()
                        V_list = V_r.tolist()
                        W_list = W_r.tolist()
                        return {"U": U_list, "V": V_list, "W": W_list}
        
        # If we found a valid decomposition at this rank, return it
        # (the loop above already returns when found)
        
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
