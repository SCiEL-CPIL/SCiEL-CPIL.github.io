import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x7 matrix.
    Uses schoolbook as fallback, then searches for lower-rank decompositions.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_sb = n * m * p  # 42
    
    # Schoolbook decomposition as fallback
    U_sb = np.zeros((nm, rank_sb), dtype=np.int64)
    V_sb = np.zeros((mp, rank_sb), dtype=np.int64)
    W_sb = np.zeros((pn, rank_sb), dtype=np.int64)
    
    for idx in range(rank_sb):
        i = idx // (m * p)
        j = (idx // p) % m
        k = idx % p
        U_sb[i * m + j, idx] = 1
        V_sb[j * p + k, idx] = 1
        W_sb[k * n + i, idx] = 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    def verify(U, V, W):
        """Verify exact decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def snap_half(arr, tol=1e-8):
        """Snap to nearest half-integer if close enough"""
        snapped = np.round(arr * 2) / 2.0
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None
    
    def try_rank(rank, max_trials=100):
        """Try to find a valid decomposition of given rank"""
        nonlocal best_U, best_V, best_W, best_rank
        
        if rank >= best_rank:
            return False
        
        # Use deterministic seed for reproducibility
        np.random.seed(42)
        
        for trial in range(max_trials):
            if time.time() > deadline:
                return False
            
            # Generate random U, V with entries in {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(nm, rank))
            V = np.random.randint(-1, 2, size=(mp, rank))
            
            W = np.zeros((pn, rank))
            valid = True
            
            # For each (i, k) pair, solve linear system for W[k*n+i, :]
            for i in range(n):
                for k in range(p):
                    # Build A_ik: m x rank matrix where A_ik[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    A_ik = np.zeros((m, rank))
                    for j in range(m):
                        A_ik[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve A_ik @ w = ones(m)
                    try:
                        # Use least squares
                        w, residuals, rank_A, s = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                        # Check if it's an exact solution
                        if not np.allclose(A_ik @ w, np.ones(m), atol=1e-9):
                            valid = False
                            break
                        W[k * n + i, :] = w
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Snap W to half-integers
            W_snapped = snap_half(W)
            if W_snapped is None:
                continue
            
            # Final verification
            if verify(U, V, W_snapped):
                best_U, best_V, best_W = U, V, W_snapped
                best_rank = rank
                return True
        
        return False
    
    # Try ranks from 41 down to 25
    for rank in range(rank_sb - 1, 24, -1):
        if time.time() > deadline:
            break
        if try_rank(rank, max_trials=50):
            # Found a valid decomposition, try to go lower
            continue
    
    # If we found something better than schoolbook, use it
    if best_rank < rank_sb:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Fallback to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
# EVOLVE-BLOCK-END
