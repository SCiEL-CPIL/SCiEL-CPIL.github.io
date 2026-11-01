"""Find rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 4]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact (within numerical tolerance)."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 for all i,j,k (matrix multiplication tensor)
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j], V[j*p + k] * W[k*n + i])
                if abs(val - 1.0) > 1e-8:
                    return False
    return True

def _check_half_integer(M):
    """Check that all entries are integer or half-integer."""
    M = np.array(M, dtype=float)
    for val in M.flat:
        if abs(val * 2 - round(val * 2)) > 1e-6:
            return False
    return True

def _snap_to_half_integer(M):
    """Snap values to nearest half-integer."""
    M = np.array(M, dtype=float)
    return np.round(M * 2) / 2.0

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    
    deadline = time.time() + 140  # safety margin
    
    # Start with schoolbook rank = 24
    # Try to find lower rank decompositions
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook solution
    U_school = np.zeros((nm, best_rank))
    V_school = np.zeros((mp, best_rank))
    W_school = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Known lower bounds and upper bounds for small matrix multiplication
    # For 2x3 * 3x4, the rank is known to be at most 24 (schoolbook)
    # Let's try to find a better decomposition using optimization
    
    # Strategy: try ranks from a lower bound up to schoolbook rank
    # Lower bound: max(nm, mp, pn) = max(6, 12, 8) = 12
    # Also, rank >= n*m*p / max(n,m,p)^2 ... not a standard bound
    # A better lower bound: rank >= max(n*m, m*p, p*n) = 12
    
    # Let's try to use a randomized approach or structured search
    # For small dimensions, we can try to find decompositions with rank < 24
    
    # Approach: Use the fact that matrix multiplication tensor can sometimes
    # be decomposed with fewer than nmp terms using clever linear combinations
    
    # Try ranks from 12 to 23
    for target_rank in range(12, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 1000 + target_rank)
            np.random.seed(attempt * 1000 + target_rank)
            
            # Initialize with small random values
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            # Use gradient descent to minimize the error
            # Error = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            lr = 0.01
            for step in range(500):
                if time.time() > deadline:
                    break
                
                # Compute gradient
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            
                            T_val = np.dot(U[idx_u], V[idx_v] * W[idx_w])
                            err = T_val - 1.0
                            
                            for r in range(target_rank):
                                dU[idx_u, r] += err * V[idx_v, r] * W[idx_w, r]
                                dV[idx_v, r] += err * U[idx_u, r] * W[idx_w, r]
                                dW[idx_w, r] += err * U[idx_u, r] * V[idx_v, r]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence
                if np.max(np.abs(dU)) < 1e-6 and np.max(np.abs(dV)) < 1e-6 and np.max(np.abs(dW)) < 1e-6:
                    break
            
            # Check if solution is valid
            if _verify_decomposition(U, V, W, n, m, p):
                # Try to snap to half-integers
                U_snap = _snap_to_half_integer(U)
                V_snap = _snap_to_half_integer(V)
                W_snap = _snap_to_half_integer(W)
                
                if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                    best_rank = target_rank
                    best_U = U_snap
                    best_V = V_snap
                    best_W = W_snap
                    break
                elif _verify_decomposition(U, V, W, n, m, p):
                    # Use non-snapped if it works
                    best_rank = target_rank
                    best_U = U
                    best_V = V
                    best_W = W
                    break
        
        if best_rank == target_rank:
            break
    
    # Convert to lists of lists for the required format
    U_list = best_U.tolist() if hasattr(best_U, 'tolist') else best_U
    V_list = best_V.tolist() if hasattr(best_V, 'tolist') else best_V
    W_list = best_W.tolist() if hasattr(best_W, 'tolist') else best_W
    
    return {"U": U_list, "V": V_list, "W": W_list}
