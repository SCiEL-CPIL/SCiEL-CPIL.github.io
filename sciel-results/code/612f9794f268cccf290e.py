import numpy as np
import time
from itertools import product
import sys

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Target tensor: T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # T[a,b,c] = 0 otherwise
    # We need: T = sum_r U[:,r] * V[:,r] * W[:,r]
    
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # Try ranks from a reasonable lower bound upward
    # Lower bound: at least max(n*m, m*p, p*n) / something, but let's start from 21
    # Actually, for matrix multiplication, rank >= n*m*p / min(n,m,p) is not quite right
    # Let's just try from 20 upward
    
    best_result = None
    
    # First, try the schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    # Strategy: for each rank r, try to find integer/half-integer matrices
    # that exactly decompose the tensor
    
    # Use a structured approach: try small ranks first
    for r in range(20, min(40, rank_school)):
        if time.time() > deadline - 10:
            break
        
        # Try to find a rank-r decomposition
        # Use random initialization with integer/half-integer constraints
        # and gradient-based refinement, then snap
        
        found = _try_rank(r, n, m, p, target, time.time(), deadline - 5)
        if found is not None:
            U, V, W = found
            best_result = {"U": U, "V": V, "W": W}
            break  # Found a valid lower rank, return it
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

def _try_rank(r, n, m, p, target, start_time, deadline):
    """Try to find a rank-r decomposition with integer/half-integer entries."""
    np.random.seed(42)  # For reproducibility
    
    # Try multiple random initializations
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Normalize to avoid numerical issues
        for c in range(r):
            norm = np.sqrt(np.sum(U[:,c]**2) * np.sum(V[:,c]**2) * np.sum(W[:,c]**2))
            if norm > 0:
                U[:,c] /= (norm ** (1/3))
                V[:,c] /= (norm ** (1/3))
                W[:,c] /= (norm ** (1/3))
        
        # Gradient descent to minimize ||T - target||^2
        lr = 0.01
        for step in range(200):
            if time.time() > deadline:
                break
            
            T = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T - target
            
            grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
            grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
            grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        # Check if we're close to a solution
        T_final = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.sum((T_final - target)**2)
        
        if error < 1e-4:
            # Try to snap to integer/half-integer
            for scale in [1, 2]:
                U_snap = np.round(U * scale) / scale
                V_snap = np.round(V * scale) / scale
                W_snap = np.round(W * scale) / scale
                
                T_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
                snap_error = np.sum((T_snap - target)**2)
                
                if snap_error < 1e-9:
                    # Verify exactness
                    if _verify_exact(U_snap, V_snap, W_snap, n, m, p, target):
                        return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
    
    return None

def _verify_exact(U, V, W, n, m, p, target, tol=1e-9):
    """Verify exact decomposition."""
    T = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, target, atol=tol, rtol=0)
