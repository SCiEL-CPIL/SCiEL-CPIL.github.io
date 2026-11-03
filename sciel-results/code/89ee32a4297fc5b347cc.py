"""Matrix multiplication 3,3,4 rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

def _validate(U, V, W, n=3, m=3, p=4):
    """Validate that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            expected = 1 if (i == ip and j == jp and k == kp) else 0
                            actual = 0
                            for c in range(rank):
                                actual += U[i * m + j][c] * V[jp * p + k][c] * W[ip * n + kp][c]
                            if abs(actual - expected) > 1e-9:
                                return False
    return True

def _schoolbook(n=3, m=3, p=4):
    """Return the schoolbook (rank n*m*p) decomposition."""
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

def _try_rank_r(rank, deadline, n=3, m=3, p=4):
    """Try to find a rank-r decomposition using linearized approach with fixed U,V."""
    for attempt in range(20):
        if time.time() > deadline - 5:
            return None
        
        random.seed(rank * 1000 + attempt)
        
        # Initialize U and V with small random integers
        U = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(rank)] for _ in range(m * p)]
        
        # Solve for W: for each (ip, kp), we have a linear system
        # For fixed (ip, kp), the equations are:
        # For each j, jp: sum_c U[ip*m+j, c] * V[jp*p+kp, c] * W[ip*n+kp, c] = delta(j=jp)
        
        W = [[0] * rank for _ in range(p * n)]
        valid = True
        
        for ip in range(n):
            for kp in range(p):
                # Build the linear system for W[ip*n+kp, :]
                # Unknowns: W[ip*n+kp, 0], ..., W[ip*n+kp, rank-1]
                # Equations: for each j, jp:
                #   sum_c U[ip*m+j, c] * V[jp*p+kp, c] * W[ip*n+kp, c] = delta(j=jp)
                
                A_sys = np.zeros((m * m, rank))
                b_sys = np.zeros(m * m)
                
                eq_idx = 0
                for j in range(m):
                    for jp in range(m):
                        for c in range(rank):
                            A_sys[eq_idx, c] = U[ip * m + j][c] * V[jp * p + kp][c]
                        b_sys[eq_idx] = 1 if j == jp else 0
                        eq_idx += 1
                
                # Solve the linear system
                if rank <= m * m:
                    # Underdetermined or square system
                    try:
                        W_row, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        if residuals and residuals[0] > 1e-10:
                            valid = False
                            break
                        W[ip * n + kp] = [round(val * 2) / 2 for val in W_row]
                    except:
                        valid = False
                        break
                else:
                    valid = False
                    break
                
                if not valid:
                    break
            
            if not valid:
                break
        
        if valid:
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x4 matrix.
    """
    deadline = time.time() + 140
    n, m, p = 3, 3, 4
    
    # Try ranks from 24 upward (20 is too low for 3x3x4 likely)
    for rank in range(24, 37):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_r(rank, deadline, n, m, p)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)
