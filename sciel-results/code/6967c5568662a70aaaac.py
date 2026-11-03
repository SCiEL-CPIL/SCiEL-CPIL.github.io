import numpy as np
from fractions import Fraction
import time
import random

def construct():
    """
    Return a tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' where:
        - U has shape (n*m, rank) = (9, rank)
        - V has shape (m*p, rank) = (12, rank)  
        - W has shape (p*n, rank) = (12, rank)
    
    The decomposition satisfies:
        For all i, k: C[i,k] = sum_j sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    """
    n, m, p = 3, 3, 4
    deadline = time.time() + 140
    
    # Build the multiplication tensor
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try to find a decomposition with rank < 36
    for target_rank in [30, 27, 24, 21, 18, 15, 12]:
        if time.time() > deadline - 15:
            break
        result = _try_rank(n, m, p, T, target_rank, deadline)
        if result is not None:
            return result
    
    # Schoolbook fallback
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
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

def _try_rank(n, m, p, T, rank, deadline):
    """Try to find a rank-r decomposition using alternating linear solves with exact verification."""
    nm, mp, pn = n*m, m*p, p*n
    
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
        
        # Random sparse initialization
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        for r in range(rank):
            # Each rank-1 term should have some structure
            # Randomly assign non-zero entries
            u_nonzero = random.sample(range(nm), random.randint(1, 3))
            v_nonzero = random.sample(range(mp), random.randint(1, 3))
            w_nonzero = random.sample(range(pn), random.randint(1, 3))
            
            for a in u_nonzero:
                U[a, r] = random.choice([-1, 1, 0.5, -0.5, 2, -2])
            for b in v_nonzero:
                V[b, r] = random.choice([-1, 1, 0.5, -0.5, 2, -2])
            for c in w_nonzero:
                W[c, r] = random.choice([-1, 1, 0.5, -0.5, 2, -2])
        
        # Alternating minimization
        for iteration in range(200):
            if time.time() > deadline - 10:
                break
            
            # Fix V, W; solve for U
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For each (b,c): T[:,b,c] = U @ (V[b,:]*W[c,:])
            T_bc = T.reshape(nm, mp * pn)
            D_u = np.zeros((mp * pn, rank))
            for b in range(mp):
                for c in range(pn):
                    D_u[b * pn + c, :] = V[b, :] * W[c, :]
            
            try:
                U_new = np.linalg.lstsq(D_u.T, T_bc.T, rcond=None)[0].T
                U = U_new
            except:
                break
            
            # Fix U, W; solve for V
            # T[a,b,c] = sum_r V[b,r] * U[a,r] * W[c,r]
            # For each (a,c): T[a,:,c] = V @ (U[a,:]*W[c,:])
            T_ac = np.zeros((nm * pn, mp))
            for a in range(nm):
                for c in range(pn):
                    T_ac[a * pn + c, :] = T[a, :, c]
            
            C_v = np.zeros((nm * pn, rank))
            for a in range(nm):
                for c in range(pn):
                    C_v[a * pn + c, :] = U[a, :] * W[c, :]
            
            try:
                V_new = np.linalg.lstsq(C_v.T, T_ac.T, rcond=None)[0].T
                V = V_new
            except:
                break
            
            # Fix U, V; solve for W
            # T[a,b,c] = sum_r W[c,r] * U[a,r] * V[b,r]
            # For each (a,b): T[a,b,:] = W @ (U[a,:]*V[b,:])
            T_ab = np.zeros((nm * mp, pn))
            for a in range(nm):
                for b in range(mp):
                    T_ab[a * mp + b, :] = T[a, b, :]
            
            D_w = np.zeros((nm * mp, rank))
            for a in range(nm):
                for b in range(mp):
                    D_w[a * mp + b, :] = U[a, :] * V[b, :]
            
            try:
                W_new = np.linalg.lstsq(D_w.T, T_ab.T, rcond=None)[0].T
                W = W_new
            except:
                break
            
            # Check if we've found an exact solution
            residual = 0.0
            for r_idx in range(rank):
                term = np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
                residual += np.sum((T - term)**2)
            
            if residual < 1e-10:
                # Try to convert to exact rationals
                U_exact = _floats_to_exact(U)
                V_exact = _floats_to_exact(V)
                W_exact = _floats_to_exact(W)
                
                if U_exact is not None and V_exact is not None and W_exact is not None:
                    if _verify_exact(n, m, p, U_exact, V_exact, W_exact, T):
                        return {"U": U_exact, "V": V_exact, "W": W_exact}
    
    return None

def _floats_to_exact(arr, tol=1e-8):
    """Convert float array to exact rationals, or None if fails."""
    result = []
    for row in arr:
        exact_row = []
        for val in row:
            if abs(val) < tol:
                exact_row.append(0)
            else:
                found = False
                for denom in [1, 2, 4, 8, 16, 3, 6, 12, 5, 10, 20, 7, 14, 28]:
                    num = round(val * denom)
                    if num != 0 and abs(val - num / denom) < tol:
                        exact_row.append(Fraction(num, denom))
                        found = True
                        break
                    elif num == 0 and abs(val) < tol:
                        exact_row.append(0)
                        found = True
                        break
                if not found:
                    return None
        result.append(exact_row)
    return result

def _verify_exact(n, m, p, U, V, W, T):
    """Verify exact decomposition."""
    nm, mp, pn = n*m, m*p, p*n
    rank = len(U[0])
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if val != T[a, b, c]:
                    return False
    return True
