import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal tensor decomposition for 2x4 by 4x5 matrix multiplication.
    Tries ranks from lower bound (10) up to schoolbook (40).
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 115  # safety margin
    
    # Build target tensor indices
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[(a, b, c)] = 1.0
    
    # Try ranks from lower bound to schoolbook
    lower_bound = max(n * p, m * n, m * p)  # = max(10, 8, 20) = 20? No, lower bound is n*p=10
    # Actually the true lower bound is at least n*p = 10
    for rank in range(10, nm * mp // pn + 1):  # up to 40
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, nm, mp, pn, rank, target, deadline)
        if result is not None:
            return result
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p, nm, mp, pn)

def _schoolbook(n, m, p, nm, mp, pn):
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(n, m, p, nm, mp, pn, rank, target, deadline):
    """Attempt to find exact rank-r decomposition using structured search."""
    # Use the structure: for n=2, we can parameterize more cleverly
    # Each output C[i,k] = sum_j A[i,j]*B[j,k]
    # Try to find rank-1 terms that cover these
    
    np.random.seed(42 + rank)
    
    for trial in range(50):
        if time.time() > deadline:
            return None
        
        # Initialize with random values
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        # Optimize using alternating linear least squares
        # For fixed V, W: solve for U via least squares
        # For fixed U, W: solve for V via least squares
        # For fixed U, V: solve for W via least squares
        
        for it in range(200):
            if time.time() > deadline:
                return None
            
            # Update U: for each row a, minimize sum_{b,c} (sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c])^2
            # This is linear in U[a,:]
            # Build system: for each (b,c), we have V[b,:]*W[c,:]^T as coefficient vector
            # Let's use a simpler gradient-based approach
            
            # Compute residuals
            # T_hat[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            
            # Gradient descent with adaptive step
            lr = 0.01 / (1 + it * 0.01)
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for (a, b, c), tval in target.items():
                # d/dU[a,r] = V[b,r]*W[c,r]
                # d/dV[b,r] = U[a,r]*W[c,r]
                # d/dW[c,r] = U[a,r]*V[b,r]
                for r in range(rank):
                    uar = U[a, r]
                    vbr = V[b, r]
                    wcr = W[c, r]
                    # residual for this (a,b,c)
                    resid = 0.0
                    for r2 in range(rank):
                        resid += U[a, r2] * V[b, r2] * W[c, r2]
                    resid -= tval
                    grad_U[a, r] += 2 * resid * vbr * wcr
                    grad_V[b, r] += 2 * resid * uar * wcr
                    grad_W[c, r] += 2 * resid * uar * vbr
            
            # Also need to ensure non-target entries are zero
            # Sample some non-target entries
            np.random.seed(it)
            for _ in range(20):
                a = np.random.randint(nm)
                b = np.random.randint(mp)
                c = np.random.randint(pn)
                if (a, b, c) not in target:
                    for r in range(rank):
                        uar = U[a, r]
                        vbr = V[b, r]
                        wcr = W[c, r]
                        resid = 0.0
                        for r2 in range(rank):
                            resid += U[a, r2] * V[b, r2] * W[c, r2]
                        grad_U[a, r] += 2 * resid * vbr * wcr
                        grad_V[b, r] += 2 * resid * uar * wcr
                        grad_W[c, r] += 2 * resid * uar * vbr
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        # Check if solution is valid
        if _check_exact(U, V, W, nm, mp, pn, target):
            # Round to half-integers
            U_r, V_r, W_r = _round_to_half_int(U, V, W)
            if U_r is not None and _verify_exact(U_r, V_r, W_r, nm, mp, pn, target):
                return {"U": U_r, "V": V_r, "W": W_r}
    
    return None

def _check_exact(U, V, W, nm, mp, pn, target, tol=1e-6):
    rank = U.shape[1]
    for (a, b, c), tval in target.items():
        s = 0.0
        for r in range(rank):
            s += U[a, r] * V[b, r] * W[c, r]
        if abs(s - tval) > tol:
            return False
    # Check non-target entries (sample)
    np.random.seed(123)
    for _ in range(50):
        a = np.random.randint(nm)
        b = np.random.randint(mp)
        c = np.random.randint(pn)
        if (a, b, c) not in target:
            s = 0.0
            for r in range(rank):
                s += U[a, r] * V[b, r] * W[c, r]
            if abs(s) > tol:
                return False
    return True

def _round_to_half_int(U, V, W, max_denom=4):
    """Round entries to multiples of 1/max_denom."""
    def round_mat(M):
        M_r = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]
                best = 0
                best_err = abs(val)
                for den in range(1, max_denom + 1):
                    for num in range(-100, 101):
                        cand = num / den
                        err = abs(val - cand)
                        if err < best_err:
                            best_err = err
                            best = cand
                M_r[i, j] = best
        return M_r.tolist()
    
    U_r = round_mat(U)
    V_r = round_mat(V)
    W_r = round_mat(W)
    return U_r, V_r, W_r

def _verify_exact(U, V, W, nm, mp, pn, target, tol=1e-9):
    rank = len(U[0])
    for (a, b, c), tval in target.items():
        s = 0.0
        for r in range(rank):
            s += U[a][r] * V[b][r] * W[c][r]
        if abs(s - tval) > tol:
            return False
    # Check all non-target entries
    target_set = set(target.keys())
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in target_set:
                    s = 0.0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s) > tol:
                        return False
    return True
# EVOLVE-BLOCK-END
