"""Program for M_{2,3,6} rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 6]

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s
    n, m, p = DIMENSIONS
    
    # Schoolbook rank is n*m*p = 36
    rank_schoolbook = n * m * p
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def verify_float(U, V, W, tol=1e-9):
        """Verify decomposition using float arithmetic."""
        T_hat = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(T_hat, T, atol=tol)
    
    def verify_exact(U_frac, V_frac, W_frac):
        """Verify decomposition using exact Fraction arithmetic."""
        r = len(U_frac[0])
        # Check all nonzero entries of T
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx1 = i * m + j
                    idx2 = j * p + k
                    idx3 = k * n + i
                    total = Fraction(0)
                    for c in range(r):
                        total += U_frac[idx1][c] * V_frac[idx2][c] * W_frac[idx3][c]
                    if total != 1:
                        return False
        # Check all zero entries
        for idx1 in range(n*m):
            for idx2 in range(m*p):
                for idx3 in range(p*n):
                    # Skip the known 1 entries
                    i1 = idx1 // m
                    j1 = idx1 % m
                    j2 = idx2 // p
                    k2 = idx2 % p
                    k3 = idx3 // n
                    i3 = idx3 % n
                    if i1 == i3 and j1 == j2 and k1 == k2 if False else (i1 == i3 and j1 == j2 and (idx2 % p) == (idx3 // n)):
                        continue
                    # This is a zero entry - check it sums to 0
                    # Actually let's just check all entries
        # Simpler: check all entries
        for idx1 in range(n*m):
            for idx2 in range(m*p):
                for idx3 in range(p*n):
                    i1 = idx1 // m
                    j1 = idx1 % m
                    j2 = idx2 // p
                    k2 = idx2 % p
                    k3 = idx3 // n
                    i3 = idx3 % n
                    expected = 1 if (i1 == i3 and j1 == j2 and k2 == k3) else 0
                    total = Fraction(0)
                    r = len(U_frac[0])
                    for c in range(r):
                        total += U_frac[idx1][c] * V_frac[idx2][c] * W_frac[idx3][c]
                    if total != expected:
                        return False
        return True
    
    # Construct schoolbook decomposition
    U = np.zeros((n*m, rank_schoolbook), dtype=float)
    V = np.zeros((m*p, rank_schoolbook), dtype=float)
    W = np.zeros((p*n, rank_schoolbook), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    
    if not verify_float(U, V, W):
        raise ValueError("Schoolbook decomposition failed verification")
    
    best_rank = rank_schoolbook
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Strategy: Try to find a lower-rank decomposition
    # Key insight: For M_{n,m,p}, we can sometimes reduce rank by exploiting
    # the structure. Let's try a structured approach.
    
    # Approach 1: Try to merge pairs of terms
    # For each pair of schoolbook terms, see if we can represent their sum
    # as a single rank-1 term (unlikely in general) or combine with others.
    
    # Approach 2: Use the fact that the tensor has a specific structure
    # and try to find a decomposition with rank < 36.
    
    # Let's try a numerical optimization approach:
    # Start with rank r < 36 and try to find U, V, W that satisfy the constraints.
    
    # For a given rank r, we need to find:
    # U: (6, r), V: (18, r), W: (12, r)
    # such that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i*m+j, j*p+k, k*n+i]
    
    # This is a system of equations. Let's try to solve it.
    
    # Number of constraints: n*m*p = 36 (the 1 entries)
    # Number of variables: 6*r + 18*r + 12*r = 36*r
    # For r < 36, we have more variables than constraints, so solutions may exist.
    
    # Let's try to find a solution for various r values.
    
    def try_rank(r, max_attempts=10):
        """Try to find a rank-r decomposition."""
        if r >= best_rank:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
            
            # Initialize with random values
            U_test = np.random.randn(n*m, r)
            V_test = np.random.randn(m*p, r)
            W_test = np.random.randn(p*n, r)
            
            # Try to optimize using gradient descent on the residual
            # Residual = T - einsum('ic,jc,kc->ijk', U, V, W)
            # Loss = sum(residual^2)
            
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 10:
                    return None
                
                T_hat = np.einsum('ic,jc,kc->ijk', U_test, V_test, W_test)
                residual = T - T_hat
                
                # Gradients
                # dL/dU[i,c] = -2 * sum_{j,k} residual[i,j,k] * V[j,c] * W[k,c]
                # dL/dV[j,c] = -2 * sum_{i,k} residual[i,j,k] * U[i,c] * W[k,c]
                # dL/dW[k,c] = -2 * sum_{i,j} residual[i,j,k] * U[i,c] * V[j,c]
                
                dU = -2 * np.einsum('ijk,jc,kc->ic', residual, V_test, W_test)
                dV = -2 * np.einsum('ijk,ic,kc->jc', residual, U_test, W_test)
                dW = -2 * np.einsum('ijk,ic,jc->kc', residual, U_test, V_test)
                
                U_test -= lr * dU
                V_test -= lr * dV
                W_test -= lr * dW
                
                # Check convergence
                if np.max(np.abs(residual)) < 1e-8:
                    if verify_float(U_test, V_test, W_test, tol=1e-7):
                        return (U_test, V_test, W_test)
            
            # Check if we got close
            T_hat = np.einsum('ic,jc,kc->ijk', U_test, V_test, W_test)
            if np.max(np.abs(T - T_hat)) < 1e-6:
                if verify_float(U_test, V_test, W_test, tol=1e-5):
                    return (U_test, V_test, W_test)
        
        return None
    
    # Try ranks from 35 down to some lower bound
    # The matricization lower bound is max(6, 18, 12) = 18
    # But realistically, for matrix multiplication, the rank is likely higher.
    
    # Let's try a range of ranks
    for r in range(rank_schoolbook - 1, 25, -1):
        if time.time() > deadline - 20:
            break
        
        result = try_rank(r, max_attempts=5)
        if result is not None:
            U_test, V_test, W_test = result
            if verify_float(U_test, V_test, W_test, tol=1e-6):
                best_rank = r
                best_U = U_test.copy()
                best_V = V_test.copy()
                best_W = W_test.copy()
                break  # Found a lower rank, stop
    
    # If we found a lower rank, try to polish it
    if best_rank < rank_schoolbook:
        # Try to verify with higher precision
        if verify_float(best_U, best_V, best_W, tol=1e-10):
            pass  # Good
        else:
            # Try to polish
            U_polish = best_U.copy()
            V_polish = best_V.copy()
            W_polish = best_W.copy()
            lr = 0.001
            for step in range(1000):
                if time.time() > deadline - 5:
                    break
                T_hat = np.einsum('ic,jc,kc->ijk', U_polish, V_polish, W_polish)
                residual = T - T_hat
                dU = -2 * np.einsum('ijk,jc,kc->ic', residual, V_polish, W_polish)
                dV = -2 * np.einsum('ijk,ic,kc->jc', residual, U_polish, W_polish)
                dW = -2 * np.einsum('ijk,ic,jc->kc', residual, U_polish, V_polish)
                U_polish -= lr * dU
                V_polish -= lr * dV
                W_polish -= lr * dW
                if np.max(np.abs(residual)) < 1e-12:
                    break
            if verify_float(U_polish, V_polish, W_polish, tol=1e-9):
                best_U = U_polish
                best_V = V_polish
                best_W = W_polish
    
    # Convert to lists of lists
    U_list = [[float(best_U[i, j]) for j in range(best_U.shape[1])] for i in range(best_U.shape[0])]
    V_list = [[float(best_V[i, j]) for j in range(best_V.shape[1])] for i in range(best_V.shape[0])]
    W_list = [[float(best_W[i, j]) for j in range(best_W.shape[1])] for i in range(best_W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
