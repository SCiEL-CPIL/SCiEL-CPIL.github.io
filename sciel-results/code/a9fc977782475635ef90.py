"""Matrix multiplication 2,3,5 rank-minimal search with deadline guard."""
import time
import random
import numpy as np
from numpy.linalg import norm, matrix_rank
from itertools import combinations, product

DIMENSIONS = [2, 3, 5]
T_START = time.time()
DEADLINE = T_START + 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Helper to check validity and rank
    def check(U, V, W):
        """Check if decomposition is valid. U: (nm, r), V: (mp, r), W: (pn, r).
        Returns True if valid, False otherwise."""
        r = len(U[0]) if U else 0
        # Tensor T[i,j,k] = U[i,j,col] * V[j,k,col] * W[k,i,col]
        # But our indexing: U row = i*m+j, V row = j*p+k, W row = k*n+i
        # T should equal 1 for all (i,j,k), 0 otherwise
        # Check: sum_col U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col] == 1 for all i,j,k
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # Reshape for easier computation
        U3 = U_arr.reshape(n, m, r)
        V3 = V_arr.reshape(m, p, r)
        W3 = W_arr.reshape(p, n, r)
        # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # Use einsum
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        # Check if T equals the identity tensor (1 for all i,j,k)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-6)
    
    def check_rank_exact(U, V, W):
        """Check if the decomposition is exact (not just approximate).
        For integer/half-integer entries, we can use exact arithmetic."""
        from fractions import Fraction
        r = len(U[0]) if U else 0
        U3 = [[[[Fraction(U[i*m+j][c]) for c in range(r)] for j in range(m)] for i in range(n)]]
        # Actually let's just use float check with tight tolerance for half-integers
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        U3 = U_arr.reshape(n, m, r)
        V3 = V_arr.reshape(m, p, r)
        W3 = W_arr.reshape(p, n, r)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-10)
    
    # Start with schoolbook
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    
    # Try to find lower rank decompositions
    # For 2x3x5, the tensor has dimension 30
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(6, 15, 10) = 15
    # Upper bound: 30 (schoolbook)
    # Let's try ranks from 15 upward
    
    # For each candidate rank, try random initializations and gradient descent
    # to find a valid decomposition
    
    def random_init(rank, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Use half-integers: multiply by 2, use integers, then divide by 2
        # For simplicity, use random integers in a small range
        scale = 2
        U = np.random.randint(-scale, scale+1, (nm, rank)).astype(float) / scale
        V = np.random.randint(-scale, scale+1, (mp, rank)).astype(float) / scale
        W = np.random.randint(-scale, scale+1, (pn, rank)).astype(float) / scale
        return U, V, W
    
    def refine(U, V, W, max_iter=2000, lr=0.01):
        """Simple gradient descent to minimize ||T - 1||^2."""
        n_, m_, p_ = n, m, p
        r = U.shape[1]
        U3 = U.reshape(n_, m_, r)
        V3 = V.reshape(m_, p_, r)
        W3 = W.reshape(p_, n_, r)
        
        for it in range(max_iter):
            if time.time() > DEADLINE:
                break
            T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            loss = np.sum((T - 1.0)**2)
            if loss < 1e-12:
                break
            
            # Gradients
            # dL/dU3[i,j,r] = 2*(T[i,j,k]-1) * V3[j,k,r] * W3[k,i,r] summed over k
            dU3 = 2 * np.einsum('ijk,jkr,kir->ijr', T - 1.0, V3, W3)
            dV3 = 2 * np.einsum('ijk,ijr,kir->jkr', T - 1.0, U3, W3)
            dW3 = 2 * np.einsum('ijk,ijr,jkr->kir', T - 1.0, U3, V3)
            
            U3 -= lr * dU3
            V3 -= lr * dV3
            W3 -= lr * dW3
            
            # Snap to nearest half-integer periodically
            if it % 100 == 0:
                U3 = np.round(U3 * 2) / 2
                V3 = np.round(V3 * 2) / 2
                W3 = np.round(W3 * 2) / 2
        
        U = U3.reshape(nm, r)
        V = V3.reshape(mp, r)
        W = W3.reshape(pn, r)
        return U, V, W
    
    def to_list(U, V, W):
        return U.tolist(), V.tolist(), W.tolist()
    
    # Try ranks from 15 to 29
    for rank in range(15, 30):
        if time.time() > DEADLINE - 5:
            break
        
        found = False
        # Try multiple random inits
        for trial in range(50):
            if time.time() > DEADLINE - 5:
                break
            
            U, V, W = random_init(rank, seed=42 + trial)
            U, V, W = refine(U, V, W, max_iter=500, lr=0.05)
            
            # Snap to half-integers
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            if check(U, V, W):
                # Verify with exact check
                U_l, V_l, W_l = to_list(U, V, W)
                if check_rank_exact(U_l, V_l, W_l):
                    best_U, best_V, best_W = U_l, V_l, W_l
                    found = True
                    break
        
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
