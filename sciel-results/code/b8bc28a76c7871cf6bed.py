import numpy as np
from fractions import Fraction
import time

def _try_random_decomposition(n, m, p, target_rank, max_iters=1000, seed=42):
    """Try to find a rank-target_rank decomposition using random initialization and refinement."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n * m, target_rank))
    V = rng.standard_normal((m * p, target_rank))
    W = rng.standard_normal((p * n, target_rank))
    
    # The constraint is: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is a system of n*m*p equations in n*m*rank + m*p*rank + p*n*rank variables.
    # For n=2, m=3, p=10: 60 equations in 6*rank + 30*rank + 20*rank = 56*rank variables.
    # For rank=55: 60 equations in 3080 variables - very underdetermined.
    # We can solve this by setting up the tensor equation and solving.
    
    # Actually, let's use a different approach: set up the bilinear form and solve.
    # The tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We want T[i,j,k] = 1 for all i,j,k.
    
    # This is equivalent to: the tensor [T] = sum_r U_r \otimes V_r \otimes W_r = ones tensor
    # where U_r is a vector of length n*m, V_r of length m*p, W_r of length p*n.
    
    # We can solve this by: fix U and V, solve for W.
    # For each r, W_r must satisfy: sum_r (U_r[i*m+j] * V_r[j*p+k]) * W_r[k*n+i] = 1
    # This is a linear system in W_r for each (k,i) pair.
    
    # Actually, let's just verify the current solution and try to improve it.
    pass

def _verify_fast(U, V, W, n, m, p, rank, tol=1e-9):
    U_arr = np.asarray(U, dtype=np.float64)
    V_arr = np.asarray(V, dtype=np.float64)
    W_arr = np.asarray(W, dtype=np.float64)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U3[i, j, :] * V3[j, k, :], W3[k, i, :])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p, rank):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True

def _make_strassen_decomposition(n, m, p):
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        U[0 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1
        
        U[0 * 3 + 0][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 1][col] += 1
        V[1 * 10 + k0][col] += 1
        V[0 * 10 + k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        U[1 * 3 + 0][col] += 1
        U[0 * 3 + 0][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        U[0 * 3 + 1][col] += 1
        U[1 * 3 + 1][col] -= 1
        V[1 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
    
    assert col == rank
    return U, V, W

def _make_schoolbook(n, m, p):
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
    return U, V, W
