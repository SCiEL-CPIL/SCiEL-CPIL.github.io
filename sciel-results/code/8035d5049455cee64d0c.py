import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

def _verify_exact(U, V, W, n, m, p):
    rank = len(U[0])
    nm, mp, pn = n * m, m * p, p * n
    for u in range(nm):
        i, j = u // m, u % m
        for v in range(mp):
            j_v, k = v // p, v % p
            for w in range(pn):
                i_w, k_w = w % n, w // n
                expected = 1 if (j == j_v and i == i_w and k == k_w) else 0
                val = Fraction(0)
                for c in range(rank):
                    val += U[u][c] * V[v][c] * W[w][c]
                if val != expected:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U); V = np.array(V); W = np.array(W)
    nm, mp, pn = n*m, m*p, p*n
    for u in range(nm):
        i, j = u // m, u % m
        for v in range(mp):
            j_v, k = v // p, v % p
            for w in range(pn):
                i_w, k_w = w % n, w // n
                expected = 1.0 if (j == j_v and i == i_w and k == k_w) else 0.0
                if abs(np.dot(U[u] * V[v], W[w]) - expected) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115
    n, m, p = 3, 5, 7
    nm, mp, pn = n*m, m*p, p*n
    
    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find a lower-rank exact decomposition using structured search
    # Strategy: use ALS with exact rational arithmetic and structured initialization
    
    def try_structured_als(r, num_iters=30, num_restarts=10):
        if time.time() > deadline:
            return None
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                return None
            
            # Structured initialization: use random sparse matrices
            np.random.seed(12345 + r * 100 + restart)
            
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            
            # Initialize with sparse random values from {0, 0.5, 1, -0.5, -1}
            for c in range(r):
                # Each column has a few non-zero entries
                nz_u = np.random.choice(nm, size=min(3, nm), replace=False)
                nz_v = np.random.choice(mp, size=min(4, mp), replace=False)
                nz_w = np.random.choice(pn, size=min(3, pn), replace=False)
                for idx in nz_u:
                    U[idx, c] = np.random.choice([0.5, 1.0, -0.5, -1.0])
                for idx in nz_v:
                    V[idx, c] = np.random.choice([0.5, 1.0, -0.5, -1.0])
                for idx in nz_w:
                    W[idx, c] = np.random.choice([0.5, 1.0, -0.5, -1.0])
            
            for it in range(num_iters):
                if time.time() > deadline:
                    return None
                
                # ALS step for U
                M = np.zeros((mp * pn, r))
                for c in range(r):
                    M[:, c] = np.outer(V[:, c], W[:, c]).flatten()
                T_flat = T.reshape(nm, mp * pn)
                try:
                    U_sol, _, _, _ = np.linalg.lstsq(M, T_flat.T, rcond=None)
                    U = U_sol.T
                except:
                    continue
                
                # ALS step for V
                T_flat_v = np.zeros((mp, nm * pn))
                for v in range(mp):
                    T_flat_v[v, :] = T[:, v, :].reshape(-1)
                M_v = np.zeros((nm * pn, r))
                for c in range(r):
                    M_v[:, c] = np.outer(U[:, c], W[:, c]).flatten()
                try:
                    V_sol, _, _, _ = np.linalg.lstsq(M_v, T_flat_v.T, rcond=None)
                    V = V_sol.T
                except:
                    continue
                
                # ALS step for W
                T_flat_w = np.zeros((pn, nm * mp))
                for w in range(pn):
                    T_flat_w[w, :] = T[:, :, w].reshape(-1)
                M_w = np.zeros((nm * mp, r))
                for c in range(r):
                    M_w[:, c] = np.outer(U[:, c], V[:, c]).flatten()
                try:
                    W_sol, _, _, _ = np.linalg.lstsq(M_w, T_flat_w.T, rcond=None)
                    W = W_sol.T
                except:
                    continue
            
            # Round to half-integers
            U_r = np.round(U * 2.0) / 2.0
            V_r = np.round(V * 2.0) / 2.0
            W_r = np.round(W * 2.0) / 2.0
            
            # Check closeness
            if (np.max(np.abs(U - U_r)) > 0.05 or 
                np.max(np.abs(V - V_r)) > 0.05 or 
                np.max(np.abs(W - W_r)) > 0.05):
                continue
            
            # Fast verification
            if not _verify_np(U_r, V_r, W_r, n, m, p):
                continue
            
            # Exact verification
            U_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in U_r]
            V_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in V_r]
            W_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in W_r]
            
            if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                return {
                    "U": [[float(x) for x in row] for row in U_frac],
                    "V": [[float(x) for x in row] for row in V_frac],
                    "W": [[float(x) for x in row] for row in W_frac]
                }
        
        return None
    
    # Try ranks from 104 down to 90
    for r in range(104, 89, -1):
        if time.time() > deadline:
            break
        result = try_structured_als(r, num_iters=20, num_restarts=5)
        if result is not None:
            return result
    
    return schoolbook()
