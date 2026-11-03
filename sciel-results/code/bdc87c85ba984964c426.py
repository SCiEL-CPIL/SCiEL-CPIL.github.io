"""Matrix multiplication 3,3,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    nm, mp, pn = n*m, m*p, p*n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn), dtype=float)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T[i, j, k] = 1.0

    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if U else 0
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        T_rec = np.einsum('ic,jc,kc->ijk', U_arr, V_arr, W_arr)
        
        if not np.allclose(T_rec, T, atol=1e-9):
            return False
        return True

    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    def try_rank_als(r, max_iters=200, seed=42, tol=1e-10):
        """
        Alternating Least Squares for tensor decomposition.
        T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
        """
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(nm, r) * 0.5
        V = np.random.randn(mp, r) * 0.5
        W = np.random.randn(pn, r) * 0.5
        
        # Pre-compute non-zero indices for faster updates
        # T is sparse: T[i,j,k] = 1 if i = a*m+b, j = b*p+c, k = c*n+a for some a,b,c
        # For each i, find all (j,k) where T[i,j,k] != 0
        # T[i,j,k] = 1 iff there exists (a,b,c) such that i = a*m+b, j = b*p+c, k = c*n+a
        # Given i, a = i // m, b = i % m
        # Then j = b*p + c, k = c*n + a for c in range(p)
        
        # For each i, the non-zero (j,k) pairs are:
        # a = i // m, b = i % m
        # for c in range(p): j = b*p + c, k = c*n + a
        
        # Pre-compute for each i the list of (j, k) pairs
        T_nz = {}
        for i in range(nm):
            a = i // m
            b = i % m
            pairs = []
            for c in range(p):
                j = b * p + c
                k = c * n + a
                pairs.append((j, k))
            T_nz[i] = pairs
        
        # Similarly for each j, find all (i, k) pairs
        T_nz_j = {}
        for j in range(mp):
            b = j // p
            c = j % p
            pairs = []
            for a in range(n):
                i = a * m + b
                k = c * n + a
                pairs.append((i, k))
            T_nz_j[j] = pairs
        
        # For each k, find all (i, j) pairs
        T_nz_k = {}
        for k in range(pn):
            c = k // n
            a = k % n
            pairs = []
            for b in range(m):
                i = a * m + b
                j = b * p + c
                pairs.append((i, j))
            T_nz_k[k] = pairs
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U
            for i in range(nm):
                num = np.zeros(r)
                den = np.zeros(r)
                for j, k in T_nz[i]:
                    vw = V[j, :] * W[k, :]
                    num += vw
                    den += vw ** 2
                mask = den > 1e-14
                U[i, mask] = num[mask] / den[mask]
            
            # Update V
            for j in range(mp):
                num = np.zeros(r)
                den = np.zeros(r)
                for i, k in T_nz_j[j]:
                    uw = U[i, :] * W[k, :]
                    num += uw
                    den += uw ** 2
                mask = den > 1e-14
                V[j, mask] = num[mask] / den[mask]
            
            # Update W
            for k in range(pn):
                num = np.zeros(r)
                den = np.zeros(r)
                for i, j in T_nz_k[k]:
                    uv = U[i, :] * V[j, :]
                    num += uv
                    den += uv ** 2
                mask = den > 1e-14
                W[k, mask] = num[mask] / den[mask]
            
            # Compute error
            if it % 5 == 0 or it == max_iters - 1:
                T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
                err = np.max(np.abs(T_rec - T))
                if err < tol:
                    # Try to round to rationals
                    U_frac = [[Fraction(x).limit_denominator(10000) for x in U[i, :]] for i in range(nm)]
                    V_frac = [[Fraction(x).limit_denominator(10000) for x in V[j, :]] for j in range(mp)]
                    W_frac = [[Fraction(x).limit_denominator(10000) for x in W[k, :]] for k in range(pn)]
                    
                    U_list = [[float(f) for f in row] for row in U_frac]
                    V_list = [[float(f) for f in row] for row in V_frac]
                    W_list = [[float(f) for f in row] for row in W_frac]
                    
                    if verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
                
                # Check for convergence
                if abs(prev_err - err) < 1e-12 and err > tol:
                    break
                prev_err = err
        
        return None

    # First, get the schoolbook as fallback
    result = schoolbook()
    
    # Try lower ranks with multiple seeds
    # For (3,3,5), try ranks from 44 down to 35
    # Use multiple seeds for each rank to improve chances of finding a solution
    for r in range(44, 34, -1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple seeds for this rank
        found = False
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            
            res = try_rank_als(r, max_iters=100, seed=42 + r * 100 + seed)
            if res is not None:
                if verify(res["U"], res["V"], res["W"]):
                    result = res
                    found = True
                    break
        
        if found:
            # Continue trying even lower ranks
            continue
    
    return result
# EVOLVE-BLOCK-END
