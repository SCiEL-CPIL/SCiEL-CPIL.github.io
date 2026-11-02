"""Program for matrix multiplication tensor decomposition (3x4x5)."""
import numpy as np
import time
import random
import sys
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    schoolbook_rank = n * m * p  # 60
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe within 150s limit
    
    # 1. Construct the initial schoolbook decomposition
    U0 = np.zeros((dim_u, schoolbook_rank))
    V0 = np.zeros((dim_v, schoolbook_rank))
    W0 = np.zeros((dim_w, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
                
    def verify_numeric(U, V, W, tol=1e-10):
        """Fast numeric verification for screening"""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target = 1.0
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - target) > tol:
                        return False
        return True

    def verify_exact(U, V, W):
        """Exact verification using Fractions. Only called on candidates."""
        r = U.shape[1]
        U_frac = [[Fraction(U[i, j]) for j in range(r)] for i in range(dim_u)]
        V_frac = [[Fraction(V[i, j]) for j in range(r)] for i in range(dim_v)]
        W_frac = [[Fraction(W[i, j]) for j in range(r)] for i in range(dim_w)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target = Fraction(1)
                    val = Fraction(0)
                    for t in range(r):
                        val += U_frac[i*m+j][t] * V_frac[j*p+k][t] * W_frac[k*n+i][t]
                    if val != target:
                        return False
        return True

    if not verify_numeric(U0, V0, W0):
        raise ValueError("Schoolbook decomposition failed verification")

    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = schoolbook_rank

    # ALS (Alternating Least Squares) search
    def run_als_search(rank, max_iter=50, restarts=5):
        """Run ALS for a specific rank. Returns best solution found."""
        best_err = float('inf')
        best_sol = None
        
        for _ in range(restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize with small random noise
            U = np.random.randn(dim_u, rank)
            V = np.random.randn(dim_v, rank)
            W = np.random.randn(dim_w, rank)
            
            prev_err = float('inf')
            
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # Solve M U_{ij,:}^T = T_{ij,:}
                        # M[k, t] = V[j*p+k, t] * W[k*n+i, t]
                        M = np.zeros((p, rank))
                        for k in range(p):
                            for t in range(rank):
                                M[k, t] = V[j*p+k, t] * W[k*n+i, t]
                        T_vec = np.ones(p) # Target is 1
                        # Least squares
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(M, T_vec, rcond=None)[0]
                        except np.linalg.LinAlgError:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        # Solve N V_{jk,:}^T = T_{jk,:}
                        # N[i, t] = U[i*m+j, t] * W[k*n+i, t]
                        N = np.zeros((n, rank))
                        for i in range(n):
                            for t in range(rank):
                                N[i, t] = U[i*m+j, t] * W[k*n+i, t]
                        T_vec = np.ones(n)
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(N, T_vec, rcond=None)[0]
                        except np.linalg.LinAlgError:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        # Solve P W_{ki,:}^T = T_{ki,:}
                        # P[j, t] = U[i*m+j, t] * V[j*p+k, t]
                        P = np.zeros((m, rank))
                        for j in range(m):
                            for t in range(rank):
                                P[j, t] = U[i*m+j, t] * V[j*p+k, t]
                        T_vec = np.ones(m)
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(P, T_vec, rcond=None)[0]
                        except np.linalg.LinAlgError:
                            pass
                
                # Calculate error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err += (val - 1.0)**2
                
                if err < best_err:
                    best_err = err
                    best_sol = (U.copy(), V.copy(), W.copy())
                
                if err < 1e-10:
                    return best_sol
                
                # Check for stagnation
                if abs(prev_err - err) < 1e-12:
                    break
                prev_err = err
        
        return best_sol

    # Try ranks from lower bound up to best_rank - 1
    # Lower bound is roughly 20 for 3x4x5 based on volume/complexity heuristics
    # We will try a few specific ranks
    ranks_to_try = [20, 21, 22, 23, 24, 25, 30, 40, 50]
    ranks_to_try = [r for r in ranks_to_try if r < best_rank]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        sol = run_als_search(r, max_iter=100, restarts=10)
        if sol is not None:
            U_cand, V_cand, W_cand = sol
            
            # Try to snap to half-integers
            # We check if the solution is close to half-integers
            U_snap = np.round(U_cand * 2) / 2.0
            V_snap = np.round(V_cand * 2) / 2.0
            W_snap = np.round(W_cand * 2) / 2.0
            
            # Verify snapped version
            if verify_numeric(U_snap, V_snap, W_snap, tol=1e-8):
                if verify_exact(U_snap, V_snap, W_snap):
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    best_rank = r
                    break # Found a valid decomposition for this rank, try lower? 
                    # Actually, we want the *minimal* rank. 
                    # So we should continue to lower ranks? 
                    # But we are iterating upwards in the list above? 
                    # Let's reverse the list to try lower ranks first? 
                    # No, lower ranks are harder. 
                    # If we find one at r=20, that's great. 
                    # If we find one at r=30, we should still try r=25, etc.
                    # But the list is fixed. 
                    # Let's just keep the best found.
            
            # Also check if the unsnapped solution is "close enough" to be considered 
            # but the task requires exact integer/half-integer entries.
            # So we MUST snap.

    # If we didn't find anything better, return schoolbook
    if not verify_numeric(best_U, best_V, best_W, tol=1e-10):
        best_U, best_V, best_W = U0, V0, W0
        best_rank = schoolbook_rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
