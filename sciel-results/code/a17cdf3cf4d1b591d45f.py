"""Rewritten program for matrix multiplication tensor decomposition (3x5 by 5x7)."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 15
    mp = m * p  # 35
    pn = p * n  # 21
    
    schoolbook_rank = n * m * p  # 105
    
    # Schoolbook solution as fallback
    U_school = [[0] * schoolbook_rank for _ in range(nm)]
    V_school = [[0] * schoolbook_rank for _ in range(mp)]
    W_school = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    def verify(U, V, W, rank):
        """Verify if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def alternating_minimization(U, V, W, rank, max_iters=100, tol=1e-12):
        """Run alternating minimization to improve the decomposition."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
            
            # Fix V, W; solve for U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        for r in range(rank):
                            M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    try:
                        u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U_new[i*m+j, :] = u
                    except:
                        U_new[i*m+j, :] = U[i*m+j, :]
            U = U_new
            
            # Fix U, W; solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    N = np.zeros((n, rank))
                    for i in range(n):
                        for r in range(rank):
                            N[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    try:
                        v, _, _, _ = np.linalg.lstsq(N, np.ones(n), rcond=None)
                        V_new[j*p+k, :] = v
                    except:
                        V_new[j*p+k, :] = V[j*p+k, :]
            V = V_new
            
            # Fix U, V; solve for W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    P = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            P[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    try:
                        w, _, _, _ = np.linalg.lstsq(P, np.ones(m), rcond=None)
                        W_new[k*n+i, :] = w
                    except:
                        W_new[k*n+i, :] = W[k*n+i, :]
            W = W_new
            
            # Check convergence
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1.0) ** 2
            if err < tol:
                break
        
        return U.tolist(), V.tolist(), W.tolist(), err
    
    # Try ranks from lower bound upward
    lower_bound = max(nm, mp, pn)  # 35
    
    for rank in range(lower_bound + 1, min(schoolbook_rank, 100)):
        if time.time() > deadline - 30:
            break
        
        found = False
        for attempt in range(15):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Run alternating minimization
            U_list, V_list, W_list, err = alternating_minimization(U, V, W, rank, max_iters=80)
            
            if err < 1e-10:
                # Try to round to half-integers
                U_rounded = [[round(x * 2) / 2 for x in row] for row in U_list]
                V_rounded = [[round(x * 2) / 2 for x in row] for row in V_list]
                W_rounded = [[round(x * 2) / 2 for x in row] for row in W_list]
                
                if verify(U_rounded, V_rounded, W_rounded, rank):
                    best_result = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                    best_rank = rank
                    found = True
                    break
                else:
                    # Check if already close to half-integers
                    max_diff = max(max(abs(x - round(x*2)/2) for x in row) for row in U_list)
                    max_diff = max(max_diff, max(max(abs(x - round(x*2)/2) for x in row) for row in V_list))
                    max_diff = max(max_diff, max(max(abs(x - round(x*2)/2) for x in row) for row in W_list))
                    
                    if max_diff < 1e-6 and verify(U_list, V_list, W_list, rank):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        found = True
                        break
        
        if found:
            break
    
    return best_result
