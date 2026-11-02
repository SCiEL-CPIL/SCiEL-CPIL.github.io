import time
import numpy as np
import random

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_solution = (rank_sb, U_sb, V_sb, W_sb)
    
    def verify(U, V, W, r, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for t in range(r):
                        val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def total_error(U, V, W, r):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for t in range(r):
                        val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    err += (val - 1.0) ** 2
        return err
    
    # Try to find lower rank solutions
    for r in range(40, rank_sb):
        if time.time() > deadline - 20:
            break
        
        found = False
        for restart in range(50):
            if time.time() > deadline - 20:
                break
            
            vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
            U = np.array([[random.choice(vals) for _ in range(r)] for _ in range(nm)])
            V = np.array([[random.choice(vals) for _ in range(r)] for _ in range(mp)])
            W = np.array([[random.choice(vals) for _ in range(r)] for _ in range(pn)])
            
            # Iterative refinement
            for it in range(300):
                if time.time() > deadline - 20:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, r))
                        b = np.ones(p)
                        for k in range(p):
                            for t in range(r):
                                A[k, t] = V[j*p+k, t] * W[k*n+i, t]
                        try:
                            sol, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                            U[i*m+j, :] = sol
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        b = np.ones(n)
                        for i in range(n):
                            for t in range(r):
                                A[i, t] = U[i*m+j, t] * W[k*n+i, t]
                        try:
                            sol, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                            V[j*p+k, :] = sol
                        except:
                            pass
                
                # Update W
                for i in range(n):
                    for k in range(p):
                        A = np.zeros((m, r))
                        b = np.ones(m)
                        for j in range(m):
                            for t in range(r):
                                A[j, t] = U[i*m+j, t] * V[j*p+k, t]
                        try:
                            sol, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                            W[k*n+i, :] = sol
                        except:
                            pass
                
                err = total_error(U, V, W, r)
                if err < 1e-10:
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if verify(U_r, V_r, W_r, r):
                        best_solution = (r, U_r, V_r, W_r)
                        found = True
                        break
                    # Local search for half-integer solution
                    for ls_step in range(2000):
                        if time.time() > deadline - 20:
                            break
                        which = random.randint(0, 2)
                        if which == 0:
                            idx = random.randint(0, nm-1)
                            t = random.randint(0, r-1)
                            old = U_r[idx, t]
                            U_r[idx, t] = old + random.choice([-1, -0.5, 0.5, 1])
                        elif which == 1:
                            idx = random.randint(0, mp-1)
                            t = random.randint(0, r-1)
                            old = V_r[idx, t]
                            V_r[idx, t] = old + random.choice([-1, -0.5, 0.5, 1])
                        else:
                            idx = random.randint(0, pn-1)
                            t = random.randint(0, r-1)
                            old = W_r[idx, t]
                            W_r[idx, t] = old + random.choice([-1, -0.5, 0.5, 1])
                        
                        if verify(U_r, V_r, W_r, r):
                            best_solution = (r, U_r, V_r, W_r)
                            found = True
                            break
                        else:
                            if which == 0:
                                U_r[idx, t] = old
                            elif which == 1:
                                V_r[idx, t] = old
                            else:
                                W_r[idx, t] = old
                    
                    if found:
                        break
            
            if found:
                break
        
        if found:
            break
    
    r, U, V, W = best_solution
    U_out = [[U[i, j] for j in range(r)] for i in range(nm)]
    V_out = [[V[i, j] for j in range(r)] for i in range(mp)]
    W_out = [[W[i, j] for j in range(r)] for i in range(pn)]
    
    return {"U": U_out, "V": V_out, "W": W_out}
