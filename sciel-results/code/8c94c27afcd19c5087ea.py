import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 130.0  # safety margin
    
    # Verification function
    def verify(U, V, W, tol=1e-9):
        """Verify decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Build initial Strassen-based decomposition
    def build_strassen():
        U_cols = []
        V_cols = []
        W_cols = []
        
        def add_term(u, v, w):
            U_cols.append(u.copy())
            V_cols.append(v.copy())
            W_cols.append(w.copy())
        
        def u_idx(i, j):
            return i * m + j
        
        def v_idx(j, k):
            return j * p + k
        
        def w_idx(k, i):
            return k * n + i
        
        def strassen_2x2(ai, aj, bj, bk):
            # 7 Strassen terms for 2x2*2x2
            # M1 = (a+d)(e+h)
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            u[u_idx(ai,aj)]=1; u[u_idx(ai+1,aj+1)]=1
            v[v_idx(bj,bk)]=1; v[v_idx(bj+1,bk+1)]=1
            w[w_idx(bk,ai)]=1; w[w_idx(bk+1,ai+1)]=1
            add_term(u,v,w)
            # M2 = (c+d)e
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            u[u_idx(ai+1,aj)]=1; u[u_idx(ai+1,aj+1)]=1
            v[v_idx(bj,bk)]=1
            w[w_idx(bk,ai+1)]=1; w[w_idx(bk+1,ai+1)]=-1
            add_term(u,v,w)
            # M3 = a(f-h)
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            u[u_idx(ai,aj)]=1
            v[v_idx(bj,bk+1)]=1; v[v_idx(bj+1,bk+1)]=-1
            w[w_idx(bk+1,ai)]=1; w[w_idx(bk+1,ai+1)]=1
            add_term(u,v,w)
            # M4 = d(g-e)
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            u[u_idx(ai+1,aj+1)]=1
            v[v_idx(bj+1,bk)]=1; v[v_idx(bj,bk)]=-1
            w[w_idx(bk,ai)]=1; w[w_idx(bk,ai+1)]=1
            add_term(u,v,w)
            # M5 = (a+b)h
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            u[u_idx(ai,aj)]=1; u[u_idx(ai,aj+1)]=1
            v[v_idx(bj+1,bk+1)]=1
            w[w_idx(bk,ai)]=-1; w[w_idx(bk+1,ai)]=1
            add_term(u,v,w)
            # M6 = (c-a)(e+f)
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            u[u_idx(ai+1,aj)]=1; u[u_idx(ai,aj)]=-1
            v[v_idx(bj,bk)]=1; v[v_idx(bj,bk+1)]=1
            w[w_idx(bk+1,ai+1)]=1
            add_term(u,v,w)
            # M7 = (b-d)(g+h)
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            u[u_idx(ai,aj+1)]=1; u[u_idx(ai+1,aj+1)]=-1
            v[v_idx(bj+1,bk)]=1; v[v_idx(bj+1,bk+1)]=1
            w[w_idx(bk,ai)]=1
            add_term(u,v,w)
        
        # 2x2x3 block: 7 Strassen + 4 direct = 11 terms
        def block_2x2x3(ai, aj, bj, bk):
            strassen_2x2(ai, aj, bj, bk)
            # 4 extra terms for the 3rd column of B
            for di in range(2):
                for dj in range(2):
                    u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                    u[u_idx(ai+di, aj+dj)]=1
                    v[v_idx(bj+dj, bk+2)]=1
                    w[w_idx(bk+2, ai+di)]=1
                    add_term(u,v,w)
        
        # Build 4x4x5 using blocks
        # Rows 0-1, Cols 0-1: 2x2x2 -> 7 terms
        strassen_2x2(0, 0, 0, 0)
        # Rows 0-1, Cols 2-3: 2x2x2 -> 7 terms  
        strassen_2x2(0, 2, 2, 0)
        # Rows 0-1, Col 4: 2x1x1 -> 2 terms
        for j in range(2):
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            u[u_idx(0, j)]=1
            v[v_idx(j, 4)]=1
            w[w_idx(4, 0)]=1
            add_term(u,v,w)
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            u[u_idx(1, j)]=1
            v[v_idx(j, 4)]=1
            w[w_idx(4, 1)]=1
            add_term(u,v,w)
        # Rows 2-3, Cols 0-1: 2x2x2 -> 7 terms
        strassen_2x2(2, 0, 0, 0)
        # Rows 2-3, Cols 2-3: 2x2x2 -> 7 terms
        strassen_2x2(2, 2, 2, 0)
        # Rows 2-3, Col 4: 2x1x1 -> 2 terms
        for j in range(2):
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            u[u_idx(2, j)]=1
            v[v_idx(j, 4)]=1
            w[w_idx(4, 2)]=1
            add_term(u,v,w)
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            u[u_idx(3, j)]=1
            v[v_idx(j, 4)]=1
            w[w_idx(4, 3)]=1
            add_term(u,v,w)
        
        U = np.array(U_cols).T
        V = np.array(V_cols).T
        W = np.array(W_cols).T
        return U, V, W

    # Round to half-integers
    def round_to_half(x):
        return np.round(2*x)/2.0

    # ALS decomposition: given target tensor, find low-rank U,V,W
    def als_decompose(target_U, target_V, target_W, rank, max_iter=200, tol=1e-10):
        """
        Given an initial solution, try to find a rank-rank decomposition via ALS.
        We minimize ||U*V*W - T|| where T is the multiplication tensor.
        """
        n, m, p = DIMENSIONS
        rng = random.Random(42)
        
        # Initialize with random rank-rank matrices
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Initialize from the given solution (project to lower rank)
        if rank <= target_U.shape[1]:
            U = target_U[:, :rank].copy()
            V = target_V[:, :rank].copy()
            W = target_W[:, :rank].copy()
        else:
            # Random initialization
            U = rng.gauss(0, 1, (n*m, rank))
            V = rng.gauss(0, 1, (m*p, rank))
            W = rng.gauss(0, 1, (p*n, rank))
        
        # Precompute the tensor structure
        # For each (i,j,k): we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # ALS: fix two, solve for third
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (i,j), solve linear system
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    # sum_r U[u_idx,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                    # Let A_k = V[j*p+k,:] * W[k*n+i,:] (element-wise product)
                    # We need: sum_r U[u_idx,r] * A_k[r] = 1 for all k
                    # This is: A @ U[u_idx,:] = ones(p)
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve A @ x = ones(p)
                    try:
                        U[u_idx, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k)
            for j in range(m):
                for k in range(p):
                    v_idx = j * p + k
                    # sum_r U[i*m+j,r] * V[v_idx,r] * W[k*n+i,r] = 1 for all i
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[v_idx, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i)
            for k in range(p):
                for i in range(n):
                    w_idx = k * n + i
                    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[w_idx,r] = 1 for all j
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[w_idx, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            if it % 10 == 0 and verify(U, V, W, tol=1e-6):
                # Try rounding
                U_r = round_to_half(U)
                V_r = round_to_half(V)
                W_r = round_to_half(W)
                if verify(U_r, V_r, W_r, tol=1e-9):
                    return U_r, V_r, W_r, True
        
        return U, V, W, verify(U, V, W, tol=1e-9)

    # Build initial solution
    U0, V0, W0 = build_strassen()
    initial_rank = U0.shape[1]
    
    # Verify initial
    assert verify(U0, V0, W0), "Initial Strassen decomposition failed!"
    
    best_rank = initial_rank
    best_U = U0.copy()
    best_V = V0.copy()
    best_W = W0.copy()
    
    # Try ALS with decreasing rank
    for target_rank in range(initial_rank - 1, max(initial_rank - 20, 40), -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            if attempt == 0:
                # Start from the full solution, project
                U_init = U0[:, :target_rank].copy()
                V_init = V0[:, :target_rank].copy()
                W_init = W0[:, :target_rank].copy()
            else:
                # Perturbed start
                rng = random.Random(100 + attempt)
                U_init = U0[:, :target_rank].copy() + rng.gauss(0, 0.1, (n*m, target_rank))
                V_init = V0[:, :target_rank].copy() + rng.gauss(0, 0.1, (m*p, target_rank))
                W_init = W0[:, :target_rank].copy() + rng.gauss(0, 0.1, (p*n, target_rank))
            
            U_res, V_res, W_res, ok = als_decompose(U_init, V_init, W_init, target_rank, max_iter=300)
            
            if ok:
                best_rank = target_rank
                best_U = U_res
                best_V = V_res
                best_W = W_res
                break  # Found a solution at this rank, try lower
        
        if best_rank < initial_rank:
            # Update initial for next iterations
            U0 = best_U
            V0 = best_V
            W0 = best_W
    
    # Final verification
    if not verify(best_U, best_V, best_W):
        # Fall back to original
        U0, V0, W0 = build_strassen()
        best_U = U0
        best_V = V0
        best_W = W0
        best_rank = U0.shape[1]
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
