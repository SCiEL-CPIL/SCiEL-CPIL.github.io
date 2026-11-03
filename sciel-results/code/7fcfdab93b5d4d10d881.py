import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Helper to build tensor for sub-problem
    def build_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    def check_solution(U, V, W, T):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        D = np.einsum('at,bt,ct->abc', U, V, W)
        return np.allclose(D, T, atol=1e-6)

    def als_step(U, V, W, rank, n, m, p):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Update U
        for i in range(n):
            for j in range(m):
                u = i * m + j
                B = np.zeros((p, rank))
                b = np.ones(p)
                for k in range(p):
                    v = j * p + k
                    w = k * n + i
                    for r in range(rank):
                        B[k, r] = V[v, r] * W[w, r]
                U[u, :], residuals, rk, s = np.linalg.lstsq(B, b, rcond=None)
        
        # Update V
        for j in range(m):
            for k in range(p):
                v = j * p + k
                B = np.zeros((n, rank))
                b = np.ones(n)
                for i in range(n):
                    u = i * m + j
                    w = k * n + i
                    for r in range(rank):
                        B[i, r] = U[u, r] * W[w, r]
                V[v, :], residuals, rk, s = np.linalg.lstsq(B, b, rcond=None)
        
        # Update W
        for k in range(p):
            for i in range(n):
                w = k * n + i
                B = np.zeros((m, rank))
                b = np.ones(m)
                for j in range(m):
                    u = i * m + j
                    v = j * p + k
                    for r in range(rank):
                        B[j, r] = U[u, r] * V[v, r]
                W[w, :], residuals, rk, s = np.linalg.lstsq(B, b, rcond=None)
        
        return U, V, W

    def search_decomp(n, m, p, target_rank, max_restarts=5, max_iter=100):
        T = build_tensor(n, m, p)
        best_result = None
        
        # Try to find decomposition of target_rank
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            for iter in range(max_iter):
                if time.time() > deadline:
                    break
                U, V, W = als_step(U, V, W, target_rank, n, m, p)
                
                # Snap to integers/half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if check_solution(U_snap, V_snap, W_snap, T):
                    return U_snap, V_snap, W_snap
        
        return None

    # Strategy: Split into 3x3x3 and 3x3x1
    # Part 1: 3x3x3
    n1, m1, p1 = 3, 3, 3
    # Part 2: 3x3x1
    n2, m2, p2 = 3, 3, 1
    
    # Try to find rank 21 for part 1
    U1, V1, W1 = search_decomp(n1, m1, p1, 21, max_restarts=3, max_iter=50)
    
    if U1 is None:
        # Fallback to schoolbook for part 1 (rank 27)
        U1 = np.zeros((n1*m1, n1*m1*p1))
        V1 = np.zeros((m1*p1, n1*m1*p1))
        W1 = np.zeros((p1*n1, n1*m1*p1))
        col = 0
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    U1[i*m1+j, col] = 1.0
                    V1[j*p1+k, col] = 1.0
                    W1[k*n1+i, col] = 1.0
                    col += 1
        rank1 = 27
    else:
        rank1 = 21
    
    # Part 2 is always rank 9 (schoolbook is optimal for 3x3x1)
    U2 = np.zeros((n2*m2, n2*m2*p2))
    V2 = np.zeros((m2*p2, n2*m2*p2))
    W2 = np.zeros((p2*n2, n2*m2*p2))
    col = 0
    for i in range(n2):
        for j in range(m2):
            # k is always 0 in the local 3x3x1 system
            U2[i*m2+j, col] = 1.0
            V2[j*p2+0, col] = 1.0
            W2[0*n2+i, col] = 1.0
            col += 1
    rank2 = 9
    
    # Combine
    total_rank = rank1 + rank2
    U_final = np.zeros((n*m, total_rank))
    V_final = np.zeros((m*p, total_rank))
    W_final = np.zeros((p*n, total_rank))
    
    # Map Part 1 indices
    # U1 indices: i*3+j (i in 0..2, j in 0..2) -> Same as U_final indices for i in 0..2, j in 0..2
    # V1 indices: j*3+k (j in 0..2, k in 0..2) -> V_final indices j*4+k for k in 0..2
    # W1 indices: k*3+i (k in 0..2, i in 0..2) -> W_final indices k*3+i for k in 0..2
    
    # Map Part 2 indices
    # U2 indices: i*3+j (i in 0..2, j in 0..2) -> Same as U_final indices
    # V2 indices: j*1+0 (j in 0..2) -> V_final indices j*4+3
    # W2 indices: 0*3+i (i in 0..2) -> W_final indices 3*3+i
    
    # Fill U_final
    for i in range(3):
        for j in range(3):
            u_idx = i*3 + j
            U_final[u_idx, :rank1] = U1[u_idx, :]
            U_final[u_idx, rank1:rank1+rank2] = U2[u_idx, :]
            
    # Fill V_final
    for j in range(3):
        for k in range(3):
            v_idx = j*4 + k
            V_final[v_idx, :rank1] = V1[j*3+k, :]
        # k=3
        v_idx = j*4 + 3
        V_final[v_idx, rank1:rank1+rank2] = V2[j*1+0, :]
        
    # Fill W_final
    for k in range(3):
        for i in range(3):
            w_idx = k*3 + i
            W_final[w_idx, :rank1] = W1[k*3+i, :]
    # k=3
    for i in range(3):
        w_idx = 3*3 + i
        W_final[w_idx, rank1:rank1+rank2] = W2[0*3+i, :]
        
    # Final verification
    T = build_tensor(n, m, p)
    if not check_solution(U_final, V_final, W_final, T):
        # Fallback to full schoolbook
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

