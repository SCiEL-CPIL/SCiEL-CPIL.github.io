# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    rank_schoolbook = n * m * p
    
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Pre-compute the target tensor indices for fast checking
    # T is sparse. Indices where T=1 are (i*m+j, j*p+k, k*n+i)
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_indices.append((u, v, w))
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the tensor."""
        R = U.shape[1]
        # Check the '1' entries
        for u, v, w in target_indices:
            val = np.dot(U[u] * V[v], W[w])
            if abs(val - 1.0) > tol:
                return False
        
        # Check a sample of '0' entries to ensure sparsity is maintained
        # We can't check all (nm*mp*pn) entries if they are many.
        # nm=12, mp=20, pn=15. Total 3600.
        # 3600 dot products of length R is fast.
        # Let's check all of them for correctness.
        for u in range(n * m):
            for v in range(m * p):
                for w in range(p * n):
                    if (u, v, w) in target_indices:
                        continue
                    val = np.dot(U[u] * V[v], W[w])
                    if abs(val) > tol:
                        return False
        return True

    def round_to_half(x):
        return np.round(x * 2) / 2

    def solve_perturbed(target_rank, U_init, V_init, W_init, max_iters=500):
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update W
            # For each (k, i), A = sum_j U[i*m+j] * V[j*p+k]
            # W[k*n+i] = A / (A.A)
            for k in range(p):
                for i in range(n):
                    A = np.zeros(target_rank)
                    for j in range(m):
                        A += U[i * m + j] * V[j * p + k]
                    norm_sq = np.dot(A, A)
                    if norm_sq > 1e-12:
                        W[k * n + i] = A / norm_sq
            
            # Update U
            # For each (i, j), B[k, r] = V[j*p+k, r] * W[k*n+i, r]
            # U[i*m+j] = B^T (B B^T)^{-1} 1
            for i in range(n):
                for j in range(m):
                    B = np.zeros((p, target_rank))
                    for k in range(p):
                        B[k] = V[j * p + k] * W[k * n + i]
                    try:
                        BBT = B.dot(B.T)
                        BBT += 1e-9 * np.eye(p)
                        U[i * m + j] = np.linalg.solve(BBT, np.ones(p)).dot(B)
                    except np.linalg.LinAlgError:
                        pass
            
            # Update V
            # For each (j, k), C[i, r] = U[i*m+j, r] * W[k*n+i, r]
            # V[j*p+k] = C^T (C C^T)^{-1} 1
            for j in range(m):
                for k in range(p):
                    C = np.zeros((n, target_rank))
                    for i in range(n):
                        C[i] = U[i * m + j] * W[k * n + i]
                    try:
                        CCT = C.dot(C.T)
                        CCT += 1e-9 * np.eye(n)
                        V[j * p + k] = np.linalg.solve(CCT, np.ones(n)).dot(C)
                    except np.linalg.LinAlgError:
                        pass
            
            # Early exit check
            # Just check the '1' entries for speed during iteration
            err = 0
            for u, v, w in target_indices:
                val = np.dot(U[u] * V[v], W[w])
                err += (val - 1.0)**2
            if err < 1e-12:
                # Full check
                if check_decomposition(U, V, W):
                    return U, V, W, True
        
        if check_decomposition(U, V, W):
            return U, V, W, True
        return U, V, W, False

    # Try to find a lower rank decomposition
    # Start from 59 down to 50
    for target_rank in range(59, 49, -1):
        if time.time() > deadline - 10:
            break
        
        U_init = U[:, :target_rank].copy()
        V_init = V[:, :target_rank].copy()
        W_init = W[:, :target_rank].copy()
        
        U_sol, V_sol, W_sol, success = solve_perturbed(target_rank, U_init, V_init, W_init)
        
        if success:
            U_rounded = round_to_half(U_sol)
            V_rounded = round_to_half(V_sol)
            W_rounded = round_to_half(W_sol)
            
            if check_decomposition(U_rounded, V_rounded, W_rounded):
                best_rank = target_rank
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                break
        
    U_list = [[float(best_U[i, j]) for j in range(best_rank)] for i in range(n * m)]
    V_list = [[float(best_V[i, j]) for j in range(best_rank)] for i in range(m * p)]
    W_list = [[float(best_W[i, j]) for j in range(best_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
