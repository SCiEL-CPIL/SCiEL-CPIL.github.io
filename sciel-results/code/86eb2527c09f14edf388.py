"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    """
    n, m, p = 3, 3, 5
    start_time = time.time()
    deadline = 115  # safety margin before 120s limit
    
    # Build the multiplication tensor T with shape (n*m, m*p, p*n)
    # T[i, j, k] = 1 if there's a multiplication term, else 0
    # Actually, the tensor T is defined such that:
    # T_{(i,j),(j,k),(k,i)} = 1 for the standard basis
    # More precisely: T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a, b, c] = 1 if j == j' and k == k' and i == i', else 0
    
    # Actually, let me think more carefully.
    # The matrix multiplication tensor T is in V* ⊗ W* ⊗ U* where:
    # V = R^{n*m} (entries of A), W = R^{m*p} (entries of B), U = R^{p*n} (entries of C)
    # T(A, B, C) = sum_{i,j,k} C_{k,i} * A_{i,j} * B_{j,k}
    # So T is a trilinear form: T = sum_{i,j,k} e_{(k,i)}^* ⊗ e_{(i,j)}^* ⊗ e_{(j,k)}^*
    # As a tensor in V ⊗ W ⊗ U, we have T_{a,b,c} = 1 if (a,b,c) = ((k,i),(i,j),(j,k)) for some i,j,k
    
    # Let me build T as a 3D array of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j  # index in A
                b = j * p + k  # index in B
                c = k * n + i  # index in C (trace convention: W[k*n+i])
                T[a, b, c] = 1
    
    # The trivial schoolbook decomposition has rank n*m*p = 45
    rank_schoolbook = n * m * p
    
    # Create the schoolbook decomposition as a fallback
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    # Try to find a lower rank decomposition using iterative refinement
    best_rank = rank_schoolbook
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Try ranks from a reasonable lower bound up to schoolbook
    for target_rank in range(23, rank_schoolbook + 1):
        if time.time() - start_time > deadline - 10:
            break
        
        # Initialize with the schoolbook decomposition, truncated
        U_try = U[:, :target_rank].copy()
        V_try = V[:, :target_rank].copy()
        W_try = W[:, :target_rank].copy()
        
        # Also try random initialization
        np.random.seed(42 + target_rank)
        U_rand = np.random.randn(n * m, target_rank)
        V_rand = np.random.randn(m * p, target_rank)
        W_rand = np.random.randn(p * n, target_rank)
        
        # Try both initializations
        for init_U, init_V, init_W in [(U_try, V_try, W_try), (U_rand, V_rand, W_rand)]:
            if time.time() - start_time > deadline - 5:
                break
            
            U_cur = init_U.copy()
            V_cur = init_V.copy()
            W_cur = init_W.copy()
            
            # Iterative refinement
            for iteration in range(200):
                if time.time() - start_time > deadline - 3:
                    break
                
                # Update U
                for r in range(target_rank):
                    residual = T.copy()
                    for rr in range(target_rank):
                        if rr != r:
                            residual -= U_cur[:, rr, np.newaxis] * V_cur[np.newaxis, :, rr] * W_cur[np.newaxis, np.newaxis, :]
                    # This approach is O(n*m*p*n) per update, which is fine for small sizes
                    denom = np.sum(V_cur[:, r]**2) * np.sum(W_cur[:, r]**2)
                    if abs(denom) < 1e-12:
                        continue
                    for a in range(n*m):
                        U_cur[a, r] = np.sum(residual[a, :, :] * V_cur[:, r, np.newaxis] * W_cur[np.newaxis, :, r]) / denom
                
                # Update V
                for r in range(target_rank):
                    residual = T.copy()
                    for rr in range(target_rank):
                        if rr != r:
                            residual -= U_cur[:, rr, np.newaxis] * V_cur[np.newaxis, :, rr] * W_cur[np.newaxis, np.newaxis, :]
                    denom = np.sum(U_cur[:, r]**2) * np.sum(W_cur[:, r]**2)
                    if abs(denom) < 1e-12:
                        continue
                    for b in range(m*p):
                        V_cur[b, r] = np.sum(residual[:, b, :] * U_cur[:, r, np.newaxis] * W_cur[np.newaxis, :, r]) / denom
                
                # Update W
                for r in range(target_rank):
                    residual = T.copy()
                    for rr in range(target_rank):
                        if rr != r:
                            residual -= U_cur[:, rr, np.newaxis] * V_cur[np.newaxis, :, rr] * W_cur[np.newaxis, np.newaxis, :]
                    denom = np.sum(U_cur[:, r]**2) * np.sum(V_cur[:, r]**2)
                    if abs(denom) < 1e-12:
                        continue
                    for c in range(p*n):
                        W_cur[c, r] = np.sum(residual[:, :, c] * U_cur[:, r, np.newaxis] * V_cur[np.newaxis, :, r]) / denom
                
                # Compute error
                T_approx = np.zeros((n*m, m*p, p*n))
                for r in range(target_rank):
                    T_approx += U_cur[:, r, np.newaxis] * V_cur[np.newaxis, :, r] * W_cur[np.newaxis, np.newaxis, :]
                err = np.linalg.norm(T - T_approx)
            
            if err < 1e-6:
                best_rank = target_rank
                best_U, best_V, best_W = U_cur, V_cur, W_cur
                break
        
        if best_rank == target_rank:
            break
    
    if best_rank < rank_schoolbook:
        U, V, W = best_U, best_V, best_W
        rank = best_rank
    else:
        rank = rank_schoolbook
    
    # Convert to lists for the return format
    U_list = [[int(U[i, j]) if abs(U[i, j]) < 1e-10 else float(U[i, j]) for j in range(rank)] for i in range(n * m)]
    V_list = [[int(V[i, j]) if abs(V[i, j]) < 1e-10 else float(V[i, j]) for j in range(rank)] for i in range(m * p)]
    W_list = [[int(W[i, j]) if abs(W[i, j]) < 1e-10 else float(W[i, j]) for j in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
