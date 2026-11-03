"""Matrix multiplication 2x3 by 3x9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    num_a = n * m  # 6
    num_b = m * p  # 27
    num_c = p * n  # 18
    
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Build target tensor
    target = np.zeros((num_a, num_b, num_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # Strategy: Use Kronecker product structure
    # 2x3 by 3x9 can be seen as: for each of 3 column-blocks of B (each 3x3),
    # do a 2x3 by 3x3 multiplication
    # 
    # For 2x3 by 3x3, the schoolbook rank is 2*3*3 = 18
    # But we can do better. Let's use the fact that 2x3x3 has a rank-6 decomposition
    # via the "Strassen-like" approach or just use rank 6.
    #
    # Actually, let's think about this more carefully.
    # The tensor for 2x3 by 3x3 multiplication:
    # A is 2x3, B is 3x3, C is 2x3
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # In the tensor T[a,b,c] where a=(i,j), b=(j',k'), c=(k'',i''):
    # T[a,b,c] = 1 if i=i'' and j=j' and k=k''
    # 
    # This is a "slice" tensor. The rank of this specific tensor...
    # Let's just try alternating minimization with good initialization.
    
    def compute_residual(U, V, W):
        return np.einsum('ar,br,cr->abc', U, V, W) - target
    
    def compute_error(U, V, W):
        res = compute_residual(U, V, W)
        return np.sum(res**2)
    
    def solve_for_U(V, W, target):
        M = np.einsum('br,cr->bcr', V, W).reshape(num_b * num_c, -1)
        U = np.zeros((num_a, M.shape[1]))
        for a in range(num_a):
            rhs = target[a, :, :].flatten()
            U[a, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
        return U
    
    def solve_for_V(U, W, target):
        M = np.einsum('ar,cr->acr', U, W).reshape(num_a * num_c, -1)
        V = np.zeros((num_b, M.shape[1]))
        for b in range(num_b):
            rhs = target[:, b, :].transpose(1, 0).flatten()
            V[b, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
        return V
    
    def solve_for_W(U, V, target):
        M = np.einsum('ar,br->abr', U, V).reshape(num_a * num_b, -1)
        W = np.zeros((num_c, M.shape[1]))
        for c in range(num_c):
            rhs = target[:, :, c].flatten()
            W[c, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
        return W
    
    def alternating_minimization(R, n_iters=50, n_restarts=3):
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            U = np.random.randn(num_a, R) * 0.5
            V = np.random.randn(num_b, R) * 0.5
            W = np.random.randn(num_c, R) * 0.5
            
            for it in range(n_iters):
                if time.time() > deadline:
                    break
                
                U = solve_for_U(V, W, target)
                V = solve_for_V(U, W, target)
                W = solve_for_W(U, V, target)
                
                err = compute_error(U, V, W)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-12:
                    break
            
            if best_err < 1e-12:
                break
        
        if best_err < 1e-10:
            return best_U, best_V, best_W, best_err
        return None, None, None, best_err
    
    # Try structured Kronecker approach first
    # 2x3x9 = sum over 3 blocks of 2x3x3
    # For each block, use a rank-R_sub decomposition of 2x3x3
    
    # Let's try rank 18 first (which is 3 * 6, if 2x3x3 has rank 6)
    # But let's also try direct AM for lower ranks
    
    # Try various ranks
    for R in [18, 24, 30, 36, 42, 48, 54]:
        if time.time() > deadline - 5:
            break
        
        U, V, W, err = alternating_minimization(R, n_iters=30, n_restarts=5)
        
        if U is not None and err < 1e-10:
            final_err = compute_error(U, V, W)
            if final_err < 1e-10:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(num_a)]
    V = [[0] * rank for _ in range(num_b)]
    W = [[0] * rank for _ in range(num_c)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
