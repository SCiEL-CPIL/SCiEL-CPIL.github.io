"""Find a rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        Um = U.reshape(n, m, r)
        Vm = V.reshape(m, p, r)
        Wm = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', Um, Vm, Wm)
        return np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0)

    def error_of(U, V, W):
        r = U.shape[1]
        Um = U.reshape(n, m, r)
        Vm = V.reshape(m, p, r)
        Wm = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', Um, Vm, Wm)
        return np.max(np.abs(T - 1.0))

    def solve_U(U, V, W):
        """Solve for U given V, W. U[i,j,:] = (V[j,:] * W[:,i,:])^T b"""
        r = U.shape[1]
        Vm = V.reshape(m, p, r)
        Wm = W.reshape(p, n, r)
        for i in range(n):
            for j in range(m):
                # A[c, k] = V[j,k,c] * W[k,i,c]
                A = Vm[j] * Wm[:, i]  # (p, r)
                b = np.ones(p)
                # Solve A^T x = b  =>  x = (A A^T)^{-1} A b
                # A^T is (r, p), b is (p,)
                try:
                    # Use lstsq for stability
                    x, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                    U[i * m + j] = x
                except np.linalg.LinAlgError:
                    pass
        return U

    def solve_V(U, V, W):
        """Solve for V given U, W."""
        r = V.shape[1]
        Um = U.reshape(n, m, r)
        Wm = W.reshape(p, n, r)
        for j in range(m):
            for k in range(p):
                # A[i, c] = U[i,j,c] * W[k,i,c]
                A = Um[:, j] * Wm[k]  # (n, r)
                b = np.ones(n)
                try:
                    x, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                    V[j * p + k] = x
                except np.linalg.LinAlgError:
                    pass
        return V

    def solve_W(U, V, W):
        """Solve for W given U, V."""
        r = W.shape[1]
        Um = U.reshape(n, m, r)
        Vm = V.reshape(m, p, r)
        for k in range(p):
            for i in range(n):
                # A[j, c] = U[i,j,c] * V[j,k,c]
                A = Um[i] * Vm[:, k]  # (m, r)
                b = np.ones(m)
                try:
                    x, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                    W[k * n + i] = x
                except np.linalg.LinAlgError:
                    pass
        return W

    def alternating_minimization(target_rank, n_restarts=10, max_iters=300, tol=1e-10, seed=None):
        """Alternating minimization to find a rank-target_rank decomposition."""
        if seed is not None:
            np.random.seed(seed)
        
        best_err = float('inf')
        best_result = None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Add schoolbook components if possible
            if target_rank >= n * m * p:
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            col = i * m * p + j * p + k
                            if col < target_rank:
                                U[i * m + j, col] += 1.0
                                V[j * p + k, col] += 1.0
                                W[k * n + i, col] += 1.0
            
            err = error_of(U, V, W)
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                U = solve_U(U, V, W)
                V = solve_V(U, V, W)
                W = solve_W(U, V, W)
                
                err = error_of(U, V, W)
                
                if err < tol:
                    if err < best_err:
                        best_err = err
                        best_result = (U.copy(), V.copy(), W.copy())
                    break
                
                if it % 50 == 49 and err > 1.0:
                    break
            
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
        
        return best_result, best_err

    # Try to find a decomposition with rank less than schoolbook
    # Start from rank 20 and go up to schoolbook rank
    # Use alternating minimization with multiple restarts
    
    # First, try a few target ranks
    target_ranks = [20, 24, 28, 32, 36, 40, 44, 48, 52]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10:
            break
        
        if target_rank >= best_rank:
            continue
        
        result, err = alternating_minimization(target_rank, n_restarts=20, max_iters=200, tol=1e-10)
        
        if result is not None:
            U_try, V_try, W_try = result
            if verify(U_try, V_try, W_try, tol=1e-9):
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                break
    
    # If still at schoolbook, try a more aggressive search
    if best_rank == n * m * p and time.time() < deadline - 30:
        for target_rank in range(15, 25):
            if time.time() > deadline - 10:
                break
            
            if target_rank >= best_rank:
                continue
            
            result, err = alternating_minimization(target_rank, n_restarts=30, max_iters=400, tol=1e-10)
            
            if result is not None:
                U_try, V_try, W_try = result
                if verify(U_try, V_try, W_try, tol=1e-9):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_try, V_try, W_try
                    break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
