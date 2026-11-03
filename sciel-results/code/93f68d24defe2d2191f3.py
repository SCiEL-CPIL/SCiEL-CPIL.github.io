import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = 2, 4, 6
    deadline = time.time() + 145.0
    
    # Build the full multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def check_decomposition(U, V, W, tol=1e-9):
        reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.max(np.abs(reconstructed - T)) < tol
    
    def to_lists(U, V, W):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Build rank-42 block-Strassen decomposition
    def make_rank42():
        rank = 42
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Strassen 7-term for 2x2
        U_s = [
            {(0,0):1, (1,1):1},
            {(1,0):1, (1,1):1},
            {(0,0):1},
            {(1,1):1},
            {(0,0):1, (0,1):1},
            {(1,0):1, (0,0):-1},
            {(0,1):1, (1,1):-1},
        ]
        V_s = [
            {(0,0):1, (1,1):1},
            {(0,0):1},
            {(0,1):1, (1,1):-1},
            {(1,0):1, (0,0):-1},
            {(1,1):1},
            {(0,0):1, (0,1):1},
            {(1,0):1, (1,1):1},
        ]
        W_s = [
            {(0,0):1, (1,1):1},
            {(0,1):1, (1,1):-1},
            {(1,0):1, (1,1):1},
            {(0,0):1, (0,1):1},
            {(0,0):-1, (1,0):1},
            {(1,1):1},
            {(0,0):1},
        ]
        
        j_groups = [(0,1), (2,3)]
        k_groups = [(0,1), (2,3), (4,5)]
        
        col = 0
        for jg in j_groups:
            j_base = jg[0]
            for kg in k_groups:
                k_base = kg[0]
                for s in range(7):
                    for (i_loc, j_loc), coeff in U_s[s].items():
                        U[i_loc*m + (j_base+j_loc), col] += coeff
                    for (j_loc, k_loc), coeff in V_s[s].items():
                        V[(j_base+j_loc)*p + (k_base+k_loc), col] += coeff
                    for (k_loc, i_loc), coeff in W_s[s].items():
                        W[(k_base+k_loc)*n + i_loc, col] += coeff
                    col += 1
        return U, V, W
    
    U42, V42, W42 = make_rank42()
    
    # Verify rank-42 works
    if not check_decomposition(U42, V42, W42, tol=1e-8):
        # Fallback: return what we have
        return to_lists(U42, V42, W42)
    
    best_U, best_V, best_W = U42.copy(), V42.copy(), W42.copy()
    best_rank = 42
    
    # Strategy 1: ALS refinement to try to reduce rank
    # For a target rank r, project the rank-42 solution and refine
    def try_als_reduce(target_rank, U_init, V_init, W_init, max_iters=200, tol=1e-10):
        """Try to find a rank-r decomposition using ALS starting from projection of higher-rank solution."""
        if time.time() > deadline - 25:
            return None
        
        U = U_init[:, :target_rank].copy()
        V = V_init[:, :target_rank].copy()
        W = W_init[:, :target_rank].copy()
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 25:
                break
            
            # Solve for W given U, V
            # For each (k,i): W[k*n+i, :] = argmin sum_j (T[i*m+j, j*p+k, k*n+i] - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])^2
            solved_all = True
            for k in range(p):
                for i in range(n):
                    # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    A = np.zeros((m, target_rank))
                    b = np.zeros(m)
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        b[j] = T[i*m+j, j*p+k, k*n+i]
                    try:
                        W[k*n+i, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        solved_all = False
                        break
                if not solved_all:
                    break
            
            if not solved_all:
                break
            
            # Solve for V given U, W
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, target_rank))
                    b = np.zeros(n)
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        b[i] = T[i*m+j, j*p+k, k*n+i]
                    try:
                        V[j*p+k, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        solved_all = False
                        break
                if not solved_all:
                    break
            
            if not solved_all:
                break
            
            # Solve for U given V, W
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, target_rank))
                    b = np.zeros(p)
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        b[k] = T[i*m+j, j*p+k, k*n+i]
                    try:
                        U[i*m+j, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        solved_all = False
                        break
                if not solved_all:
                    break
            
            if not solved_all:
                break
            
            # Check error
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.max(np.abs(recon - T))
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < tol:
                    break
        
        if best_err < tol:
            return best_U, best_V, best_W
        return None
    
    # Strategy 2: Random restart ALS for target rank
    def try_random_als(target_rank, n_trials=15, max_iters=300, tol=1e-10):
        """Random initialization ALS."""
        if time.time() > deadline - 25:
            return None
        
        for trial in range(n_trials):
            if time.time() > deadline - 25:
                break
            
            np.random.seed(trial * 7919 + target_rank * 13)
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            best_err = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(max_iters):
                if time.time() > deadline - 25:
                    break
                
                # Solve W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, target_rank))
                        b = np.zeros(m)
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                            b[j] = T[i*m+j, j*p+k, k*n+i]
                        W[k*n+i, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                
                # Solve V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, target_rank))
                        b = np.zeros(n)
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                            b[i] = T[i*m+j, j*p+k, k*n+i]
                        V[j*p+k, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                
                # Solve U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, target_rank))
                        b = np.zeros(p)
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                            b[k] = T[i*m+j, j*p+k, k*n+i]
                        U[i*m+j, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                
                recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = np.max(np.abs(recon - T))
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if err < tol:
                        break
                
                # Early stop if not improving
                if it > 50 and err > best_err * 1.01:
                    break
            
            if best_err < tol:
                return best_U, best_V, best_W
        
        return None
    
    # Strategy 3: Structured reduction - try to merge Strassen terms across blocks
    # The key observation: for n=2, the W matrix is small (12 entries).
    # Try rank 40, 38, 36, etc.
    
    # First try ALS from rank-42 projection
    for target_rank in [41, 40, 39, 38, 37, 36]:
        if time.time() > deadline - 30:
            break
        
        result = try_als_reduce(target_rank, U42, V42, W42, max_iters=500)
        if result is not None:
            U_r, V_r, W_r = result
            if check_decomposition(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
                break
    
    # If ALS from projection didn't work, try random restarts for lower ranks
    if best_rank == 42:
        for target_rank in [41, 40, 39, 38]:
            if time.time() > deadline - 30:
                break
            
            result = try_random_als(target_rank, n_trials=10, max_iters=400)
            if result is not None:
                U_r, V_r, W_r = result
                if check_decomposition(U_r, V_r, W_r):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    break
    
    # Final: try a few more random restarts at rank 40 and 39 with more trials
    if best_rank >= 40 and time.time() < deadline - 40:
        for target_rank in [40, 39]:
            if time.time() > deadline - 30:
                break
            result = try_random_als(target_rank, n_trials=20, max_iters=300)
            if result is not None:
                U_r, V_r, W_r = result
                if check_decomposition(U_r, V_r, W_r):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    break
    
    return to_lists(best_U, best_V, best_W)
