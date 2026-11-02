import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    
    # Strassen coefficients for 2x2 matrix multiplication
    strassen_x = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)],
    ]
    strassen_y = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)],
    ]
    strassen_c = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)],
    ]
    
    nb = 3
    bs = 2
    rank_bs = nb * nb * nb * 7
    
    U = np.zeros((n*m, rank_bs), dtype=np.float64)
    V = np.zeros((m*p, rank_bs), dtype=np.float64)
    W = np.zeros((p*n, rank_bs), dtype=np.float64)
    
    col = 0
    for ib in range(nb):
        for kb in range(nb):
            for jb in range(nb):
                for s in range(7):
                    for (lr, lc, coeff) in strassen_x[s]:
                        gi = ib * bs + lr
                        gj = jb * bs + lc
                        U[gi * m + gj, col] += coeff
                    for (lr, lc, coeff) in strassen_y[s]:
                        gj = jb * bs + lr
                        gk = kb * bs + lc
                        V[gj * p + gk, col] += coeff
                    for (lr, lc, coeff) in strassen_c[s]:
                        gi = ib * bs + lr
                        gk = kb * bs + lc
                        W[gk * n + gi, col] += coeff
                    col += 1
    
    def verify(U, V, W, tol=1e-9):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_expected = A @ B
        U_reshaped = U.reshape(n, m, U.shape[1])
        V_reshaped = V.reshape(m, p, V.shape[1])
        W_reshaped = W.reshape(p, n, W.shape[1])
        U_A = np.einsum('ijr,ij->ir', U_reshaped, A)
        V_B = np.einsum('jkr,jk->kr', V_reshaped, B)
        C_computed = np.einsum('kir,ir,kr->ik', W_reshaped, U_A, V_B)
        return np.allclose(C_computed, C_expected, atol=tol)
        
    if not verify(U, V, W):
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
        V_sb = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
        W_sb = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1.0
                    V_sb[j*p + k, col] = 1.0
                    W_sb[k*n + i, col] = 1.0
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
        
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_bs
    
    # Structural change: Use a randomized rank-reduction approach
    # We look for linear combinations of rank-1 terms that are redundant.
    # Specifically, we try to find a subset of columns that can be removed
    # while maintaining the decomposition property.
    
    # First, let's try the greedy removal again but more carefully
    # We check each term one by one, and if removing it still works, we keep it removed.
    
    if time.time() < deadline - 30:
        current_U = best_U.copy()
        current_V = best_V.copy()
        current_W = best_W.copy()
        current_rank = best_rank
        
        # We'll use a different strategy: try to merge pairs of terms.
        # If two rank-1 terms can be combined into one (unlikely), or if one is a linear combination.
        # A better approach: check if the tensor defined by removing term r is still in the span of the others.
        # This is equivalent to checking if the r-th term is linearly dependent in a certain sense.
        
        # Actually, the simplest structural change is to try a different base decomposition.
        # Let's try a 3x3 block Strassen with rank 168 (if 3x3 rank is 21).
        # But I don't have the 3x3 Strassen coefficients.
        
        # Alternative: Use the fact that the tensor is symmetric in some way.
        # Let's try to find a lower rank decomposition by solving a system of equations.
        # This is too complex for the time limit.
        
        # Let's stick with the greedy removal but make it more aggressive.
        # We try to remove multiple terms at once.
        
        removed = []
        for r in range(current_rank):
            if time.time() > deadline - 10:
                break
            mask = np.ones(current_rank, dtype=bool)
            mask[r] = False
            U_test = current_U[:, mask]
            V_test = current_V[:, mask]
            W_test = current_W[:, mask]
            
            if verify(U_test, V_test, W_test):
                removed.append(r)
        
        if removed:
            mask = np.ones(current_rank, dtype=bool)
            mask[removed] = False
            best_U = current_U[:, mask]
            best_V = current_V[:, mask]
            best_W = current_W[:, mask]
            best_rank = current_rank - len(removed)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
