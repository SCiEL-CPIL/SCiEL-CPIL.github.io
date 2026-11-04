"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # Safety margin before 150s kill
    
    nm = n * m  # 16
    mp = m * p  # 28
    pn = p * n  # 28
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    T_flat_2d = T.reshape(nm, mp * pn)  # (nm, mp*pn)
    T_flat_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)  # (mp, nm*pn)
    T_flat_w = T.transpose(2, 0, 1).reshape(pn, nm * mp)  # (pn, nm*mp)
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p  # 112
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    def check_decomposition(U, V, W):
        """Check if the decomposition is exact."""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        max_err = np.max(np.abs(T_approx - T))
        return max_err < 1e-9, max_err

    def try_rank_als(target_rank, n_restarts=8, max_iter=80):
        """
        Try to find a rank-target_rank decomposition using ALS with multiple restarts.
        """
        nonlocal best_U, best_V, best_W, best_rank
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with different scales
            scale = 1.0 / np.sqrt(max(target_rank, 1))
            U = np.random.randn(nm, target_rank) * scale
            V = np.random.randn(mp, target_rank) * scale
            W = np.random.randn(pn, target_rank) * scale
            
            # Also try structured initializations
            if restart == 1:
                # Initialize with schoolbook-like structure
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                for r in range(target_rank):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n*m)) % p
                    U[i*m+j, r] = 1.0
                    V[j*p+k, r] = 1.0
                    W[k*n+i, r] = 1.0
                    # Add small random perturbation
                    U[r % nm, r] += np.random.randn() * 0.1
                    V[r % mp, r] += np.random.randn() * 0.1
                    W[r % pn, r] += np.random.randn() * 0.1
            
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                # Update U: T_flat_2d = U @ M_U where M_U[r, bc] = V[b,r]*W[c,r]
                M_U = np.stack([np.outer(V[:, r], W[:, r]).reshape(-1) for r in range(target_rank)])
                # M_U shape: (target_rank, mp*pn)
                try:
                    U_new = T_flat_2d @ np.linalg.pinv(M_U)
                except:
                    break
                U = U_new
                
                # Update V
                M_V = np.stack([np.outer(U[:, r], W[:, r]).reshape(-1) for r in range(target_rank)])
                try:
                    V_new = T_flat_v @ np.linalg.pinv(M_V)
                except:
                    break
                V = V_new
                
                # Update W
                M_W = np.stack([np.outer(U[:, r], V[:, r]).reshape(-1) for r in range(target_rank)])
                try:
                    W_new = T_flat_w @ np.linalg.pinv(M_W)
                except:
                    break
                W = W_new
                
                # Check convergence
                ok, err = check_decomposition(U, V, W)
                if ok:
                    if target_rank < best_rank:
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = target_rank
                    return True
                    
            # Final check after ALS
            ok, err = check_decomposition(U, V, W)
            if ok:
                if target_rank < best_rank:
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = target_rank
                return True
                
        return False

    def try_rank_polish(target_rank, n_restarts=4):
        """
        Try to find a rank-target_rank decomposition using gradient-based polish.
        """
        nonlocal best_U, best_V, best_W, best_rank
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            scale = 1.0 / np.sqrt(target_rank)
            U = np.random.randn(nm, target_rank) * scale
            V = np.random.randn(mp, target_rank) * scale
            W = np.random.randn(pn, target_rank) * scale
            
            # Gradient descent with momentum
            lr = 0.01
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            for it in range(200):
                if time.time() > deadline - 10:
                    break
                
                T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                diff = T_approx - T
                
                grad_U = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
                grad_V = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
                grad_W = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
                
                # Momentum update
                vU = momentum * vU + grad_U
                vV = momentum * vV + grad_V
                vW = momentum * vW + grad_W
                
                U -= lr * vU
                V -= lr * vV
                W -= lr * vW
                
                # Adaptive learning rate
                if it % 20 == 0 and it > 0:
                    err = np.sum(diff * diff)
                    if err < 1e-10:
                        lr *= 0.5
                
                ok, err = check_decomposition(U, V, W)
                if ok:
                    if target_rank < best_rank:
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = target_rank
                    return True
                    
        return False

    # Try ranks from 111 down to a reasonable limit
    # We know 112 works. We want to find if we can go lower.
    for target_rank in range(111, 49, -1):
        if time.time() > deadline - 20:
            break
        # Try ALS first (often faster for structured problems)
        if try_rank_als(target_rank, n_restarts=5, max_iter=60):
            continue
        # Then try gradient polish
        if try_rank_polish(target_rank, n_restarts=3):
            continue

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
