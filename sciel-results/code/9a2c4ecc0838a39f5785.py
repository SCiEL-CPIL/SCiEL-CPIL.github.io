import numpy as np
import time
import sys

_START = time.time()
_DEADLINE = _START + 145  # Leave 5s margin for cleanup/return

def _verify(U, V, W, tol=1e-9):
    """
    Vectorized verification of the tensor decomposition.
    U: (9, R), V: (9, R), W: (9, R)
    Target T[a,b,c] = 1 if (a,b,c) corresponds to (i,j,k) with a=i*3+j, b=j*3+k, c=k*3+i, else 0.
    """
    # Compute T
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Construct Target M
    M = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                M[a, b, c] = 1.0
                
    # Check difference
    diff = T - M
    max_diff = np.max(np.abs(diff))
    
    if max_diff > tol:
        return False, max_diff
        
    # Check if entries are effectively half-integers
    # The problem requires integer/half-integer entries.
    # If we started from rounding, they are. But if we did local search, they are.
    # Just to be safe, check if they are close to half-integers.
    if not np.allclose(U * 2, np.round(U * 2), atol=1e-6): return False, max_diff
    if not np.allclose(V * 2, np.round(V * 2), atol=1e-6): return False, max_diff
    if not np.allclose(W * 2, np.round(W * 2), atol=1e-6): return False, max_diff

    return True, max_diff

def _polish(U, V, W, M, max_iters=2000, tol=1e-9):
    """
    Local search on the half-integer lattice to minimize ||T - M||.
    U, V, W are assumed to be on the half-integer grid.
    """
    def err():
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(T - M)
    
    current_err = err()
    if current_err < tol:
        return U, V, W, True
        
    # Indices for local search
    # We will try to change one element at a time by +/- 0.5
    # To speed up, we precompute the "influence" of each element?
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # Changing U[a0, r0] by delta affects T[a0, b, c] for all b,c by delta * V[b, r0] * W[c, r0].
    
    # This is complex to update incrementally without recomputing T.
    # Given R is small (~23) and size is 9, recomputing T (9*9*9*23 ~ 17k ops) is fast.
    # 2000 iters * 17k ops = 34M ops. Very fast in Python/NumPy.
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE - 2:
            break
            
        improved = False
        # Try U
        for a in range(9):
            for r in range(U.shape[1]):
                orig = U[a, r]
                # Try orig - 0.5
                if orig > -1e-9: # Avoid going below -inf? No, just try neighbors
                    U[a, r] = orig - 0.5
                    e = err()
                    if e < current_err - 1e-12:
                        current_err = e
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        improved = True
                        break
                    else:
                        U[a, r] = orig
                # Try orig + 0.5
                U[a, r] = orig + 0.5
                e = err()
                if e < current_err - 1e-12:
                    current_err = e
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    improved = True
                    break
                else:
                    U[a, r] = orig
            
            if improved:
                U = best_U; V = best_V; W = best_W
                if current_err < tol: return U, V, W, True
                continue
                
        # Try V
        for b in range(9):
            for r in range(V.shape[1]):
                orig = V[b, r]
                V[b, r] = orig - 0.5
                e = err()
                if e < current_err - 1e-12:
                    current_err = e
                    best_U = U.copy(); best_V = V.copy(); best_W = W.copy()
                    improved = True
                    break
                else:
                    V[b, r] = orig
                V[b, r] = orig + 0.5
                e = err()
                if e < current_err - 1e-12:
                    current_err = e
                    best_U = U.copy(); best_V = V.copy(); best_W = W.copy()
                    improved = True
                    break
                else:
                    V[b, r] = orig
            if improved:
                U = best_U; V = best_V; W = best_W
                if current_err < tol: return U, V, W, True
                continue
                
        # Try W
        for c in range(9):
            for r in range(W.shape[1]):
                orig = W[c, r]
                W[c, r] = orig - 0.5
                e = err()
                if e < current_err - 1e-12:
                    current_err = e
                    best_U = U.copy(); best_V = V.copy(); best_W = W.copy()
                    improved = True
                    break
                else:
                    W[c, r] = orig
                W[c, r] = orig + 0.5
                e = err()
                if e < current_err - 1e-12:
                    current_err = e
                    best_U = U.copy(); best_V = V.copy(); best_W = W.copy()
                    improved = True
                    break
                else:
                    W[c, r] = orig
            if improved:
                U = best_U; V = best_V; W = best_W
                if current_err < tol: return U, V, W, True
                continue
                
        if not improved:
            break
            
    return best_U, best_V, best_W, current_err < tol

def _als_decompose(rank, M, max_iters=50, tol=1e-8):
    # Initialize U, V, W randomly
    # Using smaller variance might help convergence?
    U = np.random.randn(9, rank) * 0.1
    V = np.random.randn(9, rank) * 0.1
    W = np.random.randn(9, rank) * 0.1
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
            
        # Update U
        B_mat = np.zeros((81, rank))
        for r in range(rank):
            B_mat[:, r] = np.outer(V[:, r], W[:, r]).flatten()
        M_mat = M.reshape(9, 81)
        try:
            U_new, _, _, _ = np.linalg.lstsq(B_mat.T, M_mat.T, rcond=None)
            U = U_new.T
        except:
            pass
            
        # Update V
        M_mat2 = M.transpose(1, 0, 2).reshape(9, 81)
        B_mat2 = np.zeros((81, rank))
        for r in range(rank):
            B_mat2[:, r] = np.outer(U[:, r], W[:, r]).flatten()
        try:
            V_new, _, _, _ = np.linalg.lstsq(B_mat2.T, M_mat2.T, rcond=None)
            V = V_new.T
        except:
            pass
            
        # Update W
        M_mat3 = M.transpose(2, 0, 1).reshape(9, 81)
        B_mat3 = np.zeros((81, rank))
        for r in range(rank):
            B_mat3[:, r] = np.outer(U[:, r], V[:, r]).flatten()
        try:
            W_new, _, _, _ = np.linalg.lstsq(B_mat3.T, M_mat3.T, rcond=None)
            W = W_new.T
        except:
            pass
        
        # Check convergence
        T = np.einsum('ar,br,cr->abc', U, V, W)
        if np.linalg.norm(T - M) < tol:
            break
            
    return U, V, W

def construct():
    # Construct Target Tensor M
    M = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                M[a, b, c] = 1.0

    # Try ranks 23, 24, 25
    # We will do multiple restarts for each rank
    best_result = None
    best_rank = 99
    
    for rank in [23, 24, 25]:
        if time.time() > _DEADLINE - 5:
            break
            
        found = False
        # Number of restarts depends on time left
        # Let's do 5 restarts for rank 23, 3 for 24, 2 for 25
        num_restarts = 5 if rank == 23 else (3 if rank == 24 else 2)
        
        for _ in range(num_restarts):
            if time.time() > _DEADLINE - 5:
                break
                
            U, V, W = _als_decompose(rank, M)
            
            # Round to nearest 0.5
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            valid, err = _verify(U_r, V_r, W_r)
            if valid:
                best_result = (U_r, V_r, W_r, rank)
                found = True
                break # Found valid for this rank, move to next lower rank? No, we go 23->24. So we want the lowest rank.
            
            # If not valid, try polishing
            if not found:
                U_p, V_p, W_p, valid_p = _polish(U_r, V_r, W_r, M)
                if valid_p:
                    best_result = (U_p, V_p, W_p, rank)
                    found = True
                    break

        if found:
            # We found a valid solution for this rank. Since we iterate 23, 24, 25, 
            # the first one found is the lowest rank.
            # However, if 23 fails, we try 24. If 24 succeeds, we return 24.
            # We should return the best (lowest rank) found.
            # Since we break on success, and iterate ascending, the first success is the best.
            break

    if best_result:
        U, V, W, r = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to 27
    U_fb = np.zeros((9, 27), dtype=float)
    V_fb = np.zeros((9, 27), dtype=float)
    W_fb = np.zeros((9, 27), dtype=float)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_fb[i*3+j, col] = 1.0
                V_fb[j*3+k, col] = 1.0
                W_fb[k*3+i, col] = 1.0
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
