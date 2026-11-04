import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 3, 4]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _error(U, V, W, T):
    """Frobenius norm squared of reconstruction error."""
    T_comp = np.einsum('al,bl,cl->abc', U, V, W)
    diff = T_comp - T
    return np.sum(diff * diff)

def _snap_to_half_integers(U, V, W, T, tol=1e-9):
    """Try to snap to half-integers and verify exactness."""
    U_r = np.round(U * 2) / 2.0
    V_r = np.round(V * 2) / 2.0
    W_r = np.round(W * 2) / 2.0
    T_comp = np.einsum('al,bl,cl->abc', U_r, V_r, W_r)
    if np.max(np.abs(T_comp - T)) < tol:
        return U_r, V_r, W_r
    return None

def _schoolbook_init(n, m, p):
    """Return schoolbook factor matrices (rank = n*m*p)."""
    nm, mp, pn = n * m, m * p, p * n
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_merge(U, V, W, T, nm, mp, pn, deadline, max_iter=500):
    """Try to reduce rank by merging pairs of columns via local search."""
    rank = U.shape[1]
    best_err = _error(U, V, W, T)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try all pairs of columns to merge
        best_pair = None
        best_pair_err = best_err
        best_merged = None
        
        for l1 in range(rank):
            for l2 in range(l1 + 1, rank):
                # Try to merge columns l1 and l2 into a single column
                # The merged column should approximate the sum of the two rank-1 terms
                # We can try different ways to split the contribution
                
                # Simple approach: average the two columns
                U_new = U.copy()
                V_new = V.copy()
                W_new = W.copy()
                
                # Try to find a single column that best approximates both
                # Use least squares: find u, v, w such that u*v*w ≈ U[:,l1]*V[:,l1]*W[:,l1] + U[:,l2]*V[:,l2]*W[:,l2]
                target = np.einsum('a,b,c->abc', U[:, l1], V[:, l1], W[:, l1]) + \
                         np.einsum('a,b,c->abc', U[:, l2], V[:, l2], W[:, l2])
                
                # Try a few random initializations for the merged column
                for _ in range(5):
                    u_init = (U[:, l1] + U[:, l2]) / 2 + np.random.randn(nm) * 0.1
                    v_init = (V[:, l1] + V[:, l2]) / 2 + np.random.randn(mp) * 0.1
                    w_init = (W[:, l1] + W[:, l2]) / 2 + np.random.randn(pn) * 0.1
                    
                    # Remove columns l1, l2 and add the new column
                    keep = [i for i in range(rank) if i != l1 and i != l2]
                    U_m = np.column_stack([U[:, keep], u_init]) if keep else u_init.reshape(nm, 1)
                    V_m = np.column_stack([V[:, keep], v_init]) if keep else v_init.reshape(mp, 1)
                    W_m = np.column_stack([W[:, keep], w_init]) if keep else w_init.reshape(pn, 1)
                    
                    err = _error(U_m, V_m, W_m, T)
                    if err < best_pair_err - 1e-10:
                        best_pair_err = err
                        best_pair = (l1, l2)
                        best_merged = (U_m, V_m, W_m)
        
        if best_merged is None:
            break
        
        U, V, W = best_merged
        rank = U.shape[1]
        best_err = best_pair_err
        
        if best_err < 1e-8:
            break
    
    return U, V, W, best_err

def _try_rank_structured(rank, T, nm, mp, pn, deadline, n_restarts=10):
    """Attempt to find a rank-`rank` decomposition via structured search."""
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Start from schoolbook and merge
        U0, V0, W0 = _schoolbook_init(2, 3, 4)
        current_rank = U0.shape[1]
        
        # If we need a specific rank, we may need to add or remove columns
        if current_rank > rank:
            # Merge until we reach the target rank
            U, V, W, err = _try_merge(U0, V0, W0, T, nm, mp, pn, deadline, max_iter=200)
            if U.shape[1] <= rank:
                # Pad if necessary
                if U.shape[1] < rank:
                    pad = rank - U.shape[1]
                    U = np.column_stack([U, np.zeros((nm, pad))])
                    V = np.column_stack([V, np.zeros((mp, pad))])
                    W = np.column_stack([W, np.zeros((pn, pad))])
                if _error(U, V, W, T) < 1e-8:
                    snapped = _snap_to_half_integers(U, V, W, T)
                    if snapped is not None:
                        return snapped
        elif current_rank < rank:
            # Add random columns and refine
            U = U0.copy()
            V = V0.copy()
            W = W0.copy()
            pad = rank - current_rank
            U = np.column_stack([U, np.random.randn(nm, pad) * 0.5])
            V = np.column_stack([V, np.random.randn(mp, pad) * 0.5])
            W = np.column_stack([W, np.random.randn(pn, pad) * 0.5])
            
            # Local search
            err = _error(U, V, W, T)
            for step in range(100):
                if time.time() > deadline:
                    break
                improved = False
                for _ in range(10):
                    mat_choice = np.random.randint(3)
                    delta = np.random.choice([0.1, 0.2, 0.5]) * np.random.choice([-1, 1])
                    if mat_choice == 0:
                        idx = (np.random.randint(nm), np.random.randint(rank))
                        U_new = U.copy()
                        U_new[idx] += delta
                        err_new = _error(U_new, V, W, T)
                        if err_new < err:
                            U = U_new
                            err = err_new
                            improved = True
                    elif mat_choice == 1:
                        idx = (np.random.randint(mp), np.random.randint(rank))
                        V_new = V.copy()
                        V_new[idx] += delta
                        err_new = _error(U, V_new, W, T)
                        if err_new < err:
                            V = V_new
                            err = err_new
                            improved = True
                    else:
                        idx = (np.random.randint(pn), np.random.randint(rank))
                        W_new = W.copy()
                        W_new[idx] += delta
                        err_new = _error(U, V, W_new, T)
                        if err_new < err:
                            W = W_new
                            err = err_new
                            improved = True
                if not improved:
                    break
            
            if err < 1e-8:
                snapped = _snap_to_half_integers(U, V, W, T)
                if snapped is not None:
                    return snapped
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for safety
    
    nm, mp, pn = n * m, m * p, p * n
    T = _build_tensor(n, m, p)
    
    # Try ranks from 19 upward
    for rank in [19, 20, 21, 22, 23]:
        if time.time() > deadline:
            break
        
        result = _try_rank_structured(rank, T, nm, mp, pn, deadline, n_restarts=5)
        if result is not None:
            U_r, V_r, W_r = result
            U_list = [[float(U_r[i][j]) for j in range(rank)] for i in range(nm)]
            V_list = [[float(V_r[i][j]) for j in range(rank)] for i in range(mp)]
            W_list = [[float(W_r[i][j]) for j in range(rank)] for i in range(pn)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    rank = n * m * p
    U, V, W = _schoolbook_init(n, m, p)
    U_list = [[float(U[i][j]) for j in range(rank)] for i in range(nm)]
    V_list = [[float(V[i][j]) for j in range(rank)] for i in range(mp)]
    W_list = [[float(W[i][j]) for j in range(rank)] for i in range(pn)]
    return {"U": U_list, "V": V_list, "W": W_list}
