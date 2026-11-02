import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 4 matrix.
    
    Uses vectorized ALS to search for low-rank decompositions.
    Falls back to schoolbook rank 24 if not found.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    n, m, p = 2, 3, 4
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the multiplication tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0
    
    # Schoolbook rank as fallback
    def _schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = _schoolbook()
    
    # Try ranks from 23 down to 10
    for rank in range(23, 9, -1):
        if time.time() > deadline - 5.0:
            break
        result = _try_rank_als(T, nm, mp, pn, rank, deadline)
        if result is not None:
            best_result = result
            break  # Found lower rank, stop
    
    return best_result

def _try_rank_als(T, nm, mp, pn, rank, deadline):
    """Try to find a rank-`rank` decomposition using vectorized ALS."""
    np.random.seed(42 + rank)
    
    # Multiple trials with diverse initializations
    for trial in range(10):
        if time.time() > deadline - 3.0:
            return None
        
        # Diverse random initialization
        scale = 0.5 + 0.1 * (trial % 4)
        U = np.random.randn(nm, rank) * scale
        V = np.random.randn(mp, rank) * scale
        W = np.random.randn(pn, rank) * scale
        
        # ALS iterations
        prev_err = np.inf
        for it in range(500):
            if time.time() > deadline - 3.0:
                break
            
            # Solve for U: for each (i,j), minimize ||T[i,j,:] - V * W^T * U[i,j,:]||
            # T[i,j,:] is a vector of length pn
            # We need: sum_r U[i*m+j, r] * (V[:, r] outer W[:, r]) flattened == T[i,j,:]
            # This is a linear system in U[i*m+j, :]
            for idx in range(nm):
                # Build the design matrix: each column r is V[:, r] outer W[:, r] flattened
                # Shape: (pn, rank)
                A = np.einsum('br,cr->bc', V, W).reshape(pn, rank)
                b = T[idx, :, :].reshape(pn)
                try:
                    U[idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except Exception:
                    pass
            
            # Solve for V
            for idx in range(mp):
                A = np.einsum('ar,cr->ac', U, W).reshape(nm, rank)
                b = T[:, idx, :].reshape(nm)
                try:
                    V[idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except Exception:
                    pass
            
            # Solve for W
            for idx in range(pn):
                A = np.einsum('ar,br->ab', U, V).reshape(nm * mp, rank)
                b = T.reshape(nm * mp)[:, idx] if T.shape[2] == pn else np.zeros(nm * mp)
                # Actually T has shape (nm, mp, pn), so for fixed idx (which is k*n+i),
                # we need T[:, :, idx] reshaped
                b = T[:, :, idx].reshape(nm * mp)
                try:
                    W[idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except Exception:
                    pass
            
            # Check error
            tensor = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((tensor - T) ** 2)
            
            if err < 1e-10:
                # Round to nearest 0.5
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify rounded solution
                tensor_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                err_r = np.sum((tensor_r - T) ** 2)
                if err_r < 1e-10:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
                # Also try without rounding
                if err < 1e-10:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Early stop if not improving
            if err > prev_err * 0.99 and it > 50:
                break
            prev_err = err
    
    return None
# EVOLVE-BLOCK-END
