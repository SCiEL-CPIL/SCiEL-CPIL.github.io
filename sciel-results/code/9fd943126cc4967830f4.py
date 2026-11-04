import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-6):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Reshape to 3D for einsum
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # Compute reconstruction
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if np.max(np.abs(recon - 1.0)) > tol:
        return False
    return True

def _snap_to_rational(arr, tol=1e-9):
    """Snap values to nearest integer or half-integer if close enough."""
    result = np.array(arr, dtype=float)
    # Try integers first
    int_snapped = np.round(result)
    if np.max(np.abs(result - int_snapped)) < tol:
        return int_snapped.astype(int).tolist()
    
    # Try half-integers
    half_snapped = np.round(result * 2) / 2
    if np.max(np.abs(result - half_snapped)) < tol:
        lst = []
        for row in half_snapped:
            r = []
            for v in row:
                if abs(v - round(v)) < 1e-9:
                    r.append(int(round(v)))
                else:
                    r.append(int(round(v * 2)) / 2.0)
            lst.append(r)
        return lst
        
    # Return None if cannot snap
    return None

def _als_decompose(n, m, p, rank, max_iters=2000, seeds=20):
    """
    Attempt to find a rank-R decomposition for n x m x p matrix multiplication tensor.
    Returns (U, V, W) as lists of snapped rationals, or None if failed.
    """
    nm, mp, pn = n*m, m*p, p*n
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    best_snapped = None
    
    for seed in range(seeds):
        np.random.seed(seed)
        # Initialize with small random integers to help snapping
        U = np.random.randint(-1, 2, (nm, rank)) * 0.5
        V = np.random.randint(-1, 2, (mp, rank)) * 0.5
        W = np.random.randint(-1, 2, (pn, rank)) * 0.5
        
        for it in range(max_iters):
            # Update W
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            M = np.einsum('ijc,jkc->kic', U3, V3)
            for k in range(p):
                for i in range(n):
                    vec = M[k, i, :]
                    norm_sq = np.dot(vec, vec)
                    if norm_sq > 1e-12:
                        W[k*n+i, :] = vec / norm_sq
            
            # Update U
            W3 = W.reshape(p, n, rank)
            N = np.einsum('jkc,kic->ijc', V3, W3)
            for i in range(n):
                for j in range(m):
                    vec = N[i, j, :]
                    norm_sq = np.dot(vec, vec)
                    if norm_sq > 1e-12:
                        U[i*m+j, :] = vec / norm_sq
            
            # Update V
            U3 = U.reshape(n, m, rank)
            P = np.einsum('ijc,kic->jkc', U3, W3)
            for j in range(m):
                for k in range(p):
                    vec = P[j, k, :]
                    norm_sq = np.dot(vec, vec)
                    if norm_sq > 1e-12:
                        V[j*p+k, :] = vec / norm_sq
            
            # Check error periodically
            if it % 50 == 0:
                U3 = U.reshape(n, m, rank)
                V3 = V.reshape(m, p, rank)
                W3 = W.reshape(p, n, rank)
                recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                err = np.max(np.abs(recon - 1.0))
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                    # Try to snap
                    U_snapped = _snap_to_rational(U)
                    V_snapped = _snap_to_rational(V)
                    W_snapped = _snap_to_rational(W)
                    
                    if U_snapped is not None and V_snapped is not None and W_snapped is not None:
                        if _verify_decomposition(U_snapped, V_snapped, W_snapped, n, m, p, rank, tol=1e-4):
                            return U_snapped, V_snapped, W_snapped

    # Final check on best found
    if best_U is not None:
        U_snapped = _snap_to_rational(best_U)
        V_snapped = _snap_to_rational(best_V)
        W_snapped = _snap_to_rational(best_W)
        
        if U_snapped is not None and V_snapped is not None and W_snapped is not None:
            if _verify_decomposition(U_snapped, V_snapped, W_snapped, n, m, p, rank, tol=1e-4):
                return U_snapped, V_snapped, W_snapped

    return None

def _get_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U.tolist(), V.tolist(), W.tolist(), rank

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook baseline
    U_sb, V_sb, W_sb, rank_sb = _get_schoolbook(n, m, p)
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Strategy: Compose
    # Split B (3x7) into B1 (3x3), B2 (3x3), B3 (3x1)
    
    # 1. Get decomposition for 2x3x1
    U3 = None
    V3 = None
    W3 = None
    rank_231 = None
    
    for r in [3, 4, 5, 6]:
        if time.time() > deadline - 60:
            break
        res = _als_decompose(2, 3, 1, r, max_iters=500, seeds=10)
        if res is not None:
            U3, V3, W3 = res
            rank_231 = r
            break
            
    if rank_231 is None:
        U3, V3, W3, rank_231 = _get_schoolbook(2, 3, 1)

    # 2. Get decomposition for 2x3x3
    U233 = None
    V233 = None
    W233 = None
    rank_233 = None
    
    for r in [5, 6, 7, 8]:
        if time.time() > deadline - 30:
            break
        res = _als_decompose(2, 3, 3, r, max_iters=1000, seeds=10)
        if res is not None:
            U233, V233, W233 = res
            rank_233 = r
            break
            
    if rank_233 is None:
        U233, V233, W233, rank_233 = _get_schoolbook(2, 3, 3)

    # Compose
    # Total rank = 2 * rank_233 + rank_231
    total_rank = 2 * rank_233 + rank_231
    
    if total_rank < best_rank:
        U_comp = np.zeros((nm, total_rank))
        V_comp = np.zeros((mp, total_rank))
        W_comp = np.zeros((pn, total_rank))
        
        # Block 1: k=0,1,2
        U_comp[:, 0:rank_233] = np.array(U233)
        for j in range(3):
            for k in range(3):
                V_comp[j*7+k, 0:rank_233] = np.array(V233)[j*3+k, :]
        for k in range(3):
            for i in range(2):
                W_comp[k*2+i, 0:rank_233] = np.array(W233)[k*2+i, :]
                
        # Block 2: k=3,4,5
        offset = rank_233
        U_comp[:, offset:2*rank_233] = np.array(U233)
        for j in range(3):
            for k in range(3, 6):
                V_comp[j*7+k, offset:2*rank_233] = np.array(V233)[j*3+(k-3), :]
        for k in range(3, 6):
            for i in range(2):
                W_comp[k*2+i, offset:2*rank_233] = np.array(W233)[(k-3)*2+i, :]
                
        # Block 3: k=6
        offset2 = 2 * rank_233
        U_comp[:, offset2:total_rank] = np.array(U3)
        for j in range(3):
            V_comp[j*7+6, offset2:total_rank] = np.array(V3)[j*1, :]
        for i in range(2):
            W_comp[6*2+i, offset2:total_rank] = np.array(W3)[0*2+i, :]
            
        # Verify composed
        if _verify_decomposition(U_comp, V_comp, W_comp, n, m, p, total_rank, tol=1e-4):
            best = {"U": U_comp.tolist(), "V": V_comp.tolist(), "W": W_comp.tolist()}
            best_rank = total_rank

    return best
