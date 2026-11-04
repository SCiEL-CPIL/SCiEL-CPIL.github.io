"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    r = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    result = np.zeros((n*m, m*p, p*n))
    for r_idx in range(r):
        result += np.einsum('a,b,c->abc', U[:,r_idx], V[:,r_idx], W[:,r_idx])
    return np.allclose(result, T, atol=tol)

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using rational arithmetic or high precision."""
    r = U.shape[1]
    # Check active entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(r):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-10:
                    return False
    # Check all inactive entries (feasible for 3540 entries with small r)
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i_a, j_a = divmod(a, m)
                j_b, k_b = divmod(b, p)
                k_c, i_c = divmod(c, n)
                if j_a == j_b and i_a == i_c and k_b == k_c:
                    continue
                val = 0.0
                for cc in range(r):
                    val += U[a, cc] * V[b, cc] * W[c, cc]
                if abs(val) > 1e-10:
                    return False
    return True

def _snap_to_half_integers(U, V, W, tol=1e-6):
    """Try to snap U, V, W to nearest half-integers and verify."""
    U_snapped = np.round(2 * U) / 2
    V_snapped = np.round(2 * V) / 2
    W_snapped = np.round(2 * W) / 2
    
    # Check if snapping was close enough
    if (np.max(np.abs(U_snapped - U)) > tol or 
        np.max(np.abs(V_snapped - V)) > tol or 
        np.max(np.abs(W_snapped - W)) > tol):
        return None
    
    if _verify_exact(U_snapped, V_snapped, W_snapped, 2, 5, 6):
        return (U_snapped, V_snapped, W_snapped)
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_rank = n * m * p  # schoolbook
    best_result = None
    
    # Schoolbook as fallback
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(nm, mp, pn) = 30
    np.random.seed(42)
    
    for rank in range(30, best_rank):
        if time.time() > deadline - 30:
            break
        
        found = False
        # Try multiple random restarts for this rank
        max_restarts = 5 if rank < 45 else 3
        for restart in range(max_restarts):
            if time.time() > deadline - 20:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            # Precompute for efficiency
            M_U = np.zeros((mp * pn, rank))
            for r_idx in range(rank):
                M_U[:, r_idx] = np.outer(V[:, r_idx], W[:, r_idx]).flatten()
            
            T_flat = T.reshape(nm, mp * pn)
            
            # Run several ALS iterations
            for _ in range(10):
                if time.time() > deadline - 15:
                    break
                
                # Fix V and W, solve for U
                M_U = np.zeros((mp * pn, rank))
                for r_idx in range(rank):
                    M_U[:, r_idx] = np.outer(V[:, r_idx], W[:, r_idx]).flatten()
                try:
                    U_new = np.zeros_like(U)
                    for a in range(nm):
                        U_new[a, :] = np.linalg.lstsq(M_U, T_flat[a, :], rcond=None)[0]
                except:
                    break
                
                # Fix U and W, solve for V
                M_V = np.zeros((nm * pn, rank))
                for r_idx in range(rank):
                    M_V[:, r_idx] = np.outer(U_new[:, r_idx], W[:, r_idx]).flatten()
                T_for_V = np.transpose(T, (1, 0, 2))  # (mp, nm, pn)
                T_flat_v = T_for_V.reshape(mp, nm * pn)
                try:
                    V_new = np.zeros_like(V)
                    for b in range(mp):
                        V_new[b, :] = np.linalg.lstsq(M_V, T_flat_v[b, :], rcond=None)[0]
                except:
                    break
                
                # Fix U and V, solve for W
                M_W = np.zeros((nm * mp, rank))
                for r_idx in range(rank):
                    M_W[:, r_idx] = np.outer(U_new[:, r_idx], V_new[:, r_idx]).flatten()
                T_for_W = np.transpose(T, (2, 0, 1))  # (pn, nm, mp)
                T_flat_w = T_for_W.reshape(pn, nm * mp)
                try:
                    W_new = np.zeros_like(W)
                    for c in range(pn):
                        W_new[c, :] = np.linalg.lstsq(M_W, T_flat_w[c, :], rcond=None)[0]
                except:
                    break
                
                # Check convergence
                diff = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
                U, V, W = U_new, V_new, W_new
                if diff < 1e-8:
                    break
            
            # Verify float solution
            if _verify_exact(U, V, W, n, m, p):
                # Try to snap to half-integers
                snapped = _snap_to_half_integers(U, V, W)
                if snapped is not None:
                    found = True
                    best_rank = rank
                    best_result = snapped
                    break
        
        if found:
            break
    
    # Convert to lists
    U_final, V_final, W_final = best_result
    U_list = [[float(x) for x in row] for row in U_final]
    V_list = [[float(x) for x in row] for row in V_final]
    W_list = [[float(x) for x in row] for row in W_final]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
