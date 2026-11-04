"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T with T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    AND == 0 for all inconsistent (a,b,c) indices.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    nm, mp, pn = n * m, m * p, p * n

    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Compute the decomposed tensor efficiently
    # result[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # Use einsum: for each r, outer product of U[:,r], V[:,r], W[:,r]
    # More efficient: reshape approach
    # result = np.einsum('ar,br,cr->abc', U, V, W)
    result = np.einsum('ar,br,cr->abc', U, V, W)

    if not np.allclose(result, T, atol=tol):
        return False

    return True


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification checking only active entries and a sample of inactive."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    nm, mp, pn = n * m, m * p, p * n

    # Check all active entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False

    # Check all inactive entries (3540 total, feasible for small rank)
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                i_a, j_a = divmod(a, m)
                j_b, k_b = divmod(b, p)
                k_c, i_c = divmod(c, n)
                if j_a == j_b and i_a == i_c and k_b == k_c:
                    continue
                val = 0.0
                for r in range(rank):
                    val += U[a, r] * V[b, r] * W[c, r]
                if abs(val) > tol:
                    return False

    return True


def _als_polish(U, V, W, n, m, p, tol=1e-12, max_iter=1000):
    """
    Alternating Least Squares polish for tensor decomposition.
    Tries to refine U, V, W so that sum_r U[:,r] V[:,r] W[:,r] matches T.
    Returns polished matrices or None if it fails to converge.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    nm, mp, pn = n * m, m * p, p * n

    T = _build_target_tensor(n, m, p)

    for iteration in range(max_iter):
        # Update U: for each a, U[a, :] = argmin ||sum_r U[a,r] V[:,r] W[:,r] - T[a,:,:]||
        # This is a linear system: V @ (U[a,:] outer W) ... 
        # Actually: sum_r U[a,r] * V[:,r] * W[:,r] should equal T[a, :, :]
        # Let M = V * W (element-wise, shape mp x rank x pn... no)
        # sum_r U[a,r] * (V[:,r] * W[:,r]) = T[a, :, :]
        # Let B = V * W reshaped: B[r, b, c] = V[b,r] * W[c,r]
        # Then sum_r U[a,r] * B[r, b, c] = T[a, b, c]
        # This is: (U[a,:] @ B_flat) = T[a].flatten() where B_flat[b*n_pn + c, r] = V[b,r]*W[c,r]
        
        # Build B_flat: shape (mp*pn, rank)
        B_flat = np.zeros((mp * pn, rank))
        for r in range(rank):
            B_flat[:, r] = (V[:, r] * W[:, r]).flatten()
        
        # Solve: U[a, :] @ B_flat.T = T[a, :, :].flatten()
        # i.e., B_flat @ U[a, :].T = T[a, :, :].flatten()
        # U[a, :].T = pinv(B_flat) @ T[a].flatten()
        
        # Similarly for V and W.
        
        # Update U
        B_flat = np.zeros((mp * pn, rank))
        for r in range(rank):
            B_flat[:, r] = (V[:, r] * W[:, r]).flatten()
        
        new_U = np.zeros_like(U)
        for a in range(nm):
            rhs = T[a, :, :].flatten()
            # Solve B_flat @ x = rhs
            try:
                new_U[a, :] = np.linalg.lstsq(B_flat, rhs, rcond=None)[0]
            except:
                new_U[a, :] = U[a, :]
        
        # Update V
        A_flat = np.zeros((nm * pn, rank))
        for r in range(rank):
            A_flat[:, r] = (U[:, r] * W[:, r]).flatten()
        
        new_V = np.zeros_like(V)
        for b in range(mp):
            rhs = T[:, b, :].reshape(nm * pn)
            try:
                new_V[b, :] = np.linalg.lstsq(A_flat, rhs, rcond=None)[0]
            except:
                new_V[b, :] = V[b, :]
        
        # Update W
        C_flat = np.zeros((nm * mp, rank))
        for r in range(rank):
            C_flat[:, r] = (U[:, r] * V[:, r]).flatten()
        
        new_W = np.zeros_like(W)
        for c in range(pn):
            rhs = T[:, :, c].reshape(nm * mp)
            try:
                new_W[c, :] = np.linalg.lstsq(C_flat, rhs, rcond=None)[0]
            except:
                new_W[c, :] = W[c, :]
        
        U, V, W = new_U, new_V, new_W
        
        # Check convergence
        if iteration % 50 == 0:
            result = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(result - T))
            if err < tol:
                return U, V, W
    
    # Final check
    result = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(result - T))
    if err < 1e-8:
        return U, V, W
    return None


def _construct_structural_decomposition(n, m, p):
    """
    Construct a decomposition exploiting the structure of the all-ones tensor.
    
    For the matrix multiplication tensor T[i,j,k] = 1 (for all valid i,j,k),
    we can use the following construction:
    
    Key insight: T[i,j,k] = 1 can be written as sum_r u[i,j,r] * v[j,k,r] * w[k,i,r]
    
    One approach: Use the identity that for the all-ones tensor, we can factor
    through intermediate sums. 
    
    Construction with rank = m + p + n - 1 (if valid) or similar.
    
    Actually, a known construction for the all-ones tensor of shape (n,m,p):
    rank <= n*m + m*p + n*p - n - m - p + 1 is NOT generally correct.
    
    Let me try a different approach: 
    For n=2, we can write:
    T[i,j,k] = 1 = (1/2) * (1 + 1) ... 
    
    Better: Use the fact that 1 = sum over some basis.
    
    Let me try a concrete construction:
    For each j in range(m), and for the i=0 and i=1 cases:
    
    Actually, let me try rank = m + p + n - 1 = 5 + 6 + 2 - 1 = 12.
    This is likely too optimistic.
    
    A more realistic construction:
    rank = 2*m + 2*p - 2 = 10 + 12 - 2 = 20? Let me verify if this works.
    
    Construction idea:
    For r = 0 to m-1: handle the "j=r" terms for i=0
    For r = m to m+p-1: handle the "k=r-m" terms for i=1
    ... this doesn't quite work.
    
    Let me try a simpler but valid construction with rank < 60.
    
    Approach: For each (i, k) pair (2*6=12 pairs), we need to compute C[i,k] = sum_j A[i,j]*B[j,k].
    We can use rank-1 terms for each (i,j,k) but that's schoolbook (60).
    
    Better: For fixed i, C[i,:] = A[i,:] @ B. This is a vector-matrix product.
    The rank of the bilinear map (A[i,:], B) -> A[i,:] @ B is at most min(m, p) per i,
    but we need to handle both i values simultaneously.
    
    Let me try: rank = 2 * min(m, p) = 2 * 5 = 10? Probably not enough.
    
    Let me just try to find a valid decomposition with rank < 60 using a heuristic.
    """
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    
    # Try rank from lower bound to schoolbook
    # Lower bound: max(nm, mp, pn) = 30
    best_rank = n * m * p
    best_decomp = None
    
    # First, try a structured construction
    # For the all-ones tensor, one known approach:
    # T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[k,i]
    
    # Try rank = m + p + n - 1 = 12 (optimistic)
    # Try rank = 2*m + 2*p - 2 = 20
    # Try rank = m*p = 30 (schoolbook for one i)
    
    # Let me try a concrete construction with rank = 2*p + m = 12 + 5 = 17
    # or rank = 2*m + p = 10 + 6 = 16
    
    # Actually, let me just use a simple valid construction:
    # For each i in {0,1}, for each k in range(p):
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    # We can write this as p terms for each i, so 2*p = 12 terms.
    # But each term needs to be rank-1 in the tensor sense.
    
    # Construction with rank = 2*p:
    # For r = i*p + k (i in {0,1}, k in range(p)):
    #   U[i*m + j, r] = delta[i, r//p] * 1 (for all j)
    #   V[j*p + k, r] = delta[k, r%p] * 1 (for all j)  
    #   W[k'*n + i', r] = delta[k', r%p] * delta[i', r//p] * 1
    # 
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i', r]
    # = sum_{i', k''} delta[i, i'] * delta[k, k''] * delta[k', k''] * delta[i', i]
    # = delta[i, i'] * delta[k, k'] * delta[k', k] ... this is getting confusing.
    
    # Let me just verify the schoolbook and try to find improvements via ALS.
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, c] = 1.0
                V_sb[j * p + k, c] = 1.0
                W_sb[k * n + i, c] = 1.0
                c += 1
    
    best_decomp = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Try to find a lower rank decomposition using random initialization + ALS
    np.random.seed(42)
    
    for target_rank in range(30, rank_sb):
        # Try a few random initializations
        for restart in range(3):
            U_rand = np.random.randn(nm, target_rank)
            V_rand = np.random.randn(mp, target_rank)
            W_rand = np.random.randn(pn, target_rank)
            
            polished = _als_polish(U_rand, V_rand, W_rand, n, m, p, tol=1e-10, max_iter=500)
            
            if polished is not None:
                U_p, V_p, W_p = polished
                if _verify_decomposition(U_p, V_p, W_p, n, m, p, tol=1e-9):
                    best_rank = target_rank
                    best_decomp = (U_p, V_p, W_p)
                    return best_decomp
    
    return best_decomp


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 60)
    nm, mp, pn = n * m, m * p, p * n
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, c] = 1.0
                V_sb[j * p + k, c] = 1.0
                W_sb[k * n + i, c] = 1.0
                c += 1
    
    best_decomp = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Check time
    if time.time() > deadline:
        return {"U": best_decomp[0].tolist(), "V": best_decomp[1].tolist(), "W": best_decomp[2].tolist()}
    
    # Try to find a better decomposition
    result = _construct_structural_decomposition(n, m, p)
    
    if result is not None:
        U_r, V_r, W_r = result
        if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-9):
            best_decomp = (U_r, V_r, W_r)
            best_rank = U_r.shape[1]
    
    # Check time
    if time.time() > deadline - 5:
        return {"U": best_decomp[0].tolist(), "V": best_decomp[1].tolist(), "W": best_decomp[2].tolist()}
    
    # Final verification
    U_final, V_final, W_final = best_decomp
    if not _verify_decomposition(U_final, V_final, W_final, n, m, p, tol=1e-9):
        # Fall back to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

# EVOLVE-BLOCK-END
