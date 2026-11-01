import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=1e-10)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100

    # Standard Rank 39 Construction
    rank = 39
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    r = 0

    def add_strassen_block(k0, k1):
        nonlocal r
        v_e, v_f = k0, k1
        v_g, v_h = 11 + k0, 11 + k1
        w_c00, w_c10 = 2 * k0, 2 * k0 + 1
        w_c01, w_c11 = 2 * k1, 2 * k1 + 1

        U[0, r] += 1; U[3, r] += 1
        V[v_e, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1; W[w_c11, r] += 1
        r += 1

        U[2, r] += 1; U[3, r] += 1
        V[v_e, r] += 1
        W[w_c10, r] += 1; W[w_c11, r] += -1
        r += 1

        U[0, r] += 1
        V[v_f, r] += 1; V[v_h, r] += -1
        W[w_c01, r] += 1; W[w_c11, r] += 1
        r += 1

        U[3, r] += 1
        V[v_g, r] += 1; V[v_e, r] += -1
        W[w_c00, r] += 1; W[w_c10, r] += 1
        r += 1

        U[0, r] += 1; U[1, r] += 1
        V[v_h, r] += 1
        W[w_c00, r] += -1; W[w_c01, r] += 1
        r += 1

        U[2, r] += 1; U[0, r] += -1
        V[v_e, r] += 1; V[v_f, r] += 1
        W[w_c11, r] += 1
        r += 1

        U[1, r] += 1; U[3, r] += -1
        V[v_g, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1
        r += 1

    for block in range(5):
        if time.time() > deadline - 5:
            break
        add_strassen_block(block * 2, block * 2 + 1)

    if r < rank:
        k2 = 10
        U[0, r] += 1; V[k2, r] += 1; W[2*k2, r] += 1; r += 1
        U[1, r] += 1; V[11+k2, r] += 1; W[2*k2, r] += 1; r += 1
        U[2, r] += 1; V[k2, r] += 1; W[2*k2+1, r] += 1; r += 1
        U[3, r] += 1; V[11+k2, r] += 1; W[2*k2+1, r] += 1; r += 1

    # Structural Change: Attempt to reduce rank via linear dependency check
    # We check if the 39 rank-1 tensors are linearly independent.
    # If not, we can remove one.
    
    if time.time() < deadline - 10:
        # Flatten the 39 terms into a matrix of shape (n*m * m*p * p*n, 39)
        # This is 1936 x 39
        dim_tensor = n * m * m * p * p * n
        M = np.zeros((dim_tensor, rank), dtype=np.float64)
        
        for col in range(rank):
            # Term is U[:, col] (outer) V[:, col] (outer) W[:, col]
            # Flatten this rank-1 tensor
            term = np.einsum('a,b,c->abc', U[:, col], V[:, col], W[:, col])
            M[:, col] = term.flatten()
            
        # Compute rank of M
        # Use SVD for stability
        try:
            U_svd, s, Vh_svd = np.linalg.svd(M, full_matrices=False)
            # Count non-zero singular values
            tol = 1e-10
            num_independent = np.sum(s > tol)
            
            if num_independent < rank:
                # We found a dependency. We can potentially reduce the rank.
                # However, removing a term requires re-solving for the coefficients or just removing it if it's exactly zero in the reduced space?
                # No, if they are linearly dependent, the span is the same.
                # We can remove any one of the dependent terms and the span remains the same?
                # Yes, if $v_k = \sum_{j \neq k} c_j v_j$, then the set without $v_k$ spans the same space.
                # So we can just drop the last term?
                # Wait, we need to ensure the dropped term is in the span of the others.
                # If rank < 39, at least one is dependent.
                # We can try to remove the last term (index 38) and check if the remaining 38 span the tensor.
                
                U_reduced = U[:, :rank-1]
                V_reduced = V[:, :rank-1]
                W_reduced = W[:, :rank-1]
                
                if _verify(U_reduced, V_reduced, W_reduced):
                    return {"U": U_reduced.tolist(), "V": V_reduced.tolist(), "W": W_reduced.tolist()}
                    
                # If removing the last one didn't work, try removing others?
                # This is a heuristic. If the dependency is complex, we might need to keep a specific subset.
                # But for now, returning 39 is safe.
        except Exception:
            pass

    # Fallback to Rank 39
    if time.time() < deadline - 2:
        if _verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Schoolbook (Rank 44)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
