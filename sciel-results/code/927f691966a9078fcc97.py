import time
import numpy as np

DEADLINE_S = 110.0
N, M, P = 4, 5, 5

def _validate(U, V, W, tol=1e-9):
    """Vectorized validation: check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = N, M, P
    R = U.shape[1]
    # Reshape: U is (n*m, R), V is (m*p, R), W is (p*n, R)
    # We need: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    # Build tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # U reshaped to (n, m, R), V to (m, p, R), W to (p, n, R)
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.all(np.abs(T - 1.0) <= tol)

def _validate_partial(U, V, W, remaining_indices, tol=1e-9):
    """Validate that for remaining (i,j,k) triples not yet covered, the sum equals 1.
    remaining_indices: list of (i,j,k) tuples still needing coverage."""
    n, m, p = N, M, P
    R = U.shape[1]
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    for i, j, k in remaining_indices:
        s = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
        if abs(s - 1.0) > tol:
            return False
    return True

def _strassen_2x2_vectors():
    return [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]

def _structured_85():
    """Build the known rank-85 structured decomposition."""
    n, m, p = N, M, P
    R = 85
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    c = 0
    
    s2 = _strassen_2x2_vectors()
    
    def map_2x2_to_4x4(vec, row_off, col_off):
        out = [0.0] * 16
        for r in range(2):
            for c2 in range(2):
                out[(row_off + r) * 4 + (col_off + c2)] = vec[r * 2 + c2]
        return out

    calls = [
        ([ (0,0,1), (2,2,1) ], [ (0,0,1), (2,2,1) ], [ (0,0,1), (2,2,1) ]),
        ([ (2,0,1), (2,2,1) ], [ (0,0,1) ], [ (2,0,1), (2,2,-1) ]),
        ([ (0,0,1) ], [ (0,2,1), (2,2,-1) ], [ (0,2,1), (2,2,1) ]),
        ([ (2,2,1) ], [ (2,0,1), (0,0,-1) ], [ (0,0,1), (2,0,1) ]),
        ([ (0,0,1), (0,2,1) ], [ (2,2,1) ], [ (0,0,-1), (0,2,1) ]),
        ([ (2,0,1), (0,0,-1) ], [ (0,0,1), (0,2,1) ], [ (2,2,1) ]),
        ([ (0,2,1), (2,2,-1) ], [ (2,0,1), (2,2,1) ], [ (0,0,1) ])
    ]
    
    for u_blocks, v_blocks, w_blocks in calls:
        for u_vec, v_vec, w_vec in s2:
            u_full = [0.0] * 16
            v_full = [0.0] * 16
            w_full = [0.0] * 16
            
            for row_off, col_off, sign in u_blocks:
                mapped = map_2x2_to_4x4(u_vec, row_off, col_off)
                for i in range(16):
                    u_full[i] += sign * mapped[i]
                    
            for row_off, col_off, sign in v_blocks:
                mapped = map_2x2_to_4x4(v_vec, row_off, col_off)
                for i in range(16):
                    v_full[i] += sign * mapped[i]
                    
            for row_off, col_off, sign in w_blocks:
                mapped = map_2x2_to_4x4(w_vec, row_off, col_off)
                for i in range(16):
                    w_full[i] += sign * mapped[i]
            
            for i in range(4):
                for j in range(4):
                    u_idx = i * 5 + j
                    u_val = u_full[i * 4 + j]
                    if u_val != 0:
                        U[u_idx, c] += u_val
                        
            for j in range(4):
                for k in range(4):
                    v_idx = j * 5 + k
                    v_val = v_full[j * 4 + k]
                    if v_val != 0:
                        V[v_idx, c] += v_val
                        
            for k in range(4):
                for i in range(4):
                    w_idx = k * 4 + i
                    w_val = w_full[k * 4 + i]
                    if w_val != 0:
                        W[w_idx, c] += w_val
                        
            c += 1
            
    # Part 2: A11 B112 (4x4x1) Rank 16
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, c] = 1.0
            V[j * 5 + 4, c] = 1.0
            W[4 * 4 + i, c] = 1.0
            c += 1
            
    # Part 3: A12 B21 (4x1x5) Rank 20
    for i in range(4):
        for k in range(5):
            U[i * 5 + 4, c] = 1.0
            V[4 * 5 + k, c] = 1.0
            W[k * 4 + i, c] = 1.0
            c += 1
            
    return U, V, W

def _try_peel_rank1(U, V, W, max_attempts=20, tol=1e-9):
    """Try to peel rank-1 components from the decomposition.
    A rank-1 component corresponds to finding (u, v, w) such that
    the outer product u⊗v⊗w is a component of the multiplication tensor.
    
    We look for rows of U, V, W that are scalar multiples of each other
    across all (i,j,k) triples they contribute to."""
    n, m, p = N, M, P
    R = U.shape[1]
    
    # Try to find columns that can be combined
    # For each pair of columns, check if they can be merged
    # This is a heuristic approach
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = R
    
    # Try removing columns that are linearly dependent
    # First, check if any column is zero
    for c in range(R):
        if np.allclose(U[:, c], 0, atol=tol) or np.allclose(V[:, c], 0, atol=tol) or np.allclose(W[:, c], 0, atol=tol):
            # This column contributes nothing, can remove
            mask = [i for i in range(R) if i != c]
            new_U = U[:, mask]
            new_V = V[:, mask]
            new_W = W[:, mask]
            if _validate(new_U, new_V, new_W, tol=tol):
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = len(mask)
    
    # Try to find rank-1 components: look for columns where U[:,c], V[:,c], W[:,c]
    # are proportional to a rank-1 pattern
    # A rank-1 tensor for matmul has U[i*m+j] = a_i * b_j, V[j*p+k] = b_j * c_k, W[k*n+i] = c_k * a_i
    # Check each column for this structure
    for c in range(R):
        u_col = U[:, c]
        v_col = V[:, c]
        w_col = W[:, c]
        
        if np.allclose(u_col, 0, atol=tol):
            continue
            
        # Try to factor u_col as a_i * b_j
        # u_col is indexed by i*m + j
        u_mat = u_col.reshape(n, m)
        # Check if rank 1
        if np.allclose(u_mat, 0, atol=tol):
            continue
            
        # SVD to check rank
        try:
            sv = np.linalg.svd(u_mat, compute_uv=False)
        except:
            continue
        if len(sv) < 2 or sv[1] / max(sv[0], tol) < 1e-6:
            # u_mat is rank 1: u_mat = a * b^T
            # Get a and b
            if abs(sv[0]) < tol:
                continue
            # Use first left and right singular vectors
            Usv, _, Vsv = np.linalg.svd(u_mat, full_matrices=False)
            a = Usv[:, 0] * sv[0]
            b = Vsv[0, :]
            
            # Check v_col = b_j * c_k
            v_mat = v_col.reshape(m, p)
            if np.allclose(v_mat, 0, atol=tol):
                continue
            try:
                sv_v = np.linalg.svd(v_mat, compute_uv=False)
            except:
                continue
            if len(sv_v) < 2 or sv_v[1] / max(sv_v[0], tol) < 1e-6:
                # v_mat is rank 1
                Usv_v, _, Vsv_v = np.linalg.svd(v_mat, full_matrices=False)
                b_v = Usv_v[:, 0] * sv_v[0]
                c_v = Vsv_v[0, :]
                
                # Check if b and b_v are proportional
                if np.allclose(b, 0, atol=tol) or np.allclose(b_v, 0, atol=tol):
                    continue
                ratio = b[0] / b_v[0] if abs(b_v[0]) > tol else None
                if ratio is not None and abs(b[0]) > tol:
                    if np.allclose(b, ratio * b_v, atol=1e-6, rtol=1e-6):
                        # Check w_col = c_v * a (up to scaling)
                        w_mat = w_col.reshape(p, n)
                        if np.allclose(w_mat, 0, atol=tol):
                            continue
                        expected_w = np.outer(c_v, a)
                        if np.allclose(w_mat, expected_w, atol=1e-6, rtol=1e-6):
                            # This column is a rank-1 component!
                            # Check if we can remove it and still have valid decomposition
                            # Actually, we can't just remove it - it's part of the decomposition
                            # But we note that this structure exists
                            pass
    
    return best_U, best_V, best_W, best_rank

def _schoolbook():
    rank = N * M * P
    U = np.zeros((N*M, rank))
    V = np.zeros((M*P, rank))
    W = np.zeros((P*N, rank))
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i*M+j, c] = 1.0
                V[j*P+k, c] = 1.0
                W[k*N+i, c] = 1.0
                c += 1
    return U, V, W

def construct():
    t0 = time.time()
    
    # Construct the rank-85 decomposition
    U, V, W = _structured_85()
    
    # Validate with vectorized check
    if not _validate(U, V, W, tol=1e-9):
        # Fallback to schoolbook if validation fails
        U, V, W = _schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to peel rank-1 components if time permits
    elapsed = time.time() - t0
    if elapsed < DEADLINE_S - 10.0:
        U2, V2, W2, rank2 = _try_peel_rank1(U, V, W)
        if rank2 < U.shape[1] and _validate(U2, V2, W2, tol=1e-9):
            U, V, W = U2, V2, W2
    
    # Final time check
    elapsed = time.time() - t0
    if elapsed > DEADLINE_S - 5.0:
        # Return immediately
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
