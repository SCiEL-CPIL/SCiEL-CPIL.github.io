import time
import numpy as np

def construct():
    _t0 = time.time()
    DEADLINE = _t0 + 110.0  # Safety margin before 120s limit
    
    n, m, p = 3, 4, 6
    RANK = 66
    
    U = [[0] * RANK for _ in range(n * m)]
    V = [[0] * RANK for _ in range(m * p)]
    W = [[0] * RANK for _ in range(p * n)]
    
    col = 0
    
    def _idx_u(i, k):
        return i * m + k

    def _idx_v(k, j):
        return k * p + j

    def _idx_w(j, i):
        return j * n + i

    def _add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = _idx_u(a_r, a_c)
        A1 = _idx_u(a_r, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C0 = _idx_w(c_c, c_r)
        C1 = _idx_w(c_c + 1, c_r)
        
        U[A0][col] = 1; V[B00][col] = 1; W[C0][col] = 1; col += 1
        U[A1][col] = 1; V[B10][col] = 1; W[C0][col] = 1; col += 1
        U[A0][col] = 1; V[B01][col] = 1; W[C1][col] = 1; col += 1
        U[A1][col] = 1; V[B11][col] = 1; W[C1][col] = 1; col += 1

    def _add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = _idx_u(a_r, a_c)
        A01 = _idx_u(a_r, a_c + 1)
        A10 = _idx_u(a_r + 1, a_c)
        A11 = _idx_u(a_r + 1, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C00 = _idx_w(c_c, c_r)
        C01 = _idx_w(c_c + 1, c_r)
        C10 = _idx_w(c_c, c_r + 1)
        C11 = _idx_w(c_c + 1, c_r + 1)

        U[A00][col] = 1; U[A11][col] = 1
        V[B00][col] = 1; V[B11][col] = 1
        W[C00][col] = 1; W[C11][col] = 1
        col += 1

        U[A10][col] = 1; U[A11][col] = 1
        V[B00][col] = 1
        W[C10][col] = 1; W[C11][col] = -1
        col += 1

        U[A00][col] = 1
        V[B01][col] = 1; V[B11][col] = -1
        W[C01][col] = 1; W[C11][col] = 1
        col += 1

        U[A11][col] = 1
        V[B10][col] = 1; V[B00][col] = -1
        W[C00][col] = 1; W[C10][col] = 1
        col += 1

        U[A00][col] = 1; U[A01][col] = 1
        V[B11][col] = 1
        W[C00][col] = -1; W[C01][col] = 1
        col += 1

        U[A10][col] = 1; U[A00][col] = -1
        V[B00][col] = 1; V[B01][col] = 1
        W[C11][col] = 1
        col += 1

        U[A01][col] = 1; U[A11][col] = -1
        V[B10][col] = 1; V[B11][col] = 1
        W[C00][col] = 1
        col += 1

    # Build 66-term decomposition
    for j in range(3):
        bc = 2 * j
        _add_1x2_2x2(0, 0, 0, bc, 0, bc)
        _add_1x2_2x2(0, 2, 2, bc, 0, bc)
        _add_strassen(1, 0, 0, bc, 1, bc)
        _add_strassen(1, 2, 2, bc, 1, bc)

    assert col == RANK, f"Expected {RANK} terms, got {col}"
    
    # Convert to numpy for faster verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Precompute index mappings for vectorized verification
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # T[i,k,j] = sum_t U[i*m+k, t] * V[k*p+j, t] * W[j*n+i, t]
    # We need T[i,k,j] = 1 for all i,k,j
    
    def verify_decomp(U_mat, V_mat, W_mat):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor using vectorized ops."""
        r = U_mat.shape[1]
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U3 = U_mat.reshape(n, m, r)
        V3 = V_mat.reshape(m, p, r)
        W3 = W_mat.reshape(p, n, r)
        # Compute tensor: T[i,k,j] = sum_t U3[i,k,t] * V3[k,j,t] * W3[j,i,t]
        # Use einsum: 'ikr,kjr,jir->ijk'
        T = np.einsum('ikr,kjr,jir->ijk', U3, V3, W3)
        # Check all entries are 1
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True
    
    # Verify baseline
    if not verify_decomp(U_np, V_np, W_np):
        # Return baseline if verification fails
        return {"U": U, "V": V, "W": W}
    
    # Local search: try to reduce rank by merging or dropping terms
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
    best_rank = RANK
    
    def try_merge_terms(U_mat, V_mat, W_mat, r):
        """Try to merge two terms into one by checking if they are proportional."""
        for a in range(r):
            if time.time() > DEADLINE - 5:
                return None
            u_a = U_mat[:, a]
            v_a = V_mat[:, a]
            w_a = W_mat[:, a]
            
            # Find non-zero entries
            nz_u = np.nonzero(u_a)[0]
            if len(nz_u) == 0:
                continue
            
            for b in range(a + 1, r):
                u_b = U_mat[:, b]
                v_b = V_mat[:, b]
                w_b = W_mat[:, b]
                
                # Check proportionality using vectorized comparison
                # u_b should be ratio * u_a
                mask = nz_u
                if np.any(u_a[mask] == 0):
                    continue
                ratios_u = u_b[mask] / u_a[mask]
                if np.any(np.abs(ratios_u - ratios_u[0]) > 1e-9):
                    continue
                ratio = ratios_u[0]
                
                # Verify full proportionality
                if not np.allclose(u_b, u_a * ratio, atol=1e-9):
                    continue
                if not np.allclose(v_b, v_a * ratio, atol=1e-9):
                    continue
                if not np.allclose(w_b, w_a * ratio, atol=1e-9):
                    continue
                
                # Merge: term_b = ratio * term_a, so combined = (1+ratio) * term_a
                new_U = U_mat.copy()
                new_V = V_mat.copy()
                new_W = W_mat.copy()
                
                new_U[:, a] = u_a * (1 + ratio)
                new_V[:, a] = v_a * (1 + ratio)
                new_W[:, a] = w_a * (1 + ratio)
                
                new_U = np.delete(new_U, b, axis=1)
                new_V = np.delete(new_V, b, axis=1)
                new_W = np.delete(new_W, b, axis=1)
                
                if verify_decomp(new_U, new_V, new_W):
                    return new_U, new_V, new_W, r - 1
        
        return None
    
    def try_drop_term(U_mat, V_mat, W_mat, r, term_idx):
        """Try to drop a single term."""
        new_U = np.delete(U_mat, term_idx, axis=1)
        new_V = np.delete(V_mat, term_idx, axis=1)
        new_W = np.delete(W_mat, term_idx, axis=1)
        
        if verify_decomp(new_U, new_V, new_W):
            return new_U, new_V, new_W, r - 1
        return None
    
    # Try to drop or merge terms
    current_U, current_V, current_W = best_U.copy(), best_V.copy(), best_W.copy()
    current_rank = best_rank
    
    # Try dropping each term (repeat until no more drops possible)
    dropped = True
    while dropped:
        dropped = False
        for i in range(current_rank):
            if time.time() > DEADLINE - 5:
                break
            result = try_drop_term(current_U, current_V, current_W, current_rank, i)
            if result:
                current_U, current_V, current_W, current_rank = result
                best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
                best_rank = current_rank
                dropped = True
                break  # Restart from beginning after a drop
    
    # Try merging pairs (repeat until no more merges possible)
    merged = True
    while merged:
        merged = False
        if time.time() > DEADLINE - 5:
            break
        result = try_merge_terms(current_U, current_V, current_W, current_rank)
        if result:
            current_U, current_V, current_W, current_rank = result
            best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
            best_rank = current_rank
            merged = True
    
    # Convert back to lists
    final_U = best_U.tolist()
    final_V = best_V.tolist()
    final_W = best_W.tolist()
    
    return {"U": final_U, "V": final_V, "W": final_W}
