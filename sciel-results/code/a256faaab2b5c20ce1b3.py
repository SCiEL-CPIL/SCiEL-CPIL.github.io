import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    deadline = time.time() + 120.0
    
    # Build rank-28 decomposition
    rank28 = 28
    U = [[0] * rank28 for _ in range(n * m)]
    V = [[0] * rank28 for _ in range(m * p)]
    W = [[0] * rank28 for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        U[(0 + i_off) * m + (0 + j_off)][col] = 1
        U[(1 + i_off) * m + (1 + j_off)][col] = 1
        V[(0 + j_off) * p + (0 + k_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = 1
        W[(0 + k_off) * n + (0 + i_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        U[(1 + i_off) * m + (0 + j_off)][col] = 1
        U[(1 + i_off) * m + (1 + j_off)][col] = 1
        V[(0 + j_off) * p + (0 + k_off)][col] = 1
        W[(0 + k_off) * n + (1 + i_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = -1
        col += 1
        
        U[(0 + i_off) * m + (0 + j_off)][col] = 1
        V[(0 + j_off) * p + (1 + k_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = -1
        W[(1 + k_off) * n + (0 + i_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        U[(1 + i_off) * m + (1 + j_off)][col] = 1
        V[(1 + j_off) * p + (0 + k_off)][col] = 1
        V[(0 + j_off) * p + (0 + k_off)][col] = -1
        W[(0 + k_off) * n + (0 + i_off)][col] = 1
        W[(0 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        U[(0 + i_off) * m + (0 + j_off)][col] = 1
        U[(0 + i_off) * m + (1 + j_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = 1
        W[(0 + k_off) * n + (0 + i_off)][col] = -1
        W[(1 + k_off) * n + (0 + i_off)][col] = 1
        col += 1
        
        U[(1 + i_off) * m + (0 + j_off)][col] = 1
        U[(0 + i_off) * m + (0 + j_off)][col] = -1
        V[(0 + j_off) * p + (0 + k_off)][col] = 1
        V[(0 + j_off) * p + (1 + k_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        U[(0 + i_off) * m + (1 + j_off)][col] = 1
        U[(1 + i_off) * m + (1 + j_off)][col] = -1
        V[(1 + j_off) * p + (0 + k_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = 1
        W[(0 + k_off) * n + (0 + i_off)][col] = 1
        col += 1

    # Block 1: j=0 (rank 10)
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 2: j=1,2
    add_strassen(0, 1, 0)
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
    add_strassen(0, 1, 3)
    
    assert col == rank28, f"Column count mismatch: {col} != {rank28}"
    
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    # Try to reduce to rank 27 by finding a linear dependency
    # Flatten each rank-1 term
    terms = []
    for r in range(rank28):
        t = np.einsum('i,j,k->ijk', U_mat[:, r], V_mat[:, r], W_mat[:, r])
        terms.append(t.flatten())
    M = np.array(terms).T  # shape (900, 28)
    
    rank_M = np.linalg.matrix_rank(M, tol=1e-10)
    
    if rank_M < rank28:
        # Found dependency, reduce
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        c = Vt[-1, :]
        k_idx = np.argmax(np.abs(c))
        if np.abs(c[k_idx]) > 1e-10:
            new_U = []
            new_V = []
            new_W = []
            for r in range(rank28):
                if r == k_idx:
                    continue
                coeff = 1 - c[r] / c[k_idx]
                new_U.append(U_mat[:, r] * coeff)
                new_V.append(V_mat[:, r] * coeff)
                new_W.append(W_mat[:, r] * coeff)
            U_new = np.array(new_U).T
            V_new = np.array(new_V).T
            W_new = np.array(new_W).T
            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    # If no simple dependency, try solving for rank 27 directly
    # Use the fact that we can try to eliminate one term by expressing it
    # as a combination of others in the tensor space
    
    # Try each pair of terms to see if their difference can be absorbed
    # This is heuristic, but let's try a more systematic approach
    
    # For rank 27, we need to find 27 vectors u_r, v_r, w_r such that
    # sum_r u_r ⊗ v_r ⊗ w_r = T
    
    # Start from rank 28 and try to merge two terms
    # If T_a + T_b can be written as a single rank-1 term, we save 1
    
    # T_a + T_b = u_a⊗v_a⊗w_a + u_b⊗v_b⊗w_b
    # This is rank 1 iff (u_a⊗v_a⊗w_a) and (u_b⊗v_b⊗w_b) are linearly dependent
    # which they're not in general
    
    # Alternative: try to find a different partition
    # Let's try: j=0,1 together and j=2 separate
    
    # j=0,1: 2x2 x 2x5, j=2: 2x1 x 1x5 (rank 10)
    # For 2x2 x 2x5: split k into 2,2,1
    # k=0,1: Strassen (7), k=2,3: Strassen (7), k=4: schoolbook (4) = 18
    # Total: 18 + 10 = 28, same
    
    # Try: j=1,2 with k=0,1,2,3,4 in different splits
    # What if k=0,1,2 (3 outputs) and k=3,4 (2 outputs)?
    # 2x2 x 2x3: can we do better than 7+4=11?
    # 2x2 x 2x2 is rank 7 (Strassen), plus 2x2 x 2x1 is rank 4, total 11
    # 2x2 x 2x2 is rank 7, 2x2 x 2x1 is rank 4, total 11
    # No improvement
    
    # Let's try a numerical optimization approach
    # Fix 27 terms from the 28, and see if we can adjust them to match T
    
    # Pick the term to eliminate (try each)
    for elim_idx in range(rank28):
        if time.time() > deadline - 30:
            break
            
        # Keep all terms except elim_idx
        U_keep = [U_mat[:, r] for r in range(rank28) if r != elim_idx]
        V_keep = [V_mat[:, r] for r in range(rank28) if r != elim_idx]
        W_keep = [W_mat[:, r] for r in range(rank28) if r != elim_idx]
        
        # Current sum of kept terms
        kept_sum = np.zeros((n, m, p))
        for r in range(len(U_keep)):
            kept_sum += np.einsum('i,j,k->ijk', U_keep[r], V_keep[r], W_keep[r])
        
        # Target: T - kept_sum = u_elim ⊗ v_elim ⊗ w_elim
        target = np.ones((n, m, p)) - kept_sum
        
        # Check if target is rank 1
        # Flatten and check if it's a single outer product
        # A tensor is rank 1 iff it can be written as u⊗v⊗w
        # This means all 2x2 minors of certain matricizations are zero
        
        # Matricize as (n*m) x p
        T_mn_p = target.reshape(n * m, p)
        # Rank 1 tensor means this matrix has rank 1
        # Check if all 2x2 minors are zero
        is_rank1 = True
        for i in range(n * m - 1):
            for j in range(i + 1, n * m):
                for k in range(p - 1):
                    for l in range(k + 1, p):
                        minor = T_mn_p[i, k] * T_mn_p[j, l] - T_mn_p[i, l] * T_mn_p[j, k]
                        if abs(minor) > 1e-10:
                            is_rank1 = False
                            break
                    if not is_rank1:
                        break
                if not is_rank1:
                    break
            if not is_rank1:
                break
        
        if is_rank1:
            # Extract u, v, w from target
            # Find a nonzero entry
            nz = np.unravel_index(np.argmax(np.abs(target)), target.shape)
            if abs(target[nz]) > 1e-10:
                # u, v, w such that u⊗v⊗w = target
                # Fix u[nz[0]] = 1, then v[nz[1]] = target[nz] / (u[nz[0]] * w[nz[2]])
                # This is underdetermined; use the matricization
                # target = u ⊗ v ⊗ w
                # Reshape to (n, m*p): target[i, j*p+k] = u[i] * v[j] * w[k]
                # For fixed i, the row is u[i] * (v ⊗ w)
                # So all rows are proportional to v⊗w
                
                # Find v⊗w from the first nonzero row
                for i in range(n):
                    row = target[i, :, :].flatten()
                    if np.any(np.abs(row) > 1e-10):
                        # row = u[i] * (v ⊗ w)
                        # Normalize: let v⊗w = row / norm, then u[i] = norm
                        norm = np.linalg.norm(row)
                        if norm > 1e-10:
                            vw = row / norm
                            u = np.zeros(n)
                            u[i] = 1
                            # For other i', target[i'] = u[i'] * vw
                            for i2 in range(n):
                                if i2 != i:
                                    row2 = target[i2, :, :].flatten()
                                    if np.any(np.abs(row2) > 1e-10):
                                        # row2 should be u[i2] * vw
                                        # u[i2] = <row2, vw> / <vw, vw>
                                        u[i2] = np.dot(row2, vw) / (norm * norm)
                            # Now extract v and w from vw
                            # vw is of length m*p = 15
                            # vw[j*p+k] = v[j] * w[k]
                            # This is a rank-1 matrix of shape (m, p)
                            vw_mat = vw.reshape(m, p)
                            # Check if rank 1
                            sv = np.linalg.svd(vw_mat, full_matrices=False)
                            if sv[0] > 1e-10 and (len(sv) < 2 or sv[1] < 1e-10):
                                # Rank 1: v = sv[0] * U[:,0], w = U[:,0] (up to scaling)
                                U_svd, s, Vt = np.linalg.svd(vw_mat, full_matrices=False)
                                v = U_svd[:, 0] * s[0]
                                w = Vt[0, :]
                                # Verify
                                recon = np.einsum('i,j,k->ijk', u, v, w)
                                if np.allclose(recon, target, atol=1e-10):
                                    # Found rank-27 decomposition!
                                    U_final = U_keep + [u.tolist()]
                                    V_final = V_keep + [v.tolist()]
                                    W_final = W_keep + [w.tolist()]
                                    return {
                                        "U": np.array(U_final).T.tolist(),
                                        "V": np.array(V_final).T.tolist(),
                                        "W": np.array(W_final).T.tolist()
                                    }
                        break
    
    # If all else fails, return the rank-28 solution
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
