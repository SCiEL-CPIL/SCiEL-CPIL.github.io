"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 16, 28, 28
    deadline = time.time() + 130  # safety margin before 150s kill

    # Build target tensor indices (sparse representation)
    # T[i*m+j, j*p+k, k*n+i] = 1
    a_idx = []
    b_idx = []
    c_idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx.append(i * m + j)
                b_idx.append(j * p + k)
                c_idx.append(k * n + i)
    a_idx = np.array(a_idx)
    b_idx = np.array(b_idx)
    c_idx = np.array(c_idx)
    n_entries = len(a_idx)  # 112

    # Schoolbook decomposition as baseline
    rank_school = n * m * p  # 112
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    def check_validity(U, V, W, tol=1e-9):
        """Check if decomposition is valid by verifying all tensor entries."""
        for idx in range(n_entries):
            a, b, c = a_idx[idx], b_idx[idx], c_idx[idx]
            val = np.dot(U[a], V[b] * W[c])  # sum_r U[a,r]*V[b,r]*W[c,r]
            if abs(val - 1.0) > tol:
                return False
        # Also check that entries where T=0 are approximately 0
        # Sample a few zero entries to verify
        for _ in range(50):
            a = np.random.randint(0, nm)
            b = np.random.randint(0, mp)
            c = np.random.randint(0, pn)
            # Check if this is a nonzero entry
            is_nonzero = False
            for idx in range(n_entries):
                if a_idx[idx] == a and b_idx[idx] == b and c_idx[idx] == c:
                    is_nonzero = True
                    break
            if not is_nonzero:
                val = np.dot(U[a], V[b] * W[c])
                if abs(val) > tol * 10:
                    return False
        return True

    def compute_error(U, V, W):
        """Compute max error on nonzero entries."""
        max_err = 0.0
        for idx in range(n_entries):
            a, b, c = a_idx[idx], b_idx[idx], c_idx[idx]
            val = np.dot(U[a], V[b] * W[c])
            err = abs(val - 1.0)
            if err > max_err:
                max_err = err
        return max_err

    def build_design_matrices(U, V, W, rank):
        """Build sparse design matrices for each mode update."""
        # For U update: T[a, bc] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Design matrix M_U: M_U[r, (b,c)] = V[b,r] * W[c,r]
        # We only need entries where T is nonzero
        M_U_rows = []
        M_U_cols = []
        M_U_vals = []
        for r in range(rank):
            for idx in range(n_entries):
                b, c = b_idx[idx], c_idx[idx]
                val = V[b, r] * W[c, r]
                if abs(val) > 1e-15:
                    M_U_rows.append(r)
                    M_U_cols.append(idx)
                    M_U_vals.append(val)
        
        # For V update: T[ab, c] = sum_r V[b,r] * U[a,r] * W[c,r]
        # Design matrix M_V: M_V[r, (a,c)] = U[a,r] * W[c,r]
        M_V_rows = []
        M_V_cols = []
        M_V_vals = []
        for r in range(rank):
            for idx in range(n_entries):
                a, c = a_idx[idx], c_idx[idx]
                val = U[a, r] * W[c, r]
                if abs(val) > 1e-15:
                    M_V_rows.append(r)
                    M_V_cols.append(idx)
                    M_V_vals.append(val)
        
        # For W update: T[ab, c] = sum_r W[c,r] * U[a,r] * V[b,r]
        # Design matrix M_W: M_W[r, (a,b)] = U[a,r] * V[b,r]
        M_W_rows = []
        M_W_cols = []
        M_W_vals = []
        for r in range(rank):
            for idx in range(n_entries):
                a, b = a_idx[idx], b_idx[idx]
                val = U[a, r] * V[b, r]
                if abs(val) > 1e-15:
                    M_W_rows.append(r)
                    M_W_cols.append(idx)
                    M_W_vals.append(val)
        
        from scipy.sparse import csr_matrix
        M_U = csr_matrix((M_U_vals, (M_U_rows, M_U_cols)), shape=(rank, n_entries))
        M_V = csr_matrix((M_V_vals, (M_V_rows, M_V_cols)), shape=(rank, n_entries))
        M_W = csr_matrix((M_W_vals, (M_W_rows, M_W_cols)), shape=(rank, n_entries))
        
        return M_U, M_V, M_W

    def als_iteration(U, V, W, rank, M_U, M_V, M_W, max_iter=200):
        """Run ALS iterations."""
        T_flat = np.ones(n_entries)  # All nonzero entries are 1
        
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            # Update U: solve U @ M_U = T_flat
            # U is (nm, rank), M_U is (rank, n_entries)
            # For each a: U[a,:] @ M_U = T_flat[a]... wait, this isn't right
            # Actually: T[a, idx] = sum_r U[a,r] * M_U[r, idx]
            # So for fixed a: T[:, a]... no
            # T_flat[idx] for entries where a_idx[idx] == a
            # Let's group by a
            U_new = np.zeros((nm, rank))
            for a in range(nm):
                mask = (a_idx == a)
                if not np.any(mask):
                    continue
                T_a = T_flat[mask]
                M_a = M_U[:, mask]  # (rank, n_a)
                # Solve: U[a,:] @ M_a = T_a
                # U[a,:] = T_a @ pinv(M_a)
                try:
                    if M_a.shape[1] >= rank:
                        # Use lstsq
                        U_new[a], _, _, _ = np.linalg.lstsq(M_a.T, T_a, rcond=None)
                    else:
                        U_new[a] = np.linalg.pinv(M_a.T) @ T_a
                except:
                    U_new[a] = U[a]
            U = U_new
            
            # Update V
            V_new = np.zeros((mp, rank))
            for b in range(mp):
                mask = (b_idx == b)
                if not np.any(mask):
                    continue
                T_b = T_flat[mask]
                M_b = M_V[:, mask]
                try:
                    if M_b.shape[1] >= rank:
                        V_new[b], _, _, _ = np.linalg.lstsq(M_b.T, T_b, rcond=None)
                    else:
                        V_new[b] = np.linalg.pinv(M_b.T) @ T_b
                except:
                    V_new[b] = V[b]
            V = V_new
            
            # Update W
            W_new = np.zeros((pn, rank))
            for c in range(pn):
                mask = (c_idx == c)
                if not np.any(mask):
                    continue
                T_c = T_flat[mask]
                M_c = M_W[:, mask]
                try:
                    if M_c.shape[1] >= rank:
                        W_new[c], _, _, _ = np.linalg.lstsq(M_c.T, T_c, rcond=None)
                    else:
                        W_new[c] = np.linalg.pinv(M_c.T) @ T_c
                except:
                    W_new[c] = W[c]
            W = W_new
            
            # Check convergence
            if it % 10 == 0:
                err = compute_error(U, V, W)
                if err < 1e-10:
                    break
        
        return U, V, W

    # Try decreasing ranks
    for target_rank in range(80, rank_school):
        if time.time() > deadline - 20:
            break
        
        n_restarts = 3 if target_rank < 90 else 2
        found = False
        
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            # Initialize
            if restart == 0:
                # Use previous best as warm start if possible
                if best_rank == target_rank + 1:
                    U = best_U[:, :target_rank] + np.random.randn(nm, target_rank) * 0.1
                    V = best_V[:, :target_rank] + np.random.randn(mp, target_rank) * 0.1
                    W = best_W[:, :target_rank] + np.random.randn(pn, target_rank) * 0.1
                else:
                    U = np.random.randn(nm, target_rank) * 0.5
                    V = np.random.randn(mp, target_rank) * 0.5
                    W = np.random.randn(pn, target_rank) * 0.5
            else:
                U = np.random.randn(nm, target_rank) * 0.5
                V = np.random.randn(mp, target_rank) * 0.5
                W = np.random.randn(pn, target_rank) * 0.5
            
            # Build design matrices
            M_U, M_V, M_W = build_design_matrices(U, V, W, target_rank)
            
            # Run ALS
            U, V, W = als_iteration(U, V, W, target_rank, M_U, M_V, M_W, max_iter=150)
            
            # Check validity
            if check_validity(U, V, W):
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = target_rank
                found = True
                break
        
        if found:
            # Try to go lower
            continue
        else:
            # If we can't find valid at this rank, stop (ranks below are harder)
            break

    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
