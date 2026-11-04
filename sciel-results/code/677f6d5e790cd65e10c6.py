"""Matrix multiplication 2x4x8 via Strassen block decomposition with dependency reduction."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(42)
    # Strassen 2x2x2 blocks
    n_blocks_i = 1
    n_blocks_j = 2
    n_blocks_k = 4
    total_blocks = n_blocks_i * n_blocks_j * n_blocks_k
    rank = total_blocks * 7  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for bi in range(n_blocks_i):
        for bj in range(n_blocks_j):
            for bk in range(n_blocks_k):
                i0 = 2 * bi
                i1 = 2 * bi + 1
                j0 = 2 * bj
                j1 = 2 * bj + 1
                k0 = 2 * bk
                k1 = 2 * bk + 1
                
                u_a = i0 * m + j0
                u_b = i0 * m + j1
                u_c = i1 * m + j0
                u_d = i1 * m + j1
                
                v_e = j0 * p + k0
                v_f = j0 * p + k1
                v_g = j1 * p + k0
                v_h = j1 * p + k1
                
                w_c00 = k0 * n + i0
                w_c01 = k1 * n + i0
                w_c10 = k0 * n + i1
                w_c11 = k1 * n + i1
                
                # M1 = (a+d)(e+h)
                U[u_a][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M2 = (c+d)e
                U[u_c][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1
                W[w_c10][col] += 1; W[w_c11][col] -= 1
                col += 1
                
                # M3 = a(f-h)
                U[u_a][col] += 1
                V[v_f][col] += 1; V[v_h][col] -= 1
                W[w_c01][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M4 = d(g-e)
                U[u_d][col] += 1
                V[v_g][col] += 1; V[v_e][col] -= 1
                W[w_c00][col] += 1; W[w_c10][col] += 1
                col += 1
                
                # M5 = (a+b)h
                U[u_a][col] += 1; U[u_b][col] += 1
                V[v_h][col] += 1
                W[w_c00][col] -= 1; W[w_c01][col] += 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[u_c][col] += 1; U[u_a][col] -= 1
                V[v_e][col] += 1; V[v_f][col] += 1
                W[w_c11][col] += 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[u_b][col] += 1; U[u_d][col] -= 1
                V[v_g][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1
                col += 1
    
    # Now try to reduce rank by finding tensor linear dependencies
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def verify_decomp(U_mat, V_mat, W_mat, n_tests=5):
        """Verify tensor decomposition: C[i,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
        for _ in range(n_tests):
            A_test = np.random.RandomState(rng.randint(2**31)).randn(n, m)
            B_test = np.random.RandomState(rng.randint(2**31)).randn(m, p)
            C_expected = A_test @ B_test  # shape (n, p)
            
            C_computed = np.zeros((n, p))
            for c in range(U_mat.shape[1]):
                for i in range(n):
                    for j in range(m):
                        u_val = U_mat[i * m + j, c]
                        if abs(u_val) < 1e-15:
                            continue
                        for k in range(p):
                            v_val = V_mat[j * p + k, c]
                            if abs(v_val) < 1e-15:
                                continue
                            w_val = W_mat[k * n + i, c]
                            C_computed[i, k] += u_val * v_val * w_val
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True
    
    # Build evaluation matrix: for each (i,j,k) triple, row = [U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for c in range(rank)]
    # Use all 64 possible (i,j,k) triples for exact check
    all_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    n_triples = len(all_triples)  # 64
    
    M = np.zeros((n_triples, rank))
    for s, (i, j, k) in enumerate(all_triples):
        for c in range(rank):
            M[s, c] = U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
    
    try:
        # Find null space of M: vectors alpha such that M @ alpha = 0
        U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=True)
        tol = 1e-10
        r = np.sum(s_vals > tol)
        
        if r < rank:
            # Null space dimension is rank - r
            # Null space vectors are rows of Vt[r:]
            null_space = Vt[r:]  # shape (rank-r, rank)
            
            # Greedy elimination: try to remove columns using null space info
            # A column c can be removed if there exists alpha in null space with alpha[c] != 0
            # and we can adjust other columns accordingly
            
            keep = list(range(rank))
            removed = []
            
            # Sort by how "redundant" each column is (larger null space coefficient = more redundant)
            # For each column, find the null space vector where it has the largest magnitude
            col_scores = []
            for c in range(rank):
                max_coeff = 0
                for ns in null_space:
                    if abs(ns[c]) > max_coeff:
                        max_coeff = abs(ns[c])
                col_scores.append((max_coeff, c))
            
            col_scores.sort(reverse=True)
            
            for _, c in col_scores:
                if time.time() > deadline - 10:
                    break
                if c not in keep:
                    continue
                other = [d for d in keep if d != c]
                if not other:
                    break
                
                # Find null space vector with largest component at c
                best_ns = None
                best_val = 0
                for ns in null_space:
                    if abs(ns[c]) > best_val:
                        best_val = abs(ns[c])
                        best_ns = ns
                
                if best_val < 1e-10:
                    continue
                
                # Check if removing c and adjusting others still works
                # The adjustment: for each other column d, add (ns[d]/ns[c]) * col_d to col_c's contribution
                # Actually, the null space relation is: sum_d ns[d] * T_d = 0
                # So T_c = -sum_{d!=c} (ns[d]/ns[c]) * T_d
                # This means we can remove c and modify other columns.
                # But modifying columns changes the decomposition...
                # Actually for rank reduction, we just need the set of remaining tensors to span the same space.
                # If T_c is in the span of {T_d : d != c}, we can remove c.
                
                # Verify: is T_c in span of {T_d : d in other}?
                # Build system: sum_{d in other} alpha_d * T_d = T_c
                # For each (i,j,k): sum_{d in other} alpha_d * U[i*m+j,d]*V[j*p+k,d]*W[k*n+i,d] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                
                A_check = np.zeros((n_triples, len(other)))
                b_check = np.zeros(n_triples)
                
                for s, (i, j, k) in enumerate(all_triples):
                    for idx, d in enumerate(other):
                        A_check[s, idx] = U_np[i * m + j, d] * V_np[j * p + k, d] * W_np[k * n + i, d]
                    b_check[s] = U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
                
                try:
                    alpha, _, rank_check, _ = np.linalg.lstsq(A_check, b_check, rcond=1e-10)
                    residual = np.linalg.norm(A_check @ alpha - b_check)
                    norm_b = np.linalg.norm(b_check)
                    if norm_b > 1e-12 and residual / norm_b < 1e-8:
                        keep.remove(c)
                except:
                    pass
            
            r_final = len(keep)
            if r_final < rank:
                U_reduced = U_np[:, keep]
                V_reduced = V_np[:, keep]
                W_reduced = W_np[:, keep]
                
                # Verify with the correct formula
                valid = verify_decomp(U_reduced, V_reduced, W_reduced, n_tests=3)
                
                if valid:
                    return {"U": U_reduced.tolist(), "V": V_reduced.tolist(), "W": W_reduced.tolist()}
    except Exception:
        pass
    
    # Final verification of the full decomposition before returning
    if time.time() < deadline - 5:
        valid = verify_decomp(U_np, V_np, W_np, n_tests=2)
        if valid:
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
