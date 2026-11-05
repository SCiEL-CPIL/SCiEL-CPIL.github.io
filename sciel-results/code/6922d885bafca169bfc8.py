import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    n, m, p = DIMENSIONS  # 4, 4, 7

    def build_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2_block(i_off, j_off, k_off, a_scale=1, b_scale=1, c_scale=1):
        """
        Adds 7 Strassen terms for a 2x2 * 2x2 block multiplication.
        A block at rows [i_off, i_off+1], cols [j_off, j_off+1]
        B block at rows [j_off, j_off+1], cols [k_off, k_off+1]
        C block at rows [i_off, i_off+1], cols [k_off, k_off+1]
        """
        terms = []
        
        # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        terms.append((u, v, w))

        # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = c_scale
        w[(k_off+1) * n + (i_off+1)] = -c_scale
        terms.append((u, v, w))

        # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = b_scale
        v[(j_off+1) * p + (k_off+1)] = -b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        terms.append((u, v, w))

        # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[j_off * p + k_off] = -b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[k_off * n + (i_off+1)] = c_scale
        terms.append((u, v, w))

        # M5 = (A11+A12)*B22 -> C12 += M5, C11 -= M5
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[i_off * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[k_off * n + i_off] = -c_scale
        terms.append((u, v, w))

        # M6 = (A21-A11)*(B11+B12) -> C22 += M6
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[i_off * m + j_off] = -a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[j_off * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = c_scale
        terms.append((u, v, w))

        # M7 = (A12-A22)*(B21+B22) -> C11 += M7
        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = a_scale
        u[(i_off+1) * m + (j_off+1)] = -a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        terms.append((u, v, w))

        return terms

    def build_4x4x7_optimized():
        all_terms = []
        
        # Part 1: 4x4 * 4x4 (columns 0-3 of B)
        # Use recursive Strassen: 7 * 7 = 49 terms
        
        outer_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]

        inner_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]

        def get_block_indices(block_idx, local_idx, dim_a, offset_k=0):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            row = br * 2 + lr
            col = bc * 2 + lc + offset_k
            if dim_a == 'A':
                return row * m + col
            elif dim_a == 'B':
                return row * p + col
            elif dim_a == 'C':
                return col * n + row

        for outer_a, outer_b, outer_c in outer_terms:
            for inner_x, inner_y, inner_z in inner_terms:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                for x_local, x_coeff in inner_x:
                    for a_block, a_coeff in outer_a:
                        idx = get_block_indices(a_block, x_local, 'A')
                        u[idx] += a_coeff * x_coeff
                for y_local, y_coeff in inner_y:
                    for b_block, b_coeff in outer_b:
                        idx = get_block_indices(b_block, y_local, 'B')
                        v[idx] += b_coeff * y_coeff
                for z_local, z_coeff in inner_z:
                    for c_block, c_coeff in outer_c:
                        idx = get_block_indices(c_block, z_local, 'C')
                        w[idx] += c_coeff * z_coeff
                all_terms.append((u, v, w))

        # Part 2: 4x4 * 4x2 (columns 4-5 of B)
        all_terms.extend(strassen_2x2_block(0, 0, 4))
        all_terms.extend(strassen_2x2_block(0, 2, 4))
        all_terms.extend(strassen_2x2_block(2, 0, 4))
        all_terms.extend(strassen_2x2_block(2, 2, 4))
        
        # Part 3: 4x4 * 4x1 (column 6 of B)
        for i in range(n):
            for j in range(m):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + 6] = 1
                w = [0] * (p * n)
                w[6 * n + i] = 1
                all_terms.append((u, v, w))

        rank = len(all_terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for c, (u, v, w) in enumerate(all_terms):
            for r in range(n * m):
                if u[r] != 0:
                    U[r][c] = u[r]
            for r in range(m * p):
                if v[r] != 0:
                    V[r][c] = v[r]
            for r in range(p * n):
                if w[r] != 0:
                    W[r][c] = w[r]
        
        return {"U": U, "V": V, "W": W}

    def rank_1_peeling():
        """
        Find a lower-rank decomposition using rank-1 peeling.
        Start from the Strassen-based decomposition and try to reduce rank.
        """
        # Get initial decomposition
        initial = build_4x4x7_optimized()
        U_init = np.array(initial["U"], dtype=float)
        V_init = np.array(initial["V"], dtype=float)
        W_init = np.array(initial["W"], dtype=float)
        
        # Build the target tensor T[i,j,k] = 1
        T = np.ones((n, m, p), dtype=float)
        
        # Current residual
        residual = T.copy()
        
        # Current decomposition terms as list of (u, v, w) tuples
        terms = []
        for c in range(U_init.shape[1]):
            u = U_init[:, c]
            v = V_init[:, c]
            w = W_init[:, c]
            if np.any(u != 0) and np.any(v != 0) and np.any(w != 0):
                terms.append((u, v, w))
        
        # Rank-1 peeling: iteratively find a rank-1 term that best matches the residual
        # A rank-1 term contributes: for all i,j,k: u[i*m+j] * v[j*p+k] * w[k*n+i]
        # We want to find (u,v,w) that maximizes <residual, u⊗v⊗w> / ||u⊗v⊗w||
        
        # Reshape residual to 3D: R[i,j,k]
        # The contribution of rank-1 term (u,v,w) is:
        # C[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
        
        # This is a 3-way tensor. To find the best rank-1 approximation,
        # we can use power iteration on the 3-way tensor.
        
        max_iterations = 50
        tol = 1e-10
        
        new_terms = []
        
        for peel_iter in range(len(terms)):
            if time.time() - start_time > DEADLINE:
                break
            
            # Find best rank-1 term for current residual
            # Use power iteration
            u = np.random.randn(n * m)
            v = np.random.randn(m * p)
            w = np.random.randn(p * n)
            u = u / np.linalg.norm(u)
            v = v / np.linalg.norm(v)
            w = w / np.linalg.norm(w)
            
            converged = False
            for it in range(max_iterations):
                if time.time() - start_time > DEADLINE:
                    break
                
                # Update u: u[i*m+j] = sum_{j,k} residual[i,j,k] * v[j*p+k] * w[k*n+i]
                # Reshape: R[i,j,k], V[j,k], W[k,i]
                # For each i,m+j: u_new[i*m+j] = sum_j sum_k R[i,j,k] * v[j*p+k] * w[k*n+i]
                
                # Efficient computation using einsum
                # u_new[i,j] = sum_{j',k} R[i,j',k] * v[j',k] * w[k,i]  -- wait, this isn't right
                # u[i*m+j] where i is row of A, j is col of A
                # v[j*p+k] where j is row of B (= col of A), k is col of B
                # w[k*n+i] where k is col of B, i is row of A (= row of C)
                
                # So: u_new[i,j] = sum_{j',k} R[i,j,k] * v[j,k] * w[k,i]
                # Note: the j index in u is the same j as in v's row and R's col
                
                # Let me be more careful:
                # u is indexed by (i,j) where i in [0,n), j in [0,m)
                # v is indexed by (j,k) where j in [0,m), k in [0,p)
                # w is indexed by (k,i) where k in [0,p), i in [0,n)
                # R is indexed by (i,j,k)
                
                # u_new[i,j] = sum_{j',k} R[i,j,k] * v[j',k] * w[k,i]
                # Wait, that's not right either. Let me re-derive.
                
                # The tensor product: T[i,j,k] = u[i,j] * v[j,k] * w[k,i]
                # So for fixed (i,j,k): contribution = u[i,j] * v[j,k] * w[k,i]
                
                # To update u: u_new[i,j] = sum_{j',k} R[i,j',k] * v[j',k] * w[k,i]
                # No wait: u[i,j] appears in T[i,j,k] for all k.
                # So: u_new[i,j] = sum_k R[i,j,k] * v[j,k] * w[k,i]
                
                # To update v: v_new[j,k] = sum_i R[i,j,k] * u[i,j] * w[k,i]
                
                # To update w: w_new[k,i] = sum_j R[i,j,k] * u[i,j] * v[j,k]
                
                R_3d = residual.reshape(n, m, p)
                v_2d = v.reshape(m, p)
                w_2d = w.reshape(p, n)
                u_2d = u.reshape(n, m)
                
                # u_new[i,j] = sum_k R[i,j,k] * v[j,k] * w[k,i]
                u_new = np.einsum('ijk,jk,ki->ij', R_3d, v_2d, w_2d)
                
                # v_new[j,k] = sum_i R[i,j,k] * u[i,j] * w[k,i]
                v_new = np.einsum('ijk,ij,ki->jk', R_3d, u_2d, w_2d)
                
                # w_new[k,i] = sum_j R[i,j,k] * u[i,j] * v[j,k]
                w_new = np.einsum('ijk,ij,jk->ki', R_3d, u_2d, v_2d)
                
                # Normalize
                u_new_norm = np.linalg.norm(u_new)
                v_new_norm = np.linalg.norm(v_new)
                w_new_norm = np.linalg.norm(w_new)
                
                if u_new_norm < tol or v_new_norm < tol or w_new_norm < tol:
                    break
                
                u_new = u_new / u_new_norm
                v_new = v_new / v_new_norm
                w_new = w_new / w_new_norm
                
                # Check convergence
                diff_u = np.linalg.norm(u_new - u)
                diff_v = np.linalg.norm(v_new - v)
                diff_w = np.linalg.norm(w_new - w)
                
                u, v, w = u_new, v_new, w_new
                u_2d = u.reshape(n, m)
                v_2d = v.reshape(m, p)
                w_2d = w.reshape(p, n)
                
                if max(diff_u, diff_v, diff_w) < 1e-8:
                    converged = True
                    break
            
            if not converged and u_new_norm < tol:
                break
            
            # Compute the best scalar alpha for this rank-1 term
            # We want: alpha * u[i,j] * v[j,k] * w[k,i] to best match residual
            # alpha = <residual, u⊗v⊗w> / ||u⊗v⊗w||^2
            
            # <residual, u⊗v⊗w> = sum_{i,j,k} R[i,j,k] * u[i,j] * v[j,k] * w[k,i]
            dot = np.einsum('ijk,ij,jk,ki->', R_3d, u_2d, v_2d, w_2d)
            
            # ||u⊗v⊗w||^2 = sum_{i,j,k} (u[i,j]*v[j,k]*w[k,i])^2
            norm_sq = np.einsum('ijk,ijk->', u_2d[:, :, np.newaxis] * v_2d[np.newaxis, :, :] * w_2d[np.newaxis, :, :], 
                                u_2d[:, :, np.newaxis] * v_2d[np.newaxis, :, :] * w_2d[np.newaxis, :, :])
            
            if norm_sq < tol:
                break
            
            alpha = dot / norm_sq
            
            if abs(alpha) < tol:
                break
            
            # Subtract this rank-1 term from residual
            # residual[i,j,k] -= alpha * u[i,j] * v[j,k] * w[k,i]
            rank1_tensor = alpha * u_2d[:, :, np.newaxis] * v_2d[np.newaxis, :, :] * w_2d[np.newaxis, :, :]
            residual -= rank1_tensor
            
            # Add to new terms
            new_terms.append((alpha * u, alpha * v, w))
            
            # Check if residual is approximately zero
            if np.max(np.abs(residual)) < 1e-9:
                break
        
        # If peeling succeeded and we have fewer terms, use it
        if len(new_terms) < len(terms) and np.max(np.abs(residual)) < 1e-9:
            rank = len(new_terms)
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            for c, (u, v, w) in enumerate(new_terms):
                for r in range(n * m):
                    if abs(u[r]) > 1e-15:
                        U[r][c] = u[r]
                for r in range(m * p):
                    if abs(v[r]) > 1e-15:
                        V[r][c] = v[r]
                for r in range(p * n):
                    if abs(w[r]) > 1e-15:
                        W[r][c] = w[r]
            
            return {"U": U, "V": V, "W": W}
        
        return initial

    if time.time() - start_time > DEADLINE:
        return build_schoolbook()

    try:
        result = build_4x4x7_optimized()
        
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        rank = U.shape[1]
        
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        tensor_decomp = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        
        target = np.ones((n, m, p), dtype=float)
        if not np.allclose(tensor_decomp, target, atol=1e-9):
            return build_schoolbook()
        
        # Try rank-1 peeling to see if we can do better
        if time.time() - start_time < DEADLINE - 30:
            try:
                peeled = rank_1_peeling()
                U_p = np.array(peeled["U"], dtype=float)
                V_p = np.array(peeled["V"], dtype=float)
                W_p = np.array(peeled["W"], dtype=float)
                rank_p = U_p.shape[1]
                
                if rank_p < rank:
                    U_pr = U_p.reshape(n, m, rank_p)
                    V_pr = V_p.reshape(m, p, rank_p)
                    W_pr = W_p.reshape(p, n, rank_p)
                    
                    tensor_decomp_p = np.einsum('ijc,jkc,kic->ijk', U_pr, V_pr, W_pr)
                    
                    if np.allclose(tensor_decomp_p, target, atol=1e-9):
                        return peeled
            except Exception:
                pass
        
        return result
    except Exception:
        return build_schoolbook()
