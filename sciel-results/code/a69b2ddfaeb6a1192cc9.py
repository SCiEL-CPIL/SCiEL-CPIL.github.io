"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # Tensor dimensions
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    total_triples = n * m * p  # 60
    
    def get_strassen_block(i0, j0, k0):
        """
        Returns lists of (u_idx, u_val), (v_idx, v_val), (w_idx, w_val) for 7 terms.
        i0, j0, k0 are global offsets.
        """
        terms = []
        
        # Term 1
        u1 = [((i0+0)*m + (j0+0), 1), ((i0+1)*m + (j0+1), 1)]
        v1 = [((j0+0)*p + (k0+0), 1), ((j0+1)*p + (k0+1), 1)]
        w1 = [((k0+0)*n + (i0+0), 1), ((k0+1)*n + (i0+1), 1)]
        terms.append((u1, v1, w1))
        
        # Term 2
        u2 = [((i0+1)*m + (j0+0), 1), ((i0+1)*m + (j0+1), 1)]
        v2 = [((j0+0)*p + (k0+0), 1)]
        w2 = [((k0+0)*n + (i0+1), 1), ((k0+1)*n + (i0+1), -1)]
        terms.append((u2, v2, w2))
        
        # Term 3
        u3 = [((i0+0)*m + (j0+0), 1)]
        v3 = [((j0+0)*p + (k0+1), 1), ((j0+1)*p + (k0+1), -1)]
        w3 = [((k0+1)*n + (i0+0), 1), ((k0+1)*n + (i0+1), 1)]
        terms.append((u3, v3, w3))
        
        # Term 4
        u4 = [((i0+1)*m + (j0+1), 1)]
        v4 = [((j0+1)*p + (k0+0), 1), ((j0+0)*p + (k0+0), -1)]
        w4 = [((k0+0)*n + (i0+0), 1), ((k0+0)*n + (i0+1), 1)]
        terms.append((u4, v4, w4))
        
        # Term 5
        u5 = [((i0+0)*m + (j0+0), 1), ((i0+0)*m + (j0+1), 1)]
        v5 = [((j0+1)*p + (k0+1), 1)]
        w5 = [((k0+0)*n + (i0+0), -1), ((k0+1)*n + (i0+0), 1)]
        terms.append((u5, v5, w5))
        
        # Term 6
        u6 = [((i0+1)*m + (j0+0), 1), ((i0+0)*m + (j0+0), -1)]
        v6 = [((j0+0)*p + (k0+0), 1), ((j0+0)*p + (k0+1), 1)]
        w6 = [((k0+1)*n + (i0+1), 1)]
        terms.append((u6, v6, w6))
        
        # Term 7
        u7 = [((i0+0)*m + (j0+1), 1), ((i0+1)*m + (j0+1), -1)]
        v7 = [((j0+1)*p + (k0+0), 1), ((j0+1)*p + (k0+1), 1)]
        w7 = [((k0+0)*n + (i0+0), 1)]
        terms.append((u7, v7, w7))
        
        return terms

    def build_decomposition():
        """Build the base 56-rank decomposition (4 Strassen blocks + schoolbook)."""
        U_cols = []
        V_cols = []
        W_cols = []
        
        # 4 Strassen blocks covering i∈{0,1}, j∈{0,1,2,3}, k∈{0,1,2,3}
        for i0 in [0]:
            for j0 in [0, 2]:
                for k0 in [0, 2]:
                    terms = get_strassen_block(i0, j0, k0)
                    for u, v, w in terms:
                        U_col = np.zeros(dim_u)
                        for idx, val in u:
                            U_col[idx] = val
                        V_col = np.zeros(dim_v)
                        for idx, val in v:
                            V_col[idx] = val
                        W_col = np.zeros(dim_w)
                        for idx, val in w:
                            W_col[idx] = val
                        U_cols.append(U_col)
                        V_cols.append(V_col)
                        W_cols.append(W_col)
        
        # Schoolbook for the rest
        covered = set()
        for i in [0, 1]:
            for j in [0, 1, 2, 3]:
                for k in [0, 1, 2, 3]:
                    covered.add((i, j, k))
                    
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in covered:
                        continue
                    U_col = np.zeros(dim_u)
                    U_col[i*m + j] = 1
                    V_col = np.zeros(dim_v)
                    V_col[j*p + k] = 1
                    W_col = np.zeros(dim_w)
                    W_col[k*n + i] = 1
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)
        
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        return U, V, W

    def verify(U, V, W):
        """Verify the decomposition is exact."""
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i*m + j
                    idx_v = j*p + k
                    idx_w = k*n + i
                    s = float(np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :]))
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def try_reduce_rank(U, V, W, max_attempts=100):
        """
        Try to reduce rank by finding linear dependencies among the rank-1 terms.
        For each term r, check if it can be expressed as a linear combination of others.
        """
        rank = U.shape[1]
        best_rank = rank
        best_U, best_V, best_W = U, V, W
        
        # Build the coefficient matrix: each column r gives a vector in R^60
        # where the (i,j,k)-th entry is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        def term_vector(U, V, W, r):
            vec = np.zeros(total_triples)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        vec[(i*m+j)*total_triples//dim_u + (j*p+k)*total_triples//dim_v + (k*n+i)*total_triples//dim_w] = 0  # placeholder
            # Actually, let's index differently
            vec = np.zeros(total_triples)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        vec[idx] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        idx += 1
            return vec
        
        # Build full coefficient matrix A where A[:, r] = term_vector(U, V, W, r)
        A = np.zeros((total_triples, rank))
        for r in range(rank):
            A[:, r] = term_vector(U, V, W, r)
        
        # For each term r, check if removing it still allows exact decomposition
        # This is equivalent to checking if A[:, r] is in the span of the other columns
        # Use SVD to find null space
        try:
            U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
            # Null space is the rows of Vt corresponding to zero singular values
            tol = 1e-10
            null_space = Vt[s < tol, :]
            null_dim = null_space.shape[0]
            
            if null_dim > 0:
                # For each null space vector, find the terms with largest absolute coefficient
                # and try removing them
                for ns_vec in null_space:
                    # Find the term with the largest absolute coefficient
                    abs_coeffs = np.abs(ns_vec)
                    max_idx = np.argmax(abs_coeffs)
                    
                    # Try removing this term
                    keep = [i for i in range(rank) if i != max_idx]
                    if len(keep) < best_rank:
                        U_new = U[:, keep]
                        V_new = V[:, keep]
                        W_new = W[:, keep]
                        err = verify(U_new, V_new, W_new)
                        if err < 1e-9:
                            best_rank = len(keep)
                            best_U, best_V, best_W = U_new, V_new, W_new
                            # Update A for next iteration
                            A = np.zeros((total_triples, best_rank))
                            for r in range(best_rank):
                                A[:, r] = term_vector(best_U, best_V, best_W, r)
                            rank = best_rank
                            break
        except np.linalg.LinAlgError:
            pass
        
        return best_U, best_V, best_W, best_rank

    # Build base decomposition
    U, V, W = build_decomposition()
    
    # Verify base decomposition
    err = verify(U, V, W)
    if err > 1e-9:
        # Fallback to schoolbook
        U_sb = np.zeros((dim_u, total_triples))
        V_sb = np.zeros((dim_v, total_triples))
        W_sb = np.zeros((dim_w, total_triples))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1
                    V_sb[j*p + k, col] = 1
                    W_sb[k*n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to reduce rank
    if time.time() < deadline - 5:
        U, V, W, rank = try_reduce_rank(U, V, W)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
