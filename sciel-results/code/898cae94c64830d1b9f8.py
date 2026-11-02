import numpy as np
import time
from fractions import Fraction
from collections import defaultdict

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    def remaining():
        return deadline - time.time()

    # Build the target tensor for verification
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def schoolbook():
        """Standard matrix multiplication - rank 120"""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return U, V, W

    def verify_decomp(U, V, W):
        """Check if U, V, W exactly decompose T"""
        if not U or not U[0]:
            return False
        rank = len(U[0])
        if rank == 0:
            return False
        # Quick dimension check
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if any(len(row) != rank for row in U):
            return False
        if any(len(row) != rank for row in V):
            return False
        if any(len(row) != rank for row in W):
            return False
        
        # Build as numpy arrays for efficient computation
        U_arr = np.array(U, dtype=np.float64)  # shape (n*m, rank)
        V_arr = np.array(V, dtype=np.float64)  # shape (m*p, rank)
        W_arr = np.array(W, dtype=np.float64)  # shape (p*n, rank)
        
        # T_recon[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
        # Efficient: for each c, add outer product
        T_recon = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for c in range(rank):
            T_recon += np.einsum('i,j,k->ijk', U_arr[:, c], V_arr[:, c], W_arr[:, c])
        
        return np.allclose(T_recon, T, atol=1e-8)

    # Strassen 2x2x2 terms
    def strassen_2x2_2x2():
        terms = []
        # M1 = (A00+A11)(B00+B11)
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        # M2 = (A10+A11)B00
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        # M3 = A00(B10-B00)
        terms.append(([1, 0, 0, 0], [-1, 0, 1, 0], [0, 1, 1, 0]))
        # M4 = A11(B11-B00)
        terms.append(([0, 0, 0, 1], [-1, 0, 0, 1], [-1, 0, 1, 0]))
        # M5 = (A00+A01)B11
        terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]))
        # M6 = (A10-A00)(B00+B01)
        terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, -1, 1]))
        # M7 = (A01-A11)(B10+B11)
        terms.append(([0, 1, 0, -1], [0, 1, 0, 1], [0, 1, -1, 0]))
        return terms

    # Winograd 2x2x2 terms
    def winograd_2x2_2x2():
        terms = []
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
        terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 1, 1, 0]))
        terms.append(([1, 1, 0, 0], [0, 0, 1, 0], [1, 1, 0, 0]))
        terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, -1, 1]))
        terms.append(([0, 1, 0, -1], [0, 1, 0, 1], [-1, 1, 1, 0]))
        return terms

    def embed_2x2x2_term(term, i_base, j_base, k_base):
        """Embed a 2x2x2 term into the full 4x5x6 space."""
        u_full = [0.0] * (n * m)
        v_full = [0.0] * (m * p)
        w_full = [0.0] * (p * n)
        u_vals, v_vals, w_vals = term
        
        for a in range(2):
            for b in range(2):
                i_idx = i_base + a
                j_idx = j_base + b
                if 0 <= i_idx < n and 0 <= j_idx < m:
                    u_full[i_idx * m + j_idx] = u_vals[a * 2 + b]
        
        for b in range(2):
            for c in range(2):
                j_idx = j_base + b
                k_idx = k_base + c
                if 0 <= j_idx < m and 0 <= k_idx < p:
                    v_full[j_idx * p + k_idx] = v_vals[b * 2 + c]
        
        for c in range(2):
            for a in range(2):
                k_idx = k_base + c
                i_idx = i_base + a
                if 0 <= k_idx < p and 0 <= i_idx < n:
                    w_full[k_idx * n + i_idx] = w_vals[c * 2 + a]
        
        return u_full, v_full, w_full

    def build_block_decomposition(use_winograd=True):
        """
        Build decomposition using block-structured approach.
        Split n=4 into [2,2], m=5 into [2,2,1], p=6 into [2,2,2].
        """
        terms_U = []
        terms_V = []
        terms_W = []
        base_terms = winograd_2x2x2_2x2() if use_winograd else strassen_2x2x2_2x2()
        
        covered = set()
        
        # i blocks: [0,1], [2,3]
        # j blocks: [0,1], [2,3], [4] (single)
        # k blocks: [0,1], [2,3], [4,5]
        
        for i_base in [0, 2]:
            for j_base in [0, 2]:  # Only 2x2 j blocks
                for k_base in [0, 2, 4]:
                    if i_base + 1 >= n or j_base + 1 >= m or k_base + 1 >= p:
                        continue
                    
                    for term in base_terms:
                        u_full, v_full, w_full = embed_2x2x2_term(term, i_base, j_base, k_base)
                        terms_U.append(u_full)
                        terms_V.append(v_full)
                        terms_W.append(w_full)
                    
                    for a in range(2):
                        for b in range(2):
                            for c in range(2):
                                covered.add((i_base + a, j_base + b, k_base + c))
        
        # Add schoolbook for uncovered
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in covered:
                        continue
                    u_full = [0.0] * (n * m)
                    v_full = [0.0] * (m * p)
                    w_full = [0.0] * (p * n)
                    u_full[i * m + j] = 1.0
                    v_full[j * p + k] = 1.0
                    w_full[k * n + i] = 1.0
                    terms_U.append(u_full)
                    terms_V.append(v_full)
                    terms_W.append(w_full)
        
        rank = len(terms_U)
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        for c, (u, v, w) in enumerate(zip(terms_U, terms_V, terms_W)):
            for i in range(n * m):
                U[i][c] = u[i]
            for j in range(m * p):
                V[j][c] = v[j]
            for k in range(p * n):
                W[k][c] = w[k]
        
        return U, V, W, rank

    # Try to reduce rank via linear dependency detection
    def try_reduce_rank(U, V, W, max_time=20.0):
        """
        Attempt to reduce rank by finding linearly dependent columns in the 
        combined tensor representation.
        """
        start = time.time()
        rank = len(U[0]) if U and U[0] else 0
        if rank <= 1:
            return U, V, W
        
        # Build the 3-mode tensor as a set of rank-1 terms
        # Try to find a subset of terms that still spans the same tensor
        # Use greedy elimination: for each term, check if removing it still gives valid decomp
        
        U_arr = np.array(U, dtype=np.float64)  # (n*m, rank)
        V_arr = np.array(V, dtype=np.float64)  # (m*p, rank)
        W_arr = np.array(W, dtype=np.float64)  # (p*n, rank)
        
        # The tensor T = sum_c u_c ⊗ v_c ⊗ w_c
        # If we have a linear dependency among the "flattened" terms, we can reduce
        
        # Approach: treat each rank-1 term as a vector in the tensor space
        # The tensor space has dimension n*m*p = 120
        # We have 'rank' vectors in this space; find the maximal independent subset
        
        # Flatten each term
        def flatten_term(c):
            return U_arr[:, c].reshape(-1, 1, 1) * V_arr[:, c].reshape(1, -1, 1) * W_arr[:, c].reshape(1, 1, -1)
        
        # More efficient: build the matrix where each column is a flattened term
        # Shape: (n*m*p, rank)
        M = np.zeros((n * m * p, rank))
        for c in range(rank):
            # Term c contributes to T[i,j,k] = U[i,c]*V[j,c]*W[k,c]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = (i * m + j) * (m * p) + (j * p + k) * p + (k * n + i)
                        # Actually, let's use a cleaner indexing
                        pass
        
        # Let me use a different approach: build the matrix directly
        M = np.zeros((n * m * p, rank))
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row_idx = (i * m + j) * (m * p) + (j * p + k) * p + k
                        # This indexing is getting messy; let me use numpy broadcasting
                        pass
        
        # Simpler: use the fact that T[i*m+j, j*p+k, k*n+i] = 1
        # The tensor has only 120 non-zero entries (one per (i,j,k) triple)
        # So the tensor space effectively has dimension 120
        
        # Build matrix where column c is the vector of T_recon values from term c
        # at the 120 positions
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # T index for (i,j,k)
                        t_idx = (i * m + j, j * p + k, k * n + i)
                        # The value contributed by term c at this position
                        val = U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                        # Map to linear index
                        lin_idx = (i * m + j) * (m * p) + (j * p + k) * p + (k * n + i)
                        M[lin_idx, c] = val
        
        # Now find the rank of M and a basis
        if time.time() - start > max_time:
            return U, V, W
        
        # Use SVD to find the rank
        try:
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
            tol = 1e-10
            effective_rank = np.sum(s > tol)
            
            if effective_rank < rank:
                # Find which columns are in the null space
                # Use the right singular vectors
                # The columns of Vt.T correspond to the original columns
                # Keep only the first effective_rank columns that form a basis
                
                # Actually, we need to be careful: the SVD gives us an orthonormal basis
                # for the column space, but we need to select which original columns to keep
                # to still span the same space.
                
                # Use QR with column pivoting
                Q, R, piv = np.linalg.qr(M, mode='complete') if hasattr(np.linalg, 'qr') else (None, None, None)
                # numpy doesn't have pivoted QR directly, so let's do it manually
                
                # Greedy approach: iterate through columns, keep if independent
                kept = []
                A = np.zeros((n * m * p, 0))
                for c in range(rank):
                    col = M[:, c].reshape(-1, 1)
                    if A.shape[1] == 0:
                        kept.append(c)
                        A = col
                    else:
                        # Check if col is in span of A
                        # Compute residual
                        try:
                            coeff, residuals, rank_sub, s_vals = np.linalg.lstsq(A, col, rcond=None)
                            residual = col - A @ coeff
                            if np.linalg.norm(residual) > 1e-8:
                                kept.append(c)
                                A = np.hstack([A, col])
                        except:
                            kept.append(c)
                            A = np.hstack([A, col])
                    
                    if time.time() - start > max_time:
                        break
                
                if len(kept) < rank:
                    new_rank = len(kept)
                    U_new = [[U[i][c] for c in kept] for i in range(n * m)]
                    V_new = [[V[i][c] for c in kept] for i in range(m * p)]
                    W_new = [[W[i][c] for c in kept] for i in range(p * n)]
                    
                    # Verify
                    if verify_decomp(U_new, V_new, W_new):
                        return U_new, V_new, W_new
            return U, V, W
        except:
            return U, V, W

    # Start with schoolbook
    U_school, V_school, W_school = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school

    # Try block decompositions
    for use_winograd in [True, False]:
        if remaining() < 15:
            break
        U_b, V_b, W_b, rank_b = build_block_decomposition(use_winograd)
        if rank_b < best_rank:
            if verify_decomp(U_b, V_b, W_b):
                best_rank = rank_b
                best_U, best_V, best_W = U_b, V_b, W_b

    # Try to further reduce the best found so far
    if remaining() > 30 and best_rank < 120:
        U_r, V_r, W_r = try_reduce_rank(best_U, best_V, best_W, max_time=25.0)
        new_rank = len(U_r[0]) if U_r and U_r[0] else 0
        if new_rank < best_rank:
            best_rank = new_rank
            best_U, best_V, best_W = U_r, V_r, W_r

    # Final verification
    if not verify_decomp(best_U, best_V, best_W):
        # Fallback to schoolbook
        return {"U": U_school, "V": V_school, "W": W_school}

    return {"U": best_U, "V": best_V, "W": best_W}
