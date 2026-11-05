"""Matrix multiplication 2,5,6 rank-minimal tensor decomposition."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Build the matrix multiplication tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    # Strategy: Use the rank-54 Strassen-based decomposition and then
    # try to reduce rank by finding linear dependencies among the rank-1 terms
    # using SVD/null-space analysis.
    
    def build_strassen_decomposition(j_pairs, j_naive, k_pairs):
        """Build decomposition given j-pair grouping and k-pair grouping."""
        rank = 7 * len(j_pairs) * len(k_pairs) + 4 * len(j_naive) * len(k_pairs)
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        col = 0
        
        def add_strassen_block(i_base, j_base, k_base):
            nonlocal col
            
            # M0: (A00+A11)(B00+B11)
            U[i_base * m + j_base, col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[j_base * p + k_base, col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            # M1: (A10+A11)B00
            U[(i_base + 1) * m + j_base, col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] -= 1
            col += 1
            
            # M2: A00(B01-B11)
            U[i_base * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] -= 1
            W[(k_base + 1) * n + i_base, col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            # M3: A11(B10-B00)
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[(j_base + 1) * p + k_base, col] += 1
            V[j_base * p + k_base, col] -= 1
            W[k_base * n + i_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            col += 1
            
            # M4: (A00+A01)B11
            U[i_base * m + j_base, col] += 1
            U[i_base * m + (j_base + 1), col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] -= 1
            W[(k_base + 1) * n + i_base, col] += 1
            col += 1
            
            # M5: (A10-A00)(B00+B01)
            U[(i_base + 1) * m + j_base, col] += 1
            U[i_base * m + j_base, col] -= 1
            V[j_base * p + k_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            # M6: (A01-A11)(B10+B11)
            U[i_base * m + (j_base + 1), col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] -= 1
            V[(j_base + 1) * p + k_base, col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] += 1
            col += 1

        def add_naive_2112_block(i_base, j_base, k_base):
            nonlocal col
            
            # 1: A00 B00 -> C00
            U[i_base * m + j_base, col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + i_base, col] += 1
            col += 1
            
            # 2: A00 B01 -> C01
            U[i_base * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + i_base, col] += 1
            col += 1
            
            # 3: A10 B00 -> C10
            U[(i_base + 1) * m + j_base, col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            col += 1
            
            # 4: A10 B01 -> C11
            U[(i_base + 1) * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1

        # Apply Strassen blocks
        for j_pair in j_pairs:
            for k_pair in k_pairs:
                add_strassen_block(0, j_pair[0], k_pair[0])
        
        # Apply naive blocks
        for j_naive_idx in j_naive:
            for k_pair in k_pairs:
                add_naive_2112_block(0, j_naive_idx, k_pair[0])
        
        return U, V, W, col

    def try_reduce_rank(U, V, W, original_rank, tolerance=1e-10):
        """Try to reduce rank by finding linear dependencies using SVD."""
        if time.time() > deadline:
            return U, V, W, original_rank
        
        # Build the tensor from the decomposition
        T_dec = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check if it matches the target tensor
        if not np.allclose(T_dec, T, atol=tolerance, rtol=tolerance):
            return U, V, W, original_rank
        
        # Try to find the minimal rank by looking at the matrix flattening
        # Flatten mode 3: T as (p*n) x (n*m*m*p) matrix
        # T[k*n+i, i*m+j, j*p+k] -> flatten (i*m+j, j*p+k) as columns
        # Actually, let's use a different approach: find the rank of the tensor
        # by checking if we can express it with fewer terms.
        
        # Use the fact that if columns of U are linearly dependent, we can
        # potentially reduce rank. But the constraint is that the tensor must
        # be preserved.
        
        # A better approach: use randomized SVD on the mode-3 flattening
        # to estimate the tensor rank, then try to construct a lower-rank
        # decomposition.
        
        # For now, try a simple approach: check if any pair of columns in U, V, W
        # are linearly dependent in a way that allows merging.
        
        # Actually, let's try a different strategy: use the null space of the
        # Gram matrix of the rank-1 terms to find linear dependencies.
        
        # The rank-1 terms are u_r ⊗ v_r ⊗ w_r for r=1..R.
        # If there's a linear dependency ∑ c_r (u_r ⊗ v_r ⊗ w_r) = 0,
        # then we can reduce the rank.
        
        # This is equivalent to checking if the R x R matrix M where
        # M[a,b] = <u_a ⊗ v_a ⊗ w_a, u_b ⊗ v_b ⊗ w_b> has a non-trivial null space.
        # But this is expensive. Instead, let's use a simpler heuristic.
        
        # Try to find if we can eliminate one term by expressing it as a
        # linear combination of others.
        
        # For each term r, try to find coefficients c_s (s≠r) such that
        # u_r ⊗ v_r ⊗ w_r = ∑_{s≠r} c_s (u_s ⊗ v_s ⊗ w_s)
        
        # This is a system of n*m*p equations in R-1 unknowns.
        # Let's set it up as a least squares problem.
        
        R = original_rank
        if R <= 1:
            return U, V, W, R
        
        # Flatten the tensor
        T_flat = T.flatten()
        n_params = n * m * p
        
        # Build the matrix where column r is the flattened u_r ⊗ v_r ⊗ w_r
        # u_r is (n*m,), v_r is (m*p,), w_r is (p*n,)
        # The tensor u_r ⊗ v_r ⊗ w_r has entries U[i,j] * V[j,k] * W[k,i]
        # where i in [0,n), j in [0,m), k in [0,p)
        # In flattened form, the index is (i*m+j) * (m*p) * (p*n) + (j*p+k) * (p*n) + (k*n+i)
        # But our T is indexed as T[i*m+j, j*p+k, k*n+i]
        
        # Let's build the matrix properly
        M = np.zeros((n_params, R), dtype=np.float64)
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = (i * m + j) * (m * p) * (p * n) + (j * p + k) * (p * n) + (k * n + i)
                        # But T is stored as (n*m, m*p, p*n) so flattened index is:
                        # (i*m+j) * (m*p) * (p*n) + (j*p+k) * (p*n) + (k*n+i)
                        M[idx, r] = U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
        
        # Now check if columns are linearly dependent
        # Use SVD to find the rank of M
        try:
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
            # The rank is the number of singular values above tolerance
            tol = max(M.shape) * np.finfo(float).eps * s[0] if len(s) > 0 else 1e-10
            rank_M = np.sum(s > tol)
        except:
            rank_M = R
        
        if rank_M < R:
            # There's linear dependence. Try to find a lower-rank decomposition.
            # Use the SVD to get a rank-rank_M approximation, but we need exactness.
            # Since the entries are integers/half-integers, let's try to find
            # an exact lower-rank decomposition by solving a linear system.
            
            # For now, just return the original decomposition if we can't
            # easily find an exact lower-rank one.
            pass
        
        return U, V, W, R

    # Try multiple block partition strategies
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Original - j_pairs=(0,1),(2,3), j_naive=4, k_pairs=(0,1),(2,3),(4,5)
    if time.time() < deadline:
        U, V, W, rank = build_strassen_decomposition([(0,1),(2,3)], [4], [(0,1),(2,3),(4,5)])
        U, V, W, rank = try_reduce_rank(U, V, W, rank)
        if rank < best_rank:
            # Verify
            T_dec = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_dec, T, atol=1e-10, rtol=1e-10):
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy 2: j_pairs=(0,1),(3,4), j_naive=2, k_pairs=(0,1),(2,3),(4,5)
    if time.time() < deadline:
        U, V, W, rank = build_strassen_decomposition([(0,1),(3,4)], [2], [(0,1),(2,3),(4,5)])
        U, V, W, rank = try_reduce_rank(U, V, W, rank)
        if rank < best_rank:
            T_dec = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_dec, T, atol=1e-10, rtol=1e-10):
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy 3: j_pairs=(0,2),(1,3), j_naive=4, k_pairs=(0,1),(2,3),(4,5)
    if time.time() < deadline:
        U, V, W, rank = build_strassen_decomposition([(0,2),(1,3)], [4], [(0,1),(2,3),(4,5)])
        U, V, W, rank = try_reduce_rank(U, V, W, rank)
        if rank < best_rank:
            T_dec = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_dec, T, atol=1e-10, rtol=1e-10):
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy 4: Try different k_pairings - (0,2),(1,3),(4,5)
    if time.time() < deadline:
        U, V, W, rank = build_strassen_decomposition([(0,1),(2,3)], [4], [(0,2),(1,3),(4,5)])
        U, V, W, rank = try_reduce_rank(U, V, W, rank)
        if rank < best_rank:
            T_dec = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_dec, T, atol=1e-10, rtol=1e-10):
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy 5: Try k_pairs=(0,3),(1,4),(2,5)
    if time.time() < deadline:
        U, V, W, rank = build_strassen_decomposition([(0,1),(2,3)], [4], [(0,3),(1,4),(2,5)])
        U, V, W, rank = try_reduce_rank(U, V, W, rank)
        if rank < best_rank:
            T_dec = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_dec, T, atol=1e-10, rtol=1e-10):
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy 6: Try all possible j_pairings with 2 pairs and 1 naive
    if time.time() < deadline:
        # All ways to choose 2 pairs from 5 j's, leaving 1 naive
        # Pairs must be disjoint
        all_j = list(range(5))
        for j1, j2 in itertools.combinations(all_j, 2):
            for j3, j4 in itertools.combinations([x for x in all_j if x not in (j1, j2)], 2):
                j_naive = [x for x in all_j if x not in (j1, j2, j3, j4)][0]
                j_pairs = [(j1, j2), (j3, j4)]
                for k1, k2 in itertools.combinations(range(6), 2):
                    for k3, k4 in itertools.combinations([x for x in range(6) if x not in (k1, k2)], 2):
                        k5 = [x for x in range(6) if x not in (k1, k2, k3, k4)][0]
                        k_pairs = [(k1, k2), (k3, k4), (k5, k5)]  # Last one is a single
                        # But this doesn't match the block structure. Let's skip this.
                        break
                    break
                break
            break
    
    # Strategy 7: Try k_pairs with different pairings
    if time.time() < deadline:
        k_pairings = [
            [(0,1),(2,3),(4,5)],
            [(0,1),(2,4),(3,5)],
            [(0,1),(2,5),(3,4)],
            [(0,2),(1,3),(4,5)],
            [(0,2),(1,4),(3,5)],
            [(0,2),(1,5),(3,4)],
            [(0,3),(1,2),(4,5)],
            [(0,3),(1,4),(2,5)],
            [(0,3),(1,5),(2,4)],
            [(0,4),(1,2),(3,5)],
            [(0,4),(1,3),(2,5)],
            [(0,4),(1,5),(2,3)],
            [(0,5),(1,2),(3,4)],
            [(0,5),(1,3),(2,4)],
            [(0,5),(1,4),(2,3)],
        ]
        j_pairings = [
            ([(0,1),(2,3)], [4]),
            ([(0,1),(3,4)], [2]),
            ([(0,2),(1,3)], [4]),
            ([(0,2),(1,4)], [3]),
            ([(0,2),(3,4)], [1]),
            ([(0,3),(1,2)], [4]),
            ([(0,3),(1,4)], [2]),
            ([(0,3),(2,4)], [1]),
            ([(0,4),(1,2)], [3]),
            ([(0,4),(1,3)], [2]),
            ([(0,4),(2,3)], [1]),
            ([(1,2),(3,4)], [0]),
        ]
        
        for j_pairs, j_naive in j_pairings:
            if time.time() > deadline:
                break
            for k_pairs in k_pairings:
                if time.time() > deadline:
                    break
                U, V, W, rank = build_strassen_decomposition(j_pairs, j_naive, k_pairs)
                U, V, W, rank = try_reduce_rank(U, V, W, rank)
                if rank < best_rank:
                    T_dec = np.einsum('ar,br,cr->abc', U, V, W)
                    if np.allclose(T_dec, T, atol=1e-10, rtol=1e-10):
                        best_rank = rank
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
