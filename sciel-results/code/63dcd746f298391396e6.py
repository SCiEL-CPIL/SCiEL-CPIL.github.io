<<<<<<< SEARCH
"""Matrix multiplication 4x4x6 - Winograd/Strassen-based rank reduction."""
import time
import sys
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 24
    pn = p * n  # 24
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p  # 96
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Strassen-based decomposition for 4x4x6
    # Strategy: 
    #   - Top level: treat A (4x4) as 2x2 blocks of 2x2, B (4x6) as 2x3 blocks of 2x3
    #   - Use 7 Strassen multiplications of (2x2) * (2x3)
    #   - Each (2x2)*(2x3) is decomposed as:
    #       * Inner Strassen (rank 7) for the first 2 columns: (2x2)*(2x2)
    #       * Schoolbook (rank 4) for the last 1 column: (2x2)*(2x1)
    #     Total per (2x2)*(2x3): 7 + 4 = 11
    #   - Total rank: 7 * 11 = 77
    
    R = 77
    U = np.zeros((nm, R), dtype=int)
    V = np.zeros((mp, R), dtype=int)
    W = np.zeros((pn, R), dtype=int)
    
    # Strassen coefficients for 2x2x2
    # A blocks indexed 0..3: 00, 01, 10, 11
    # B blocks indexed 0..3: 00, 01, 10, 11
    alpha = [
        [1, 0, 0, 1],  # s0: A00 + A11
        [0, 0, 1, 1],  # s1: A10 + A11
        [1, 0, 0, 0],  # s2: A00
        [0, 0, 0, 1],  # s3: A11
        [1, 1, 0, 0],  # s4: A00 + A01
        [-1, 0, 1, 0], # s5: A10 - A00
        [0, 1, 0, -1]  # s6: A01 - A11
    ]
    beta = [
        [1, 0, 0, 1],  # s0: B00 + B11
        [1, 0, 0, 0],  # s1: B00
        [0, 1, 0, -1], # s2: B01 - B11
        [-1, 0, 1, 0], # s3: B10 - B00
        [0, 0, 0, 1],  # s4: B11
        [1, 1, 0, 0],  # s5: B00 + B01
        [0, 0, 1, 1]   # s6: B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s0
        [(1, 0, 1), (1, 1, -1)],    # s1
        [(0, 1, 1), (1, 1, 1)],     # s2
        [(0, 0, 1), (1, 0, 1)],     # s3
        [(0, 0, -1), (0, 1, 1)],    # s4
        [(1, 1, 1)],                # s5
        [(0, 0, 1)]                 # s6
    ]
    
    col = 0
    for s in range(7):
        # For each top-level Strassen term s:
        # X_s is 2x2, Y_s is 2x3
        # Compute X_s * Y_s using inner Strassen (cols 0-1) + schoolbook (col 2)
        
        # Part 1: Inner Strassen for (2x2) * (2x2) [columns 0,1 of Y_s]
        for si in range(7):
            if time.time() > deadline:
                break
            # U part: inner_alpha[si][a] * alpha[s][r]
            # a indexes position in X_s: row=a//2, col=a%2
            # r indexes block in A: ar=r//2, br=r%2
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            # V part: inner_beta[si][b] * beta[s][c]
            # b indexes position in Y_s[:,0:2]: row=b//2, col=b%2
            # c indexes block in B: ac=c//2, bc=c%2
            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            
            # W part: recon[s] and inner_recon[si]
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    # za, zb are positions in the 2x2 inner output
                    # This is P_s[za, zb]
                    # It contributes to C block (a,b) at position (za, zb)
                    # C global: row = a*2 + za, col = b*3 + zb
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for (2x2) * (2x1) [column 2 of Y_s]
        # X_s (2x2) * Y_s[:,2] (2x1) -> Z (2x1)
        # 4 schoolbook multiplications
        for u in range(2):
            if time.time() > deadline:
                break
            for v in range(2):
                # Term: X_s[u,v] * Y_s[v,2]
                
                # U part
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                # V part
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        # B_c[v, 2]
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                # W part
                for a, b, gamma in recon[s]:
                    # Output Z[u] contributes to C block (a,b) at row u, col 2
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    # Verify the decomposition
    def verify_decomp(U_mat, V_mat, W_mat):
        """Check if U, V, W exactly decompose the tensor."""
        rank = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                    if s != 1:
                        return False
        return True
    
    if not verify_decomp(U, V, W):
        # Fallback to schoolbook if verification fails
        return schoolbook()
    
    # Try to reduce rank by finding truly redundant terms
    # Strategy: for each column r, check if the tensor contribution can be
    # expressed as a linear combination of other columns
    
    def try_reduce_rank(U_mat, V_mat, W_mat, max_iter=200):
        """Try to reduce rank by removing truly redundant terms."""
        current_U = U_mat.copy().astype(float)
        current_V = V_mat.copy().astype(float)
        current_W = W_mat.copy().astype(float)
        current_rank = current_U.shape[1]
        
        removed = set()
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            if current_rank <= 1:
                break
            
            # Try to remove each column in order
            found_removal = False
            for r in range(current_rank):
                if time.time() > deadline:
                    break
                
                # Try removing column r
                test_U = np.delete(current_U, r, axis=1)
                test_V = np.delete(current_V, r, axis=1)
                test_W = np.delete(current_W, r, axis=1)
                
                # Quick check: does this still decompose the tensor?
                # We can do a fast check using matrix multiplication
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for rr in range(test_U.shape[1]):
                                s += test_U[i * m + j, rr] * test_V[j * p + k, rr] * test_W[k * n + i, rr]
                            if abs(s - 1) > 1e-10:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    current_U = test_U
                    current_V = test_V
                    current_W = test_W
                    current_rank -= 1
                    found_removal = True
                    break
            
            if not found_removal:
                break
        
        # Convert back to int if all entries are integers
        if np.allclose(current_U, np.round(current_U)):
            current_U = np.round(current_U).astype(int)
        if np.allclose(current_V, np.round(current_V)):
            current_V = np.round(current_V).astype(int)
        if np.allclose(current_W, np.round(current_W)):
            current_W = np.round(current_W).astype(int)
        
        return current_U, current_V, current_W, current_rank
    
    # Attempt rank reduction
    U_red, V_red, W_red, rank_red = try_reduce_rank(U, V, W, max_iter=200)
    
    if verify_decomp(U_red, V_red, W_red):
        U, V, W = U_red, V_red, W_red
    else:
        # Keep original if reduction failed verification
        pass
    
    # Convert to lists of lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
=======
"""Matrix multiplication 4x4x6 - randomized rank reduction search."""
import time
import sys
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 24
    pn = p * n  # 24
    
    # Build the matmul tensor T of shape (nm, mp, pn) where
    # T[i*m+j, j*p+k, k*n+i] = 1 and 0 otherwise
    T = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    # Schoolbook decomposition (rank 96)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.int64)
        V = np.zeros((mp, rank), dtype=np.int64)
        W = np.zeros((pn, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U0, V0, W0 = schoolbook()
    
    # Fast verification using einsum-like approach
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        """Check decomposition: sum_r U[i,r]*V[j,r]*W[k,r] == T[i,j,k] for all i,j,k"""
        rank = U_mat.shape[1]
        # Compute the decomposed tensor
        # For each rank index r, outer product of U[:,r], V[:,r], W[:,r]
        # Use broadcasting: shape (rank, nm, mp, pn)
        # But that's rank*16*24*24 which could be large. Use loop over rank.
        result = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(rank):
            result += U_mat[:, r] * V_mat[:, r] * W_mat[:, r]
        return np.allclose(result, T.astype(np.float64), atol=tol)
    
    # Strassen-based initial decomposition (rank 77)
    def strassen_77():
        R = 77
        U = np.zeros((nm, R), dtype=np.int64)
        V = np.zeros((mp, R), dtype=np.int64)
        W = np.zeros((pn, R), dtype=np.int64)
        
        alpha = [
            [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0],
            [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
        ]
        beta = [
            [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1],
            [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
        ]
        recon = [
            [(0, 0, 1), (1, 1, 1)], [(1, 0, 1), (1, 1, -1)],
            [(0, 1, 1), (1, 1, 1)], [(0, 0, 1), (1, 0, 1)],
            [(0, 0, -1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, 1)]
        ]
        
        col = 0
        for s in range(7):
            for si in range(7):
                for a in range(4):
                    if alpha[si][a] != 0:
                        x_row, x_col = a // 2, a % 2
                        for r in range(4):
                            if alpha[s][r] != 0:
                                ar, br = r // 2, r % 2
                                u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                                U[u_idx, col] += alpha[si][a] * alpha[s][r]
                for b in range(4):
                    if beta[si][b] != 0:
                        y_row, y_inner_col = b // 2, b % 2
                        for c in range(4):
                            if beta[s][c] != 0:
                                ac, bc = c // 2, c % 2
                                v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                                V[v_idx, col] += beta[si][b] * beta[s][c]
                for a, b, gamma in recon[s]:
                    for za, zb, zgamma in recon[si]:
                        w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                        W[w_idx, col] += gamma * zgamma
                col += 1
            
            for u in range(2):
                for v in range(2):
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                            U[u_idx, col] += alpha[s][r]
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                            V[v_idx, col] += beta[s][c]
                    for a, b, gamma in recon[s]:
                        w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                        W[w_idx, col] += gamma
                    col += 1
        
        return U, V, W
    
    U_str, V_str, W_str = strassen_77()
    
    # Start with the better initial decomposition
    if verify(U_str, V_str, W_str):
        U_cur, V_cur, W_cur = U_str.copy(), V_str.copy(), W_str.copy()
    else:
        U_cur, V_cur, W_cur = U0.copy(), V0.copy(), W0.copy()
    
    best_rank = U_cur.shape[1]
    best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
    
    # Structural rank reduction: try to express one term as a combination of others
    # For a set of rank-1 terms, if term r can be written as sum_{s != r} c_s * term_s
    # (where term_r means the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]),
    # then we can eliminate it.
    # 
    # More practically: if there's a linear dependency among the "flattened" rank-1 tensors,
    # we can reduce rank. But the space is 16*24*24 = 9216 dimensional and rank is ~77-96,
    # so generic linear dependencies are unlikely. Instead, we look for structured ones.
    #
    # Alternative structural approach: use the fact that the matmul tensor has a special
    # algebraic structure. Try "splitting" and "merging" terms.
    #
    # Key insight: if we have terms where U vectors are linearly dependent, or V vectors,
    # or W vectors, we can potentially combine them.
    
    def try_structural_reduction(U_mat, V_mat, W_mat, max_rounds=50):
        """Try structured rank reduction by finding and eliminating dependent terms."""
        U_c = U_mat.astype(np.float64).copy()
        V_c = V_mat.astype(np.float64).copy()
        W_c = W_mat.astype(np.float64).copy()
        cur_rank = U_c.shape[1]
        
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            
            found = False
            # Try each column for removal
            for r in range(cur_rank):
                if time.time() > deadline:
                    break
                
                # Check if removing column r still gives a valid decomposition
                # Fast check: compute the contribution of term r and see if it's zero
                # or can be absorbed
                # The contribution of term r to the tensor is:
                #   U[:,r] ⊗ V[:,r] ⊗ W[:,r]
                # If this is zero, we can remove it.
                term_r = U_c[:, r] * V_c[:, r] * W_c[:, r]
                if np.allclose(term_r, 0, atol=1e-12):
                    U_c = np.delete(U_c, r, axis=1)
                    V_c = np.delete(V_c, r, axis=1)
                    W_c = np.delete(W_c, r, axis=1)
                    cur_rank -= 1
                    found = True
                    break
                
                # Check if term r is a linear combination of other terms
                # This is expensive, so only do it for small remaining rank
                if cur_rank > 40:
                    continue
                
                # Try to solve: term_r = sum_{s != r} c_s * term_s
                # Flatten all terms into vectors of length nm*mp*pn
                # This is too expensive for large rank, so skip for now
                pass
            
            if not found:
                # Try a different strategy: look for pairs of terms that can be merged
                # If term_a + term_b = term_c (another existing term), we can remove one
                # This is O(n^3) in rank, so only for small rank
                if cur_rank <= 30:
                    merged = False
                    for a in range(cur_rank):
                        if time.time() > deadline:
                            break
                        for b in range(a + 1, cur_rank):
                            if time.time() > deadline:
                                break
                            # Check if term_a + term_b equals some other term
                            sum_ab = U_c[:, a] * V_c[:, a] * W_c[:, a] + U_c[:, b] * V_c[:, b] * W_c[:, b]
                            for c in range(cur_rank):
                                if c == a or c == b:
                                    continue
                                term_c = U_c[:, c] * V_c[:, c] * W_c[:, c]
                                if np.allclose(sum_ab, term_c, atol=1e-10):
                                    # Remove terms a and b, keep c (which already accounts for their sum)
                                    # Actually we need to remove a and b and adjust c
                                    # term_c = term_a + term_b, so we can remove a and b
                                    # But we need to be careful: the total tensor is sum of all terms
                                    # If we remove a and b, we lose term_a + term_b = term_c
                                    # But term_c is still there, so the total is unchanged
                                    # Wait: total = ... + term_a + term_b + term_c + ...
                                    # = ... + term_c + term_c + ...
                                    # That's wrong. We need total = ... + term_c + ...
                                    # So we should remove a, b, AND subtract term_c
                                    # This gets complicated. Skip this approach.
                                    pass
                            break
                        if merged:
                            break
                    if merged:
                        found = True
                        break
                if not found:
                    break
        
        return U_c, V_c, W_c, cur_rank
    
    # Apply structural reduction
    U_red, V_red, W_red, rank_red = try_structural_reduction(U_cur, V_cur, W_cur, max_rounds=50)
    
    if rank_red < best_rank and verify(U_red, V_red, W_red):
        best_rank = rank_red
        best_U, best_V, best_W = U_red.copy(), V_red.copy(), W_red.copy()
    
    # Randomized search: try random linear transformations of the factor matrices
    # If we have a valid decomposition with rank R, we can apply any invertible
    # R x R matrix M to the rank index:
    #   U' = U @ M, V' = V @ inv(M), W' = W (or W' = W @ something)
    # This preserves the decomposition but changes the representation.
    # After the transformation, some entries may become zero, allowing removal.
    
    def randomized_search(U_init, V_init, W_init, num_trials=200):
        """Try random transformations to find sparser representations."""
        U_c = U_init.astype(np.float64).copy()
        V_c = V_init.astype(np.float64).copy()
        W_c = W_init.astype(np.float64).copy()
        rank = U_c.shape[1]
        best_local_rank = rank
        best_local_U, best_local_V, best_local_W = U_c.copy(), V_c.copy(), W_c.copy()
        
        rng = np.random.RandomState(42)
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Random invertible transformation
            M = np.eye(rank)
            # Apply a few random elementary operations
            for _ in range(rank // 2):
                i, j = rng.randint(0, rank, size=2)
                if i != j:
                    factor = rng.uniform(-2, 2)
                    M[i, j] += factor
            
            # Ensure M is invertible
            try:
                M_inv = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                continue
            
            U_t = U_c @ M
            V_t = V_c @ M_inv
            W_t = W_c  # W stays the same (or we could transform it differently)
            
            # Actually, the correct transformation is:
            # sum_r U[:,r] V[:,r] W[:,r] = sum_s U'[:,s] V'[:,s] W'[:,s]
            # where U' = U @ M, V' = V @ M^{-1}, W' = W
            # Let's verify: sum_s (U M)[:,s] (V M^{-1})[:,s] W[:,s]
            # = sum_s sum_a U[:,a] M[a,s] * sum_b V[:,b] M^{-1}[b,s] * W[:,s]
            # This doesn't simplify nicely. The correct identity is:
            # sum_r U_r V_r W_r (outer products)
            # = sum_s (sum_a U_a M[a,s]) (sum_b V_b (M^{-1})[b,s]) W_s
            # which is NOT equal to sum_s U'_s V'_s W_s in general.
            #
            # The correct transformation for tensor rank:
            # If T = sum_r u_r ⊗ v_r ⊗ w_r, then for any invertible M:
            # T = sum_s (sum_a M[a,s] u_a) ⊗ (sum_b (M^{-1})[b,s] v_b) ⊗ w_s
            # Wait, that's not right either. Let me think more carefully.
            #
            # T = sum_r u_r ⊗ v_r ⊗ w_r
            # Let u'_s = sum_a M[a,s] u_a, v'_s = sum_b (M^{-1})[b,s] v_b
            # Then sum_s u'_s ⊗ v'_s ⊗ w_s = sum_s sum_a sum_b M[a,s] (M^{-1})[b,s] u_a ⊗ v_b ⊗ w_s
            # = sum_a sum_b u_a ⊗ v_b ⊗ (sum_s M[a,s] (M^{-1})[b,s] w_s)
            # = sum_a sum_b u_a ⊗ v_b ⊗ (M M^{-1})[a,b] w_b  ... no this is getting confused.
            #
            # Actually the standard result is:
            # T = sum_r u_r ⊗ v_r ⊗ w_r
            # For any invertible R×R matrix P:
            # T = sum_s (sum_a P[a,s] u_a) ⊗ (sum_b P^{-1}[s,b] v_b) ⊗ w_s
            # Let me verify: sum_s sum_a sum_b P[a,s] P^{-1}[s,b] u_a ⊗ v_b ⊗ w_s
            # = sum_a sum_b u_a ⊗ v_b ⊗ (sum_s P[a,s] P^{-1}[s,b] w_s)
            # The inner sum is (P P^{-1})[a,b] = δ_{a,b}
            # So = sum_a sum_b u_a ⊗ v_b ⊗ δ_{a,b} w_b = sum_a u_a ⊗ v_a ⊗ w_a ✓
            #
            # So: U'[:,s] = sum_a P[a,s] U[:,a], V'[:,s] = sum_b P^{-1}[s,b] V[:,b], W' = W
            # In matrix form: U' = U @ P, V' = V @ P^{-T} (since V'[:,s] = sum_b P^{-1}[s,b] V[:,b] means V' = V @ (P^{-1})^T)
            
            U_t = U_c @ M
            V_t = V_c @ np.linalg.inv(M).T
            W_t = W_c
            
            # Now try to remove zero or near-zero columns
            for s in range(rank):
                if time.time() > deadline:
                    break
                # Check if this term contributes nothing
                contrib = U_t[:, s] * V_t[:, s] * W_t[:, s]
                if np.allclose(contrib, 0, atol=1e-10):
                    U_t = np.delete(U_t, s, axis=1)
                    V_t = np.delete(V_t, s, axis=1)
                    W_t = np.delete(W_t, s, axis=1)
                    rank -= 1
                    # Adjust indices
                    break  # Restart from 0 since indices changed
            
            if rank < best_local_rank:
                best_local_rank = rank
                best_local_U, best_local_V, best_local_W = U_t.copy(), V_t.copy(), W_t.copy()
        
        return best_local_U, best_local_V, best_local_W, best_local_rank
    
    # Apply randomized search
    U_rnd, V_rnd, W_rnd, rank_rnd = randomized_search(best_U, best_V, best_W, num_trials=100)
    
    if rank_rnd < best_rank and verify(U_rnd, V_rnd, W_rnd):
        best_rank = rank_rnd
        best_U, best_V, best_W = U_rnd.copy(), V_rnd.copy(), W_rnd.copy()
    
    # Final verification and conversion
    if not verify(best_U, best_V, best_W):
        # Fallback to schoolbook
        U_f, V_f, W_f = schoolbook()
        best_U, best_V, best_W = U_f, V_f, W_f
        best_rank = U_f.shape[1]
    
    # Round to integers if close
    def round_if_close(mat):
        if np.allclose(mat, np.round(mat), atol=1e-8):
            return np.round(mat).astype(np.int64)
        return mat
    
    best_U = round_if_close(best_U)
    best_V = round_if_close(best_V)
    best_W = round_if_close(best_W)
    
    # Final check
    if not verify(best_U, best_V, best_W):
        U_f, V_f, W_f = schoolbook()
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
