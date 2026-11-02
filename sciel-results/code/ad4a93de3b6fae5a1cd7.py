"""Matrix multiplication 3x4x7 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 4, 7]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    def time_left():
        return deadline - time.time()

    def schoolbook():
        """Standard rank-84 decomposition."""
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

    def verify_decomp(U, V, W, atol=1e-8):
        """Check if U,V,W exactly decompose the multiplication tensor."""
        try:
            Ua = np.array(U, dtype=np.float64)
            Va = np.array(V, dtype=np.float64)
            Wa = np.array(W, dtype=np.float64)
            r = Ua.shape[1]
            Ur = Ua.reshape(n, m, r)
            Vr = Va.reshape(m, p, r)
            Wr = Wa.reshape(p, n, r)
            # T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
            T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
            if not np.allclose(T, 1.0, atol=atol):
                return False
            return True
        except Exception:
            return False

    def check_half_int(mat):
        """All entries are integers or half-integers."""
        for row in mat:
            for v in row:
                if not np.isfinite(v):
                    return False
                if abs(v * 2.0 - round(v * 2.0)) > 1e-7:
                    return False
        return True

    def valid(U, V, W):
        if not check_half_int(U) or not check_half_int(V) or not check_half_int(W):
            return False
        return verify_decomp(U, V, W)

    def try_reduce(U, V, W, max_iter=2000):
        """
        Attempt to reduce rank by finding linear dependencies among columns.
        If column c is a linear combination of other columns in all three matrices,
        we can eliminate it.
        """
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        r = Ua.shape[1]

        # Stack into a single matrix: each column c gives a vector in R^(nm+mp+pn)
        # If a column is in the span of others, we can remove it.
        # But we need the same linear combination to work across all three matrices.
        # So we look for: U[:,c] = sum_a alpha_a U[:,a], V[:,c] = sum_a alpha_a V[:,a], W[:,c] = sum_a alpha_a W[:,a]
        
        # Build the combined matrix where each "feature" is a column
        # We want to find if any column is in the span of the others in the combined space
        # Combined: for each column c, vector = [U[:,c]; V[:,c]; W[:,c]]
        # But the constraint is that the SAME coefficients work for all three.
        # This is equivalent to: the column c in the stacked matrix [U;V;W] is in the span
        # of the other columns, AND the coefficients give valid reconstructions.
        
        # Actually, the correct condition: there exist alpha_1..alpha_r (not all zero, with alpha_c = 1)
        # such that for all a != c: U[:,a] - alpha_a * U[:,c] = 0 when projected...
        
        # Simpler approach: try to find a set of columns that are linearly dependent
        # in the "product space" and eliminate one.
        
        # Let's try: for each column c, check if it's in the span of the others
        # in the combined matrix C = [U | V | W] (vertically stacked)
        # If yes, find the null vector and use it to eliminate.
        
        best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
        best_rank = r

        for _ in range(max_iter):
            if time_left() < 5:
                break
            cr = best_U.shape[1]
            if cr <= 1:
                break
            
            # Build combined matrix: (nm + mp + pn) x cr
            C = np.vstack([best_U, best_V, best_W])
            
            # Find rank and null space
            try:
                # Use SVD for numerical stability
                U_sv, s_vals, Vt = np.linalg.svd(C, full_matrices=False)
                tol = 1e-10 * max(s_vals) if len(s_vals) > 0 else 1e-10
                rank_C = np.sum(s_vals > tol)
                
                if rank_C < cr:
                    # There's a null space vector
                    # The null vectors are the last (cr - rank_C) rows of Vt
                    null_vecs = Vt[rank_C:]  # shape (cr-rank_C, cr)
                    # Take the first null vector
                    nv = null_vecs[0]
                    
                    # Find a column with large coefficient to eliminate
                    # We want to eliminate column c where |nv[c]| is large
                    # and nv[c] != 0
                    c_idx = np.argmax(np.abs(nv))
                    if abs(nv[c_idx]) < 1e-10:
                        continue
                    
                    # Normalize: nv[c_idx] should be 1 for elimination
                    # The relation is: sum_c nv[c] * C[:,c] = 0
                    # So C[:,c_idx] = -sum_{a != c_idx} (nv[a]/nv[c_idx]) * C[:,a]
                    # We can eliminate column c_idx by expressing it in terms of others
                    
                    # Build new matrices without column c_idx
                    mask = [a for a in range(cr) if a != c_idx]
                    new_U = best_U[:, mask].copy()
                    new_V = best_V[:, mask].copy()
                    new_W = best_W[:, mask].copy()
                    
                    # Verify the reduced decomposition still works
                    if verify_decomp(new_U.tolist(), new_V.tolist(), new_W.tolist(), atol=1e-7):
                        best_U = new_U
                        best_V = new_V
                        best_W = new_W
                        best_rank = cr - 1
                        continue
                    else:
                        # The null space vector in the combined space doesn't
                        # necessarily give a valid decomposition reduction
                        # Try next null vector or skip
                        pass
            except Exception:
                break
        
        return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank

    def greedy_random_search(U_init, V_init, W_init, max_iters=5000):
        """
        Randomized search for rank reduction.
        Try random linear combinations to see if we can replace a set of columns
        with fewer columns.
        """
        Ua = np.array(U_init, dtype=np.float64)
        Va = np.array(V_init, dtype=np.float64)
        Wa = np.array(W_init, dtype=np.float64)
        r = Ua.shape[1]
        best_rank = r
        best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
        
        # Try to find pairs of columns that can be merged
        # If U[:,a] + U[:,b], V[:,a]+V[:,b], W[:,a]+W[:,b] can replace both
        # (i.e., the product tensor contribution is preserved)
        # This requires: for all i,j,k:
        #   U[i,j,a]*V[j,k,a]*W[k,i,a] + U[i,j,b]*V[j,k,b]*W[k,i,b]
        #   = U[i,j,a+b]*V[j,k,a+b]*W[k,i,a+b]
        # which is generally NOT true unless special structure exists.
        
        # Alternative: try rank-1 updates
        # Replace two columns with one by finding a, b, c such that
        # the sum of contributions equals a single column contribution
        
        # For now, try the null-space approach more aggressively
        for _ in range(max_iters):
            if time_left() < 5:
                break
            cr = best_U.shape[1]
            if cr <= 1:
                break
            
            # Randomly pick a column to try to eliminate
            c = random.randint(0, cr - 1)
            
            # Try to express column c as a combination of others
            # using least squares on the combined matrix
            others = [a for a in range(cr) if a != c]
            if not others:
                break
            
            C = np.vstack([best_U, best_V, best_W])
            C_target = C[:, c]
            C_others = C[:, others]
            
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(C_others, C_target, rcond=1e-10)
                # Check if the fit is good
                C_recon = C_others @ coeffs
                if np.linalg.norm(C_target - C_recon) < 1e-7 * np.linalg.norm(C_target):
                    # Try removing column c
                    mask = others
                    new_U = best_U[:, mask].copy()
                    new_V = best_V[:, mask].copy()
                    new_W = best_W[:, mask].copy()
                    if verify_decomp(new_U.tolist(), new_V.tolist(), new_W.tolist(), atol=1e-7):
                        best_U = new_U
                        best_V = new_V
                        best_W = new_W
                        best_rank = cr - 1
            except Exception:
                pass
        
        return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank

    # Start with schoolbook
    best_result = schoolbook()
    best_rank = n * m * p

    # Try to reduce
    U0 = best_result["U"]
    V0 = best_result["V"]
    W0 = best_result["W"]

    # Method 1: Null-space reduction
    U1, V1, W1, r1 = try_reduce(U0, V0, W0, max_iter=5000)
    if r1 < best_rank and valid(U1, V1, W1):
        best_result = {"U": U1, "V": V1, "W": W1}
        best_rank = r1

    # Method 2: Randomized greedy
    if time_left() > 10:
        U2, V2, W2, r2 = greedy_random_search(U1, V1, W1, max_iters=3000)
        if r2 < best_rank and valid(U2, V2, W2):
            best_result = {"U": U2, "V": V2, "W": W2}
            best_rank = r2

    # Final verification
    if not valid(best_result["U"], best_result["V"], best_result["W"]):
        # Fallback to schoolbook
        best_result = schoolbook()

    return best_result
# EVOLVE-BLOCK-END
