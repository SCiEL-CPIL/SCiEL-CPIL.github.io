"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    deadline = time.time() + 130  # safety margin before 150s kill

    # Build the multiplication tensor T[a,b,c]
    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T = build_tensor()

    def verify(U, V, W, tol=1e-8):
        """Check exact decomposition."""
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def to_lists(U, V, W, r):
        """Convert numpy arrays to list-of-lists format."""
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    # === STRASSEN-BASED CONSTRUCTION (rank 28) ===
    def strassen_2x2_decomp():
        """
        Return Strassen decomposition of 2x2 * 2x2.
        U: (4, 7), V: (4, 7), W: (4, 7)
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # t=0: M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1.0; U[3, 0] = 1.0
        V[0, 0] = 1.0; V[3, 0] = 1.0
        W[0, 0] = 1.0; W[3, 0] = 1.0

        # t=1: M2 = (A10+A11)*B00
        U[2, 1] = 1.0; U[3, 1] = 1.0
        V[0, 1] = 1.0
        W[1, 1] = 1.0; W[3, 1] = -1.0

        # t=2: M3 = A00*(B01-B11)
        U[0, 2] = 1.0
        V[1, 2] = 1.0; V[3, 2] = -1.0
        W[2, 2] = 1.0; W[3, 2] = 1.0

        # t=3: M4 = A11*(B10-B00)
        U[3, 3] = 1.0
        V[2, 3] = 1.0; V[0, 3] = -1.0
        W[0, 3] = 1.0; W[1, 3] = 1.0

        # t=4: M5 = (A00+A01)*B11
        U[0, 4] = 1.0; U[1, 4] = 1.0
        V[3, 4] = 1.0
        W[0, 4] = -1.0; W[2, 4] = 1.0

        # t=5: M6 = (A10-A00)*(B00+B01)
        U[2, 5] = 1.0; U[0, 5] = -1.0
        V[0, 5] = 1.0; V[1, 5] = 1.0
        W[3, 5] = 1.0

        # t=6: M7 = (A01-A11)*(B10+B11)
        U[1, 6] = 1.0; U[3, 6] = -1.0
        V[2, 6] = 1.0; V[3, 6] = 1.0
        W[0, 6] = 1.0

        return U, V, W

    # Build the full 2x4 * 4x4 decomposition using 4 Strassen blocks.
    U_s, V_s, W_s = strassen_2x2_decomp()

    full_rank = 28
    U_full = np.zeros((n * m, full_rank))  # 8 x 28
    V_full = np.zeros((m * p, full_rank))  # 16 x 28
    W_full = np.zeros((p * n, full_rank))  # 8 x 28

    # Block 1: A_left * B_top_left -> C_left
    for t in range(7):
        col = 0 * 7 + t
        for li in range(2):
            for lj in range(2):
                local_a = li * 2 + lj
                global_a = li * 4 + lj
                U_full[global_a, col] = U_s[local_a, t]
        for lj in range(2):
            for lk in range(2):
                local_b = lj * 2 + lk
                global_b = lj * 4 + lk
                V_full[global_b, col] = V_s[local_b, t]
        for lk in range(2):
            for li in range(2):
                local_c = lk * 2 + li
                global_c = lk * 2 + li
                W_full[global_c, col] = W_s[local_c, t]

    # Block 2: A_right * B_bot_left -> C_left
    for t in range(7):
        col = 1 * 7 + t
        for li in range(2):
            for lj in range(2):
                local_a = li * 2 + lj
                global_a = li * 4 + (lj + 2)
                U_full[global_a, col] = U_s[local_a, t]
        for lj in range(2):
            for lk in range(2):
                local_b = lj * 2 + lk
                global_b = (lj + 2) * 4 + lk
                V_full[global_b, col] = V_s[local_b, t]
        for lk in range(2):
            for li in range(2):
                local_c = lk * 2 + li
                global_c = lk * 2 + li
                W_full[global_c, col] = W_s[local_c, t]

    # Block 3: A_left * B_top_right -> C_right
    for t in range(7):
        col = 2 * 7 + t
        for li in range(2):
            for lj in range(2):
                local_a = li * 2 + lj
                global_a = li * 4 + lj
                U_full[global_a, col] = U_s[local_a, t]
        for lj in range(2):
            for lk in range(2):
                local_b = lj * 2 + lk
                global_b = lj * 4 + (lk + 2)
                V_full[global_b, col] = V_s[local_b, t]
        for lk in range(2):
            for li in range(2):
                local_c = lk * 2 + li
                global_c = (lk + 2) * 2 + li
                W_full[global_c, col] = W_s[local_c, t]

    # Block 4: A_right * B_bot_right -> C_right
    for t in range(7):
        col = 3 * 7 + t
        for li in range(2):
            for lj in range(2):
                local_a = li * 2 + lj
                global_a = li * 4 + (lj + 2)
                U_full[global_a, col] = U_s[local_a, t]
        for lj in range(2):
            for lk in range(2):
                local_b = lj * 2 + lk
                global_b = (lj + 2) * 4 + (lk + 2)
                V_full[global_b, col] = V_s[local_b, t]
        for lk in range(2):
            for li in range(2):
                local_c = lk * 2 + li
                global_c = (lk + 2) * 2 + li
                W_full[global_c, col] = W_s[local_c, t]

    # Verify the rank-28 construction
    if not verify(U_full, V_full, W_full, tol=1e-8):
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return to_lists(U, V, W, rank)

    best_U, best_V, best_W, best_rank = U_full.copy(), V_full.copy(), W_full.copy(), full_rank

    # === RANK REDUCTION SEARCH ===
    # Try to reduce rank by finding pairs of rank-1 terms that can be merged.
    # For a pair of columns (r1, r2), we look for coefficients alpha, beta such that
    # the tensor contribution of alpha*term(r1) + beta*term(r2) can be represented
    # as a single rank-1 term. This is equivalent to finding a 2D subspace of the
    # factor space where the tensor has rank 1.
    
    def try_reduce_pair(U, V, W, r1, r2, tol=1e-6, max_iters=500):
        """
        Try to merge rank-1 terms r1 and r2 into a single term.
        Returns (new_U, new_V, new_W) with rank reduced by 1, or None.
        """
        if time.time() > deadline - 5:
            return None
        
        # The combined contribution of terms r1 and r2 is:
        # T_pair = U[:,r1]⊗V[:,r1]⊗W[:,r1] + U[:,r2]⊗V[:,r2]⊗W[:,r2]
        # We want to find u, v, w such that u⊗v⊗w = T_pair
        # This means T_pair must be a rank-1 tensor.
        
        # Check if the 2D subspace spanned by {term(r1), term(r2)} has tensor rank 1.
        # Method: form the 2x2 matrix of inner products and check determinant.
        # Actually, a better approach: parameterize u = a*u1 + b*u2, etc.
        # and find (a,b,c) such that (a*u1+b*u2)⊗(c*v1+d*v2)⊗(e*w1+f*w2) = T_pair
        
        # Simpler: check if T_pair is rank-1 by flattening and checking SVD.
        u1, v1, w1 = U[:, r1], V[:, r1], W[:, r1]
        u2, v2, w2 = U[:, r2], V[:, r2], W[:, r2]
        
        # T_pair flattened: sum over a,b,c of T_pair[a,b,c] * x[a] * y[b] * z[c]
        # For rank-1, T_pair = u ⊗ v ⊗ w for some u,v,w
        # Check: T_pair[a,b,c] * T_pair[a',b',c'] = T_pair[a,b',c'] * T_pair[a',b,c']
        # (the "rank-1 consistency" condition)
        
        # More efficiently: flatten T_pair and check if it's a rank-1 tensor
        # by checking if the matrix unfolding has rank 1.
        
        # Build T_pair as a 3D array
        T_pair = np.outer(np.outer(u1, v1), w1) + np.outer(np.outer(u2, v2), w2)
        # T_pair shape: (n*m, m*p, p*n)
        
        # Check rank-1 by SVD of mode-1 unfolding
        T_pair_2d = T_pair.reshape(n * m, -1)
        sv = np.linalg.svd(T_pair_2d, compute_uv=False)
        
        # If rank-1, the second singular value should be ~0
        if len(sv) < 2 or sv[1] > tol * (sv[0] + tol):
            return None
        
        # Extract the rank-1 factor
        # The dominant left singular vector gives u (up to scale)
        U_mat, s, Vt = np.linalg.svd(T_pair_2d, full_matrices=False)
        u_new = U_mat[:, 0] * np.sqrt(s[0])
        
        # Now find v and w from T_pair = u_new ⊗ v ⊗ w
        # Reshape: T_pair[a, b, c] = u_new[a] * v[b] * w[c]
        # For each a: T_pair[a, :, :] = u_new[a] * (v ⊗ w)
        # So (v ⊗ w) = T_pair[a, :, :] / u_new[a] for any a where u_new[a] != 0
        
        # Find a with significant u_new[a]
        idx = np.argmax(np.abs(u_new))
        if abs(u_new[idx]) < tol:
            return None
        
        vw = T_pair[idx, :, :] / u_new[idx]
        # vw shape: (m*p, p*n)
        # vw should be v ⊗ w, i.e., rank-1 matrix
        vw_2d = vw.reshape(m * p, p * n)
        U_vw, s_vw, Vt_vw = np.linalg.svd(vw_2d, full_matrices=False)
        
        if len(s_vw) < 1 or s_vw[0] < tol:
            return None
            
        if len(s_vw) >= 2 and s_vw[1] > tol * s_vw[0]:
            return None
        
        v_new = U_vw[:, 0] * np.sqrt(s_vw[0])
        w_new = Vt_vw[0, :] * np.sqrt(s_vw[0])
        
        # Verify
        T_check = np.outer(np.outer(u_new, v_new), w_new)
        if np.max(np.abs(T_check - T_pair)) > tol * np.max(np.abs(T_pair)):
            return None
        
        # Build new factor matrices with rank reduced by 1
        # Remove columns r1 and r2, add the new column
        cols_keep = [c for c in range(len(U)) if c != r1 and c != r2]
        new_rank = len(cols_keep) + 1
        
        U_new = np.zeros((n * m, new_rank))
        V_new = np.zeros((m * p, new_rank))
        W_new = np.zeros((p * n, new_rank))
        
        for new_i, old_i in enumerate(cols_keep):
            U_new[:, new_i] = U[:, old_i]
            V_new[:, new_i] = V[:, old_i]
            W_new[:, new_i] = W[:, old_i]
        
        U_new[:, new_rank - 1] = u_new
        V_new[:, new_rank - 1] = v_new
        W_new[:, new_rank - 1] = w_new
        
        return U_new, V_new, W_new

    # Attempt rank reduction: try all pairs of columns
    # To save time, only try pairs within the same Strassen block first,
    # then try cross-block pairs.
    current_U, current_V, current_W = best_U.copy(), best_V.copy(), best_W.copy()
    current_rank = best_rank
    
    # Try reducing by 1 at a time, up to a few reductions
    max_reductions = 3
    for reduction_round in range(max_reductions):
        if time.time() > deadline - 10:
            break
            
        found_reduction = False
        # Try all pairs
        for r1 in range(current_rank):
            if time.time() > deadline - 10:
                break
            for r2 in range(r1 + 1, current_rank):
                if time.time() > deadline - 10:
                    break
                
                result = try_reduce_pair(current_U, current_V, current_W, r1, r2)
                if result is not None:
                    U_new, V_new, W_new = result
                    if verify(U_new, V_new, W_new, tol=1e-7):
                        current_U, current_V, current_W = U_new, V_new, W_new
                        current_rank = len(U_new[0])
                        found_reduction = True
                        break
            
            if found_reduction:
                break
        
        if not found_reduction:
            break
    
    # Update best
    if current_rank < best_rank:
        best_U, best_V, best_W, best_rank = current_U, current_V, current_W, current_rank

    # Final verification
    if verify(best_U, best_V, best_W, tol=1e-7):
        return to_lists(best_U, best_V, best_W, best_rank)
    
    # Fallback: return the verified rank-28
    if verify(U_full, V_full, W_full, tol=1e-8):
        return to_lists(U_full, V_full, W_full, full_rank)
    
    # Last resort: schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return to_lists(U, V, W, rank)
# EVOLVE-BLOCK-END
