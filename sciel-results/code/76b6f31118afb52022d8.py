"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # n=4, m=4, p=8

    # Build the multiplication tensor T[a, b, c]
    # a = i*m + j (index into A: n x m = 4x4 = 16)
    # b = j*p + k (index into B: m x p = 4x8 = 32)
    # c = k*n + i (index into C: p x n = 8x4 = 32)
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Schoolbook rank = n*m*p = 128
    schoolbook_rank = n * m * p

    # Build schoolbook decomposition as fallback
    U_sb = np.zeros((n * m, schoolbook_rank), dtype=np.int64)
    V_sb = np.zeros((m * p, schoolbook_rank), dtype=np.int64)
    W_sb = np.zeros((p * n, schoolbook_rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def verify_decomposition(U, V, W, T, tol=1e-6):
        """Verify that sum_r U[:,r] V[:,r] W[:,r] == T."""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=tol)

    def to_half_integer(mat, tol=1e-8):
        """Check if all entries are half-integers and round them."""
        doubled = 2 * mat
        rounded = np.round(doubled)
        if np.allclose(doubled, rounded, atol=tol):
            return rounded / 2.0
        return None

    def try_rank(rank, T, dim_a, dim_b, dim_c, max_iters=5000, seed=42):
        """
        Try to find a rank-`rank` decomposition using iterative randomized search.
        This is a different strategy: instead of constructing from known algorithms,
        we use random initialization + gradient-free optimization (coordinate descent on 
        the tensor fitting problem).
        """
        rng = np.random.RandomState(seed)
        
        # Initialize with random factors
        U = rng.randn(dim_a, rank)
        V = rng.randn(dim_b, rank)
        W = rng.randn(dim_c, rank)
        
        # Normalize columns to prevent divergence
        for r in range(rank):
            norm = np.sqrt(np.sum(U[:,r]**2) + np.sum(V[:,r]**2) + np.sum(W[:,r]**2))
            if norm > 1e-10:
                U[:,r] /= norm
                V[:,r] /= norm
                W[:,r] /= norm
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random rank-1 term to update
            r = rng.randint(rank)
            
            # Compute residual
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            R = T - T_recon
            # Subtract current term's contribution
            R -= np.outer(U[:,r], np.outer(V[:,r], W[:,r]))
            
            # Update U[:,r] by projecting R onto V[:,r] and W[:,r]
            # R[a,b,c] = U[a,r] * V[b,r] * W[c,r] + ...
            # For fixed V[:,r], W[:,r]: U[a,r] = sum_{b,c} R[a,b,c] V[b,r] W[c,r] / (sum_b V[b,r]^2 * sum_c W[c,r]^2)
            v_norm_sq = np.sum(V[:,r]**2)
            w_norm_sq = np.sum(W[:,r]**2)
            if v_norm_sq < 1e-10 or w_norm_sq < 1e-10:
                continue
            U_new = np.sum(R * np.outer(V[:,r], W[:,r]), axis=(1,2)) / (v_norm_sq * w_norm_sq)
            
            # Compute new error contribution
            old_term = np.outer(U[:,r], np.outer(V[:,r], W[:,r]))
            new_term = np.outer(U_new, np.outer(V[:,r], W[:,r]))
            
            old_err = np.sum((T_recon + old_term - T)**2)
            new_err = np.sum((T_recon + new_term - T)**2)
            
            if new_err < old_err - 1e-10:
                U[:,r] = U_new
                best_err = min(best_err, new_err)
                if best_err < 1e-8:
                    break
                # Occasionally save best
                if it % 100 == 0:
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Similarly update V
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            R = T - T_recon
            R -= np.outer(U[:,r], np.outer(V[:,r], W[:,r]))
            u_norm_sq = np.sum(U[:,r]**2)
            w_norm_sq = np.sum(W[:,r]**2)
            if u_norm_sq < 1e-10 or w_norm_sq < 1e-10:
                continue
            V_new = np.sum(R * np.outer(U[:,r], W[:,r]), axis=(0,2)) / (u_norm_sq * w_norm_sq)
            old_term = np.outer(U[:,r], np.outer(V[:,r], W[:,r]))
            new_term = np.outer(U[:,r], np.outer(V_new, W[:,r]))
            old_err = np.sum((T_recon + old_term - T)**2)
            new_err = np.sum((T_recon + new_term - T)**2)
            if new_err < old_err - 1e-10:
                V[:,r] = V_new
                if new_err < 1e-8:
                    best_err = new_err
                    break
                if it % 100 == 0:
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Update W
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            R = T - T_recon
            R -= np.outer(U[:,r], np.outer(V[:,r], W[:,r]))
            u_norm_sq = np.sum(U[:,r]**2)
            v_norm_sq = np.sum(V[:,r]**2)
            if u_norm_sq < 1e-10 or v_norm_sq < 1e-10:
                continue
            W_new = np.sum(R * np.outer(U[:,r], V[:,r]), axis=(0,1)) / (u_norm_sq * v_norm_sq)
            old_term = np.outer(U[:,r], np.outer(V[:,r], W[:,r]))
            new_term = np.outer(U[:,r], np.outer(V[:,r], W_new))
            old_err = np.sum((T_recon + old_term - T)**2)
            new_err = np.sum((T_recon + new_term - T)**2)
            if new_err < old_err - 1e-10:
                W[:,r] = W_new
                if new_err < 1e-8:
                    best_err = new_err
                    break
                if it % 100 == 0:
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Check if we found a valid decomposition
        T_recon = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
        if np.allclose(T_recon, T, atol=1e-6):
            # Try to convert to half-integers
            Ui = to_half_integer(best_U)
            Vi = to_half_integer(best_V)
            Wi = to_half_integer(best_W)
            if Ui is not None and Vi is not None and Wi is not None:
                # Verify the integer version
                if verify_decomposition(Ui, Vi, Wi, T, tol=1e-6):
                    return Ui, Vi, Wi
        return None

    # Strategy: Try to find a decomposition with rank less than 128 using 
    # iterative optimization. Start from rank 98 (Strassen-based bound) and 
    # try to reduce.
    
    # First, build the Strassen-based rank-98 decomposition as a starting point
    def build_strassen_2x2x2():
        """Build rank-7 decomposition for 2x2 x 2x2 matrix multiplication."""
        U7 = np.zeros((4, 7), dtype=np.float64)
        V7 = np.zeros((4, 7), dtype=np.float64)
        W7 = np.zeros((4, 7), dtype=np.float64)

        # P1 = (a+b)*(e+f), contributes to C00 (+1) and C11 (+1)
        U7[0, 0] = 1; U7[1, 0] = 1
        V7[0, 0] = 1; V7[1, 0] = 1
        W7[0, 0] = 1; W7[3, 0] = 1

        # P2 = (c+d)*e, contributes to C10 (+1)
        U7[2, 1] = 1; U7[3, 1] = 1
        V7[0, 1] = 1
        W7[2, 1] = 1

        # P3 = a*(f-h), contributes to C01 (+1) and C11 (+1)
        U7[0, 2] = 1
        V7[1, 2] = 1; V7[3, 2] = -1
        W7[1, 2] = 1; W7[3, 2] = 1

        # P4 = d*(g-e), contributes to C00 (+1)
        U7[3, 3] = 1
        V7[2, 3] = 1; V7[0, 3] = -1
        W7[0, 3] = 1

        # P5 = (a+d)*(e+h), contributes to C00 (-1), C01 (+1), C10 (+1)
        U7[0, 4] = 1; U7[3, 4] = 1
        V7[0, 4] = 1; V7[3, 4] = 1
        W7[0, 4] = -1; W7[1, 4] = 1; W7[2, 4] = 1

        # P6 = (b-d)*(g+h), contributes to C11 (+1)
        U7[1, 5] = 1; U7[3, 5] = -1
        V7[2, 5] = 1; V7[3, 5] = 1
        W7[3, 5] = 1

        # P7 = (c-a)*(f+g), contributes to C00 (+1) and C10 (+1)
        U7[2, 6] = 1; U7[0, 6] = -1
        V7[1, 6] = 1; V7[2, 6] = 1
        W7[0, 6] = 1; W7[2, 6] = 1

        return U7, V7, W7

    def build_4x4x4_strassen():
        """Build rank-49 decomposition for 4x4 x 4x4 matrix multiplication."""
        U7, V7, W7 = build_strassen_2x2x2()

        U4 = np.zeros((16, 49), dtype=np.float64)
        V4 = np.zeros((16, 49), dtype=np.float64)
        W4 = np.zeros((16, 49), dtype=np.float64)

        A_combos = [
            [(0, 1), (3, 1)],   # P1: A11 + A22
            [(2, 1), (3, 1)],   # P2: A21 + A22
            [(0, 1)],           # P3: A11
            [(3, 1)],           # P4: A22
            [(0, 1), (1, 1)],   # P5: A11 + A12
            [(2, 1), (0, -1)],  # P6: A21 - A11
            [(1, 1), (3, -1)],  # P7: A12 - A22
        ]

        B_combos = [
            [(0, 1), (3, 1)],   # P1: B11 + B22
            [(0, 1)],           # P2: B11
            [(1, 1), (3, -1)],  # P3: B12 - B22
            [(2, 1), (0, -1)],  # P4: B21 - B11
            [(3, 1)],           # P5: B22
            [(0, 1), (1, 1)],   # P6: B11 + B12
            [(2, 1), (3, 1)],   # P7: B21 + B22
        ]

        C_coeffs = [
            [1, 0, 0, -1, 1, 0, -1],   # C11: P1 - P4 + P5 - P7
            [0, 0, 1, 0, 1, 0, 0],     # C12: P3 + P5
            [0, 1, 0, 0, 0, 0, 1],     # C21: P2 + P7
            [1, -1, 1, 0, 0, 1, 0],    # C22: P1 - P2 + P3 + P6
        ]

        block_offsets = [0, 4, 8, 12]

        for pi in range(7):
            for t in range(7):
                col = pi * 7 + t
                for blk, sign in A_combos[pi]:
                    offset = block_offsets[blk]
                    for r in range(4):
                        U4[offset + r, col] += sign * U7[r, t]
                for blk, sign in B_combos[pi]:
                    offset = block_offsets[blk]
                    for r in range(4):
                        V4[offset + r, col] += sign * V7[r, t]
                for cj in range(4):
                    coeff = C_coeffs[cj][pi]
                    if coeff != 0:
                        offset = block_offsets[cj]
                        for r in range(4):
                            W4[offset + r, col] += coeff * W7[r, t]

        return U4, V4, W4

    # Build the base 4x4x8 decomposition (rank 98)
    U4, V4, W4 = build_4x4x4_strassen()
    rank4 = 49
    total_rank_base = 2 * rank4

    U_base = np.zeros((16, total_rank_base), dtype=np.float64)
    V_base = np.zeros((32, total_rank_base), dtype=np.float64)
    W_base = np.zeros((32, total_rank_base), dtype=np.float64)

    for col in range(rank4):
        U_base[:, col] = U4[:, col]
        for j in range(4):
            for k in range(4):
                V_base[j * 8 + k, col] = V4[j * 4 + k, col]
        for k in range(4):
            for i in range(4):
                W_base[k * 4 + i, col] = W4[k * 4 + i, col]

        col2 = rank4 + col
        U_base[:, col2] = U4[:, col]
        for j in range(4):
            for k in range(4):
                V_base[j * 8 + (k + 4), col2] = V4[j * 4 + k, col]
        for k in range(4):
            for i in range(4):
                W_base[(k + 4) * 4 + i, col2] = W4[k * 4 + i, col]

    # Verify base decomposition
    if not verify_decomposition(U_base, V_base, W_base, T, tol=1e-6):
        # Fallback to schoolbook
        U = U_sb.astype(np.float64)
        V = V_sb.astype(np.float64)
        W = W_sb.astype(np.float64)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = total_rank_base
    best_U, best_V, best_W = U_base.copy(), V_base.copy(), W_base.copy()

    # Now try to reduce the rank using iterative optimization
    # Try various ranks below 98
    for target_rank in [96, 94, 92, 90, 88, 86, 84, 82, 80]:
        if time.time() > deadline - 20:
            break
        
        result = try_rank(target_rank, T, 16, 32, 32, max_iters=3000, seed=target_rank)
        if result is not None:
            U_r, V_r, W_r = result
            if verify_decomposition(U_r, V_r, W_r, T, tol=1e-6):
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break  # Found a valid decomposition at this rank
    
    # Also try a few more aggressive ranks with different seeds
    for target_rank in [80, 78, 76, 74, 72, 70, 68, 66, 64]:
        if time.time() > deadline - 10:
            break
        for seed in [target_rank, target_rank + 1000]:
            if time.time() > deadline - 10:
                break
            result = try_rank(target_rank, T, 16, 32, 32, max_iters=2000, seed=seed)
            if result is not None:
                U_r, V_r, W_r = result
                if verify_decomposition(U_r, V_r, W_r, T, tol=1e-6):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                    break
        if best_rank <= target_rank:
            break

    # Final verification
    if not verify_decomposition(best_U, best_V, best_W, T, tol=1e-6):
        best_U, best_V, best_W = U_base.copy(), V_base.copy(), W_base.copy()

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
