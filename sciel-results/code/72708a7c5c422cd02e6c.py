import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    # Build the target tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

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

    def strassen_2x2_decomp():
        """Returns U, V, W for 2x2 * 2x2 -> 2x2 with rank 7."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # Indices: A00=0, A01=1, A10=2, A11=3
        #           B00=0, B01=1, B10=2, B11=3
        #           C00=0, C01=1, C10=2, C11=3
        
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

    U_s, V_s, W_s = strassen_2x2_decomp()

    # Build the full rank-28 decomposition
    full_rank = 28
    U_full = np.zeros((n * m, full_rank))
    V_full = np.zeros((m * p, full_rank))
    W_full = np.zeros((p * n, full_rank))

    # Block structure:
    # A is 2x4, B is 4x4.
    # A = [A_LL, A_LR] (2x2 each)
    # B = [B_TL, B_TR; B_BL, B_BR] (2x2 each)
    # C = A * B = [A_LL B_TL + A_LR B_BL, A_LL B_TR + A_LR B_BR]
    
    # Block 1: A_LL * B_TL -> C_LL
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

    # Block 2: A_LR * B_BL -> C_LL
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

    # Block 3: A_LL * B_TR -> C_LR
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

    # Block 4: A_LR * B_BR -> C_LR
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

    if not verify(U_full, V_full, W_full, tol=1e-8):
        # Fallback schoolbook if Strassen construction failed (should not happen)
        def build_schoolbook():
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
            return U, V, W
        U, V, W = build_schoolbook()
        return to_lists(U, V, W, n * m * p)

    best_rank = 28
    best_U, best_V, best_W = U_full, V_full, W_full

    def try_reduce_rank(target_rank, max_restarts=20):
        """
        Attempts to find a decomposition of target_rank using least_squares.
        Returns (U, V, W) if successful, else None.
        """
        if time.time() > deadline - 10:
            return None
            
        size_U = n * m * target_rank
        size_V = m * p * target_rank
        size_W = p * n * target_rank
        
        # Function to calculate residuals
        def residuals(params):
            U = params[:size_U].reshape(n*m, target_rank)
            V = params[size_U:size_U+size_V].reshape(m*p, target_rank)
            W = params[size_U+size_V:].reshape(p*n, target_rank)
            recon = np.einsum('at,bt,ct->abc', U, V, W)
            return (recon - T).flatten()

        # Strategy: Start from rank 28, drop 'target_rank' terms? No, we need exactly target_rank terms.
        # We will try to initialize with a subset of the rank-28 terms or a perturbed version.
        
        # Candidate drops: We try dropping different sets of (28 - target_rank) terms.
        # For rank 27, we drop 1 term.
        # For rank 26, we drop 2 terms.
        
        drops_needed = 28 - target_rank
        
        # Generate combinations of drops
        # To save time, we prioritize drops that might be symmetric or redundant.
        # In the block structure, blocks 0 and 1 contribute to C_LL, blocks 2 and 3 to C_LR.
        # Terms within a block are Strassen terms.
        
        import itertools
        
        # Pre-compute indices for each block's terms
        block_indices = []
        for b in range(4):
            block_indices.append(list(range(b*7, b*7+7)))
            
        # We will try dropping 1 term from each block first, then pairs, etc.
        # But for efficiency, let's just try random drops and structured drops.
        
        attempts = []
        
        # Structured drops: drop one term from each of the 4 blocks? No, that's 4 drops.
        # If target_rank == 27, drop 1 term.
        # If target_rank == 26, drop 2 terms.
        
        if drops_needed == 1:
            # Try dropping each of the 28 terms
            for i in range(28):
                attempts.append([i])
        elif drops_needed == 2:
            # Try dropping pairs. Focus on pairs within the same block or corresponding terms in different blocks.
            # Corresponding terms: t in block 0 and t in block 1.
            for t in range(7):
                attempts.append([t, 7+t]) # Block 0 and 1
                attempts.append([14+t, 21+t]) # Block 2 and 3
                attempts.append([t, 14+t]) # Block 0 and 2
                attempts.append([7+t, 21+t]) # Block 1 and 3
            # Also try random pairs
            for _ in range(20):
                pair = np.random.choice(28, 2, replace=False)
                attempts.append(list(pair))
        else:
            # General random drops
            for _ in range(50):
                drops = np.random.choice(28, drops_needed, replace=False)
                attempts.append(list(drops))

        # Shuffle attempts to mix structured and random
        np.random.shuffle(attempts)

        for attempt_idx, drops in enumerate(attempts):
            if time.time() > deadline - 5:
                break
                
            # Construct initial guess
            keep_indices = [i for i in range(28) if i not in drops]
            
            U_init = U_full[:, keep_indices]
            V_init = V_full[:, keep_indices]
            W_init = W_full[:, keep_indices]
            
            # Add small noise to escape local minima
            noise_scale = 0.01
            U_init = U_init + np.random.randn(n*m, target_rank) * noise_scale
            V_init = V_init + np.random.randn(m*p, target_rank) * noise_scale
            W_init = W_init + np.random.randn(p*n, target_rank) * noise_scale
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                from scipy.optimize import least_squares
                # LM is fast but can be sensitive. Use 'lm' for small problems.
                res = least_squares(residuals, x0, method='lm', max_nfev=1000)
                
                if res.success and res.cost < 1e-10:
                    # Check if we can round to half-integers
                    U_opt = res.x[:size_U].reshape(n*m, target_rank)
                    V_opt = res.x[size_U:size_U+size_V].reshape(m*p, target_rank)
                    W_opt = res.x[size_U+size_V:].reshape(p*n, target_rank)
                    
                    # Round to nearest 0.5
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    if verify(U_round, V_round, W_round, tol=1e-8):
                        return U_round, V_round, W_round
                    elif verify(U_opt, V_opt, W_opt, tol=1e-8):
                        # If it doesn't round nicely but is exact, it might be valid if evaluator allows floats?
                        # Task says "integer/half-integer entries". So we must round.
                        # If rounding fails verification, try to adjust?
                        # For now, return None if rounding fails.
                        pass
                        
            except Exception:
                continue
                
        return None

    # Try rank 27
    if time.time() < deadline - 20:
        res_27 = try_reduce_rank(27, max_restarts=30)
        if res_27 is not None:
            best_rank = 27
            best_U, best_V, best_W = res_27

    # Try rank 26 if time permits
    if time.time() < deadline - 10 and best_rank == 27:
        res_26 = try_reduce_rank(26, max_restarts=20)
        if res_26 is not None:
            best_rank = 26
            best_U, best_V, best_W = res_26

    return to_lists(best_U, best_V, best_W, best_rank)

# EVOLVE-BLOCK-END
