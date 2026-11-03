import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    def compute_error(U, V, W, R):
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        M = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.sum((M - 1.0)**2)

    def solve_for_U(V, W, R):
        U = np.zeros((n*m, R))
        V_reshaped = V.reshape(m, p, R)
        W_reshaped = W.reshape(p, n, R)
        for j in range(m):
            for i in range(n):
                A = V_reshaped[j] * W_reshaped[:, i]
                b = np.ones(p)
                try:
                    U[i*m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return U

    def solve_for_V(U, W, R):
        V = np.zeros((m*p, R))
        U_reshaped = U.reshape(n, m, R)
        W_reshaped = W.reshape(p, n, R)
        for j in range(m):
            for k in range(p):
                A = U_reshaped[:, j] * W_reshaped[k]
                b = np.ones(n)
                try:
                    V[j*p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return V

    def solve_for_W(U, V, R):
        W = np.zeros((p*n, R))
        U_reshaped = U.reshape(n, m, R)
        V_reshaped = V.reshape(m, p, R)
        for k in range(p):
            for i in range(n):
                A = U_reshaped[i] * V_reshaped[:, k]
                b = np.ones(m)
                try:
                    W[k*n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return W

    def is_exact_solution(U, V, W, R):
        err = compute_error(U, V, W, R)
        return err < 1e-9

    def round_to_half_int(M):
        return np.round(M * 2) / 2.0

    def polish(U, V, W, R):
        """
        Local search to reduce error after rounding.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        current_err = compute_error(U, V, W, R)
        if current_err < 1e-9:
            return U, V, W, True

        candidates = [i/2.0 for i in range(-12, 13)]
        
        improved = True
        max_iters = 80
        iter_count = 0
        
        while improved and iter_count < max_iters and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # Try flipping in U
            for idx in range(n*m):
                for r in range(R):
                    old_val = U[idx, r]
                    close_candidates = [c for c in candidates if abs(c - old_val) < 2.5]
                    if not close_candidates:
                        close_candidates = [c for c in candidates if abs(c - old_val) < 6.0]
                    
                    for new_val in close_candidates:
                        if np.isclose(old_val, new_val, atol=1e-12):
                            continue
                        U[idx, r] = new_val
                        new_err = compute_error(U, V, W, R)
                        if new_err < current_err - 1e-12:
                            current_err = new_err
                            improved = True
                            break
                        else:
                            U[idx, r] = old_val
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                continue

            # Try flipping in V
            for idx in range(m*p):
                for r in range(R):
                    old_val = V[idx, r]
                    close_candidates = [c for c in candidates if abs(c - old_val) < 2.5]
                    if not close_candidates:
                        close_candidates = [c for c in candidates if abs(c - old_val) < 6.0]
                    
                    for new_val in close_candidates:
                        if np.isclose(old_val, new_val, atol=1e-12):
                            continue
                        V[idx, r] = new_val
                        new_err = compute_error(U, V, W, R)
                        if new_err < current_err - 1e-12:
                            current_err = new_err
                            improved = True
                            break
                        else:
                            V[idx, r] = old_val
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                continue

            # Try flipping in W
            for idx in range(p*n):
                for r in range(R):
                    old_val = W[idx, r]
                    close_candidates = [c for c in candidates if abs(c - old_val) < 2.5]
                    if not close_candidates:
                        close_candidates = [c for c in candidates if abs(c - old_val) < 6.0]
                    
                    for new_val in close_candidates:
                        if np.isclose(old_val, new_val, atol=1e-12):
                            continue
                        W[idx, r] = new_val
                        new_err = compute_error(U, V, W, R)
                        if new_err < current_err - 1e-12:
                            current_err = new_err
                            improved = True
                            break
                        else:
                            W[idx, r] = old_val
                    if improved:
                        break
                if improved:
                    break
                    
        return U, V, W, current_err < 1e-9

    def get_rank28_block_decomposition():
        """
        Constructs a rank-28 decomposition using block-based Strassen + schoolbook.
        This is a reliable baseline that beats the schoolbook rank 30.
        """
        rank = 28
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Helper to add a Strassen block for 2x2 * 2x2
        def add_strassen_block(rank_offset, j_offset, k_offset):
            # A indices
            a_idx = 0 * m + j_offset
            b_idx = 0 * m + (j_offset + 1)
            c_idx = 1 * m + j_offset
            d_idx = 1 * m + (j_offset + 1)
            
            # B indices
            e_idx = j_offset * p + k_offset
            f_idx = j_offset * p + (k_offset + 1)
            g_idx = (j_offset + 1) * p + k_offset
            h_idx = (j_offset + 1) * p + (k_offset + 1)
            
            # Ranks
            r0 = rank_offset
            r1 = rank_offset + 1
            r2 = rank_offset + 2
            r3 = rank_offset + 3
            r4 = rank_offset + 4
            r5 = rank_offset + 5
            r6 = rank_offset + 6
            
            # U (L_A)
            U[a_idx, r0] += 1; U[d_idx, r0] += 1
            U[c_idx, r1] += 1; U[d_idx, r1] += 1
            U[a_idx, r2] += 1
            U[d_idx, r3] += 1
            U[a_idx, r4] += 1; U[b_idx, r4] += 1
            U[c_idx, r5] += 1; U[a_idx, r5] -= 1
            U[b_idx, r6] += 1; U[d_idx, r6] -= 1
            
            # V (L_B)
            V[e_idx, r0] += 1; V[h_idx, r0] += 1
            V[e_idx, r1] += 1
            V[f_idx, r2] += 1; V[h_idx, r2] -= 1
            V[g_idx, r3] += 1; V[e_idx, r3] -= 1
            V[h_idx, r4] += 1
            V[e_idx, r5] += 1; V[f_idx, r5] += 1
            V[g_idx, r6] += 1; V[h_idx, r6] += 1
            
            # W (Coefficients)
            # C_00 (i=0, k=k_offset) -> idx k_offset*n + 0
            W[k_offset * n + 0, r0] += 1
            W[k_offset * n + 0, r3] += 1
            W[k_offset * n + 0, r4] -= 1
            W[k_offset * n + 0, r6] += 1
            
            # C_01 (i=0, k=k_offset+1) -> idx (k_offset+1)*n + 0
            W[(k_offset + 1) * n + 0, r2] += 1
            W[(k_offset + 1) * n + 0, r4] += 1
            
            # C_10 (i=1, k=k_offset) -> idx k_offset*n + 1
            W[k_offset * n + 1, r1] += 1
            W[k_offset * n + 1, r3] += 1
            
            # C_11 (i=1, k=k_offset+1) -> idx (k_offset+1)*n + 1
            W[(k_offset + 1) * n + 1, r0] += 1
            W[(k_offset + 1) * n + 1, r1] -= 1
            W[(k_offset + 1) * n + 1, r2] += 1
            W[(k_offset + 1) * n + 1, r5] += 1

        # Part 1: A_L (2x2) * B_T1 (2x2) -> C_1 (cols 0,1)
        add_strassen_block(0, 0, 0)

        # Part 2: A_L (2x2) * B_T2 (2x2) -> C_2 (cols 2,3)
        add_strassen_block(7, 0, 2)

        # Part 3: A_L (2x2) * B_T3 (2x1) -> C_3 (col 4)
        # Schoolbook for 2x2 * 2x1 = 4 terms
        U[0, 14] += 1
        V[0*5 + 4, 14] += 1
        W[4*2 + 0, 14] += 1
        
        U[1, 15] += 1
        V[1*5 + 4, 15] += 1
        W[4*2 + 0, 15] += 1
        
        U[3, 16] += 1
        V[0*5 + 4, 16] += 1
        W[4*2 + 1, 16] += 1
        
        U[4, 17] += 1
        V[1*5 + 4, 17] += 1
        W[4*2 + 1, 17] += 1

        # Part 4: A_R (2x1) * B_B (1x5) -> C_4 (all cols)
        # A col 2. B row 2.
        # Schoolbook for 2x1 * 1x5 = 10 terms
        U[2, 18] += 1
        V[2*5 + 0, 18] += 1
        W[0*2 + 0, 18] += 1
        
        U[2, 19] += 1
        V[2*5 + 1, 19] += 1
        W[1*2 + 0, 19] += 1
        
        U[2, 20] += 1
        V[2*5 + 2, 20] += 1
        W[2*2 + 0, 20] += 1
        
        U[2, 21] += 1
        V[2*5 + 3, 21] += 1
        W[3*2 + 0, 21] += 1
        
        U[2, 22] += 1
        V[2*5 + 4, 22] += 1
        W[4*2 + 0, 22] += 1
        
        U[5, 23] += 1
        V[2*5 + 0, 23] += 1
        W[0*2 + 1, 23] += 1
        
        U[5, 24] += 1
        V[2*5 + 1, 24] += 1
        W[1*2 + 1, 24] += 1
        
        U[5, 25] += 1
        V[2*5 + 2, 25] += 1
        W[2*2 + 1, 25] += 1
        
        U[5, 26] += 1
        V[2*5 + 3, 26] += 1
        W[3*2 + 1, 26] += 1
        
        U[5, 27] += 1
        V[2*5 + 4, 27] += 1
        W[4*2 + 1, 27] += 1

        return U, V, W

    def get_schoolbook():
        R = n*m*p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # 1. Try to find a rank < 28 solution using the search machinery
    best_rank = 28
    best_U, best_V, best_W = None, None, None
    
    # Initialize with the known rank 28 solution as a fallback
    U28, V28, W28 = get_rank28_block_decomposition()
    
    # Check if rank 28 is valid (should be)
    if is_exact_solution(U28, V28, W28, 28):
        best_rank = 28
        best_U, best_V, best_W = U28, V28, W28
    else:
        # Fallback to schoolbook if the hardcoded one is broken (should not happen)
        best_rank = 30
        best_U, best_V, best_W = get_schoolbook()

    # Search for lower ranks (22-27)
    for R in range(22, 28):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        trials = 1000
        found = False
        for trial in range(trials):
            if time.time() > deadline:
                break
            
            # Use deterministic seed based on trial to ensure reproducibility
            np.random.seed(42 + trial + R)
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Iterate ALS
            for it in range(30):
                if time.time() > deadline:
                    break
                U = solve_for_U(V, W, R)
                V = solve_for_V(U, W, R)
                W = solve_for_W(U, V, R)
                
            # Round and Polish
            U_rounded = round_to_half_int(U)
            V_rounded = round_to_half_int(V)
            W_rounded = round_to_half_int(W)
            
            # Only polish if the error is reasonably small
            err = compute_error(U_rounded, V_rounded, W_rounded, R)
            if err < 5.0:
                U_polished, V_polished, W_polished, is_exact = polish(U_rounded, V_rounded, W_rounded, R)
                if is_exact:
                    best_rank = R
                    best_U, best_V, best_W = U_polished, V_polished, W_polished
                    found = True
                    break
            else:
                # Check if it was already exact before polishing
                if is_exact_solution(U_rounded, V_rounded, W_rounded, R):
                    best_rank = R
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    found = True
                    break
        
        if found:
            break

    if best_U is None:
        # Final fallback
        best_U, best_V, best_W = get_schoolbook()

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
