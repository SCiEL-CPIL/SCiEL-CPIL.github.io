import numpy as np
import time
import random

def verify_decomposition_233(U, V, W):
    """Verify rank-r decomposition for 2x3x3 matrix multiplication."""
    # U: (6, r), V: (9, r), W: (6, r)
    # A: 2x3 (idx 0..5), B: 3x3 (idx 0..8), C: 2x3 (idx 0..5)
    # C index: k*2 + i
    r = U.shape[1]
    for a in range(2):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    for i in range(2):
                        for k in range(3):
                            target = 1.0 if (a == i and c == k and b == d) else 0.0
                            s = np.sum(U[a * 3 + b] * V[c * 3 + d] * W[k * 2 + i])
                            if not np.isclose(s, target, atol=1e-9):
                                return False
    return True

def verify_decomposition_236(U, V, W):
    """Verify rank-r decomposition for 2x3x6 matrix multiplication."""
    # U: (6, r), V: (18, r), W: (12, r)
    # A: 2x3 (idx 0..5), B: 3x6 (idx 0..17), C: 2x6 (idx 0..11)
    # C index: k*2 + i
    r = U.shape[1]
    for a in range(2):
        for b in range(3):
            for c in range(3):
                for d in range(6):
                    for i in range(2):
                        for k in range(6):
                            target = 1.0 if (a == i and c == k and b == d) else 0.0
                            s = np.sum(U[a * 3 + b] * V[c * 6 + d] * W[k * 2 + i])
                            if not np.isclose(s, target, atol=1e-9):
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 3, 6

    # Schoolbook fallback (rank 36)
    def schoolbook():
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Verified rank-34: two copies of rank-17 2x3x3
    def rank34_decomposition():
        rank = 34
        U = np.zeros((6, rank))
        V = np.zeros((18, rank))
        W = np.zeros((12, rank))

        # Rank-17 decomposition for 2x3x3
        # A: 2x3 (indices 0..5), B: 3x3 (indices 0..8), C: 2x3 (indices 0..5)
        
        # 2x2x2 Strassen (7 terms) for A(0,1)x(0,1), B(0,1)x(0,1) -> C(0,1)x(0,1)
        # U: A00=0, A01=1, A10=2, A11=3
        # V: B00=0, B01=1, B10=2, B11=3
        # W: C00=0, C10=1, C01=2, C11=3

        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1

        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1

        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1

        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1

        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1

        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1

        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1

        # Column 2 of B: A(0,1,2) x B col 2 -> C col 2
        U[0, 7] = 1
        V[4, 7] = 1
        W[4, 7] = 1

        U[1, 8] = 1
        V[5, 8] = 1
        W[4, 8] = 1

        U[2, 9] = 1
        V[4, 9] = 1
        W[5, 9] = 1

        U[3, 10] = 1
        V[5, 10] = 1
        W[5, 10] = 1

        # Row 2 of B: A(0,1,2) x B row 2 -> C(0,1)
        U[4, 11] = 1
        V[6, 11] = 1
        W[0, 11] = 1

        U[4, 12] = 1
        V[7, 12] = 1
        W[2, 12] = 1

        U[5, 13] = 1
        V[6, 13] = 1
        W[1, 13] = 1

        U[5, 14] = 1
        V[7, 14] = 1
        W[3, 14] = 1

        U[4, 15] = 1
        V[8, 15] = 1
        W[4, 15] = 1

        U[5, 16] = 1
        V[8, 16] = 1
        W[5, 16] = 1

        # Verify 2x3x3
        if verify_decomposition_233(U[:, :17], V[:, :17], W[:, :17]):
            # Combine two 2x3x3 blocks for 2x3x6
            U_comb = np.zeros((6, 34))
            U_comb[:, :17] = U[:, :17]
            U_comb[:, 17:] = U[:, :17]

            V_comb = np.zeros((18, 34))
            for j in range(3):
                for k in range(3):
                    V_comb[j * 6 + k, :17] = V[j * 3 + k, :17]
                    V_comb[j * 6 + 3 + k, 17:] = V[j * 3 + k, :17]

            W_comb = np.zeros((12, 34))
            for k in range(3):
                for i in range(2):
                    W_comb[k * 2 + i, :17] = W[k * 2 + i, :17]
                    W_comb[(k + 3) * 2 + i, 17:] = W[k * 2 + i, :17]

            if verify_decomposition_236(U_comb, V_comb, W_comb):
                return {"U": U_comb.tolist(), "V": V_comb.tolist(), "W": W_comb.tolist()}

        return None

    # Try rank-34 first (verified approach)
    result = rank34_decomposition()
    if result is not None:
        return result

    if time.time() > deadline - 20:
        return schoolbook()

    # Structural Change: Search for Rank-11 2x3x3 decomposition
    # If found, construct Rank-22 2x3x6 decomposition.
    
    def search_rank11_233():
        """
        Search for a rank-11 decomposition of 2x3x3 matrix multiplication.
        Uses random restarts with local search (Nelder-Mead) on the 33x33 space.
        """
        rank = 11
        # Total variables: 6*11 + 9*11 + 6*11 = 66 + 99 + 66 = 231 variables
        
        def objective(x):
            U = x[:66].reshape(6, 11)
            V = x[66:165].reshape(9, 11)
            W = x[165:].reshape(6, 11)
            error = 0.0
            for a in range(2):
                for b in range(3):
                    for c in range(3):
                        for d in range(3):
                            for i in range(2):
                                for k in range(3):
                                    target = 1.0 if (a == i and c == k and b == d) else 0.0
                                    s = np.sum(U[a * 3 + b] * V[c * 3 + d] * W[k * 2 + i])
                                    error += (s - target) ** 2
            return error

        # Use the first 11 columns of the rank-17 solution as initial guess
        # This is a good starting point because it's already "close"
        U_init = np.zeros((6, 11))
        V_init = np.zeros((9, 11))
        W_init = np.zeros((6, 11))
        
        # Copy from rank-17 (first 11 terms)
        # We need to reconstruct the rank-17 matrices locally or hardcode them
        # Let's hardcode the rank-17 part again for initialization
        U17 = np.zeros((6, 17))
        V17 = np.zeros((9, 17))
        W17 = np.zeros((6, 17))
        
        U17[0, 0] = 1; U17[3, 0] = 1
        V17[0, 0] = 1; V17[3, 0] = 1
        W17[0, 0] = 1; W17[3, 0] = 1

        U17[2, 1] = 1; U17[3, 1] = 1
        V17[0, 1] = 1
        W17[1, 1] = 1

        U17[0, 2] = 1
        V17[1, 2] = 1; V17[3, 2] = -1
        W17[2, 2] = 1; W17[3, 2] = 1

        U17[3, 3] = 1
        V17[2, 3] = 1; V17[0, 3] = -1
        W17[0, 3] = 1; W17[1, 3] = 1

        U17[0, 4] = 1; U17[1, 4] = 1
        V17[3, 4] = 1
        W17[0, 4] = -1; W17[2, 4] = 1

        U17[2, 5] = 1; U17[0, 5] = -1
        V17[0, 5] = 1; V17[1, 5] = 1
        W17[3, 5] = 1

        U17[1, 6] = 1; U17[3, 6] = -1
        V17[2, 6] = 1; V17[3, 6] = 1
        W17[0, 6] = 1; W17[3, 6] = 1

        U17[0, 7] = 1
        V17[4, 7] = 1
        W17[4, 7] = 1

        U17[1, 8] = 1
        V17[5, 8] = 1
        W17[4, 8] = 1

        U17[2, 9] = 1
        V17[4, 9] = 1
        W17[5, 9] = 1

        U17[3, 10] = 1
        V17[5, 10] = 1
        W17[5, 10] = 1
        
        # Remaining 6 terms in rank-17
        U17[4, 11] = 1
        V17[6, 11] = 1
        W17[0, 11] = 1

        U17[4, 12] = 1
        V17[7, 12] = 1
        W17[2, 12] = 1

        U17[5, 13] = 1
        V17[6, 13] = 1
        W17[1, 13] = 1

        U17[5, 14] = 1
        V17[7, 14] = 1
        W17[3, 14] = 1

        U17[4, 15] = 1
        V17[8, 15] = 1
        W17[4, 15] = 1

        U17[5, 16] = 1
        V17[8, 16] = 1
        W17[5, 16] = 1

        U_init = U17[:, :11]
        V_init = V17[:, :11]
        W_init = W17[:, :11]
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Try a few random restarts with perturbations
        for restart in range(5):
            if time.time() > deadline - 10:
                break
                
            x_start = x0.copy()
            if restart > 0:
                # Add small noise
                noise = np.random.normal(0, 0.1, x_start.shape)
                x_start = x_start + noise
            
            # Use Nelder-Mead
            try:
                from scipy.optimize import minimize
                res = minimize(objective, x_start, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
                x_opt = res.x
                U_opt = x_opt[:66].reshape(6, 11)
                V_opt = x_opt[66:165].reshape(9, 11)
                W_opt = x_opt[165:].reshape(6, 11)
                
                # Round to nearest half-integer
                U_round = np.round(2 * U_opt) / 2
                V_round = np.round(2 * V_opt) / 2
                W_round = np.round(2 * W_opt) / 2
                
                if verify_decomposition_233(U_round, V_round, W_round):
                    return U_round, V_round, W_round
                    
                # Try integer rounding
                U_int = np.round(U_opt)
                V_int = np.round(V_opt)
                W_int = np.round(W_opt)
                if verify_decomposition_233(U_int, V_int, W_int):
                    return U_int, V_int, W_int
                    
            except Exception:
                continue
                
        return None

    # Attempt to find rank-11 2x3x3
    decomp_233 = search_rank11_233()
    
    if decomp_233 is not None:
        U11, V11, W11 = decomp_233
        # Construct rank-22 2x3x6
        U_comb = np.zeros((6, 22))
        U_comb[:, :11] = U11
        U_comb[:, 11:] = U11
        
        V_comb = np.zeros((18, 22))
        for j in range(3):
            for k in range(3):
                V_comb[j * 6 + k, :11] = V11[j * 3 + k, :11]
                V_comb[j * 6 + 3 + k, 11:] = V11[j * 3 + k, :11]
                
        W_comb = np.zeros((12, 22))
        for k in range(3):
            for i in range(2):
                W_comb[k * 2 + i, :11] = W11[k * 2 + i, :11]
                W_comb[(k + 3) * 2 + i, 11:] = W11[k * 2 + i, :11]
                
        if verify_decomposition_236(U_comb, V_comb, W_comb):
            return {"U": U_comb.tolist(), "V": V_comb.tolist(), "W": W_comb.tolist()}

    # Final fallback
    return schoolbook()
