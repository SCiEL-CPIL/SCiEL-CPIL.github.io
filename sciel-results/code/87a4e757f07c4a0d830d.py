import time
import numpy as np
import itertools
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10

    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def snap_to_half(arr):
        return np.round(arr * 2) / 2

    def snap_to_integer(arr):
        return np.round(arr)

    def try_snap_validate(U, V, W):
        # Try half-integers first
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        # Try integers
        U_i = snap_to_integer(U)
        V_i = snap_to_integer(V)
        W_i = snap_to_integer(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
        # Try quarters
        U_q = np.round(U * 4) / 4
        V_q = np.round(V * 4) / 4
        W_q = np.round(W * 4) / 4
        if validate(U_q, V_q, W_q):
            return True, U_q, V_q, W_q
        return False, None, None, None

    def build_rank46():
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))

        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1

            # P2 = (A10 + A11) B00
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1

            # P3 = A00 (B01 - B11)
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1

            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off * n + 1][c] += 1
            c += 1

            # P5 = (A00 + A01) B11
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1

            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1

            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)

        col = 28
        for i in range(2):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1

        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

        return U, V, W

    def solve_linear_system(supports_U, supports_V, supports_W):
        """
        Given fixed supports (sets of non-zero indices for each column),
        solve for the values U, V, W.
        Returns None if no solution or solution is not close to integer/half-integer.
        """
        num_cols = len(supports_U)
        # Variables: U[a, r], V[b, r], W[c, r] for (a,b,c) in supports
        # Equations: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c] for all a,b,c
        # This is a bilinear system, not linear in U,V,W simultaneously.
        # However, if we fix V and W, it's linear in U.
        # If we fix U and W, it's linear in V.
        # If we fix U and V, it's linear in W.
        # This strategy doesn't work directly for simultaneous solve.
        
        # Alternative: Use the fact that we start from a known solution and perturb supports.
        # This function is a placeholder for a more complex solver.
        pass

    def try_rank_reduction(U_init, V_init, W_init, target_rank, max_attempts=50):
        """
        Try to reduce rank by removing columns and re-optimizing the remaining ones.
        """
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        current_rank = U.shape[1]
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Remove a random column
            col_to_remove = np.random.randint(0, current_rank)
            mask = [i for i in range(current_rank) if i != col_to_remove]
            
            U_new = U[:, mask].copy()
            V_new = V[:, mask].copy()
            W_new = W[:, mask].copy()
            
            # Try to optimize U_new, V_new, W_new to fit T
            # Use a simple gradient descent with restarts
            success = False
            for restart in range(3):
                if time.time() > deadline - 5:
                    break
                U_opt, V_opt, W_opt = U_new.copy(), V_new.copy(), W_new.copy()
                # Add small noise
                U_opt += np.random.randn(*U_opt.shape) * 0.01
                V_opt += np.random.randn(*V_opt.shape) * 0.01
                W_opt += np.random.randn(*W_opt.shape) * 0.01
                
                # Optimize
                lr = 0.05
                for step in range(200):
                    if time.time() > deadline - 5:
                        break
                    prod = np.einsum('ar,br,cr->abc', U_opt, V_opt, W_opt, optimize=True)
                    diff = prod - T
                    loss = np.sum(diff * diff)
                    if loss < 1e-8:
                        break
                    
                    gradU = np.einsum('abc,br,cr->ar', diff, V_opt, W_opt, optimize=True)
                    gradV = np.einsum('abc,ar,cr->br', diff, U_opt, W_opt, optimize=True)
                    gradW = np.einsum('abc,ar,br->cr', diff, U_opt, V_opt, optimize=True)
                    
                    U_opt -= lr * gradU
                    V_opt -= lr * gradV
                    W_opt -= lr * gradW
                    lr *= 0.995
                
                valid, U_s, V_s, W_s = try_snap_validate(U_opt, V_opt, W_opt)
                if valid:
                    return U_s, V_s, W_s, target_rank
                    
            if success:
                break
        
        return None, None, None, current_rank

    # Build and verify rank-46
    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
        # Fallback to naive rank 50
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }

    best_U, best_V, best_W = U46, V46, W46
    best_rank = 46

    np.random.seed(42)

    # Try to reduce to 45
    for target_rank in [45, 44, 43]:
        if time.time() > deadline - 30:
            break
        
        found = False
        attempts = 0
        while not found and attempts < 20 and time.time() < deadline - 10:
            attempts += 1
            U_res, V_res, W_res, res_rank = try_rank_reduction(U46, V46, W46, target_rank, max_attempts=1)
            if U_res is not None and res_rank == target_rank:
                best_U, best_V, best_W = U_res, V_res, W_res
                best_rank = target_rank
                found = True
                break

    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
# EVOLVE-BLOCK-END
