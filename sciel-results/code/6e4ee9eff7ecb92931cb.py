import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

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

    def snap_to_int(arr):
        return np.round(arr)

    def try_snap_validate(U, V, W):
        # Try integers first
        U_i = snap_to_int(U)
        V_i = snap_to_int(V)
        W_i = snap_to_int(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
        # Try half-integers
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        return False, None, None, None

    def build_rank46():
        """Rank-46 decomposition using Strassen-like structure on 2x2 blocks."""
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))

        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + j_off, c] += 1
            U[1 * m + (j_off + 1), c] += 1
            V[j_off * p + k_off, c] += 1
            V[(j_off + 1) * p + (k_off + 1), c] += 1
            W[k_off * n + 0, c] += 1
            W[(k_off + 1) * n + 1, c] += 1
            c += 1

            # P2 = (A10 + A11) B00
            U[1 * m + j_off, c] += 1
            U[1 * m + (j_off + 1), c] += 1
            V[j_off * p + k_off, c] += 1
            W[k_off * n + 1, c] += 1
            W[(k_off + 1) * n + 1, c] -= 1
            c += 1

            # P3 = A00 (B01 - B11)
            U[0 * m + j_off, c] += 1
            V[j_off * p + (k_off + 1), c] += 1
            V[(j_off + 1) * p + (k_off + 1), c] -= 1
            W[(k_off + 1) * n + 0, c] += 1
            W[(k_off + 1) * n + 1, c] += 1
            c += 1

            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1), c] += 1
            V[(j_off + 1) * p + k_off, c] += 1
            V[j_off * p + k_off, c] -= 1
            W[k_off * n + 0, c] += 1
            W[k_off * n + 1, c] += 1
            c += 1

            # P5 = (A00 + A01) B11
            U[0 * m + j_off, c] += 1
            U[0 * m + (j_off + 1), c] += 1
            V[(j_off + 1) * p + (k_off + 1), c] += 1
            W[k_off * n + 0, c] -= 1
            W[(k_off + 1) * n + 0, c] += 1
            c += 1

            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off, c] += 1
            U[0 * m + j_off, c] -= 1
            V[j_off * p + k_off, c] += 1
            V[j_off * p + (k_off + 1), c] += 1
            W[(k_off + 1) * n + 1, c] += 1
            c += 1

            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1), c] += 1
            U[1 * m + (j_off + 1), c] -= 1
            V[(j_off + 1) * p + k_off, c] += 1
            V[(j_off + 1) * p + (k_off + 1), c] += 1
            W[k_off * n + 0, c] += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)

        col = 28
        for i in range(2):
            for k in range(4):
                U[i * m + 4, col] = 1
                V[4 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                if time.time() > deadline - 10:
                    return U, V, W

        for i in range(2):
            for j in range(4):
                U[i * m + j, col] = 1
                V[j * p + 4, col] = 1
                W[4 * n + i, col] = 1
                col += 1
                if time.time() > deadline - 10:
                    return U, V, W

        for i in range(2):
            U[i * m + 4, col] = 1
            V[4 * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1
            if time.time() > deadline - 10:
                return U, V, W

        return U, V, W

    def als_polish(U, V, W, max_iter=200, tol=1e-12):
        """
        Alternating Least Squares polish.
        Fixes two matrices and solves for the third exactly via linear least squares.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            prev_loss = np.sum((np.einsum('ar,br,cr->abc', U, V, W, optimize=True) - T)**2)
            
            # Solve for U
            # T_{abc} = sum_r U_{ar} V_{br} W_{cr}
            # For each a, T_{a, :, :] = sum_r U_{ar} (V[:, r] outer W[:, r])
            # Let M be matrix of shape (V_dim*W_dim, R) where M[:, r] = V[:, r] * W[:, r]
            M_U = np.einsum('br,cr->bcr', V, W).reshape(V_dim * W_dim, -1)
            T_reshaped = T.reshape(U_dim, -1).T # Shape (V_dim*W_dim, U_dim)
            # We want U.T such that M_U @ U.T = T_reshaped
            U_new, residuals, rank, sv = np.linalg.lstsq(M_U, T_reshaped, rcond=None)
            U = U_new.T
            
            # Solve for V
            M_V = np.einsum('ar,cr->acr', U, W).reshape(U_dim * W_dim, -1)
            T_reshaped_v = T.reshape(V_dim, U_dim, W_dim).transpose(1, 2, 0).reshape(U_dim * W_dim, V_dim).T # Careful with axes
            # T_{abc} = sum_r V_{br} (U_{ar} W_{cr})
            # For each b, T_{:, b, :} = sum_r V_{br} (U[:, r] outer W[:, r])
            M_V = np.einsum('ar,cr->acr', U, W).reshape(U_dim * W_dim, -1)
            T_reshaped_v = T.transpose(1, 0, 2).reshape(V_dim, -1).T # Shape (U_dim*W_dim, V_dim)
            V_new, residuals, rank, sv = np.linalg.lstsq(M_V, T_reshaped_v, rcond=None)
            V = V_new.T
            
            # Solve for W
            M_W = np.einsum('ar,br->abr', U, V).reshape(U_dim * V_dim, -1)
            # T_{abc} = sum_r W_{cr} (U_{ar} V_{br})
            # For each c, T_{:, :, c} = sum_r W_{cr} (U[:, r] outer V[:, r])
            M_W = np.einsum('ar,br->abr', U, V).reshape(U_dim * V_dim, -1)
            T_reshaped_w = T.transpose(2, 0, 1).reshape(W_dim, -1).T # Shape (U_dim*V_dim, W_dim)
            W_new, residuals, rank, sv = np.linalg.lstsq(M_W, T_reshaped_w, rcond=None)
            W = W_new.T
            
            curr_loss = np.sum((np.einsum('ar,br,cr->abc', U, V, W, optimize=True) - T)**2)
            if abs(prev_loss - curr_loss) < tol:
                break
            if curr_loss < 1e-14:
                break
                
        return U, V, W

    def fast_optimize(U_init, V_init, W_init, target_rank, max_iter=100, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, lr_decay=0.995):
        """Adam optimizer followed by ALS polish."""
        R = target_rank
        U = U_init[:, :R].copy()
        V = V_init[:, :R].copy()
        W = W_init[:, :R].copy()

        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)

        prev_loss = float('inf')
        no_improve = 0
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        current_lr = lr
        restart_count = 0

        for it in range(max_iter):
            if time.time() > deadline - 20:
                break

            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)

            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            if loss < 1e-8:
                break

            # Stagnation detection
            if loss > prev_loss * 1.005 and no_improve > 10:
                restart_count += 1
                if restart_count > 3:
                    break
                U += np.random.randn(*U.shape) * 0.01
                V += np.random.randn(*V.shape) * 0.01
                W += np.random.randn(*W.shape) * 0.01
                mU = np.zeros_like(U)
                vU = np.zeros_like(U)
                mV = np.zeros_like(V)
                vV = np.zeros_like(V)
                mW = np.zeros_like(W)
                vW = np.zeros_like(W)
                no_improve = 0
                prev_loss = loss
                continue
            else:
                no_improve += 1
                prev_loss = loss

            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)

            t = it + 1
            mU = beta1 * mU + (1 - beta1) * gradU
            vU = beta2 * vU + (1 - beta2) * (gradU ** 2)
            mU_hat = mU / (1 - beta1 ** t)
            vU_hat = vU / (1 - beta2 ** t)
            U -= current_lr * mU_hat / (np.sqrt(vU_hat) + eps)

            mV = beta1 * mV + (1 - beta1) * gradV
            vV = beta2 * vV + (1 - beta2) * (gradV ** 2)
            mV_hat = mV / (1 - beta1 ** t)
            vV_hat = vV / (1 - beta2 ** t)
            V -= current_lr * mV_hat / (np.sqrt(vV_hat) + eps)

            mW = beta1 * mW + (1 - beta1) * gradW
            vW = beta2 * vW + (1 - beta2) * (gradW ** 2)
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            W -= current_lr * mW_hat / (np.sqrt(vW_hat) + eps)

            current_lr *= lr_decay

        # If Adam got close, use ALS to polish to high precision
        if best_loss < 1e-3:
            U_als, V_als, W_als = als_polish(best_U, best_V, best_W, max_iter=50)
            prod = np.einsum('ar,br,cr->abc', U_als, V_als, W_als, optimize=True)
            als_loss = np.sum((prod - T)**2)
            if als_loss < best_loss:
                best_U, best_V, best_W = U_als, V_als, W_als
                best_loss = als_loss

        valid, U_s, V_s, W_s = try_snap_validate(best_U, best_V, best_W)
        if valid:
            return U_s, V_s, W_s

        return None, None, None

    # Build and verify rank-46
    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
        # Fallback to standard rank 50
        rank = 50
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }

    best_result = None
    np.random.seed(42)

    # Calculate column importance for removal
    col_scores = []
    support_U = [set(np.nonzero(U46[:, c])[0]) for c in range(46)]
    support_V = [set(np.nonzero(V46[:, c])[0]) for c in range(46)]
    support_W = [set(np.nonzero(W46[:, c])[0]) for c in range(46)]
    
    for c in range(46):
        support_size = len(support_U[c]) + len(support_V[c]) + len(support_W[c])
        overlap_count = 0
        for c2 in range(46):
            if c2 == c: continue
            overlap = len(support_U[c] & support_U[c2]) + len(support_V[c] & support_V[c2]) + len(support_W[c] & support_W[c2])
            if overlap > 3:
                overlap_count += 1
        col_scores.append(support_size + overlap_count * 0.5)
        
    col_order = np.argsort(col_scores)

    # Try Rank 45
    if time.time() < deadline - 30:
        for attempt in range(min(10, len(col_order))):
            if time.time() > deadline - 25: break
            col_to_zero = col_order[attempt]
            mask = [i for i in range(46) if i != col_to_zero]
            U_test = U46[:, mask].copy()
            V_test = V46[:, mask].copy()
            W_test = W46[:, mask].copy()
            
            for seed in [42, 123, 999]:
                if time.time() > deadline - 20: break
                np.random.seed(seed + attempt)
                U_try = U_test + np.random.randn(*U_test.shape) * 0.05
                V_try = V_test + np.random.randn(*V_test.shape) * 0.05
                W_try = W_test + np.random.randn(*W_test.shape) * 0.05
                
                U_opt, V_opt, W_opt = fast_optimize(U_try, V_try, W_try, 45, max_iter=80)
                if U_opt is not None:
                    best_result = (U_opt, V_opt, W_opt)
                    break
            if best_result: break

    # Try Rank 44
    if best_result is None and time.time() < deadline - 40:
        # Try removing pairs
        pair_candidates = []
        for i in range(len(col_order)):
            for j in range(i+1, len(col_order)):
                c1, c2 = col_order[i], col_order[j]
                overlap = len(support_U[c1] & support_U[c2]) + len(support_V[c1] & support_V[c2]) + len(support_W[c1] & support_W[c2])
                pair_candidates.append((overlap, c1, c2))
        pair_candidates.sort(reverse=True)
        
        for idx in range(min(10, len(pair_candidates))):
            if time.time() > deadline - 35: break
            _, c1, c2 = pair_candidates[idx]
            mask = [i for i in range(46) if i not in (c1, c2)]
            U_test = U46[:, mask].copy()
            V_test = V46[:, mask].copy()
            W_test = W46[:, mask].copy()
            
            for seed in [42, 123, 999]:
                if time.time() > deadline - 30: break
                np.random.seed(seed + idx)
                U_try = U_test + np.random.randn(*U_test.shape) * 0.05
                V_try = V_test + np.random.randn(*V_test.shape) * 0.05
                W_try = W_test + np.random.randn(*W_test.shape) * 0.05
                
                U_opt, V_opt, W_opt = fast_optimize(U_try, V_try, W_opt, 44, max_iter=80)
                if U_opt is not None:
                    best_result = (U_opt, V_opt, W_opt)
                    break
            if best_result: break

    if best_result is not None:
        U_r, V_r, W_r = best_result
        if validate(U_r, V_r, W_r):
            return {
                "U": [[float(x) for x in row] for row in U_r],
                "V": [[float(x) for x in row] for row in V_r],
                "W": [[float(x) for x in row] for row in W_r]
            }

    return {
        "U": [[float(x) for x in row] for row in U46],
        "V": [[float(x) for x in row] for row in V46],
        "W": [[float(x) for x in row] for row in W46]
    }
# EVOLVE-BLOCK-END
