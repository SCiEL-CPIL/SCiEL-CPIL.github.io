import time
import numpy as np

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
    
    def snap_to_quarter(arr):
        return np.round(arr * 4) / 4
    
    def try_snap_validate(U, V, W):
        # Try half-integers first
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        # Try integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
        # Try quarters
        U_q = snap_to_quarter(U)
        V_q = snap_to_quarter(V)
        W_q = snap_to_quarter(W)
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

    def fast_optimize(U_init, V_init, W_init, target_rank, max_iter=150, lr=0.06, seed=42):
        """Faster Adam optimizer with aggressive early stopping."""
        R = target_rank
        U = U_init[:, :R].copy()
        V = V_init[:, :R].copy()
        W = W_init[:, :R].copy()
        
        np.random.seed(seed)
        
        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        prev_loss = float('inf')
        no_improve = 0
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        current_lr = lr
        restart_count = 0
        
        for it in range(max_iter):
            if time.time() > deadline - 8:
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
            if loss > prev_loss * 1.02 and no_improve > 15:
                restart_count += 1
                if restart_count > 5:
                    break
                scale = 0.03 * (1 + restart_count * 0.3)
                U += np.random.randn(*U.shape) * scale
                V += np.random.randn(*V.shape) * scale
                W += np.random.randn(*W.shape) * scale
                mU = np.zeros_like(U)
                vU = np.zeros_like(U)
                mV = np.zeros_like(V)
                vV = np.zeros_like(V)
                mW = np.zeros_like(W)
                vW = np.zeros_like(W)
                no_improve = 0
                prev_loss = loss
                current_lr = max(0.02, current_lr * 0.8)
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
            
            current_lr *= 0.997
        
        valid, U_s, V_s, W_s = try_snap_validate(best_U, best_V, best_W)
        if valid:
            return U_s, V_s, W_s, best_loss
        
        # Final polish if close
        if best_loss < 1e-3:
            # Simple gradient descent polish
            for _ in range(50):
                if time.time() > deadline - 6:
                    break
                prod = np.einsum('ar,br,cr->abc', best_U, best_V, best_W, optimize=True)
                diff = prod - T
                loss = np.sum(diff * diff)
                if loss < 1e-12:
                    break
                gradU = np.einsum('abc,br,cr->ar', diff, best_V, best_W, optimize=True)
                gradV = np.einsum('abc,ar,cr->br', diff, best_U, best_W, optimize=True)
                gradW = np.einsum('abc,ar,br->cr', diff, best_U, best_V, optimize=True)
                best_U -= 0.005 * gradU
                best_V -= 0.005 * gradV
                best_W -= 0.005 * gradW
            
            valid, U_s, V_s, W_s = try_snap_validate(best_U, best_V, best_W)
            if valid:
                return U_s, V_s, W_s, best_loss
        
        return None, None, None, best_loss

    # Build and verify rank-46
    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
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

    best_result = None
    best_rank = 46
    
    # Compute column importance using a more refined metric
    col_contributions = []
    for c in range(46):
        contrib = np.einsum('a,b,c->abc', U46[:, c], V46[:, c], W46[:, c], optimize=True)
        col_contributions.append(np.sum(contrib * contrib))
    
    support_U = [set(np.nonzero(U46[:, c])[0]) for c in range(46)]
    support_V = [set(np.nonzero(V46[:, c])[0]) for c in range(46)]
    support_W = [set(np.nonzero(W46[:, c])[0]) for c in range(46)]
    
    # Score columns: lower = more removable
    col_scores = []
    for c in range(46):
        support_size = len(support_U[c]) + len(support_V[c]) + len(support_W[c])
        overlap_count = 0
        for c2 in range(46):
            if c2 == c:
                continue
            overlap = len(support_U[c] & support_U[c2]) + len(support_V[c] & support_V[c2]) + len(support_W[c] & support_W[c2])
            if overlap > 2:
                overlap_count += 1
        score = col_contributions[c] * 0.05 + support_size * 0.3 + overlap_count * 0.4
        col_scores.append(score)
    
    col_order = np.argsort(col_scores)
    
    # Strategy 1: Try removing single columns (rank 45)
    for attempt in range(min(20, 46)):
        if time.time() > deadline - 30:
            break
        
        col_to_zero = col_order[attempt]
        mask = [i for i in range(46) if i != col_to_zero]
        U_test = U46[:, mask].copy()
        V_test = V46[:, mask].copy()
        W_test = W46[:, mask].copy()
        
        found = False
        for seed in [42, 123]:
            if time.time() > deadline - 28:
                break
            np.random.seed(seed + attempt * 100)
            for noise_scale in [0.005, 0.02, 0.05]:
                if time.time() > deadline - 26:
                    break
                U_try = U_test.copy()
                V_try = V_test.copy()
                W_try = W_test.copy()
                U_try += np.random.randn(*U_try.shape) * noise_scale
                V_try += np.random.randn(*V_try.shape) * noise_scale
                W_try += np.random.randn(*W_try.shape) * noise_scale
                
                U_opt, V_opt, W_opt, loss = fast_optimize(U_try, V_try, W_try, 45, max_iter=120, lr=0.05)
                
                if U_opt is not None:
                    best_result = (U_opt, V_opt, W_opt)
                    best_rank = 45
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break
    
    # Strategy 2: Try removing pairs (rank 44)
    if best_result is None and time.time() < deadline - 40:
        pair_candidates = []
        for i in range(min(12, len(col_order))):
            for j in range(i+1, min(12, len(col_order))):
                c1 = col_order[i]
                c2 = col_order[j]
                overlap = (len(support_U[c1] & support_U[c2]) + 
                          len(support_V[c1] & support_V[c2]) + 
                          len(support_W[c1] & support_W[c2]))
                combined_score = col_scores[i] + col_scores[j]
                pair_candidates.append((overlap, combined_score, c1, c2))
        
        pair_candidates.sort(key=lambda x: (-x[0], x[1]))
        
        for idx in range(min(10, len(pair_candidates))):
            if time.time() > deadline - 35:
                break
            _, _, c1, c2 = pair_candidates[idx]
            mask = [i for i in range(46) if i != c1 and i != c2]
            U_test = U46[:, mask].copy()
            V_test = V46[:, mask].copy()
            W_test = W46[:, mask].copy()
            
            for seed in [42, 123]:
                if time.time() > deadline - 33:
                    break
                np.random.seed(seed + idx * 10)
                for noise_scale in [0.01, 0.03]:
                    if time.time() > deadline - 31:
                        break
                    U_try = U_test.copy()
                    V_try = V_test.copy()
                    W_try = W_test.copy()
                    U_try += np.random.randn(*U_try.shape) * noise_scale
                    V_try += np.random.randn(*V_try.shape) * noise_scale
                    W_try += np.random.randn(*W_try.shape) * noise_scale
                    
                    U_opt, V_opt, W_opt, loss = fast_optimize(U_try, V_try, W_try, 44, max_iter=100, lr=0.04)
                    
                    if U_opt is not None:
                        best_result = (U_opt, V_opt, W_opt)
                        best_rank = 44
                        break
                if best_result is not None:
                    break
            if best_result is not None:
                break
    
    if best_result is not None:
        U_r, V_r, W_r = best_result
        if validate(U_r, V_r, W_r):
            return {
                "U": [[float(x) for x in row] for row in U_r],
                "V": [[float(x) for x in row] for row in V_r],
                "W": [[float(x) for x in row] for row in W_r]
            }
    
    # Return rank-46 as fallback
    return {
        "U": [[float(x) for x in row] for row in U46],
        "V": [[float(x) for x in row] for row in V46],
        "W": [[float(x) for x in row] for row in W46]
    }
# EVOLVE-BLOCK-END
