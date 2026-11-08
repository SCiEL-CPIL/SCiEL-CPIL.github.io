import time
import numpy as np

DIMENSIONS = [2, 5, 5]

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
        
    def try_snap_validate(U, V, W):
        # Try integers first (simplest)
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
        # Try half-integers
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        return False, None, None, None

    def build_rank50():
        """Build the trivial rank-50 decomposition."""
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
        return U, V, W

    def build_rank46():
        """Build the rank-46 Strassen-based decomposition."""
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
                if time.time() > deadline - 10:
                    return U, V, W
                
        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10:
                    return U, V, W
                
        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            if time.time() > deadline - 10:
                return U, V, W
                
        return U, V, W

    def solve_milp_fixed_V(R, V_fixed, time_budget=30):
        """
        Given a fixed V matrix (with ±1 entries, one per column),
        solve for U and W with integer/half-integer entries.
        
        The constraint is: for each (u,v,w) with T[u,v,w]=1:
            sum_r U[u,r] * V[v,r] * W[w,r] = 1
        
        With V fixed, this becomes: for each (u,w) pair that appears in T:
            sum_{r: V[v_uw,r] != 0} U[u,r] * V[v_uw,r] * W[w,r] = 1
            
        But actually each (u,v,w) is a separate constraint. Let's reformulate:
        For each target entry (u,v,w) with T[u,v,w]=1:
            sum_r U[u,r] * V[v,r] * W[w,r] = 1
        
        And for all other (u,v,w) with T[u,v,w]=0:
            sum_r U[u,r] * V[v,r] * W[w,r] = 0
        
        This is a system of bilinear equations. With V fixed, each equation
        is linear in the products U[u,r]*W[w,r].
        
        Strategy: For each column r, V[v,r] is ±1 or 0. If V[v,r]=1, then
        the term is U[u,r]*W[w,r]. If V[v,r]=-1, it's -U[u,r]*W[w,r].
        
        We'll use a greedy/iterative approach: fix U to be sparse (±1 or 0),
        then solve for W, and iterate.
        """
        elapsed_start = time.time()
        
        # Determine which (u,v,w) triples have T=1
        target_triples = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    target_triples.append((u_idx, v_idx, w_idx))
        
        # For each column r, find which v has nonzero V
        # V_fixed shape: (V_dim, R)
        # For each r, let v_r be the index where V_fixed[v_r, r] != 0
        v_nonzero = {}
        for r in range(R):
            nz = np.nonzero(V_fixed[:, r])[0]
            if len(nz) == 0:
                return None
            # Allow up to 2 nonzeros per column for more flexibility
            v_nonzero[r] = nz.tolist()
        
        # Initialize U and W with small random values
        np.random.seed(42)
        U = np.zeros((U_dim, R))
        W = np.zeros((W_dim, R))
        
        # Set U and W to have ±1 at random positions
        for r in range(R):
            # Pick random positions for U and W nonzeros
            u_pos = np.random.randint(0, U_dim)
            w_pos = np.random.randint(0, W_dim)
            U[u_pos, r] = np.random.choice([-1, 1])
            W[w_pos, r] = np.random.choice([-1, 1])
        
        best_U, best_W = U.copy(), W.copy()
        best_loss = float('inf')
        
        for iteration in range(2000):
            if time.time() - elapsed_start > time_budget:
                break
            
            # Compute current product
            prod = np.einsum('ar,br,cr->abc', U, V_fixed, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_W = U.copy(), W.copy()
            
            if loss < 1e-10:
                break
            
            # Gradient descent on U and W
            gradU = np.einsum('abc,br,cr->ar', diff, V_fixed, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V_fixed, optimize=True)
            
            # Adaptive step
            lr = 0.01
            U -= lr * gradU
            W -= lr * gradW
            
            # Occasionally snap to nearest integer/half-integer
            if iteration % 50 == 0:
                U_snap = snap_to_half(U)
                W_snap = snap_to_half(W)
                prod_snap = np.einsum('ar,br,cr->abc', U_snap, V_fixed, W_snap, optimize=True)
                loss_snap = np.sum((prod_snap - T) ** 2)
                if loss_snap < loss:
                    U = U_snap
                    W = W_snap
        
        # Check if best solution is valid
        prod_best = np.einsum('ar,br,cr->abc', best_U, V_fixed, best_W, optimize=True)
        if np.allclose(prod_best, T, atol=1e-6):
            return best_U, best_W
        
        # Try snapping the best
        U_s = snap_to_half(best_U)
        W_s = snap_to_half(best_W)
        prod_s = np.einsum('ar,br,cr->abc', U_s, V_fixed, W_s, optimize=True)
        if np.allclose(prod_s, T, atol=1e-6):
            return U_s, W_s
        
        return None

    def search_fixed_V(R, time_budget=60):
        """
        Search for a valid decomposition with fixed V structure.
        V has exactly one ±1 per column.
        """
        elapsed_start = time.time()
        
        # Generate candidate V matrices
        # Each column r has one nonzero at position v_r with value ±1
        # v_r should be in range [0, V_dim)
        
        # Strategy: use the V from rank-46 as a starting point, then modify
        U46, V46, W46 = build_rank46()
        
        # Extract the V pattern from rank-46 and try variations
        # First, try to reduce rank by removing columns from V46
        for cols_to_remove in range(0, 5):
            if time.time() - elapsed_start > time_budget:
                break
            
            if cols_to_remove == 0:
                V_test = V46[:, :R].copy() if R <= 46 else V46.copy()
            else:
                # Remove the last cols_to_remove columns
                V_test = V46[:, :R].copy() if R < 46 else V46[:, :-cols_to_remove].copy()
            
            if V_test.shape[1] != R:
                continue
            
            # Check if V_test has the right structure (one nonzero per column)
            valid_V = True
            for r in range(R):
                nz = np.nonzero(V_test[:, r])[0]
                if len(nz) != 1:
                    valid_V = False
                    break
            
            if not valid_V:
                # Try to fix by setting to single nonzero
                for r in range(R):
                    nz = np.nonzero(V_test[:, r])[0]
                    if len(nz) > 1:
                        # Keep the first nonzero, zero out others
                        keep = nz[0]
                        V_test[:, r] = 0
                        V_test[keep, r] = 1
                    elif len(nz) == 0:
                        # Assign a random position
                        v_pos = np.random.randint(0, V_dim)
                        V_test[v_pos, r] = 1
            
            # Now solve for U and W
            result = solve_milp_fixed_V(R, V_test, time_budget=min(20, time_budget - (time.time() - elapsed_start)))
            if result is not None:
                U_sol, W_sol = result
                return U_sol, V_test, W_sol
        
        # Random search over V patterns
        for attempt in range(100):
            if time.time() - elapsed_start > time_budget:
                break
            
            np.random.seed(attempt)
            V_rand = np.zeros((V_dim, R))
            for r in range(R):
                v_pos = np.random.randint(0, V_dim)
                V_rand[v_pos, r] = np.random.choice([-1, 1])
            
            result = solve_milp_fixed_V(R, V_rand, time_budget=min(15, time_budget - (time.time() - elapsed_start)))
            if result is not None:
                U_sol, W_sol = result
                return U_sol, V_rand, W_sol
        
        return None

    def fast_optimize(U_init, V_init, W_init, target_rank, max_iter=200, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, lr_decay=0.999, l1_penalty=0.0):
        """Adam optimizer with adaptive restarts, L1 penalty for sparsity, and final polish."""
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
            if time.time() > deadline - 15:
                break
            
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)
            
            if l1_penalty > 0:
                loss += l1_penalty * (np.sum(np.abs(U)) + np.sum(np.abs(V)) + np.sum(np.abs(W)))
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if np.sum(diff * diff) < 1e-10:
                break
            
            # Stagnation detection with restart
            if loss > prev_loss * 1.005 and no_improve > 15:
                restart_count += 1
                if restart_count > 5:
                    break
                # Perturb to escape local minimum
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
            
            if l1_penalty > 0:
                gradU += l1_penalty * np.sign(U)
                gradV += l1_penalty * np.sign(V)
                gradW += l1_penalty * np.sign(W)
            
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
        
        valid, U_s, V_s, W_s = try_snap_validate(best_U, best_V, best_W)
        if valid:
            return U_s, V_s, W_s
        
        # Final polish: try Newton if close enough
        if best_loss < 1e-3 and time.time() < deadline - 10:
            U2, V2, W2 = newton_polish(best_U, best_V, best_W, max_iter=80, tol=1e-15)
            valid, U_s, V_s, W_s = try_snap_validate(U2, V2, W2)
            if valid:
                return U_s, V_s, W_s
        
        return None, None, None

    def newton_polish(U, V, W, max_iter=100, tol=1e-14):
        """High-precision Newton-style polish for near-exact solutions."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)
            if loss < tol:
                break
            
            # Compute gradients
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            # Adaptive step size based on gradient magnitude
            grad_norm = np.sqrt(np.sum(gradU**2) + np.sum(gradV**2) + np.sum(gradW**2))
            if grad_norm < 1e-15:
                break
            lr = 0.002 * min(1.0, 1e-6 / grad_norm)
            
            U -= lr * gradU
            V -= lr * gradV
            W -= lr * gradW
        
        return U, V, W

    # Build and verify rank-46
    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
        # Fallback to standard rank 50 if something went wrong
        U50, V50, W50 = build_rank50()
        return {
            "U": [[float(x) for x in row] for row in U50],
            "V": [[float(x) for x in row] for row in V50],
            "W": [[float(x) for x in row] for row in W50]
        }

    best_result = None
    
    np.random.seed(42)
    
    # Strategy 1: Try fixed-V MILP approach for rank 45, 44
    for target_rank in [45, 44, 43]:
        if time.time() > deadline - 50:
            break
        
        result = search_fixed_V(target_rank, time_budget=min(30, deadline - time.time() - 20))
        if result is not None:
            U_r, V_r, W_r = result
            if validate(U_r, V_r, W_r):
                best_result = (U_r, V_r, W_r)
                break
    
    # Strategy 2: Original column removal approach
    if best_result is None and time.time() < deadline - 40:
        col_nonzeros_U = np.count_nonzero(U46, axis=0)
        col_nonzeros_V = np.count_nonzero(V46, axis=0)
        col_nonzeros_W = np.count_nonzero(W46, axis=0)
        
        support_U = [set(np.nonzero(U46[:, c])[0]) for c in range(46)]
        support_V = [set(np.nonzero(V46[:, c])[0]) for c in range(46)]
        support_W = [set(np.nonzero(W46[:, c])[0]) for c in range(46)]
        
        col_scores = []
        for c in range(46):
            support_size = len(support_U[c]) + len(support_V[c]) + len(support_W[c])
            overlap_count = 0
            max_overlap = 0
            for c2 in range(46):
                if c2 == c:
                    continue
                overlap = len(support_U[c] & support_U[c2]) + len(support_V[c] & support_V[c2]) + len(support_W[c] & support_W[c2])
                if overlap > 3:
                    overlap_count += 1
                max_overlap = max(max_overlap, overlap)
            score = support_size + overlap_count * 0.3 - max_overlap * 0.2
            col_scores.append(score)
        
        col_order = np.argsort(col_scores)
        
        max_attempts = min(15, len(col_order))
        for attempt in range(max_attempts):
            if time.time() > deadline - 30:
                break
            
            col_to_zero = col_order[attempt]
            mask = [i for i in range(46) if i != col_to_zero]
            U_test = U46[:, mask].copy()
            V_test = V46[:, mask].copy()
            W_test = W46[:, mask].copy()
            
            found = False
            for seed in [42, 123, 999]:
                if time.time() > deadline - 25:
                    break
                np.random.seed(seed + attempt * 100)
                for noise_scale in [0.005, 0.02, 0.08]:
                    if time.time() > deadline - 22:
                        break
                    U_try = U_test.copy()
                    V_try = V_test.copy()
                    W_try = W_test.copy()
                    U_try += np.random.randn(*U_try.shape) * noise_scale
                    V_try += np.random.randn(*V_try.shape) * noise_scale
                    W_try += np.random.randn(*W_try.shape) * noise_scale
                    
                    U_opt, V_opt, W_opt = fast_optimize(U_try, V_try, W_try, 45, max_iter=150, lr=0.04, l1_penalty=0.05)
                    
                    if U_opt is not None:
                        best_result = (U_opt, V_opt, W_opt)
                        found = True
                        break
                
                if found:
                    break
            
            if found:
                break
    
    if best_result is not None:
        U_r, V_r, W_r = best_result
        if validate(U_r, V_r, W_r):
            return {
                "U": [[float(x) for x in row] for row in U_r],
                "V": [[float(x) for x in row] for row in V_r],
                "W": [[float(x) for x in row] for row in W_r]
            }
    
    # Final fallback: return rank-46
    return {
        "U": [[float(x) for x in row] for row in U46],
        "V": [[float(x) for x in row] for row in V46],
        "W": [[float(x) for x in row] for row in W46]
    }
