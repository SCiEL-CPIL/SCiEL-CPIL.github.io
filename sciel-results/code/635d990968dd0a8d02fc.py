# EVOLVE-BLOCK-START
import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 585.0  # safety margin

    def evaluate(S, W):
        """
        O(n^3) vectorized evaluation using searchsorted.
        P = sum_{k,l} w_k w_l * F(2*s_l - s_k)
        where F(t) = sum_{i,j: s_i+s_j < t} w_i w_j
        For sorted S, for each i, the max j with S[i]+S[j] < t is found by searchsorted.
        """
        n = len(S)
        if n == 0 or n == 1:
            return 0.0

        cumW = np.concatenate([[0.0], np.cumsum(W)])

        # Vectorize over k: for each k, thresholds for all l are 2*S[l] - S[k]
        # For each (k, l), F = sum_i W[i] * cumW[jmax(i, threshold)]
        # jmax(i, t) = searchsorted(S, t - S[i], side='left')
        #
        # Full vectorization: build (n_k, n_l, n_i) array of jmax indices
        # Then F[k,l] = sum_i W[i] * cumW[jmax[k,l,i]]
        # This is n^3 memory which is fine for n <= ~30

        P = 0.0
        # Process in batches of k to limit memory
        batch = 32
        for k_start in range(0, n, batch):
            k_end = min(k_start + batch, n)
            Sk = S[k_start:k_end, None, None]  # (b, 1, 1)
            Sl = S[None, None, :]              # (1, 1, n)
            Si = S[None, :, None]              # (1, n, 1)

            # thresholds: (b, 1, n_l) - S[k] for each k in batch
            thresh = 2.0 * Sl - Sk  # (b, 1, n_l)
            # t_minus_Si: (b, n_i, n_l)
            t_si = thresh - Si  # (b, n_i, n_l)

            # jmax: for each (b, i, l), find largest j with S[j] < t_si
            # searchsorted needs sorted array; S is sorted
            # We need to searchsorted S against t_si values
            # t_si shape (b, n_i, n_l), S shape (n,)
            # Use broadcasting: S[None,None,None,:] vs t_si[:,:,None,:]
            # That's (b, n_i, n_l, n) - too large
            # Instead, loop over l within the batch
            for l in range(n):
                t_col = 2.0 * S[l] - S[k_start:k_end]  # (b,)
                # For each k in batch and each i, find jmax
                # t_col[:, None] - Si_flat: (b, n)
                t_vals = t_col[:, None] - S[None, :]  # (b, n_i)
                # jmax[b, i] = searchsorted(S, t_vals[b, i], 'left')
                # Vectorize: for each b, searchsorted S against t_vals[b,:]
                jmax = np.empty((k_end - k_start, n), dtype=np.intp)
                for b_idx in range(k_end - k_start):
                    jmax[b_idx] = np.searchsorted(S, t_vals[b_idx], side='left')
                # F[b] = sum_i W[i] * cumW[jmax[b, i]]
                F_vals = np.sum(W[None, :] * cumW[jmax], axis=1)  # (b,)
                P += np.dot(W[k_start:k_end], W[l] * F_vals)

        return P

    def polish_gradient(S_init, W_init, max_iters=300, lr_init=0.05, lr_min=1e-8, h=1e-5):
        """
        Gradient-based polishing with adaptive learning rate.
        Uses finite differences on the full parameter vector.
        """
        S = np.sort(S_init.copy()).astype(np.float64)
        W = W_init.copy().astype(np.float64)
        W = np.clip(W, 0.0, None)
        w_sum = np.sum(W)
        if w_sum < 1e-9:
            return S, W, evaluate(S, W)
        W /= w_sum
        
        n = len(S)
        current_P = evaluate(S, W)
        best_P = current_P
        best_S = S.copy()
        best_W = W.copy()
        
        lr = lr_init
        no_improve = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Compute gradient using forward differences (cheaper than central)
            dP_dS = np.zeros(n)
            dP_dW = np.zeros(n)
            
            for i in range(n):
                # Perturb S[i] (keep sorted)
                S_p = S.copy()
                S_p[i] += h
                S_p = np.clip(S_p, 0.0, None)
                S_p = np.sort(S_p)
                # If sorting changed the index, we still evaluate correctly
                dP_dS[i] = (evaluate(S_p, W) - current_P) / h
                
                # Perturb W[i]
                W_p = W.copy()
                W_p[i] += h
                W_p = np.clip(W_p, 0.0, None)
                sp = np.sum(W_p)
                if sp > 1e-9:
                    W_p /= sp
                    dP_dW[i] = (evaluate(S, W_p) - current_P) / h
                else:
                    dP_dW[i] = 0.0
            
            # Update S
            S_new = S + lr * dP_dS
            S_new = np.clip(S_new, 0.0, None)
            S_new = np.sort(S_new)
            
            # Update W
            W_new = W + lr * dP_dW
            W_new = np.clip(W_new, 0.0, None)
            w_sum = np.sum(W_new)
            if w_sum < 1e-9:
                W_new = W.copy()
            else:
                W_new /= w_sum
            
            P_new = evaluate(S_new, W_new)
            
            if P_new > current_P:
                current_P = P_new
                S = S_new
                W = W_new
                no_improve = 0
                if P_new > best_P:
                    best_P = P_new
                    best_S = S.copy()
                    best_W = W.copy()
            else:
                no_improve += 1
                lr *= 0.8
                if lr < lr_min:
                    break
        
        return best_S, best_W, best_P

    def optimize_local(S_init, W_init, max_iters=400):
        """
        Deterministic local search: coordinate descent on support points
        and weight ratios, with occasional random perturbations.
        """
        S = np.sort(S_init.copy())
        W = W_init.copy()
        W = np.clip(W, 0.0, None)
        W_sum = np.sum(W)
        if W_sum == 0:
            W = np.ones_like(W)
        else:
            W /= W_sum
            
        n = len(S)
        current_P = evaluate(S, W)
        best_P = current_P
        best_S = S.copy()
        best_W = W.copy()
        
        step_s = 0.08
        step_w = 0.08
        no_improve = 0
        rng = np.random.RandomState(42)
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            
            # Coordinate descent on support points
            for i in range(n):
                if time.time() > deadline - 5.0:
                    break
                for delta in [step_s, -step_s, step_s*0.5, -step_s*0.5]:
                    S_try = S.copy()
                    S_try[i] = max(0.0, min(S_try[i] + delta, S[-1]))
                    S_try = np.sort(S_try)
                    P_try = evaluate(S_try, W)
                    if P_try > current_P:
                        current_P = P_try
                        S = S_try
                        improved = True
                        break
            
            # Coordinate descent on weights (move probability mass)
            for i in range(n - 1):
                if time.time() > deadline - 5.0:
                    break
                for frac in [step_w, step_w*0.5, -step_w, -step_w*0.5]:
                    W_try = W.copy()
                    W_try[i] += frac
                    W_try[i+1] -= frac
                    W_try = np.clip(W_try, 0.0, None)
                    ws = np.sum(W_try)
                    if ws < 1e-9:
                        continue
                    W_try /= ws
                    P_try = evaluate(S, W_try)
                    if P_try > current_P:
                        current_P = P_try
                        W = W_try
                        improved = True
                        break
            
            if improved:
                no_improve = 0
                if current_P > best_P:
                    best_P = current_P
                    best_S = S.copy()
                    best_W = W.copy()
            else:
                no_improve += 1
                step_s *= 0.95
                step_w *= 0.95
                
                # Random restart perturbation every 30 stalls
                if no_improve % 30 == 0 and step_s > 0.005:
                    S_try = S.copy()
                    W_try = W.copy()
                    S_try += rng.randn(n) * 0.1
                    S_try = np.clip(S_try, 0.0, None)
                    S_try = np.sort(S_try)
                    W_try += rng.randn(n) * 0.1
                    W_try = np.clip(W_try, 0.0, None)
                    ws = np.sum(W_try)
                    if ws > 1e-9:
                        W_try /= ws
                        P_try = evaluate(S_try, W_try)
                        if P_try > current_P:
                            S = S_try
                            W = W_try
                            current_P = P_try
                            if current_P > best_P:
                                best_P = current_P
                                best_S = S.copy()
                                best_W = W.copy()
                            no_improve = 0
                            step_s = 0.08
                            step_w = 0.08
        
        return best_P, best_S, best_W

    global_best_P = 0.0
    global_best_S = np.array([0.0, 1.0])
    global_best_W = np.array([0.5, 0.5])
    
    # Analytic 2-point solution as a strong baseline
    p2 = (15.0 - np.sqrt(33.0)) / 16.0
    S2 = np.array([0.0, 1.0])
    W2 = np.array([p2, 1.0 - p2])
    P2 = evaluate(S2, W2)
    if P2 > global_best_P:
        global_best_P = P2
        global_best_S = S2.copy()
        global_best_W = W2.copy()

    # Additional structured 2-point candidates
    for p in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
        S_cand = np.array([0.0, 1.0])
        W_cand = np.array([p, 1.0 - p])
        P_cand = evaluate(S_cand, W_cand)
        if P_cand > global_best_P:
            global_best_P = P_cand
            global_best_S = S_cand.copy()
            global_best_W = W_cand.copy()

    # Structured starting points with endpoint-heavy weights
    deterministic_starts = []
    
    # 2-point variants
    for p in [0.45, 0.5, 0.55]:
        deterministic_starts.append((np.array([0.0, 1.0]), np.array([p, 1.0-p])))
    
    # 3-point: mass at endpoints, small mass in middle
    for mid in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for wmid in [0.05, 0.1, 0.15]:
            w_end = (1.0 - wmid) / 2.0
            deterministic_starts.append((np.array([0.0, mid, 1.0]), np.array([w_end, wmid, w_end])))
    
    # 4-point: symmetric
    for a, b in [(0.2, 0.8), (0.25, 0.75), (0.3, 0.7)]:
        for wmid in [0.05, 0.1]:
            w_end = (1.0 - 2*wmid) / 2.0
            deterministic_starts.append((np.array([0.0, a, b, 1.0]), np.array([w_end, wmid, wmid, w_end])))
    
    # 5-point
    deterministic_starts.append((np.array([0.0, 0.2, 0.5, 0.8, 1.0]), np.array([0.35, 0.05, 0.1, 0.05, 0.45])))
    deterministic_starts.append((np.array([0.0, 0.25, 0.5, 0.75, 1.0]), np.array([0.4, 0.05, 0.1, 0.05, 0.4])))
    
    # 6-point
    deterministic_starts.append((np.array([0.0, 0.15, 0.35, 0.65, 0.85, 1.0]), np.array([0.3, 0.05, 0.05, 0.05, 0.05, 0.5])))
    
    # Evaluate all deterministic starts
    for S_det, W_det in deterministic_starts:
        if time.time() > deadline - 60.0:
            break
        P_det = evaluate(S_det, W_det)
        if P_det > global_best_P:
            global_best_P = P_det
            global_best_S = S_det.copy()
            global_best_W = W_det.copy()

    # Multi-start coordinate descent
    max_n = 7
    restarts_by_n = {2: 20, 3: 25, 4: 20, 5: 15, 6: 10, 7: 8}
    rng = np.random.RandomState(123)
    
    for n in range(2, max_n + 1):
        if time.time() > deadline - 20.0:
            break
        
        n_restarts = restarts_by_n.get(n, 10)
        
        for r in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
            
            if r == 0:
                S_init = np.linspace(0.0, 1.0, n)
                W_init = np.ones(n) / n
            elif r == 1:
                if n > 2:
                    S_init = np.array([0.0] + list(np.linspace(0.2, 0.8, n-2)) + [1.0])
                    W_init = np.array([0.4] + list(np.ones(n-2)*0.1) + [0.3])
                    W_init = W_init / np.sum(W_init)
                else:
                    S_init = np.array([0.0, 1.0])
                    W_init = np.array([0.5, 0.5])
            elif r == 2:
                # Endpoint-heavy
                S_init = np.sort(rng.uniform(0.0, 1.0, n))
                W_init = np.zeros(n)
                W_init[0] = rng.uniform(0.3, 0.55)
                W_init[-1] = rng.uniform(0.3, 0.55)
                remaining = 1.0 - W_init[0] - W_init[-1]
                if remaining > 0 and n > 2:
                    W_init[1:-1] = rng.dirichlet(np.ones(n-2) * 0.5) * remaining
                W_init = np.clip(W_init, 0.0, None)
                W_init /= np.sum(W_init)
            else:
                # Random with endpoint bias
                S_init = np.sort(rng.uniform(0.0, 1.0, n - 1))
                S_init = np.append(S_init, 1.0)
                alpha = np.ones(n) * 2
                alpha[0] *= 3
                alpha[-1] *= 3
                W_init = rng.dirichlet(alpha)
                
            P, S, W = optimize_local(S_init, W_init, max_iters=350)
            
            if P > global_best_P:
                global_best_P = P
                global_best_S = S.copy()
                global_best_W = W.copy()

    # Gradient polishing on the best solution found
    if time.time() < deadline - 25.0:
        S_pol, W_pol, P_pol = polish_gradient(global_best_S, global_best_W, max_iters=150, lr_init=0.05)
        if P_pol > global_best_P:
            global_best_P = P_pol
            global_best_S = S_pol
            global_best_W = W_pol

    # Try gradient polish on the analytic 2-point solution
    if time.time() < deadline - 15.0:
        S2p = np.array([0.0, 1.0])
        W2p = np.array([p2, 1.0 - p2])
        S_pol2, W_pol2, P_pol2 = polish_gradient(S2p, W2p, max_iters=100, lr_init=0.03)
        if P_pol2 > global_best_P:
            global_best_P = P_pol2
            global_best_S = S_pol2
            global_best_W = W_pol2

    # Final fine-step gradient polish
    if time.time() < deadline - 5.0:
        S_final, W_final, P_final = polish_gradient(global_best_S, global_best_W, max_iters=80, lr_init=0.01)
        if P_final > global_best_P:
            global_best_P = P_final
            global_best_S = S_final
            global_best_W = W_final

    # Final validity check and normalization
    global_best_S = np.sort(global_best_S)
    global_best_W = np.clip(global_best_W, 0.0, None)
    w_sum = np.sum(global_best_W)
    if w_sum < 1e-9:
        global_best_S = np.array([0.0, 1.0])
        global_best_W = np.array([0.5, 0.5])
    else:
        global_best_W /= w_sum
        global_best_S = np.clip(global_best_S, 0.0, None)
        if len(global_best_S) < 1:
            global_best_S = np.array([0.0])
            global_best_W = np.array([1.0])
        
    return global_best_S.astype(np.float64), global_best_W.astype(np.float64)

# EVOLVE-BLOCK-END
