# EVOLVE-BLOCK-START
import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590.0  # 600s limit, keep margin

    def evaluate(S, W):
        """
        Compute P[X1 + X2 + X3 < 2*X4] for discrete measure (S, W).
        Uses vectorized approach: for each (k,l), compute sum_{i,j: S_i+S_j < 2*S_l-S_k} W_i*W_j.
        S: 1-D sorted array of support points.
        W: 1-D array of weights (sum to 1).
        """
        n = len(S)
        if n == 0:
            return 0.0
        
        # Precompute cumulative weights
        cumW = np.concatenate([[0.0], np.cumsum(W)])  # (n+1,)
        
        P = 0.0
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * S[l] - S[k]
                if threshold <= 0:
                    continue
                # For each i, find max j such that S[i] + S[j] < threshold
                # i.e., S[j] < threshold - S[i]
                max_j = np.searchsorted(S, threshold - S, side='left')  # (n,)
                prefix_sums = cumW[max_j]  # (n,)
                P += W[k] * W[l] * np.dot(W, prefix_sums)
        
        return P

    def compute_gradient(S, W, h=1e-6):
        """
        Compute gradient of P with respect to S and W using central finite differences.
        Returns (dP_dS, dP_dW) as 1-D arrays.
        """
        n = len(S)
        P0 = evaluate(S, W)
        
        dP_dS = np.zeros(n)
        dP_dW = np.zeros(n)
        
        for i in range(n):
            # Perturb S[i]
            S_p = S.copy()
            S_m = S.copy()
            S_p[i] += h
            S_m[i] -= h
            S_p = np.clip(S_p, 0.0, None)
            S_m = np.clip(S_m, 0.0, None)
            S_p = np.sort(S_p)
            S_m = np.sort(S_m)
            dP_dS[i] = (evaluate(S_p, W) - evaluate(S_m, W)) / (2 * h)
            
            # Perturb W[i]
            W_p = W.copy()
            W_m = W.copy()
            W_p[i] += h
            W_m[i] -= h
            W_p = np.clip(W_p, 0.0, None)
            W_m = np.clip(W_m, 0.0, None)
            sp = np.sum(W_p)
            sm = np.sum(W_m)
            if sp > 1e-9:
                W_p /= sp
            if sm > 1e-9:
                W_m /= sm
            dP_dW[i] = (evaluate(S, W_p) - evaluate(S, W_m)) / (2 * h)
        
        return dP_dS, dP_dW

    def polish_gradient(S_init, W_init, max_iters=300, lr_init=0.05, lr_min=1e-8):
        """
        Gradient-based polishing with adaptive learning rate.
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
            
            dP_dS, dP_dW = compute_gradient(S, W, h=1e-6)
            
            # Update S: move in gradient direction, keep sorted and in [0, 1]
            S_new = S + lr * dP_dS
            S_new = np.clip(S_new, 0.0, 1.0)
            S_new = np.sort(S_new)
            # Ensure at least some spread
            if n > 1:
                if S_new[-1] - S_new[0] < 1e-8:
                    S_new = S.copy()
            
            # Update W: move in gradient direction, keep non-negative
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
                lr *= 0.7
                if lr < lr_min:
                    break
        
        return best_S, best_W, best_P

    def optimize_local(S_init, W_init, max_iters=600, verbose=False):
        """
        Local hill climbing search with adaptive step sizes.
        """
        S = np.sort(S_init.copy())
        W = W_init.copy()
        W = np.clip(W, 0.0, None)
        W_sum = np.sum(W)
        if W_sum == 0:
            W = np.ones_like(W)
        else:
            W /= W_sum
            
        if len(S) > 1:
            S = S / S[-1]
            S[-1] = 1.0
            
        n = len(S)
        current_P = evaluate(S, W)
        best_P = current_P
        best_S = S.copy()
        best_W = W.copy()
        
        step_s = 0.05
        step_w = 0.05
        no_improve_count = 0
        
        for i in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            S_new = S.copy()
            W_new = W.copy()
            
            if n > 1:
                noise_s = np.random.randn(n - 1) * step_s
                S_new[0:n-1] += noise_s
                S_new[0:n-1] = np.clip(S_new[0:n-1], 0.0, 1.0)
                S_new[0:n-1] = np.sort(S_new[0:n-1])
            
            noise_w = np.random.randn(n) * step_w
            W_new += noise_w
            W_new = np.clip(W_new, 0.0, None)
            W_sum = np.sum(W_new)
            if W_sum < 1e-9:
                continue
            W_new /= W_sum
            
            P_new = evaluate(S_new, W_new)
            
            if P_new > current_P:
                current_P = P_new
                S = S_new
                W = W_new
                no_improve_count = 0
                if P_new > best_P:
                    best_P = P_new
                    best_S = S_new.copy()
                    best_W = W_new.copy()
            else:
                no_improve_count += 1
                # Decay steps when stuck
                step_s *= 0.97
                step_w *= 0.97
                
                # Occasionally try a larger random jump to escape local optima
                if no_improve_count % 40 == 0:
                    S_jump = S.copy()
                    W_jump = W.copy()
                    if n > 1:
                        S_jump[0:n-1] += np.random.randn(n - 1) * 0.3
                        S_jump[0:n-1] = np.clip(S_jump[0:n-1], 0.0, 1.0)
                        S_jump[0:n-1] = np.sort(S_jump[0:n-1])
                    W_jump += np.random.randn(n) * 0.3
                    W_jump = np.clip(W_jump, 0.0, None)
                    wj = np.sum(W_jump)
                    if wj > 1e-9:
                        W_jump /= wj
                        P_jump = evaluate(S_jump, W_jump)
                        if P_jump > current_P:
                            S = S_jump
                            W = W_jump
                            current_P = P_jump
                            if P_jump > best_P:
                                best_P = P_jump
                                best_S = S_jump.copy()
                                best_W = W_jump.copy()
            
            # Gradually increase step size if improving (restart-like behavior)
            if no_improve_count == 0:
                step_s = min(step_s * 1.02, 0.2)
                step_w = min(step_w * 1.02, 0.2)
                
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

    # Try several deterministic starting points
    deterministic_starts = []
    for n in range(2, 8):
        S_det = np.linspace(0.0, 1.0, n)
        W_det = np.ones(n) / n
        deterministic_starts.append((S_det, W_det))
    
    # Add some structured starts
    # Concentrate mass at 0 and 1
    for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
        S_struct = np.array([0.0, 1.0])
        W_struct = np.array([frac, 1.0 - frac])
        deterministic_starts.append((S_struct, W_struct))
    
    # Three-point structured
    S_3 = np.array([0.0, 0.5, 1.0])
    W_3 = np.array([0.4, 0.2, 0.4])
    deterministic_starts.append((S_3, W_3))
    
    S_3b = np.array([0.0, 0.3, 1.0])
    W_3b = np.array([0.5, 0.1, 0.4])
    deterministic_starts.append((S_3b, W_3b))
    
    S_3c = np.array([0.0, 0.7, 1.0])
    W_3c = np.array([0.6, 0.1, 0.3])
    deterministic_starts.append((S_3c, W_3c))
    
    # Four-point structured
    S_4 = np.array([0.0, 0.3, 0.7, 1.0])
    W_4 = np.array([0.4, 0.1, 0.1, 0.4])
    deterministic_starts.append((S_4, W_4))
    
    S_4b = np.array([0.0, 0.2, 0.8, 1.0])
    W_4b = np.array([0.5, 0.1, 0.1, 0.3])
    deterministic_starts.append((S_4b, W_4b))
    
    # Evaluate all deterministic starts
    for S_det, W_det in deterministic_starts:
        if time.time() > deadline - 60.0:
            break
        P_det = evaluate(S_det, W_det)
        if P_det > global_best_P:
            global_best_P = P_det
            global_best_S = S_det.copy()
            global_best_W = W_det.copy()

    # Multi-start local search
    # Focus on n=2 to n=6 with more restarts for smaller n
    max_n = 6
    restarts_by_n = {2: 40, 3: 30, 4: 25, 5: 20, 6: 15}
    
    for n in range(2, max_n + 1):
        if time.time() > deadline - 30.0:
            break
        
        n_restarts = restarts_by_n.get(n, 10)
        
        for r in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
            
            if r == 0:
                # Deterministic init
                S_init = np.linspace(0.0, 1.0, n)
                W_init = np.ones(n) / n
            elif r == 1:
                # Concentrated at endpoints
                S_init = np.array([0.0] + list(np.linspace(0.2, 0.8, n-2)) + [1.0]) if n > 2 else np.array([0.0, 1.0])
                W_init = np.array([0.4] + list(np.ones(n-2)*0.1) + [0.3]) if n > 2 else np.array([0.5, 0.5])
                W_init = W_init / np.sum(W_init)
            else:
                # Random init
                S_init = np.sort(np.random.uniform(0.0, 1.0, n - 1))
                S_init = np.append(S_init, 1.0)
                W_init = np.random.dirichlet(np.ones(n) * 2)
                
            P, S, W = optimize_local(S_init, W_init, max_iters=500)
            
            if P > global_best_P:
                global_best_P = P
                global_best_S = S.copy()
                global_best_W = W.copy()

    # Gradient polishing on the best solution found
    if time.time() < deadline - 20.0:
        S_pol, W_pol, P_pol = polish_gradient(global_best_S, global_best_W, max_iters=200, lr_init=0.05)
        if P_pol > global_best_P:
            global_best_P = P_pol
            global_best_S = S_pol
            global_best_W = W_pol

    # Try gradient polish on a few other good candidates too
    if time.time() < deadline - 10.0:
        # Re-evaluate the 2-point analytic solution with polish
        S2p = np.array([0.0, 1.0])
        W2p = np.array([p2, 1.0 - p2])
        S_pol2, W_pol2, P_pol2 = polish_gradient(S2p, W2p, max_iters=150, lr_init=0.03)
        if P_pol2 > global_best_P:
            global_best_P = P_pol2
            global_best_S = S_pol2
            global_best_W = W_pol2
    
    # Final fine-step gradient polish
    if time.time() < deadline - 5.0:
        S_final, W_final, P_final = polish_gradient(global_best_S, global_best_W, max_iters=100, lr_init=0.01)
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
        # Ensure at least one point
        if len(global_best_S) < 1:
            global_best_S = np.array([0.0])
            global_best_W = np.array([1.0])
        
    return global_best_S.astype(np.float64), global_best_W.astype(np.float64)

# EVOLVE-BLOCK-END
