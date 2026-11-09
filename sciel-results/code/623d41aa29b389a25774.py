# EVOLVE-BLOCK-START
import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 585.0  # 600s limit, keep margin

    def evaluate(S, W):
        """
        Compute P[X1 + X2 + X3 < 2*X4] for discrete measure (S, W).
        S: 1-D array of support points (sorted).
        W: 1-D array of weights (sum to 1).
        Uses O(n^3) approach: for each pair (i,j), precompute S_i+S_j,
        then for each (k,l) count how many (i,j) satisfy S_i+S_j < 2*S_l - S_k.
        """
        n = len(S)
        if n == 0:
            return 0.0
        
        # Precompute all pairwise sums S_i + S_j and their weights
        # PairSums[i,j] = S[i] + S[j]
        # PairWeights[i,j] = W[i] * W[j]
        PairSums = S[:, None] + S[None, :]  # (n, n)
        PairWeights = W[:, None] * W[None, :]  # (n, n)
        
        # Flatten for easier handling
        PS = PairSums.ravel()  # (n*n,)
        PW = PairWeights.ravel()  # (n*n,)
        
        # Sort by PairSums
        sort_idx = np.argsort(PS)
        PS_sorted = PS[sort_idx]
        PW_sorted = PW[sort_idx]
        PW_cumsum = np.cumsum(PW_sorted)
        
        # For each (k, l): threshold = 2*S[l] - S[k]
        # Count sum of PW where PS < threshold
        P = 0.0
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * S[l] - S[k]
                # Number of pairs with PS_sorted < threshold
                idx = np.searchsorted(PS_sorted, threshold, side='left')
                if idx > 0:
                    sum_pw = PW_cumsum[idx - 1]
                else:
                    sum_pw = 0.0
                P += W[k] * W[l] * sum_pw
        
        return P

    def optimize_local(S_init, W_init, max_iters=1000, verbose=False):
        """
        Local hill climbing search with simulated annealing.
        """
        S = np.sort(S_init.copy())
        W = W_init.copy()
        W = np.clip(W, 0.0, None)
        W_sum = np.sum(W)
        if W_sum == 0:
            W = np.ones_like(W)
        else:
            W /= W_sum
            
        # Ensure S is valid: non-negative, max normalized to 1 if n>0
        if len(S) > 1:
            S = S / S[-1]
            S[-1] = 1.0
            
        n = len(S)
        current_P = evaluate(S, W)
        best_P = current_P
        best_S = S.copy()
        best_W = W.copy()
        
        # Simulated annealing parameters
        T = 0.01  # initial temperature
        T_min = 1e-6
        cooling = 0.998
        
        for i in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Adaptive step size based on temperature
            step_s = 0.05 * T / 0.01
            step_w = 0.05 * T / 0.01
            
            # Generate perturbation
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
            
            # Evaluate
            P_new = evaluate(S_new, W_new)
            
            # Simulated annealing acceptance
            delta = P_new - current_P
            if delta > 0 or np.random.rand() < np.exp(delta / max(T, 1e-10)):
                current_P = P_new
                S = S_new
                W = W_new
                if P_new > best_P:
                    best_P = P_new
                    best_S = S_new.copy()
                    best_W = W_new.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min
            
        return best_P, best_S, best_W

    def polish_gradient(S_init, W_init, max_iters=200, lr=0.01):
        """
        Gradient ascent polishing step using finite differences.
        """
        S = np.sort(S_init.copy()).astype(np.float64)
        W = W_init.copy().astype(np.float64)
        W = np.clip(W, 0.0, None)
        W_sum = np.sum(W)
        if W_sum < 1e-9:
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
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Compute gradient via central differences
            eps = 1e-5
            dP_dS = np.zeros(n)
            dP_dW = np.zeros(n)
            
            # Gradient w.r.t. S (only interior points)
            for i in range(n - 1):
                S_p = S.copy()
                S_p[i] += eps
                S_p[0:n-1] = np.sort(S_p[0:n-1])
                P_p = evaluate(S_p, W)
                
                S_m = S.copy()
                S_m[i] -= eps
                S_m[0:n-1] = np.sort(S_m[0:n-1])
                P_m = evaluate(S_m, W)
                
                dP_dS[i] = (P_p - P_m) / (2 * eps)
            
            # Gradient w.r.t. W
            for i in range(n):
                W_p = W.copy()
                W_p[i] += eps
                W_p = np.clip(W_p, 0, None)
                ws = np.sum(W_p)
                if ws > 0:
                    W_p /= ws
                P_p = evaluate(S, W_p)
                
                W_m = W.copy()
                W_m[i] -= eps
                W_m = np.clip(W_m, 0, None)
                ws = np.sum(W_m)
                if ws > 0:
                    W_m /= ws
                P_m = evaluate(S, W_m)
                
                dP_dW[i] = (P_p - P_m) / (2 * eps)
            
            # Update S
            S_new = S.copy()
            if n > 1:
                S_new[0:n-1] += lr * dP_dS[0:n-1]
                S_new[0:n-1] = np.clip(S_new[0:n-1], 0.0, 1.0)
                S_new[0:n-1] = np.sort(S_new[0:n-1])
                S_new[-1] = 1.0
            
            # Update W
            W_new = W + lr * dP_dW
            W_new = np.clip(W_new, 0.0, None)
            W_sum = np.sum(W_new)
            if W_sum < 1e-9:
                break
            W_new /= W_sum
            
            P_new = evaluate(S_new, W_new)
            
            if P_new > current_P:
                current_P = P_new
                S = S_new
                W = W_new
                if P_new > best_P:
                    best_P = P_new
                    best_S = S.copy()
                    best_W = W.copy()
                lr *= 0.99
            else:
                lr *= 0.5
                if lr < 1e-8:
                    break
        
        return best_P, best_S, best_W

    global_best_P = 0.0
    global_best_S = np.array([0.0, 1.0])
    global_best_W = np.array([0.5, 0.5])
    
    # Analytic 2-point solution as a strong baseline
    # p = (15 - sqrt(33)) / 16
    p2 = (15.0 - np.sqrt(33.0)) / 16.0
    S2 = np.array([0.0, 1.0])
    W2 = np.array([p2, 1.0 - p2])
    P2 = evaluate(S2, W2)
    if P2 > global_best_P:
        global_best_P = P2
        global_best_S = S2.copy()
        global_best_W = W2.copy()
    
    # Try uniform distribution on [0, 1] as baseline
    S_unif = np.array([0.0, 1.0])
    W_unif = np.array([0.5, 0.5])
    P_unif = evaluate(S_unif, W_unif)
    if P_unif > global_best_P:
        global_best_P = P_unif
        global_best_S = S_unif.copy()
        global_best_W = W_unif.copy()
    
    # Try 3-point uniform: 0, 0.5, 1
    S3_unif = np.array([0.0, 0.5, 1.0])
    W3_unif = np.array([1.0/3, 1.0/3, 1.0/3])
    P3_unif = evaluate(S3_unif, W3_unif)
    if P3_unif > global_best_P:
        global_best_P = P3_unif
        global_best_S = S3_unif.copy()
        global_best_W = W3_unif.copy()
    
    # Try 4-point uniform: 0, 1/3, 2/3, 1
    S4_unif = np.array([0.0, 1.0/3, 2.0/3, 1.0])
    W4_unif = np.array([0.25, 0.25, 0.25, 0.25])
    P4_unif = evaluate(S4_unif, W4_unif)
    if P4_unif > global_best_P:
        global_best_P = P4_unif
        global_best_S = S4_unif.copy()
        global_best_W = W4_unif.copy()

    # Multi-start local search for n = 2 to 5
    # Focus on smaller n since evaluation is O(n^3) and we have time constraints
    for n in range(2, 6):
        if time.time() > deadline - 30.0:
            break
            
        # Number of restarts: more for smaller n
        n_restarts = 30 if n <= 3 else 15
        
        for r in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
            
            if r == 0:
                # Deterministic init based on n
                if n == 2:
                    S_init = np.array([0.0, 1.0])
                    W_init = np.array([p2, 1.0 - p2])
                elif n == 3:
                    S_init = np.array([0.0, 0.5, 1.0])
                    W_init = np.array([0.4, 0.2, 0.4])
                elif n == 4:
                    S_init = np.array([0.0, 0.33, 0.66, 1.0])
                    W_init = np.array([0.3, 0.2, 0.2, 0.3])
                else:
                    S_init = np.linspace(0.0, 1.0, n)
                    W_init = np.ones(n) / n
            else:
                # Random init with bias toward interesting structures
                # Support: sorted random in [0, 1], last is 1
                S_init = np.sort(np.random.uniform(0.0, 1.0, n - 1))
                S_init = np.append(S_init, 1.0)
                # Weights: Dirichlet with varying concentration
                alpha = np.random.uniform(0.5, 3.0, n)
                W_init = np.random.dirichlet(alpha)
                
            P, S, W = optimize_local(S_init, W_init, max_iters=800)
            
            if P > global_best_P:
                global_best_P = P
                global_best_S = S
                global_best_W = W

    # Gradient polishing pass on the best solution found
    if time.time() < deadline - 20.0:
        P_pol, S_pol, W_pol = polish_gradient(global_best_S, global_best_W, max_iters=300, lr=0.005)
        if P_pol > global_best_P:
            global_best_P = P_pol
            global_best_S = S_pol
            global_best_W = W_pol

    # Final high-precision polish: very small steps
    if time.time() < deadline - 10.0:
        S_fine = global_best_S.copy()
        W_fine = global_best_W.copy()
        n = len(S_fine)
        step_s = 1e-4
        step_w = 1e-4
        for it in range(200):
            if time.time() > deadline - 5.0:
                break
            S_new = S_fine.copy()
            W_new = W_fine.copy()
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
                break
            W_new /= W_sum
            P_new = evaluate(S_new, W_new)
            if P_new > global_best_P:
                global_best_P = P_new
                S_fine = S_new
                W_fine = W_new
                global_best_S = S_fine.copy()
                global_best_W = W_fine.copy()
            step_s *= 0.98
            step_w *= 0.98

    # Final validity check and normalization
    global_best_S = np.sort(global_best_S)
    global_best_W = np.clip(global_best_W, 0.0, None)
    w_sum = np.sum(global_best_W)
    if w_sum < 1e-9:
        # Fallback
        global_best_S = np.array([0.0, 1.0])
        global_best_W = np.array([0.5, 0.5])
    else:
        global_best_W /= w_sum
        # Ensure support is non-negative
        global_best_S = np.clip(global_best_S, 0.0, None)
    
    # Remove duplicate support points (keep the one with combined weight)
    if len(global_best_S) > 1:
        unique_S, inverse = np.unique(global_best_S, return_inverse=True)
        if len(unique_S) < len(global_best_S):
            new_W = np.zeros(len(unique_S))
            for i in range(len(global_best_S)):
                new_W[inverse[i]] += global_best_W[i]
            global_best_S = unique_S
            global_best_W = new_W
            global_best_W /= np.sum(global_best_W)
    
    return global_best_S.astype(np.float64), global_best_W.astype(np.float64)

# EVOLVE-BLOCK-END
