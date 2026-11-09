# EVOLVE-BLOCK-START
import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590.0  # 600s limit, keep margin

    def evaluate(S, W):
        """
        Compute P[X1 + X2 + X3 < 2*X4] for discrete measure (S, W).
        S: 1-D array of support points (sorted).
        W: 1-D array of weights (sum to 1).
        """
        n = len(S)
        if n == 0:
            return 0.0
        
        # Vectorized computation
        # S_i + S_j + S_k < 2 S_l
        # Dimensions:
        # S_i: (n, 1, 1)
        # S_j: (1, n, 1)
        # S_k: (1, 1, n)
        # TripleSums: (n, n, n)
        
        TripleSums = S[:, None, None] + S[None, :, None] + S[None, None, :]
        
        # W_i W_j W_k
        WeightsTriple = W[:, None, None] * W[None, :, None] * W[None, None, :]
        
        # Thresholds: 2 * S_l for each l
        # We need to sum over i,j,k for each l.
        # Indicators[l, i, j, k] = (TripleSums[i, j, k] < 2*S[l])
        
        # Thresholds shape: (n, 1, 1, 1)
        Thresholds = 2.0 * S[:, None, None, None]
        
        # TripleSums shape: (1, n, n, n) to broadcast against (n, 1, 1, 1)
        # Result Indicators shape: (n, n, n, n) -> (l, i, j, k)
        Indicators = (TripleSums[None, :, :, :] < Thresholds).astype(np.float64)
        
        # Calculate P_l = sum_{i,j,k} Indicators[l, i, j, k] * WeightsTriple[i, j, k]
        # Indicators: (L, I, J, K)
        # WeightsTriple: (I, J, K)
        # We want result of shape (L,)
        P_l = np.tensordot(Indicators, WeightsTriple, axes=([1, 2, 3], [0, 1, 2]))
        
        # Total P = sum_l W_l * P_l
        P = np.dot(W, P_l)
        
        return P

    def optimize_local(S_init, W_init, max_iters=1000, verbose=False):
        """
        Local hill climbing search.
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
        # We enforce S[-1] = 1.0 in the loop if n>1
        if len(S) > 1:
            S = S / S[-1]
            S[-1] = 1.0
            
        n = len(S)
        current_P = evaluate(S, W)
        best_P = current_P
        best_S = S.copy()
        best_W = W.copy()
        
        # Initial step sizes
        step_s = 0.1
        step_w = 0.1
        
        for i in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            # Generate perturbation
            S_new = S.copy()
            W_new = W.copy()
            
            # Perturb Support
            # Keep S[0] >= 0, S[-1] == 1.
            # Perturb S[1:-1] (and S[0] if we allow it to move, but keep sorted)
            # Actually, let's perturb all S[0:-1] and keep S[-1]=1.
            if n > 1:
                noise_s = np.random.randn(n - 1) * step_s
                S_new[0:n-1] += noise_s
                # Clip to [0, 1]
                S_new[0:n-1] = np.clip(S_new[0:n-1], 0.0, 1.0)
                # Sort to maintain order
                S_new[0:n-1] = np.sort(S_new[0:n-1])
            else:
                # n=1: X=0 always? P=0. Not useful.
                pass
                
            # Perturb Weights
            noise_w = np.random.randn(n) * step_w
            W_new += noise_w
            W_new = np.clip(W_new, 0.0, None)
            W_sum = np.sum(W_new)
            if W_sum < 1e-9:
                continue
            W_new /= W_sum
            
            # Evaluate
            P_new = evaluate(S_new, W_new)
            
            # Accept if better (Hill Climbing)
            if P_new > current_P:
                current_P = P_new
                S = S_new
                W = W_new
                if P_new > best_P:
                    best_P = P_new
                    best_S = S_new.copy()
                    best_W = W_new.copy()
            else:
                # Simulated annealing style? 
                # To keep it simple and robust, we just decay steps.
                # We might get stuck, but we do multiple restarts.
                pass
                
            # Decay
            step_s *= 0.995
            step_w *= 0.995
            
        return best_P, best_S, best_W

    def polish_gradient(S_init, W_init, max_iters=200, lr=0.01):
        """
        Gradient ascent polishing step.
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
            
            # Compute gradient
            dP_dS, dP_dW = compute_gradient(S, W)
            
            # Update S (keep S[-1] = 1, S[0] >= 0, sorted)
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
                # Decay learning rate slowly
                lr *= 0.99
            else:
                # Reduce learning rate if no improvement
                lr *= 0.5
                if lr < 1e-8:
                    break
        
        return best_P, best_S, best_W

    def compute_gradient(S, W, eps=1e-6):
        """
        Compute numerical gradient of P with respect to S and W.
        Returns (dP/dS, dP/dW) as 1-D arrays.
        """
        n = len(S)
        P0 = evaluate(S, W)
        
        dP_dS = np.zeros(n)
        dP_dW = np.zeros(n)
        
        # Gradient w.r.t. S
        for i in range(n):
            S_plus = S.copy()
            S_plus[i] += eps
            # Keep sorted: if we perturb, re-sort but track which index moved
            # Actually for gradient, we need to be careful about sorting
            # Let's just perturb and evaluate (the function handles unsorted via broadcasting)
            P_plus = evaluate(S_plus, W)
            
            S_minus = S.copy()
            S_minus[i] -= eps
            P_minus = evaluate(S_minus, W)
            
            dP_dS[i] = (P_plus - P_minus) / (2 * eps)
        
        # Gradient w.r.t. W
        for i in range(n):
            W_plus = W.copy()
            W_plus[i] += eps
            W_plus = np.clip(W_plus, 0, None)
            W_sum = np.sum(W_plus)
            if W_sum > 0:
                W_plus /= W_sum
            P_plus = evaluate(S, W_plus)
            
            W_minus = W.copy()
            W_minus[i] -= eps
            W_minus = np.clip(W_minus, 0, None)
            W_sum = np.sum(W_minus)
            if W_sum > 0:
                W_minus /= W_sum
            P_minus = evaluate(S, W_minus)
            
            dP_dW[i] = (P_plus - P_minus) / (2 * eps)
        
        return dP_dS, dP_dW

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
        global_best_S = S2
        global_best_W = W2

    # Multi-start local search for n = 2 to 6
    for n in range(2, 7):
        if time.time() > deadline - 30.0:
            break
            
        # Number of restarts depends on remaining time
        # Each restart takes ~max_iters * eval_cost.
        # eval_cost is very low for small n.
        # Let's try 20 restarts for small n, fewer for larger n.
        n_restarts = 20 if n <= 3 else 10
        
        for r in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
            
            if r == 0:
                # Deterministic init based on n
                if n == 2:
                    S_init = np.array([0.0, 1.0])
                    W_init = np.array([p2, 1.0 - p2])
                elif n == 3:
                    # Try 0, 0.5, 1
                    S_init = np.array([0.0, 0.5, 1.0])
                    W_init = np.array([0.4, 0.2, 0.4])
                elif n == 4:
                    S_init = np.array([0.0, 0.33, 0.66, 1.0])
                    W_init = np.array([0.3, 0.2, 0.2, 0.3])
                else:
                    S_init = np.linspace(0.0, 1.0, n)
                    W_init = np.ones(n) / n
            else:
                # Random init
                # Support: sorted random in [0, 1], last is 1
                S_init = np.sort(np.random.uniform(0.0, 1.0, n - 1))
                S_init = np.append(S_init, 1.0)
                # Weights: Dirichlet
                W_init = np.random.dirichlet(np.ones(n))
                
            P, S, W = optimize_local(S_init, W_init, max_iters=500)
            
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
        
    return global_best_S.astype(np.float64), global_best_W.astype(np.float64)

# EVOLVE-BLOCK-END
