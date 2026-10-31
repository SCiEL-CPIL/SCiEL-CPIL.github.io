import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 50

def _exact_ratio(h):
    """
    Compute the exact ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    """
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * max_conv)

def _optimize_weights_on_support(indices, deadline):
    """
    Given a fixed set of non-zero indices, optimize the weights to maximize the ratio.
    We fix L1=1 by normalizing. The ratio becomes sum(h^2) / max(conv).
    We minimize -log(sum(h^2)/max(conv)) using L-BFGS-B on log-weights.
    """
    k = len(indices)
    if k == 0:
        return None, 0.0
    
    # Initial guess: equal weights
    x0 = np.zeros(k)
    
    def objective(x, deadline):
        if time.time() > deadline:
            return 1e10
        # Softmax to ensure positive and sum=1
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        h_sub = exp_x / np.sum(exp_x)
        
        # Place into full vector
        h = np.zeros(N)
        h[indices] = h_sub
        
        l2_sq = np.sum(h**2)
        if l2_sq < 1e-15:
            return 1e10
            
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-15:
            return 1e10
            
        # We want to maximize l2_sq / max_conv
        # Minimize log(max_conv) - log(l2_sq)
        return np.log(max_conv) - np.log(l2_sq)

    try:
        res = minimize(objective, x0, args=(deadline,), method='L-BFGS-B', 
                       options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6})
        x_opt = res.x
        x_shifted = x_opt - np.max(x_opt)
        exp_x = np.exp(x_shifted)
        h_sub = exp_x / np.sum(exp_x)
        
        h = np.zeros(N)
        h[indices] = h_sub
        score = _exact_ratio(h)
        return h, score
    except Exception:
        return None, 0.0

def _sidon_penalty(indices):
    """
    Count the number of duplicate differences between pairs in indices.
    A Sidon set has all differences unique.
    """
    k = len(indices)
    if k < 2:
        return 0
    diffs = []
    for i in range(k):
        for j in range(i+1, k):
            diffs.append(abs(indices[i] - indices[j]))
    diffs.sort()
    penalty = 0
    for i in range(1, len(diffs)):
        if diffs[i] == diffs[i-1]:
            penalty += 1
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_h = None
    best_score = 0.0
    
    # Strategy: Simulated Annealing on the Support Set
    # We search for a subset of indices of size K that forms a good Sidon-like set.
    # Then we optimize weights on that support.
    
    # Target size K. 
    # Uniform (K=50) gives ~1.0.
    # Sparse (K~10-15) often gives better ratios due to reduced interference.
    # Let's try a range of K values.
    
    np.random.seed(42)
    
    # Pre-compute some random starting supports
    # We will run a SA loop for each K
    
    k_values = [10, 12, 15, 20]
    
    for K in k_values:
        if time.time() > deadline - 10:
            break
            
        # Initialize support
        current_support = np.random.choice(N, K, replace=False)
        current_support.sort()
        
        # Evaluate initial support
        h_init, score_init = _optimize_weights_on_support(current_support, deadline)
        if h_init is not None:
            best_h = h_init
            best_score = score_init
        
        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.99
        max_steps = 2000
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            # Propose a move: swap one element in support with one outside
            if K >= N:
                break
                
            # Pick an index to remove
            rem_idx = np.random.randint(0, K)
            # Pick an index to add (not in current support)
            # Efficient way: generate random candidates until one is not in support
            # Since K is small relative to N, this is fast
            while True:
                add_idx = np.random.randint(0, N)
                if add_idx not in current_support:
                    break
            
            new_support = np.delete(current_support, rem_idx)
            new_support = np.append(new_support, add_idx)
            new_support.sort()
            
            # Evaluate new support
            h_new, score_new = _optimize_weights_on_support(new_support, deadline)
            
            if h_new is None:
                continue
                
            # Calculate acceptance
            # We want to maximize score.
            # delta = score_new - score_current
            # Accept if delta > 0 or with probability exp(delta / T)
            
            # Use a heuristic: if the new support has fewer Sidon collisions, it's structurally better.
            # But score is the ultimate metric.
            
            delta = score_new - best_score # Compare against global best for simplicity in SA? 
            # Standard SA compares against current state. Let's track current score.
            # Actually, _optimize_weights is expensive. Comparing to global best is okay for a greedy-ish SA.
            # Let's do standard SA against the score of the current support.
            # We need to store current_score.
            
            # Re-evaluating current score is expensive. Let's assume the last evaluated score for current_support.
            # We didn't store it. Let's just do a greedy hill-climbing with random restarts if SA is too complex to track state.
            # Actually, let's simplify: Greedy local search with random restarts.
            
            # Let's restart the logic for clarity and speed:
            # 1. Generate random support
            # 2. Optimize weights
            # 3. Try swapping 1 element, if score improves, keep it.
            # 4. Repeat until no improvement or time out.
            
            # This is effectively the loop we are in, but let's make it a proper local search.
            
            # For this specific step in the loop, let's just do a simple random restart strategy instead of complex SA state tracking
            # to ensure robustness within the time limit.
            
            # If we are here, we are in the K loop. Let's do 5 random restarts for this K.
            # The code above already did 1. Let's do 4 more.
            # Wait, the loop structure is a bit mixed. Let's refactor construct() to be cleaner.
            
            # Break out and use a cleaner structure below.
            break 
            
    # --- Refactored Construct Logic ---
    # The above loop was a draft. Let's implement a clean multi-restart local search.
    
    best_h = None
    best_score = 0.0
    
    k_values = [8, 10, 12, 15, 20]
    
    for K in k_values:
        if time.time() > deadline - 20:
            break
            
        num_restarts = 10
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random support
            support = np.random.choice(N, K, replace=False)
            support.sort()
            
            # Optimize weights
            h_opt, score_opt = _optimize_weights_on_support(support, deadline)
            
            if h_opt is not None:
                if score_opt > best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
                
                # Local Search: Try swapping one index
                current_support = support.copy()
                current_score = score_opt
                current_h = h_opt
                
                local_improved = True
                while local_improved and time.time() < deadline - 2:
                    local_improved = False
                    # Try all possible swaps? K*(N-K) swaps.
                    # If K=10, N=50, 10*40=400 swaps.
                    # Each swap requires weight optimization (~10-50ms).
                    # 400 * 0.05s = 20s. Too slow for a single restart.
                    # We need a faster evaluation or fewer swaps.
                    
                    # Heuristic: Only try swaps that improve Sidon property?
                    # Or just random swaps?
                    
                    # Let's try a limited number of random swaps per iteration.
                    for _ in range(5): # 5 random swaps
                        if time.time() > deadline - 2:
                            break
                            
                        rem_idx = np.random.randint(0, K)
                        while True:
                            add_idx = np.random.randint(0, N)
                            if add_idx not in current_support:
                                break
                            
                        new_support = np.delete(current_support, rem_idx)
                        new_support = np.append(new_support, add_idx)
                        new_support.sort()
                        
                        h_new, score_new = _optimize_weights_on_support(new_support, deadline)
                        
                        if h_new is not None and score_new > current_score:
                            current_support = new_support
                            current_score = score_new
                            current_h = h_new
                            local_improved = True
                            
                            if current_score > best_score:
                                best_score = current_score
                                best_h = current_h.copy()
    
    # Fallback: Uniform
    if best_h is None:
        best_h = np.ones(N) / N
        best_score = _exact_ratio(best_h)
        
    # Final Polish: Optimize weights on the best support found
    # We need to recover the support from best_h
    if best_h is not None:
        support = np.where(best_h > 1e-8)[0]
        if len(support) > 0:
            h_polished, score_polished = _optimize_weights_on_support(support, deadline)
            if h_polished is not None and score_polished > best_score:
                best_h = h_polished
                best_score = score_polished

    # Ensure non-negative and return list
    best_h = np.maximum(best_h, 0)
    return best_h.tolist()

if __name__ == "__main__":
    # Quick test
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s")
    print(f"Score: {_exact_ratio(np.array(res)):.4f}")
