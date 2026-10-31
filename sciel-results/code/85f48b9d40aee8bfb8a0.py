import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _calculate_ratio(h):
    """
    Calculate the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    h is a numpy array of nonnegative heights.
    """
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
        
    l2_sq = np.sum(h**2)
    
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    
    max_conv = np.max(conv)
    
    if max_conv == 0:
        return 0.0
        
    return l2_sq / (l1 * max_conv)

def _optimize_weights_on_support(support, deadline):
    """
    Optimizes weights for a given support to maximize the ratio.
    support: list of indices
    """
    if not support:
        return None, 0.0
        
    M = len(support)
    
    # Objective: Minimize LSE(conv) - log(sum(h^2))
    # We fix sum(h)=1 by normalizing inside or using a transformation.
    # L-BFGS-B works with bounds. Let's work with h directly and normalize in objective?
    # No, better to optimize h such that sum(h)=1 is a constraint?
    # L-BFGS-B doesn't handle equality constraints well.
    # Let's use the fact that the ratio is scale invariant.
    # We can just optimize h >= 0 and the ratio is invariant.
    # But LSE needs a scale. Let's fix sum(h)=1 by dividing by sum in the objective?
    # That makes it non-smooth if sum hits 0.
    # Better: Parameterize h_i = exp(x_i) / sum(exp(x_j)).
    # This ensures h_i > 0 and sum(h)=1.
    # But we need h_i >= 0. exp is always positive.
    # This removes the non-negativity constraint (it's strictly positive).
    # If the optimal solution has zeros, this might struggle, but for Sidon sets,
    # all weights should be positive (to maximize sum h^2, we want to spread mass, 
    # but to minimize peak, we want to balance. For Sidon, all h_i equal is optimal?).
    # If h_i = 1/M for all i in Sidon set:
    # Ratio = M.
    # If we deviate, sum h^2 increases, but peak might increase more?
    # For Sidon, peak is max(h_i^2, 2 h_i h_j ...).
    # Actually, for a perfect Sidon set, each bin k has at most one pair {i,j}.
    # The bin 2i has h_i^2. The bin i+j has h_i h_j.
    # To minimize max(h_i^2, h_i h_j), we should make all h_i equal.
    # So for a perfect Sidon set, uniform weights are optimal.
    # For near-Sidon sets, optimization helps.
    
    def log_sum_exp(x):
        m = np.max(x)
        if m > 700: # overflow protection
            return m + np.log(np.sum(np.exp(x - m)))
        return np.log(np.sum(np.exp(x)))
    
    def objective_and_grad(x):
        # x are log-weights
        # h = softmax(x)
        # This ensures sum(h)=1 and h>0
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        sum_exp = np.sum(exp_x)
        h = exp_x / sum_exp
        
        # Compute convolution only on support?
        # np.convolve is fast for N=50.
        conv = np.convolve(h, h, mode='full')
        
        # LSE of conv
        # conv is non-negative.
        # We want to minimize max(conv).
        # LSE approximates max.
        # Temperature tau.
        tau = 0.05
        # conv values are small (approx 1/M^2).
        # If M=14, h=1/14, h^2=1/196 ~ 0.005.
        # tau should be smaller than the scale of differences.
        # Let's use a fixed small tau.
        lse_conv = tau * log_sum_exp(conv / tau)
        
        # l2_sq
        l2_sq = np.sum(h**2)
        
        # We want to maximize l2_sq / max_conv.
        # Equivalent to minimize log(max_conv) - log(l2_sq).
        # Approx: lse_conv - log(l2_sq)
        
        if l2_sq < 1e-10:
            return 1e10, np.zeros_like(x)
            
        val = lse_conv - np.log(l2_sq)
        
        # Gradient calculation is complex. Let's use approx_grad or analytic?
        # Analytic:
        # d/dx h = h * (I - h^T 1) ... standard softmax derivative.
        # d/dx LSE(conv) = (1/tau) * (d conv / dx) * softmax(conv/tau)
        # d conv / dx is sparse.
        # Given N=50, numerical gradient is acceptable if we limit iterations.
        # But L-BFGS-B with approx_grad is slow.
        # Let's try to derive or just use finite differences if needed.
        # Actually, for the "perfect Sidon" case, uniform is optimal.
        # So we can just check if uniform is good enough.
        # If the support is not perfect Sidon, optimization might help.
        # Let's just return the value and let L-BFGS-B handle gradients numerically if we don't provide them.
        # scipy.optimize.minimize(method='L-BFGS-B') can use approx_grad=True.
        
        return val, None # Returning None triggers approx_grad

    # Initial guess: uniform weights (x=0)
    x0 = np.zeros(M)
    
    # Bounds: x can be anything (softmax handles it).
    # But to prevent extreme values, we can bound x to [-10, 10].
    bounds = [(-10, 10)] * M
    
    try:
        res = minimize(
            objective_and_grad, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8, 'approx_grad': True}
        )
        
        x_opt = res.x
        # Convert back to h
        x_opt_shifted = x_opt - np.max(x_opt)
        exp_opt = np.exp(x_opt_shifted)
        h_support = exp_opt / np.sum(exp_opt)
        
        # Construct full h
        h_full = np.zeros(N)
        h_full[support] = h_support
        
        score = _calculate_ratio(h_full)
        return h_full, score
        
    except Exception:
        return None, 0.0

def _greedy_sidon_set(N, target_size=None):
    """
    Greedily construct a Sidon set (B2-sequence) in [0, N-1].
    A Sidon set has distinct sums i+j for i<=j.
    """
    S = []
    sums = set()
    
    # Try adding elements in order of "cost" or just sequentially?
    # Sequential greedy: add x if it doesn't create a collision.
    # This might get stuck early.
    # Better: Randomized greedy with restarts.
    
    best_set = []
    best_size = 0
    
    for _ in range(20): # 20 restarts
        np.random.seed(_)
        S = []
        sums = set()
        
        # Randomly shuffle candidates? Or just iterate 0..N-1?
        # Iterating 0..N-1 tends to cluster.
        # Let's pick randomly from available.
        available = list(range(N))
        np.random.shuffle(available)
        
        for x in available:
            # Check if adding x creates a collision
            new_sums = set()
            for y in S:
                new_sums.add(x + y)
            new_sums.add(2*x)
            
            # Check intersection with existing sums
            # Note: (x,x) sum 2x is unique to x.
            # (x,y) sum x+y might collide with (u,v).
            # If any new sum is in `sums`, it's a collision.
            # Wait, if x+y == u+v, and {x,y} != {u,v}, it's a collision.
            # Since we build incrementally, if x+y is in sums, it must come from a previous pair.
            # Is it possible x+y comes from (x, y')? No, y' < y?
            # The set `sums` contains all sums from previous elements.
            # If x+y is in sums, it means there exist u,v in S such that u+v = x+y.
            # Since x is new, u or v must be x? No, x is not in S yet.
            # So u,v are both in S.
            # So {u,v} is a pair in S.
            # And {x,y} is a new pair.
            # So if x+y is in sums, it's a collision.
            
            if new_sums & sums:
                continue
            
            # No collision
            S.append(x)
            sums.update(new_sums)
            
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
        if target_size and best_size >= target_size:
            break
            
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_score = 0.0
    best_h = None
    
    # Candidate 1: Spike (Baseline)
    h_spike = np.zeros(N)
    h_spike[0] = 1.0
    score_spike = _calculate_ratio(h_spike)
    if score_spike > best_score:
        best_score = score_spike
        best_h = h_spike.copy()
        
    # Candidate 2: Uniform (Baseline)
    h_uniform = np.ones(N)
    score_uniform = _calculate_ratio(h_uniform)
    if score_uniform > best_score:
        best_score = score_uniform
        best_h = h_uniform.copy()

    # Strategy: Find a large Sidon set and optimize weights on it.
    
    # 1. Greedy Sidon Set
    sidon_set = _greedy_sidon_set(N)
    if sidon_set:
        h_opt, score_opt = _optimize_weights_on_support(sidon_set, deadline)
        if h_opt is not None and score_opt > best_score:
            best_score = score_opt
            best_h = h_opt.copy()
            
    # 2. Random Sparse Supports
    # Try a few random supports of size ~10-15
    np.random.seed(123)
    for k in [10, 12, 14, 16]:
        if time.time() > deadline:
            break
        for _ in range(5):
            support = np.random.choice(N, k, replace=False)
            h_opt, score_opt = _optimize_weights_on_support(support, deadline)
            if h_opt is not None and score_opt > best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
    # 3. Local Search on the Best Support Found?
    # If we have a good h, we can try to swap elements to improve.
    # But the optimizer already did local optimization on the weights.
    # The support was fixed.
    # We could try to swap an element in the support with one outside.
    
    if best_h is not None:
        # Try a few swaps
        h_current = best_h.copy()
        score_current = best_score
        support_indices = np.where(h_current > 1e-6)[0]
        
        for _ in range(20):
            if time.time() > deadline:
                break
            if len(support_indices) == 0:
                break
            # Pick a random index in support to remove
            i_remove = np.random.choice(support_indices)
            # Pick a random index outside support to add
            outside = np.setdiff1d(np.arange(N), support_indices)
            if len(outside) == 0:
                break
            i_add = np.random.choice(outside)
            
            new_support = list(support_indices)
            new_support.remove(i_remove)
            new_support.append(i_add)
            
            h_opt, score_opt = _optimize_weights_on_support(new_support, deadline)
            if h_opt is not None and score_opt > score_current:
                score_current = score_opt
                best_score = score_current
                best_h = h_opt.copy()
                support_indices = np.where(h_current > 1e-6)[0] # Update support for next swap?
                # Actually, just update the list
                support_indices = np.where(best_h > 1e-6)[0]

    if best_h is None:
        return [1.0] + [0.0] * (N - 1)
        
    return list(best_h)
