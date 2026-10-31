import numpy as np
import time
from scipy.optimize import minimize
import random

N = 50

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for h >= 0.
    """
    h = np.asarray(h, dtype=np.float64)
    # Ensure non-negativity
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    
    l2_sq = np.sum(h * h)
    
    # Efficient autoconvolution using FFT
    # h*h has length 2N-1
    # Zero pad to power of 2 for FFT efficiency
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    hh = np.fft.irfft(H * H, n=n_fft)
    hh_max = np.max(hh)
    
    if hh_max < 1e-300:
        return 0.0
        
    return l2_sq / (l1 * hh_max)

def optimize_weights(support, rng, time_limit=0.1):
    """
    Optimize weights for a given support to maximize the ratio.
    """
    K = len(support)
    if K == 0:
        return 0.0, np.zeros(N)
    
    # We optimize in the space of weights summing to 1.
    # Use Nelder-Mead on the first K-1 weights, derive the K-th.
    # Or just use Nelder-Mead on K weights and project?
    # Nelder-Mead doesn't handle constraints well.
    # Let's use a simple gradient-free search on the simplex.
    # Or use `scipy.optimize.minimize` with `bounds` and a penalty for sum != 1?
    
    # Better: Parameterize weights by $K-1$ free variables $u_i$, 
    # $w_i = \exp(u_i) / \sum \exp(u_j)$. This ensures positivity and sum 1.
    # Then maximize ratio.
    
    def neg_ratio_log_w(u):
        # u is vector of length K
        # Softmax to get weights
        u_max = np.max(u)
        exp_u = np.exp(u - u_max)
        w = exp_u / np.sum(exp_u)
        
        h = np.zeros(N)
        h[support] = w
        return -compute_ratio(h)
    
    # Initial guess: uniform
    u0 = np.zeros(K)
    
    best_score = 0.0
    best_w = np.ones(K) / K
    
    # Multi-start
    n_starts = 5
    start_time = time.time()
    
    for i in range(n_starts):
        if time.time() - start_time > time_limit:
            break
            
        if i == 0:
            u_init = np.zeros(K)
        else:
            u_init = rng.normal(0, 0.5, K)
            
        res = minimize(neg_ratio_log_w, u_init, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.fun < 0: # If improved
            u_opt = res.x
            u_max = np.max(u_opt)
            exp_u = np.exp(u_opt - u_max)
            w_opt = exp_u / np.sum(exp_u)
            
            h_opt = np.zeros(N)
            h_opt[support] = w_opt
            score_opt = compute_ratio(h_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_w = w_opt
                
    h_best = np.zeros(N)
    h_best[support] = best_w
    return best_score, h_best

def greedy_sidon_set(N, K, rng, seed=0):
    """
    Greedily construct a set of size K that minimizes max collision count.
    """
    S = [0]
    # Collision counts for current S
    # r[k] = number of pairs (i,j) in S x S such that i+j=k
    # We can maintain this incrementally.
    
    # For simplicity and speed with small K, just recompute or update carefully.
    # K is small (~10). N is 50.
    
    for _ in range(K - 1):
        best_j = -1
        best_max_coll = np.inf
        best_sum_sq = np.inf
        
        # Candidates: 0..N-1 not in S
        candidates = [j for j in range(N) if j not in S]
        
        for j in candidates:
            S_new = S + [j]
            # Calculate collision counts for S_new
            # Pairs (i,j) in S_new x S_new
            # We can just compute the vector of sums
            sums = np.array([a + b for a in S_new for b in S_new])
            # Count occurrences
            vals, counts = np.unique(sums, return_counts=True)
            max_coll = np.max(counts)
            sum_sq = np.sum(counts * counts)
            
            if max_coll < best_max_coll or (max_coll == best_max_coll and sum_sq < best_sum_sq):
                best_max_coll = max_coll
                best_sum_sq = sum_sq
                best_j = j
        
        if best_j != -1:
            S.append(best_j)
        else:
            break
            
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_h = None
    best_score = -np.inf
    
    # Strategy:
    # 1. Try different support sizes K.
    # 2. For each K, generate several greedy Sidon-like sets.
    # 3. Optimize weights for each set.
    # 4. Keep the best.
    
    # Theoretical max K for Sidon set in N=50 is around 7-8.
    # Let's try K from 2 to 15.
    
    for K in range(2, 16):
        if time.time() > deadline - 10.0:
            break
            
        # Generate 5 different greedy sets for this K
        for trial in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Randomize the starting point or tie-breaking to get different sets
            # The greedy algorithm is deterministic if we don't randomize.
            # Let's add some randomness to the candidate selection or initial set.
            
            # Simple randomization: shuffle candidates?
            # Or just start with a random element instead of 0?
            # Greedy from 0 is good. Let's try random starts.
            
            start_elem = rng.integers(0, N)
            # Modify greedy to start with start_elem?
            # For now, let's just use the standard greedy but maybe randomize the order of evaluation in ties?
            # Actually, let's just generate a random set of size K and then "refine" it?
            # No, the directive says "greedy heuristic".
            
            # Let's use the standard greedy but with a random initial set of size 1.
            S = [start_elem]
            
            # We need to adapt greedy_sidon_set to accept an initial set.
            # Let's just inline it here for flexibility.
            
            for _ in range(K - 1):
                best_j = -1
                best_max_coll = np.inf
                best_sum_sq = np.inf
                
                candidates = [j for j in range(N) if j not in S]
                
                # Shuffle candidates to break ties randomly
                rng.shuffle(candidates)
                
                for j in candidates:
                    S_new = S + [j]
                    sums = np.array([a + b for a in S_new for b in S_new])
                    vals, counts = np.unique(sums, return_counts=True)
                    max_coll = np.max(counts)
                    sum_sq = np.sum(counts * counts)
                    
                    if max_coll < best_max_coll or (max_coll == best_max_coll and sum_sq < best_sum_sq):
                        best_max_coll = max_coll
                        best_sum_sq = sum_sq
                        best_j = j
                
                if best_j != -1:
                    S.append(best_j)
                else:
                    break
                    
            if len(S) < K:
                continue
                
            support = np.array(S)
            
            # Optimize weights
            score, h = optimize_weights(support, rng, time_limit=0.2)
            
            if score > best_score:
                best_score = score
                best_h = h
                
    # If no good set found, fallback to delta
    if best_h is None:
        best_h = np.zeros(N)
        best_h[0] = 1.0
        best_score = compute_ratio(best_h)
        
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
