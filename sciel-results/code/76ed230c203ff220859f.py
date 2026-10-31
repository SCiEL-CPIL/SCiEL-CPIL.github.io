"""
Strategy: Combinatorial Sparse Support Search with Analytic Weight Optimization.

The problem is to maximize R(h) = ||h||_2^2 / (||h||_1 ||h*h||_inf) for h >= 0.
Since R is scale-invariant, we can fix ||h||_1 = 1.
Let S be the support of h (indices where h_i > 0). Let h_S be the weights on S.
For a fixed support S, we want to maximize (sum_{i in S} h_i^2) / max_{k} (sum_{i,j in S, i-j=k} h_i h_j).
This is equivalent to finding the "Sidon-ness" of the set S, but allowing weights.
For a fixed S, if we set all weights equal, h_i = 1/|S|, then:
Numerator = |S| * (1/|S|)^2 = 1/|S|.
Denominator = (1) * max_k (count_k / |S|^2), where count_k is the number of pairs (i,j) in S with i-j=k.
So R = (1/|S|) / (max_count / |S|^2) = |S| / max_count.
To maximize this, we want a set S of size K such that the maximum number of differences (count_k) is minimized.
Such sets are called Sidon sets or B_2 sets.
For a fixed S, we can potentially do better than equal weights by solving a small convex-like problem or using gradient ascent on the weights.
However, equal weights are a strong baseline.
We will search over subsets S of {0..49} of size K (e.g., K=10..20) to find those with low max_count.
Then, for the best candidates, we will refine the weights using a few steps of gradient ascent or coordinate descent.
"""

import numpy as np
import time
from itertools import combinations
import random

N = 50

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    l2_sq = np.sum(h * h)
    hh = np.convolve(h, h, mode='full')
    hh_max = np.max(hh)
    if hh_max < 1e-300:
        return 0.0
    return l2_sq / (l1 * hh_max)

def get_support_score(S):
    """
    Calculate the score for a support set S assuming equal weights.
    Score = |S| / max_diff_count
    """
    K = len(S)
    if K == 0:
        return 0.0
    # Count differences
    diff_counts = {}
    S_list = list(S)
    for i in range(K):
        for j in range(K):
            d = S_list[i] - S_list[j]
            diff_counts[d] = diff_counts.get(d, 0) + 1
    max_count = max(diff_counts.values())
    return K / max_count

def refine_weights(S, initial_weights=None):
    """
    Refine weights for a fixed support S using gradient ascent on the log-likelihood 
    or similar. Here we use a simple coordinate ascent / power iteration style approach 
    for the generalized eigenvalue problem associated with maximizing the Rayleigh quotient 
    of the quadratic form defined by the convolution matrix restricted to S.
    
    Actually, maximizing (h^T h) / (h^T M h) is not standard.
    Let's stick to a simple heuristic: 
    The denominator is max_k (sum_{i-j=k} h_i h_j).
    Let's try to minimize the max autocorrelation peak.
    This is a min-max problem.
    We can use a simple iterative method:
    1. Start with uniform weights.
    2. Compute autocorrelation A.
    3. Identify the lag k with max A_k.
    4. Adjust weights to reduce A_k?
    
    Alternatively, we can just use a few steps of random perturbation + local search on the weights.
    Given the time, let's try a simple gradient step for the numerator vs denominator.
    Actually, let's just use the equal weight score as a proxy and refine with a few random restarts of weights.
    """
    K = len(S)
    if K == 0:
        return np.zeros(N)
    
    # Initialize with uniform weights
    w = np.ones(K) / K
    
    # Map support indices to positions in h
    h = np.zeros(N)
    for i, idx in enumerate(S):
        h[idx] = w[i]
        
    best_w = w.copy()
    best_score = compute_ratio(h)
    
    # Simple local search: try to shift mass from indices contributing to the peak 
    # to indices not contributing, or just random perturbations.
    # Given the small K, we can do a more thorough search.
    
    # Let's try to optimize the weights directly.
    # We want to maximize R. 
    # Let's use a simple Nelder-Mead on the K weights (normalized).
    # But K can be up to 20. Nelder-Mead in 20D is slow.
    
    # Instead, let's use a heuristic: 
    # The "bad" lags are those with high counts. 
    # If we can make the weights of pairs forming the "bad" lag small, we improve.
    # This is complex. Let's just return the uniform weight solution for now, 
    # as the support selection is the primary driver.
    
    # Actually, let's do 100 random perturbations of the weights.
    rng = np.random.default_rng(42)
    for _ in range(100):
        # Perturb weights
        w_new = w * (1 + rng.normal(0, 0.1, K))
        w_new = np.maximum(w_new, 0)
        s = np.sum(w_new)
        if s < 1e-10:
            continue
        w_new /= s
        
        h_new = np.zeros(N)
        for i, idx in enumerate(S):
            h_new[idx] = w_new[i]
            
        score = compute_ratio(h_new)
        if score > best_score:
            best_score = score
            best_w = w_new.copy()
            
    h_final = np.zeros(N)
    for i, idx in enumerate(S):
        h_final[idx] = best_w[i]
        
    return h_final

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.ones(N)
    best_score = compute_ratio(best_h)
    
    # Search for good supports S
    # We will test sizes K from 5 to 25.
    # For each K, we generate random subsets and keep the best few based on get_support_score.
    # Then we refine the weights for the top candidates.
    
    rng = np.random.default_rng(123)
    
    # Cache best supports found so far
    best_supports = []
    
    for K in range(5, 26):
        if time.time() > deadline - 10:
            break
            
        # Number of trials per K
        # Smaller K: more trials. Larger K: fewer.
        trials = 2000 if K < 10 else 500
        
        # Keep top 5 supports for this K
        top_supports = []
        
        for _ in range(trials):
            S = rng.choice(N, size=K, replace=False)
            S = tuple(sorted(S))
            
            score = get_support_score(S)
            
            # Insert into top_supports
            if len(top_supports) < 5:
                top_supports.append((score, S))
                top_supports.sort(key=lambda x: -x[0])
            else:
                if score > top_supports[-1][0]:
                    top_supports[-1] = (score, S)
                    top_supports.sort(key=lambda x: -x[0])
                    
        # Refine weights for the top supports of this K
        for score, S in top_supports:
            if time.time() > deadline - 5:
                break
                
            h_candidate = refine_weights(S)
            cand_score = compute_ratio(h_candidate)
            
            if cand_score > best_score:
                best_score = cand_score
                best_h = h_candidate.copy()
                
    # Final check: maybe a very sparse set (K=3,4) is better?
    # Let's explicitly check small K.
    for K in range(1, 5):
        if time.time() > deadline - 2:
            break
        # Exhaustive search for small K? 
        # C(50, 4) is 230k. C(50, 3) is 19k.
        # We can do it.
        if K == 1:
            h = np.zeros(N); h[0]=1
            s = compute_ratio(h)
            if s > best_score:
                best_score = s; best_h = h
        elif K == 2:
            for i in range(N):
                for j in range(i+1, N):
                    h = np.zeros(N); h[i]=0.5; h[j]=0.5
                    s = compute_ratio(h)
                    if s > best_score:
                        best_score = s; best_h = h
        elif K == 3:
            # Random sample is probably enough, but let's try a bit more
            for _ in range(1000):
                S = tuple(sorted(rng.choice(N, size=3, replace=False)))
                h = refine_weights(S)
                s = compute_ratio(h)
                if s > best_score:
                    best_score = s; best_h = h
        elif K == 4:
            for _ in range(1000):
                S = tuple(sorted(rng.choice(N, size=4, replace=False)))
                h = refine_weights(S)
                s = compute_ratio(h)
                if s > best_score:
                    best_score = s; best_h = h

    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
