import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations
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
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    hh = np.fft.irfft(H * H, n=n_fft)
    hh_max = np.max(hh)
    
    if hh_max < 1e-300:
        return 0.0
        
    return l2_sq / (l1 * hh_max)

def optimize_weights_for_support(support, rng, time_limit=0.5):
    """
    Given a fixed support set, optimize the weights to maximize the ratio.
    We use Nelder-Mead on the log-weights to handle non-negativity naturally,
    or just project onto the simplex.
    Since K is small, we can afford a precise search.
    """
    K = len(support)
    if K == 0:
        return 0.0, np.zeros(N)
    
    # Initial guess: uniform weights
    w0 = np.ones(K) / K
    
    def neg_ratio_logw(log_w):
        # Convert log weights to actual weights
        w = np.exp(log_w)
        # Normalize to sum to 1 (scale invariance)
        s = np.sum(w)
        if s < 1e-100:
            return 1e10
        w_norm = w / s
        
        h = np.zeros(N)
        h[support] = w_norm
        return -compute_ratio(h)
    
    # Start from log(1/K)
    log_w0 = np.full(K, np.log(1.0/K))
    
    # Run Nelder-Mead
    # We run multiple restarts with different initial conditions to avoid local minima
    best_score = -np.inf
    best_w = w0
    
    # Time check for this specific optimization
    start_opt = time.time()
    
    # Restart 1: Uniform
    try:
        res = minimize(neg_ratio_logw, log_w0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        score = -res.fun
        if score > best_score:
            best_score = score
            best_w = np.exp(res.x) / np.sum(np.exp(res.x))
    except Exception:
        pass
    
    # Restart 2: Random initialization
    if time.time() - start_opt < time_limit:
        for _ in range(5):
            w_rand = rng.random(K)
            w_rand /= np.sum(w_rand)
            log_w_rand = np.log(w_rand + 1e-10)
            try:
                res = minimize(neg_ratio_logw, log_w_rand, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_w = np.exp(res.x) / np.sum(np.exp(res.x))
            except Exception:
                pass
            if time.time() - start_opt > time_limit:
                break
    
    h_best = np.zeros(N)
    h_best[support] = best_w
    return best_score, h_best

def generate_sidon_candidates(rng, num_candidates=200):
    """
    Generate candidate support sets that are likely to be good.
    Strategy:
    1. Start with small random sets.
    2. Use a greedy heuristic to build Sidon sets.
    3. Randomly mutate good sets.
    """
    candidates = []
    
    # 1. Greedy Sidon-like construction
    # We try to build sets of size K=5..15
    for K in range(5, 16):
        for _ in range(20): # 20 random starts for each K
            support = set()
            # Start with a random element
            support.add(rng.integers(0, N))
            while len(support) < K:
                # Try to add a random element that minimizes collisions
                best_elem = None
                best_collisions = np.inf
                for _ in range(50): # Sample 50 candidates
                    cand = rng.integers(0, N)
                    if cand in support:
                        continue
                    new_support = support | {cand}
                    # Count collisions in autoconvolution of indicator function
                    # This is a proxy for the denominator
                    # For a set S, the number of pairs (i,j) in S such that i+j = k
                    # We want to minimize the max frequency of any sum.
                    # Quick check: compute the sum set
                    sums = [a+b for a in new_support for b in new_support if a <= b]
                    # Count frequencies
                    if not sums:
                        freqs = {}
                    else:
                        freqs = {}
                        for s in sums:
                            freqs[s] = freqs.get(s, 0) + 1
                    max_freq = max(freqs.values()) if freqs else 0
                    # We want low max_freq
                    if max_freq < best_collisions:
                        best_collisions = max_freq
                        best_elem = cand
                if best_elem is not None:
                    support.add(best_elem)
                else:
                    # Just pick random if no improvement found
                    support.add(rng.integers(0, N))
            candidates.append(tuple(sorted(support)))
            
    # 2. Random sets
    for _ in range(100):
        K = rng.integers(5, 16)
        support = set(rng.choice(N, size=K, replace=False))
        candidates.append(tuple(sorted(support)))
        
    # 3. Mutations of the best found so far? (We don't know best yet, so just return these)
    # Deduplicate
    unique_candidates = list(set(candidates))
    return unique_candidates

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_score = -np.inf
    best_h = np.zeros(N)
    
    # Phase 1: Generate a diverse pool of structured support sets
    candidates = generate_sidon_candidates(rng, num_candidates=300)
    
    # Shuffle to ensure we don't get stuck in a bad part of the list if we run out of time
    rng.shuffle(candidates)
    
    # Phase 2: Evaluate each candidate support with weight optimization
    # We prioritize supports of intermediate size (8-12) as they often balance L2 and L1 well
    # and have enough structure to minimize collisions.
    
    # Sort candidates by a quick heuristic: number of distinct sums (higher is better for Sidon)
    # Actually, let's just iterate. The optimization is the expensive part.
    
    for sup_tuple in candidates:
        if time.time() > deadline - 5.0:
            break
            
        support = list(sup_tuple)
        if len(support) < 2:
            continue
            
        # Quick rejection: if the support is "dense" in a way that creates huge collisions
        # We can estimate the "Sidon-ness" quickly
        # But let's just run the optimizer, it's fast for K<15
        
        score, h = optimize_weights_for_support(support, rng, time_limit=0.2)
        
        if score > best_score:
            best_score = score
            best_h = h
    
    # Phase 3: Local refinement around the best support
    if best_score > -np.inf and time.time() < deadline - 10.0:
        # Take the best support found and try adding/removing 1 element
        best_support = list(np.where(best_h > 1e-8)[0])
        
        # Try adding a new element
        for new_elem in range(N):
            if new_elem in best_support:
                continue
            new_support = best_support + [new_elem]
            new_support.sort()
            score, h = optimize_weights_for_support(new_support, rng, time_limit=0.1)
            if score > best_score:
                best_score = score
                best_h = h
                best_support = new_support # Update for potential further additions? 
                # Let's not chain too much to save time, just one step of expansion
                break # Found a better one, break to try others? No, let's try all additions once
                
        # Try removing an element
        for i in range(len(best_support)):
            new_support = best_support[:i] + best_support[i+1:]
            if not new_support:
                continue
            score, h = optimize_weights_for_support(new_support, rng, time_limit=0.1)
            if score > best_score:
                best_score = score
                best_h = h
        
    # Final check
    if best_h is None or np.sum(best_h) == 0:
        # Fallback: uniform distribution
        best_h = np.ones(N) / N
        best_score = compute_ratio(best_h)
        
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
