import numpy as np
import time
from itertools import combinations
import random

N = 50

def compute_ratio(h):
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_h = np.ones(N)
    best_score = compute_ratio(best_h)

    # --- Phase 1: Find optimal support sizes and structures ---
    # We know R is scale invariant. Let's normalize L1=1.
    # If we restrict to a support S of size K with equal weights 1/K:
    # R = (K * (1/K)^2) / (1 * max_conv)
    # max_conv for equal weights = max_k count_k / K^2
    # R = (1/K) / (max_count / K^2) = K / max_count
    # So we want to maximize K / max_count.
    # max_count is the maximum number of pairs (i,j) in S such that i-j = k.
    # This is the "maximum frequency of differences".
    # For a Golomb ruler, max_count = 1, so R = K.
    # For general sets, R < K.
    
    # Let's search for supports S that maximize K / max_count.
    # Then we can optimize weights on that support.
    
    def get_max_diff_count(S):
        # S is a list of indices
        S = sorted(S)
        counts = np.zeros(2*N-1, dtype=np.int32)
        for i in range(len(S)):
            for j in range(len(S)):
                diff = S[i] - S[j]
                counts[diff + (N-1)] += 1
        return np.max(counts)

    def get_support_score(S):
        K = len(S)
        if K == 0:
            return 0.0
        max_c = get_max_diff_count(S)
        if max_c == 0:
            return 0.0
        return K / max_c

    # Search for good supports
    # We can try random sets of size K.
    # K ranges from 1 to N.
    # For small K, K/max_count is high if max_count=1.
    # Max K for max_count=1 is 11 (Golomb ruler of 11 marks).
    # For K=11, R=11.
    # For K=10, R=10.
    # For K=12, max_count must be at least 2. R <= 6.
    # So the optimal support is likely a Golomb ruler of size 11, or close to it.
    # Let's verify: if we have a Golomb ruler of size 11, R=11.
    # If we have size 12, max_count >= 2, R <= 6.
    # So size 11 is likely the best support size for equal weights.
    
    # Let's find a good Golomb ruler of size 11 within 50 indices.
    # And also try size 10, 12, etc.
    
    # We will do a local search on the support.
    # Start with a random set of size K.
    # Swap elements to maximize K/max_count.
    
    rng = random.Random(42)
    
    # We'll try several K values.
    # K=11 is the peak of K/max_count for max_count=1.
    # Let's focus on K=11 and K=10.
    
    best_support = None
    best_support_score = 0.0
    
    # Try to find a good support
    for K in [11, 10, 9, 12, 13]:
        if time.time() > deadline - 60:
            break
        
        # Random restarts for this K
        for restart in range(500):
            if time.time() > deadline - 60:
                break
            
            # Random initialization
            S = list(set(rng.sample(range(N), K)))
            # Ensure size K
            while len(S) < K:
                S.append(rng.randint(0, N-1))
            S = sorted(list(set(S))[:K])
            
            # Local search: swap elements
            # We want to minimize max_count.
            # If max_count > 1, try to fix collisions.
            
            cur_score = get_support_score(S)
            if cur_score > best_support_score:
                best_support_score = cur_score
                best_support = S.copy()
            
            # Local search
            improved = True
            steps = 0
            while improved and steps < 100:
                improved = False
                steps += 1
                max_c = get_max_diff_count(S)
                if max_c == 1:
                    # Can't do better than K/1
                    break
                
                # Try swapping each element with a random new element
                # Or try moving an element to reduce max_count.
                # Let's try random swaps.
                for i in range(K):
                    if time.time() > deadline - 60:
                        break
                    old_val = S[i]
                    # Try moving to a random new position
                    new_val = rng.randint(0, N-1)
                    if new_val in S:
                        continue
                    
                    S[i] = new_val
                    S = sorted(list(set(S)))
                    if len(S) != K:
                        # Duplicates removed? No, set ensures uniqueness.
                        # But if new_val was already in S, we skipped.
                        # So len should be K.
                        pass
                    
                    new_score = get_support_score(S)
                    if new_score > cur_score:
                        cur_score = new_score
                        improved = True
                        if cur_score > best_support_score:
                            best_support_score = cur_score
                            best_support = S.copy()
                    else:
                        S[i] = old_val
                        S = sorted(S)
                        
    # Now we have a good support.
    # But we can do better than equal weights.
    # Let's optimize weights on the best support found.
    
    if best_support is None:
        # Fallback
        best_support = list(range(11))
        
    # Sort support
    best_support = sorted(best_support)
    
    # --- Phase 2: Optimize weights on the best support ---
    # We have K active indices.
    # h_i = 0 for i not in support.
    # Let w_k be the weight at support index k.
    # Normalize sum w = 1.
    # Maximize sum w_k^2 / max_conv.
    
    # We can use Nelder-Mead or simple coordinate ascent.
    # Since K is small (e.g., 11), we can do a fine grid or random search.
    
    # Let's try a local search on the weights.
    # Start with equal weights.
    K_sup = len(best_support)
    w = np.ones(K_sup) / K_sup
    
    def eval_weights(w):
        h = np.zeros(N)
        for i, idx in enumerate(best_support):
            h[idx] = w[i]
        return compute_ratio(h)
    
    cur_w_score = eval_weights(w)
    if cur_w_score > best_score:
        best_score = cur_w_score
        best_h = np.zeros(N)
        for i, idx in enumerate(best_support):
            best_h[idx] = w[i]
    
    # Local search on weights
    # We will try moving mass between pairs of weights.
    # Or use a simple gradient-free method.
    # Let's try random perturbations of w.
    
    rng_w = random.Random(123)
    
    for iter in range(5000):
        if time.time() > deadline:
            break
        
        # Random perturbation
        new_w = w.copy()
        # Pick two random indices
        i, j = rng_w.sample(range(K_sup), 2)
        # Move a small amount of mass
        delta = rng_w.uniform(-0.05, 0.05)
        new_w[i] += delta
        new_w[j] -= delta
        
        # Project to simplex
        if np.min(new_w) < 0:
            # Simple projection: clip and renormalize?
            # Or just reject if negative.
            continue
        
        s = np.sum(new_w)
        new_w = new_w / s
        
        new_score = eval_weights(new_w)
        if new_score > cur_w_score:
            cur_w_score = new_score
            w = new_w.copy()
            if cur_w_score > best_score:
                best_score = cur_w_score
                best_h = np.zeros(N)
                for i, idx in enumerate(best_support):
                    best_h[idx] = w[i]
    
    # --- Phase 3: Try to add more elements or change support slightly ---
    # Maybe the optimal support isn't a pure Golomb ruler.
    # Let's do a global search over a few random supports and optimize weights.
    
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Random support size
        K_rand = rng.randint(8, 15)
        S_rand = list(set(rng.sample(range(N), K_rand)))
        while len(S_rand) < K_rand:
            S_rand.append(rng.randint(0, N-1))
        S_rand = sorted(list(set(S_rand))[:K_rand])
        
        if len(S_rand) != K_rand:
            continue
            
        # Start with equal weights
        w_rand = np.ones(K_rand) / K_rand
        
        def eval_w_rand(w):
            h = np.zeros(N)
            for i, idx in enumerate(S_rand):
                h[idx] = w[i]
            return compute_ratio(h)
        
        cur_score_rand = eval_w_rand(w_rand)
        
        # Local search on weights
        for w_iter in range(500):
            if time.time() > deadline:
                break
            i, j = rng.sample(range(K_rand), 2)
            delta = rng.uniform(-0.05, 0.05)
            new_w = w_rand.copy()
            new_w[i] += delta
            new_w[j] -= delta
            if np.min(new_w) < 0:
                continue
            s = np.sum(new_w)
            new_w = new_w / s
            new_score = eval_w_rand(new_w)
            if new_score > cur_score_rand:
                cur_score_rand = new_score
                w_rand = new_w.copy()
        
        if cur_score_rand > best_score:
            best_score = cur_score_rand
            best_h = np.zeros(N)
            for i, idx in enumerate(S_rand):
                best_h[idx] = w_rand[i]
    
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
