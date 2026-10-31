import numpy as np
import random
import time

N = 50

def is_sidon(S):
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def greedy_sidon(N, restarts=200):
    best_set = []
    best_size = 0
    
    for _ in range(restarts):
        indices = list(range(N))
        random.shuffle(indices)
        S = []
        for i in indices:
            # Check if adding i keeps S Sidon
            # Optimization: only check sums involving i
            new_sums = set()
            valid = True
            for j in S:
                s = i + j
                if s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            if valid:
                # Check against existing sums
                # We need to store existing sums to check against
                # This greedy approach is O(N^3) per restart if not careful.
                # For N=50, it's fine.
                S.append(i)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
        # Early exit if we found a very large set (theoretical max ~ sqrt(N))
        if best_size >= 8: 
            break
            
    return best_set

def optimize_weights(S, time_limit=5.0):
    K = len(S)
    if K == 0:
        return np.zeros(N)
    
    # Map indices in S to 0..K-1
    h = np.ones(K) / K
    
    def ratio(h):
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        # Normalize to sum 1 for ratio calculation? 
        # Ratio is scale invariant.
        l2_sq = np.sum(h**2)
        
        # Compute peaks
        # h is defined on S.
        # Convolution peak for sum k is sum_{i+j=k} h_i h_j
        # We only care about the max peak.
        # Sums are distinct.
        # Peak for i=j is h_i^2
        # Peak for i!=j is 2*h_i*h_j
        max_peak = 0.0
        # Diagonal
        max_peak = max(max_peak, np.max(h**2))
        # Off-diagonal
        if K > 1:
            # Max of 2*h_i*h_j
            # This is 2 * max(h_i*h_j)
            # Which is 2 * (max(h) * second_max(h))? No, just max product.
            # Since h_i > 0, max product is product of two largest?
            # Not necessarily, but close.
            # Let's just compute all pairs for accuracy in optimization loop?
            # K is small (~7). K^2 is 49. Cheap.
            for i in range(K):
                for j in range(i+1, K):
                    p = 2 * h[i] * h[j]
                    if p > max_peak:
                        max_peak = p
        
        if max_peak == 0:
            return 0.0
        return l2_sq / max_peak

    start_time = time.time()
    best_ratio = ratio(h)
    
    # Simple local search
    while time.time() - start_time < time_limit:
        # Perturb
        h_new = h.copy()
        # Randomly pick an index to decrease and one to increase
        i_dec = random.randint(0, K-1)
        i_inc = random.randint(0, K-1)
        if i_dec == i_inc:
            continue
            
        # Move mass from i_dec to i_inc
        # Ensure h remains non-negative
        # Try a small step
        step = random.uniform(0.01, 0.1) * h[i_dec]
        h_new[i_dec] -= step
        h_new[i_inc] += step
        
        if h_new[i_dec] < 0:
            continue
            
        # Renormalize? 
        # The ratio is scale invariant, so we don't strictly need sum=1, 
        # but it helps numerical stability.
        # Let's keep sum constant.
        # Actually, my ratio function doesn't normalize.
        # r(h) = sum(h^2) / max_peak(h)
        # If we scale h by c, sum(h^2) scales by c^2, max_peak scales by c^2. Ratio invariant.
        # So we don't need to renormalize.
        
        r_new = ratio(h_new)
        if r_new > best_ratio:
            h = h_new
            best_ratio = r_new
        else:
            # Simulated annealing style? Or just reject.
            # For now, reject.
            pass
            
    # Construct full vector
    h_full = np.zeros(N)
    for idx, val in zip(S, h):
        h_full[idx] = val
        
    return h_full

def construct():
    # Find largest Sidon set
    S = greedy_sidon(N, restarts=100)
    if not S:
        S = [0]
        
    # Optimize weights
    h = optimize_weights(S, time_limit=10.0)
        
    return list(h)

def compute_ratio(h):
    h = np.asarray(h)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_peak = np.max(conv)
    if max_peak == 0:
        return 0.0
    return l2_sq / (l1 * max_peak)
